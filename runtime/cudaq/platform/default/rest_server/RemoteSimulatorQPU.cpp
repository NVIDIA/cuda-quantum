/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/RuntimeMLIR.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/NoiseModel.h"
#include "common/RestClient.h"
#include "cudaq.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/spin_op.h"
#include <fstream>
#include <iostream>
#include <spdlog/cfg/env.h>

#include "JsonConvert.h"
#include "nvqir/CircuitSimulator.h"
#include <arpa/inet.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/socket.h>

namespace {
using namespace mlir;

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class RemoteSimulatorQPU : public cudaq::QPU {
private:
  std::string m_url;
  std::string m_simName;
  cudaq::RestClient m_client;
  std::unordered_map<std::thread::id, cudaq::ExecutionContext *> m_contexts;
  std::mutex m_contextMutex;
  MLIRContext &m_mlirContext;
  static inline const std::vector<std::string> clientPasses = {};
  static inline const std::vector<std::string> serverPasses = {
      "func.func(unwind-lowering)",
      "func.func(indirect-to-direct-calls)",
      "inline",
      "canonicalize",
      "apply-op-specialization",
      "func.func(memtoreg{quantum=0})",
      "canonicalize",
      "expand-measurements",
      "cc-loop-normalize",
      "cc-loop-unroll",
      "canonicalize",
      "func.func(add-dealloc)",
      "func.func(quake-add-metadata)",
      "canonicalize",
      "func.func(lower-to-cfg)",
      "func.func(combine-quantum-alloc)",
      "canonicalize",
      "cse",
      "quake-to-qir"};

public:
  RemoteSimulatorQPU(MLIRContext &mlirContext, const std::string &url,
                     std::size_t id, const std::string &simName)
      : QPU(id), m_url(url), m_simName(simName), m_mlirContext(mlirContext) {
    cudaq::info("Create a remote QPU: id={}; url={}; simulator={}", qpu_id,
                m_url, m_simName);
  }

  std::thread::id getExecutionThreadId() const {
    return execution_queue->getExecutionThreadId();
  }

  void enqueue(cudaq::QuantumTask &task) override {
    cudaq::info("RemoteSimulatorQPU: Enqueue Task on QPU {}", qpu_id);
    execution_queue->enqueue(task);
  }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("RemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
                "(simulator = {})",
                name, qpu_id, m_simName);

    cudaq::ExecutionContext *executionContext = [&]() {
      std::scoped_lock<std::mutex> lock(m_contextMutex);
      const auto iter = m_contexts.find(std::this_thread::get_id());
      if (iter == m_contexts.end())
        throw std::runtime_error("Internal error: invalid execution context");
      return iter->second;
    }();
    cudaq::RestRequest request(*executionContext);
    request.entryPoint = name;
    if (cudaq::__internal__::isLibraryMode(name)) {
      if (args && voidStarSize > 0) {
        cudaq::info("Serialize {} bytes of args.", voidStarSize);
        request.args.resize(voidStarSize);
        std::memcpy(request.args.data(), args, voidStarSize);
      }
      // Library mode: retrieve the embedded bitcode in the executable.
      const auto path = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
      // Load the object file
      auto [objBin, objBuffer] =
          llvm::cantFail(llvm::object::ObjectFile::createObjectFile(path))
              .takeBinary();
      if (!objBin)
        throw std::runtime_error("Failed to load binary object file");

      if (kernelFunc) {
        ::Dl_info info;
        ::dladdr(reinterpret_cast<void *>(kernelFunc), &info);
        const auto funcName = cudaq::quantum_platform::demangle(info.dli_sname);
        cudaq::info("RemoteSimulatorQPU: retrieve name '{}' for kernel {}",
                    funcName, name);
        request.entryPoint = funcName;
      }

      for (const auto &section : objBin->sections()) {
        // Get the bitcode section
        if (section.isBitcode()) {
          llvm::MemoryBufferRef llvmBc(llvm::cantFail(section.getContents()),
                                       "Bitcode");
          request.format = cudaq::CodeFormat::LLVM;
          request.code = llvm::encodeBase64(llvmBc.getBuffer());
        }
      }
    } else {
      // Get the quake representation of the kernel
      auto quakeCode = cudaq::get_quake_by_name(name);
      request.passes = serverPasses;
      request.format = cudaq::CodeFormat::MLIR;
      auto module = parseSourceString<ModuleOp>(quakeCode, &m_mlirContext);
      if (!module)
        throw std::runtime_error("module cannot be parsed");

      // Extract the kernel name
      auto func = module->lookupSymbol<mlir::func::FuncOp>(
          std::string("__nvqpp__mlirgen__") + name);

      // Create a new Module to clone the function into
      auto location = FileLineColLoc::get(&m_mlirContext, "<builder>", 1, 1);
      ImplicitLocOpBuilder builder(location, &m_mlirContext);

      // FIXME this should be added to the builder.
      if (!func->hasAttr(cudaq::entryPointAttrName))
        func->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
      auto moduleOp = builder.create<ModuleOp>();
      moduleOp.push_back(func.clone());

      if (args) {
        cudaq::info("Run Quake Synth.\n");
        PassManager pm(&m_mlirContext);
        pm.addPass(cudaq::opt::createQuakeSynthesizer(name, args));
        if (failed(pm.run(moduleOp)))
          throw std::runtime_error("Could not successfully apply quake-synth.");
      }

      // Client-side passes
      if (!clientPasses.empty()) {
        PassManager pm(&m_mlirContext);
        std::string errMsg;
        llvm::raw_string_ostream os(errMsg);
        const std::string pipeline =
            std::accumulate(clientPasses.begin(), clientPasses.end(),
                            std::string(), [](const auto &ss, const auto &s) {
                              return ss.empty() ? s : ss + "," + s;
                            });
        if (failed(parsePassPipeline(pipeline, pm, os)))
          throw std::runtime_error(
              "Remote rest platform failed to add passes to pipeline (" +
              errMsg + ").");

        if (failed(pm.run(moduleOp)))
          throw std::runtime_error(
              "Remote rest platform: applying IR passes failed.");
      }
      std::string mlirCode;
      llvm::raw_string_ostream outStr(mlirCode);
      mlir::OpPrintingFlags opf;
      opf.enableDebugInfo(/*enable=*/true,
                          /*pretty=*/false);
      moduleOp.print(outStr, opf);
      request.code = llvm::encodeBase64(mlirCode);
    }

    request.simulator = m_simName;
    request.seed = cudaq::get_random_seed();
    // Don't let curl adding "Expect: 100-continue" header, which is not
    // suitable for large requests, e.g., bitcode in the JSON request.
    //  Ref: https://gms.tf/when-curl-sends-100-continue.html
    std::map<std::string, std::string> headers{
        {"Expect:", ""}, {"Content-type", "application/json"}};
    json requestJson = request;
    auto resultJs = m_client.post(m_url, "job", requestJson, headers, false);
    resultJs.get_to(*executionContext);
  }

  void setExecutionContext(cudaq::ExecutionContext *context) override {
    cudaq::info("RemoteSimulatorQPU::setExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts[std::this_thread::get_id()] = context;
  }

  void resetExecutionContext() override {
    cudaq::info("RemoteSimulatorQPU::resetExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts.erase(std::this_thread::get_id());
  }
};

// Util to query an available TCP/IP port for auto-launching a server instance.
std::optional<std::string> getAvailablePort() {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0)
    return {};
  struct sockaddr_in servAddr;
  ::bzero((char *)&servAddr, sizeof(servAddr));
  servAddr.sin_family = AF_INET;
  servAddr.sin_addr.s_addr = INADDR_ANY;
  // sin_port = 0 => auto assign
  servAddr.sin_port = 0;
  if (::bind(sock, (struct sockaddr *)&servAddr, sizeof(servAddr)) < 0)
    return {};
  socklen_t len = sizeof(servAddr);
  if (::getsockname(sock, (struct sockaddr *)&servAddr, &len) == -1)
    return {};
  if (close(sock) < 0)
    return {};
  return std::to_string(::ntohs(servAddr.sin_port));
}

// Multi-QPU platform for remotely-hosted simulator QPUs
class RemoteSimulatorQuantumPlatform : public cudaq::quantum_platform {
  std::unique_ptr<MLIRContext> m_mlirContext;
  std::vector<llvm::sys::ProcessInfo> m_serverProcesses;

public:
  RemoteSimulatorQuantumPlatform() : m_mlirContext(cudaq::initializeMLIR()) {
    platformNumQPUs = 0;
  }

  ~RemoteSimulatorQuantumPlatform() {
    for (auto &process : m_serverProcesses) {
      cudaq::info("Shutting down REST server process {}", process.Pid);
      ::kill(process.Pid, SIGKILL);
    }
  }
  bool supports_task_distribution() const override { return true; }
  void setTargetBackend(const std::string &description) override {
    const auto getOpt = [](const std::string &str,
                           const std::string &prefix) -> std::string {
      const auto prefixPos = str.find(prefix);
      if (prefixPos == std::string::npos)
        return "";
      const auto endPos = str.find_first_of(";", prefixPos);
      return (endPos == std::string::npos)
                 ? str.substr(prefixPos + prefix.size() + 1)
                 : cudaq::split(str.substr(prefixPos + prefix.size() + 1),
                                ';')[0];
    };

    auto urls = cudaq::split(getOpt(description, "url"), ',');
    auto sims = cudaq::split(getOpt(description, "backend"), ',');
    const bool autoLaunch =
        description.find("auto_launch") != std::string::npos;

    const auto formatUrl = [](const std::string &url) -> std::string {
      auto formatted = url;
      if (formatted.rfind("http", 0) != 0)
        formatted = std::string("http://") + formatted;
      if (formatted.back() != '/')
        formatted += '/';
      return formatted;
    };

    if (autoLaunch) {
      urls.clear();
      if (sims.empty())
        sims.emplace_back("qpp");
      const int numInstances = std::stoi(getOpt(description, "auto_launch"));
      cudaq::info("Auto launch {} REST servers", numInstances);
      const std::string serverExeName = "cudaq_rest_server";
      auto serverApp = llvm::sys::findProgramByName(serverExeName.c_str());
      if (!serverApp)
        throw std::runtime_error(
            "Unable to find CUDA Quantum REST server to launch.");

      for (int i = 0; i < numInstances; ++i) {
        const auto port = getAvailablePort();
        if (!port.has_value())
          throw std::runtime_error("Unable to find a TCP/IP port on the local "
                                   "machine for auto-launch a REST server.");
        urls.emplace_back(std::string("localhost:") + port.value());
        llvm::StringRef argv[] = {serverApp.get(), "--port", port.value()};
        [[maybe_unused]] auto processInfo =
            llvm::sys::ExecuteNoWait(serverApp.get(), argv, std::nullopt);
        cudaq::info("Auto launch REST server at http://localhost:{} (PID {})",
                    port.value(), processInfo.Pid);
        m_serverProcesses.emplace_back(processInfo);
      }
      // Allows some time for the servers to start
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    // List of simulator names must either be one or the same length as the URL
    // list. If one simulator name is provided, assuming that all the URL should
    // be using the same simulator.
    if (sims.size() > 1 && sims.size() != urls.size())
      throw std::runtime_error(
          fmt::format("Invalid number of remote backend simulators provided: "
                      "receiving {}, expecting {}.",
                      sims.size(), urls.size()));
    for (std::size_t qId = 0; qId < urls.size(); ++qId) {
      const auto simName = sims.size() == 1 ? sims.front() : sims[qId];
      // Populate the information and add the QPUs
      auto qpu = std::make_unique<RemoteSimulatorQPU>(
          *m_mlirContext, formatUrl(urls[qId]), qId, simName);
      threadToQpuId[std::hash<std::thread::id>{}(qpu->getExecutionThreadId())] =
          qId;
      platformQPUs.emplace_back(std::move(qpu));
    }

    platformNumQPUs = platformQPUs.size();
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(RemoteSimulatorQuantumPlatform, remote_sim)
