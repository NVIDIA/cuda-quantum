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
#include "llvm/Support/Base64.h"
#include "llvm/Support/MemoryBuffer.h"
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
#include "cuda_runtime_api.h"
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

namespace {

class RemoteSimulatorQPU : public cudaq::QPU {
private:
  std::string m_url;
  std::string m_simName;
  cudaq::RestClient m_client;
  std::unordered_map<std::thread::id, cudaq::ExecutionContext *> m_contexts;
  std::mutex m_contextMutex;
  MLIRContext &m_mlirContext;

public:
  RemoteSimulatorQPU(MLIRContext &mlirContext, const std::string &url,
                     std::size_t id, const std::string &simName)
      : QPU(id), m_url(url), m_simName(simName), m_mlirContext(mlirContext) {
    cudaq::info("Create a remote QPU: id={}; url={}; simulator={}", qpu_id,
                m_url, m_simName);
  }

  void enqueue(cudaq::QuantumTask &task) override {
    cudaq::info("Enqueue Task on QPU {}", qpu_id);
    execution_queue->enqueue(task);
  }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("QPU::launchKernel named '{}' remote QPU {} (simulator = {})",
                name, qpu_id, m_simName);

    cudaq::ExecutionContext *executionContext = [&]() {
      std::scoped_lock<std::mutex> lock(m_contextMutex);
      const auto iter = m_contexts.find(std::this_thread::get_id());
      if (iter == m_contexts.end())
        throw std::runtime_error("Internal error: invalid execution context");
      return iter->second;
    }();
    cudaq::RestRequest request(*executionContext);

    // Get the quake representation of the kernel
    auto quakeCode = cudaq::get_quake_by_name(name);
    request.entryPoint = name;
    request.simulator = m_simName;
    request.seed = cudaq::get_random_seed();

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

    llvm::raw_string_ostream outStr(request.code);
    mlir::OpPrintingFlags opf;
    opf.enableDebugInfo(/*enable=*/true,
                        /*pretty=*/false);
    moduleOp.print(outStr, opf);
    std::map<std::string, std::string> headers{};
    json requestJson = request;
    auto resultJs = m_client.post(m_url, "job", requestJson, headers);
    resultJs.get_to(*executionContext);
  }

  void setExecutionContext(cudaq::ExecutionContext *context) override {
    cudaq::info("RemoteSimulatorQPU::setExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts.emplace(std::this_thread::get_id(), context);
  }

  void resetExecutionContext() override {
    cudaq::info("RemoteSimulatorQPU::resetExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts.erase(std::this_thread::get_id());
  }
};

class RemoteSimulatorQuantumPlatform : public cudaq::quantum_platform {
  std::unique_ptr<MLIRContext> m_mlirContext;

public:
  ~RemoteSimulatorQuantumPlatform() = default;
  RemoteSimulatorQuantumPlatform() : m_mlirContext(cudaq::initializeMLIR()) {
    platformNumQPUs = 0;
  }

  bool supports_task_distribution() const override { return true; }
  void setTargetBackend(const std::string &description) override {
    const auto getOpt = [](const std::string &str,
                           const std::string &prefix) -> std::string {
      const auto prefixPos = str.find(prefix);
      if (prefixPos == std::string::npos)
        return "";
      const auto endPos = str.find_first_of(";", prefixPos);
      if (endPos == std::string::npos)
        return str.substr(prefixPos + prefix.size() + 1);
      else
        return cudaq::split(str.substr(prefixPos + prefix.size() + 1), ';')[0];
    };

    const auto urls = cudaq::split(getOpt(description, "url"), ',');
    const auto sims = cudaq::split(getOpt(description, "backend"), ',');
    // List of simulator names must either be one or the same length as the URL
    // list. If one simulator name is provided, assuming that all the URL should
    // be using the same simulator.
    if (sims.size() > 1 && sims.size() != urls.size())
      throw std::runtime_error(
          fmt::format("Invalid number of remote backend simulators provided: "
                      "receiving {}, expecting {}.",
                      sims.size(), urls.size()));
    const auto formatUrl = [](const std::string &url) -> std::string {
      auto formatted = url;
      if (formatted.rfind("http", 0) != 0)
        formatted = std::string("http://") + formatted;
      if (formatted.back() != '/')
        formatted += '/';
      return formatted;
    };
    for (std::size_t qId = 0; qId < urls.size(); ++qId) {
      const auto simName = sims.size() == 1 ? sims.front() : sims[qId];
      // Populate the information and add the QPUs
      platformQPUs.emplace_back(std::make_unique<RemoteSimulatorQPU>(
          *m_mlirContext, formatUrl(urls[qId]), qId, simName));
    }

    platformNumQPUs = platformQPUs.size();
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(RemoteSimulatorQuantumPlatform, remote_sim)
