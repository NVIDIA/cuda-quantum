/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "JsonConvert.h"
#include "common/Logger.h"
#include "common/PluginUtils.h"
#include "common/RemoteKernelExecutor.h"
#include "common/RuntimeMLIR.h"
#include "cudaq.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm_jit/JIT.h"
#include "nvqir/CircuitSimulator.h"
#include "server_impl/RestServer.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/Passes.h"

extern "C" {
void __nvqir__setCircuitSimulator(nvqir::CircuitSimulator *);
}

namespace {
using namespace mlir;

class RemoteRestRuntimeServer : public cudaq::RemoteRuntimeServer {
  int m_port = -1;
  std::unique_ptr<cudaq::RestServer> m_server;
  std::unique_ptr<MLIRContext> m_mlirContext;
  bool m_hasMpi = false;
  struct CodeTransformInfo {
    cudaq::CodeFormat format;
    std::vector<std::string> passes;
  };
  std::unordered_map<std::size_t, CodeTransformInfo> m_codeTransform;

public:
  virtual void
  init(const std::unordered_map<std::string, std::string> &configs) override {
    const auto portIter = configs.find("port");
    if (portIter != configs.end())
      m_port = stoi(portIter->second);
    // Note: port numbers 0-1023 are used for well-known ports.
    const bool portValid = m_port >= 1024 && m_port <= 65535;
    if (!portValid)
      throw std::runtime_error(
          "Invalid TCP/IP port requested. Valid range: [1024, 65535].");
    m_server = std::make_unique<cudaq::RestServer>(m_port);
    m_server->addRoute(cudaq::RestServer::Method::GET, "/",
                       [](const std::string &reqBody) {
                         // Return an empty JSON string,
                         // e.g., for client to ping the server.
                         return json();
                       });

    // New simulation request.
    m_server->addRoute(cudaq::RestServer::Method::POST, "/job",
                       [&](const std::string &reqBody) {
                         std::string mutableReq = reqBody;
                         if (m_hasMpi)
                           cudaq::mpi::broadcast(mutableReq, 0);
                         return processRequest(reqBody);
                       });
    m_mlirContext = cudaq::initializeMLIR();
    m_hasMpi = cudaq::mpi::is_initialized();
  }
  // Start the server.
  virtual void start() override {
    if (!m_server)
      throw std::runtime_error(
          "Fatal error: attempt to start the server before initialization. "
          "Please initialize the server.");
    if (!m_hasMpi || cudaq::mpi::rank() == 0) {
      // Only run this app on Rank 0;
      // the rest will wait for a broadcast.
      m_server->start();
    } else if (m_hasMpi) {
      for (;;) {
        std::string jsonRequestBody;
        cudaq::mpi::broadcast(jsonRequestBody, 0);
        // All ranks need to join, e.g., MPI-capable backends.
        processRequest(jsonRequestBody);
      }
    }
  }
  // Stop the server.
  virtual void stop() override { m_server->stop(); }

  virtual void handleRequest(std::size_t reqId,
                             cudaq::ExecutionContext &io_context,
                             const std::string &backendSimName,
                             std::string_view ir, std::string_view kernelName,
                             void *kernelArgs, std::uint64_t argsSize,
                             std::size_t seed) override {
    void *handle = loadNvqirSimLib(backendSimName);
    if (seed != 0)
      cudaq::set_random_seed(seed);
    auto &platform = cudaq::get_platform();
    platform.set_exec_ctx(&io_context);

    auto &requestInfo = m_codeTransform[reqId];

    if (requestInfo.format == cudaq::CodeFormat::LLVM)
      cudaq::invokeWrappedKernel(ir, std::string(kernelName), kernelArgs,
                                 argsSize);
    else
      invokeMlirKernel(m_mlirContext, ir, requestInfo.passes,
                       std::string(kernelName));
    platform.reset_exec_ctx();
    dlclose(handle);
  }

private:
  std::unique_ptr<ExecutionEngine>
  jitMlirCode(ModuleOp currentModule, const std::vector<std::string> &passes,
              const std::vector<std::string> &extraLibPaths = {}) {
    cudaq::info("Running jitCode.");
    auto module = currentModule.clone();
    ExecutionEngineOptions opts;
    opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
    opts.enableObjectDump = true;
    opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
    SmallVector<StringRef, 4> sharedLibs;
    for (auto &lib : extraLibPaths) {
      cudaq::info("Extra library loaded: {}", lib);
      sharedLibs.push_back(lib);
    }
    opts.sharedLibPaths = sharedLibs;

    auto ctx = module.getContext();
    PassManager pm(ctx);
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    const std::string pipeline =
        std::accumulate(passes.begin(), passes.end(), std::string(),
                        [](const auto &ss, const auto &s) {
                          return ss.empty() ? s : ss + "," + s;
                        });
    if (failed(parsePassPipeline(pipeline, pm, os)))
      throw std::runtime_error(
          "Remote rest platform failed to add passes to pipeline (" + errMsg +
          ").");

    if (failed(pm.run(module)))
      throw std::runtime_error(
          "Remote rest platform: applying IR passes failed.");

    cudaq::info("- Pass manager was applied.");

    opts.llvmModuleBuilder =
        [](Operation *module,
           llvm::LLVMContext &llvmContext) -> std::unique_ptr<llvm::Module> {
      llvmContext.setOpaquePointers(false);
      auto llvmModule = translateModuleToLLVMIR(module, llvmContext);
      if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return nullptr;
      }
      ExecutionEngine::setupTargetTriple(llvmModule.get());
      return llvmModule;
    };

    cudaq::info("- Creating the MLIR ExecutionEngine");
    auto uniqueJit = llvm::cantFail(ExecutionEngine::create(module, opts));
    cudaq::info("- MLIR ExecutionEngine created successfully.");
    return uniqueJit;
  }

  void invokeMlirKernel(std::unique_ptr<MLIRContext> &contextPtr,
                        std::string_view irString,
                        const std::vector<std::string> &passes,
                        const std::string &entryPointFn) {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(irString),
                                 llvm::SMLoc());
    auto module = parseSourceFile<ModuleOp>(sourceMgr, contextPtr.get());
    auto engine = jitMlirCode(*module, passes);
    const std::string entryPointFunc =
        std::string(cudaq::runtime::cudaqGenPrefixName) + entryPointFn;
    auto fnPtr = llvm::cantFail(engine->lookup(entryPointFunc));
    if (!fnPtr)
      throw std::runtime_error("Failed to get entry function");

    auto fn = reinterpret_cast<void (*)()>(fnPtr);
    // Invoke the kernel
    fn();
  }

  void *loadNvqirSimLib(const std::string &simulatorName) {
    const std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
#if defined(__APPLE__) && defined(__MACH__)
    const auto libSuffix = "dylib";
#else
    const auto libSuffix = "so";
#endif
    const auto simLibPath =
        cudaqLibPath.parent_path() /
        fmt::format("libnvqir-{}.{}", simulatorName, libSuffix);
    cudaq::info("Request simulator {} at {}", simulatorName,
                simLibPath.c_str());
    void *simLibHandle = dlopen(simLibPath.c_str(), RTLD_GLOBAL | RTLD_NOW);
    if (!simLibHandle) {
      char *error_msg = dlerror();
      throw std::runtime_error(fmt::format(
          "Failed to open simulator backend library: {}.",
          error_msg ? std::string(error_msg) : std::string("Unknown error")));
    }
    auto *sim = cudaq::getUniquePluginInstance<nvqir::CircuitSimulator>(
        std::string("getCircuitSimulator"), simLibPath.c_str());
    __nvqir__setCircuitSimulator(sim);

    return simLibHandle;
  }

  json processRequest(const std::string &reqBody) {
    // IMPORTANT: This assumes the REST server handles incoming requests
    // sequentially.
    static std::size_t g_requestCounter = 0;
    auto requestJson = json::parse(reqBody);
    cudaq::RestRequest request(requestJson);
    const auto reqId = g_requestCounter++;
    m_codeTransform[reqId] = CodeTransformInfo(request.format, request.passes);
    std::vector<char> decodedCodeIr;
    if (llvm::decodeBase64(request.code, decodedCodeIr)) {
      throw std::runtime_error("Failed to decode input IR");
    }
    std::string_view codeStr(decodedCodeIr.data(), decodedCodeIr.size());
    handleRequest(reqId, request.executionContext, request.simulator, codeStr,
                  request.entryPoint, request.args.data(), request.args.size(),
                  request.seed);
    json resultJson;
    resultJson["executionContext"] = request.executionContext;
    m_codeTransform.erase(reqId);
    return resultJson;
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeServer, RemoteRestRuntimeServer, rest)
