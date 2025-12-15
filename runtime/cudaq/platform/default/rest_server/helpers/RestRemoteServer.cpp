/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/JIT.h"
#include "common/JsonConvert.h"
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
#include "nvqir/CircuitSimulator.h"
#include "server_impl/RestServer.h"
#include "llvm/ADT/ScopeExit.h"
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
#include <cxxabi.h>
#include <filesystem>
#include <fstream>
#include <streambuf>

extern "C" {
void __nvqir__setCircuitSimulator(nvqir::CircuitSimulator *);
}

namespace {
using namespace mlir;
// Encapsulates a dynamically-loaded NVQIR simulator library
struct SimulatorHandle {
  std::string name;
  void *libHandle;
};

// Implementation of llvm::cantFail which throws a C++ exception rather than
// emits a signal/asserts.
template <typename T>
T getValueOrThrow(llvm::Expected<T> valOrErr,
                  const std::string &errorMsgToThrow) {
  if (valOrErr)
    return std::move(*valOrErr);
  else {
    LLVMConsumeError(llvm::wrap(valOrErr.takeError()));
    throw std::runtime_error(errorMsgToThrow);
  }
}

// Clear any registered operations in the ExecutionManager and then destroy the
// JIT. This needs to be called when the registered operations may contain
// pointers into the code objects inside the JIT.
void clearRegOpsAndDestroyJIT(std::unique_ptr<llvm::orc::LLJIT> &jit) {
  cudaq::getExecutionManager()->clearRegisteredOperations();
  // Destroys the LLJIT object
  jit.reset();
}

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
  // Currently-loaded NVQIR simulator.
  SimulatorHandle m_simHandle;
  // Default backend for initialization.
  // Note: we always need to preload a default backend on the server runtime
  // since cudaq runtime relies on that.
  static constexpr const char *DEFAULT_NVQIR_SIMULATION_BACKEND = "qpp";

protected:
  // Server to exit after each job request.
  // Note: this doesn't apply to ping ("/") endpoint.
  bool exitAfterJob = false;
  // Time-point data
  std::optional<std::chrono::time_point<std::chrono::high_resolution_clock>>
      requestStart;
  std::optional<std::chrono::time_point<std::chrono::high_resolution_clock>>
      simulationStart;
  std::optional<std::chrono::time_point<std::chrono::high_resolution_clock>>
      simulationEnd;

  // Method to filter incoming request.
  // The request is only handled iff this returns true.
  // When returning false, `outValidationMessage` can be used to report the
  // error message.
  virtual bool filterRequest(const cudaq::RestRequest &in_request,
                             std::string &outValidationMessage) const {
    // Default is no filter.
    return true;
  }

public:
  RemoteRestRuntimeServer()
      : cudaq::RemoteRuntimeServer(),
        m_simHandle(DEFAULT_NVQIR_SIMULATION_BACKEND,
                    loadNvqirSimLib(DEFAULT_NVQIR_SIMULATION_BACKEND)) {}

  virtual std::pair<int, int> version() const override {
    return std::make_pair(cudaq::RestRequest::REST_PAYLOAD_VERSION,
                          cudaq::RestRequest::REST_PAYLOAD_MINOR_VERSION);
  }
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
    m_server->addRoute(
        cudaq::RestServer::Method::GET, "/",
        [](const std::string &reqBody,
           const std::unordered_multimap<std::string, std::string> &headers) {
          // Return an empty JSON string,
          // e.g., for client to ping the server.
          return json();
        });

    // New simulation request.
    m_server->addRoute(
        cudaq::RestServer::Method::POST, "/job",
        [&](const std::string &reqBody,
            const std::unordered_multimap<std::string, std::string> &headers) {
          requestStart = std::chrono::high_resolution_clock::now();
          auto shutdownAfterHandlingRequest = llvm::make_scope_exit([&] {
            if (this->exitAfterJob)
              m_server->stop();
          });

          std::string mutableReq;
          for (const auto &[k, v] : headers)
            CUDAQ_INFO("Request Header: {} : {}", k, v);
          mutableReq = reqBody;

          if (m_hasMpi)
            cudaq::mpi::broadcast(mutableReq, 0);
          auto resultJs = processRequest(mutableReq);

          return resultJs;
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
        // Break the loop if the server is operating in one-shot mode.
        if (exitAfterJob)
          break;
      }
    }
  }
  // Stop the server.
  virtual void stop() override { m_server->stop(); }

  virtual void handleVQERequest(std::size_t reqId,
                                cudaq::ExecutionContext &io_context,
                                const std::string &backendSimName,
                                std::string_view ir, cudaq::gradient *gradient,
                                cudaq::optimizer &optimizer, const int n_params,
                                std::string_view kernelName,
                                std::size_t seed) override {
    cudaq::optimization_result result;

    // Treat the shots as a signed number, and if it is <= 0, then shots-based
    // sampling is disabled. This is standard VQE/observe behavior.
    std::int64_t shots = *reinterpret_cast<std::int64_t *>(&io_context.shots);

    // If we're changing the backend, load the new simulator library from file.
    if (m_simHandle.name != backendSimName) {
      if (m_simHandle.libHandle)
        dlclose(m_simHandle.libHandle);

      m_simHandle =
          SimulatorHandle(backendSimName, loadNvqirSimLib(backendSimName));
    }

    if (seed != 0)
      cudaq::set_random_seed(seed);
    simulationStart = std::chrono::high_resolution_clock::now();

    auto &requestInfo = m_codeTransform[reqId];
    if (requestInfo.format == cudaq::CodeFormat::LLVM) {
      throw std::runtime_error("CodeFormat::LLVM is not supported with VQE. "
                               "Use CodeFormat::MLIR instead.");
    } else {
      llvm::SourceMgr sourceMgr;
      sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(ir),
                                   llvm::SMLoc());
      auto module = parseSourceFile<ModuleOp>(sourceMgr, m_mlirContext.get());
      if (!module)
        throw std::runtime_error("Failed to parse the input MLIR code");
      auto engine = jitMlirCode(*module, requestInfo.passes);
      const std::string entryPointFunc =
          std::string(cudaq::runtime::cudaqGenPrefixName) +
          std::string(kernelName);
      auto fnPtr =
          getValueOrThrow(engine->lookup(entryPointFunc),
                          "Failed to look up entry-point function symbol");
      if (!fnPtr)
        throw std::runtime_error("Failed to get entry function");

      // quake-to-qir translates cc.stdvec<f64> to !llvm.struct<(ptr<f64>,
      // i64)>, so we need to provide the inputs in this format. Make a lambda
      // to convert between the two formats.
      struct stdvec_struct {
        const double *ptr;
        std::size_t size;
      };
      auto fn = reinterpret_cast<void (*)(stdvec_struct)>(fnPtr);
      auto fnWrapper = [fn](const std::vector<double> &x) {
        fn({x.data(), x.size()});
      };

      // Construct the gradient object.
      if (gradient)
        gradient->setKernel(fnWrapper);

      bool requiresGrad = optimizer.requiresGradients();
      auto theSpin = *io_context.spin;
      assert(cudaq::spin_op::canonicalize(theSpin) == theSpin);

      result = optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                                std::vector<double> &grad_vec) {
        double e = shots <= 0 ? cudaq::observe(fnWrapper, theSpin, x)
                              : cudaq::observe(shots, fnWrapper, theSpin, x);
        if (requiresGrad)
          gradient->compute(x, grad_vec, theSpin, e);
        return e;
      });
    }
    simulationEnd = std::chrono::high_resolution_clock::now();
    io_context.optResult = result;
  }

  virtual void handleRequest(std::size_t reqId,
                             cudaq::ExecutionContext &io_context,
                             const std::string &backendSimName,
                             std::string_view ir, std::string_view kernelName,
                             void *kernelArgs, std::uint64_t argsSize,
                             std::size_t seed) override {

    // If we're changing the backend, load the new simulator library from file.
    if (m_simHandle.name != backendSimName) {
      if (m_simHandle.libHandle)
        dlclose(m_simHandle.libHandle);

      m_simHandle =
          SimulatorHandle(backendSimName, loadNvqirSimLib(backendSimName));
    }
    if (seed != 0)
      cudaq::set_random_seed(seed);
    auto &platform = cudaq::get_platform();
    auto &requestInfo = m_codeTransform[reqId];

    // The lifetime of this pointer should be just as long as `platform` because
    // any calls to `platform` functions could invoke code that relies on the
    // JIT being present.
    std::unique_ptr<llvm::orc::LLJIT> llvmJit;
    if (requestInfo.format == cudaq::CodeFormat::LLVM) {
      if (io_context.name == "sample") {
        // In library mode (LLVM), check to see if we have mid-circuit measures
        // by tracing the kernel function.
        cudaq::ExecutionContext context("tracer");
        platform.set_exec_ctx(&context);
        llvmJit = cudaq::invokeWrappedKernel(ir, std::string(kernelName),
                                             kernelArgs, argsSize);
        platform.reset_exec_ctx();
        // In trace mode, if we have a measure result
        // that is passed to an if statement, then
        // we'll have collected registerNames
        if (!context.registerNames.empty()) {
          // append new register names to the main sample context
          for (std::size_t i = 0; i < context.registerNames.size(); ++i)
            io_context.registerNames.emplace_back("auto_register_" +
                                                  std::to_string(i));
          io_context.hasConditionalsOnMeasureResults = true;
          // Need to run simulation shot-by-shot
          cudaq::sample_result counts;
          platform.set_exec_ctx(&io_context);
          // Since registered operations may contain pointers to classes defined
          // in an LLVM JIT, we must clear them before any prior LLVM JIT gets
          // deleted.
          clearRegOpsAndDestroyJIT(llvmJit);
          // If it has conditionals, loop over individual circuit executions
          llvmJit = cudaq::invokeWrappedKernel(
              ir, std::string(kernelName), kernelArgs, argsSize,
              io_context.shots, [&](std::size_t i) {
                // Reset the context and get the single
                // measure result, add it to the
                // sample_result and clear the context
                // result
                platform.reset_exec_ctx();
                counts += io_context.result;
                io_context.result.clear();
                if (i != (io_context.shots - 1))
                  platform.set_exec_ctx(&io_context);
              });
          io_context.result = counts;
        } else {
          // If no conditionals, nothing special to do for library mode
          platform.set_exec_ctx(&io_context);
          // Since registered operations may contain pointers to classes defined
          // in an LLVM JIT, we must clear them before any prior LLVM JIT gets
          // deleted.
          clearRegOpsAndDestroyJIT(llvmJit);
          llvmJit = cudaq::invokeWrappedKernel(ir, std::string(kernelName),
                                               kernelArgs, argsSize);
          platform.reset_exec_ctx();
        }
      } else {
        platform.set_exec_ctx(&io_context);
        llvmJit = cudaq::invokeWrappedKernel(ir, std::string(kernelName),
                                             kernelArgs, argsSize);
        platform.reset_exec_ctx();
      }
    } else {
      platform.set_exec_ctx(&io_context);
      if (io_context.name == "sample" &&
          io_context.hasConditionalsOnMeasureResults) {
        // Need to run simulation shot-by-shot
        cudaq::sample_result counts;
        invokeMlirKernel(io_context, m_mlirContext, ir, requestInfo.passes,
                         std::string(kernelName), io_context.shots,
                         [&](std::size_t i) {
                           // Reset the context and get the single
                           // measure result, add it to the
                           // sample_result and clear the context
                           // result
                           platform.reset_exec_ctx();
                           counts += io_context.result;
                           io_context.result.clear();
                           if (i != (io_context.shots - 1))
                             platform.set_exec_ctx(&io_context);
                         });
        io_context.result = counts;
      } else if (io_context.name == "run") {
        // Handle cudaq::run: it should be executed in a context-free manner;
        // the output log is accumulated in the simulator output log.
        platform.reset_exec_ctx();
        //  Clear the outputLog.
        auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();
        circuitSimulator->outputLog.clear();
        // Invoke the kernel multiple times.
        invokeMlirKernel(io_context, m_mlirContext, ir, requestInfo.passes,
                         std::string(kernelName), io_context.shots);
        // Save the output log to the result buffer to be sent back to the
        // client.
        io_context.invocationResultBuffer.assign(
            circuitSimulator->outputLog.c_str(),
            circuitSimulator->outputLog.c_str() +
                circuitSimulator->outputLog.size());
        circuitSimulator->outputLog.clear();
      } else {
        invokeMlirKernel(io_context, m_mlirContext, ir, requestInfo.passes,
                         std::string(kernelName));
        platform.reset_exec_ctx();
      }
    }
    // Clear the registered operations before the `llvmJit` goes out of scope
    // so that destruction of registered operations doesn't cause segfaults
    // during shutdown.
    clearRegOpsAndDestroyJIT(llvmJit);
    simulationEnd = std::chrono::high_resolution_clock::now();
  }

protected:
  std::unique_ptr<ExecutionEngine>
  jitMlirCode(ModuleOp currentModule, const std::vector<std::string> &passes,
              const std::vector<std::string> &extraLibPaths = {}) {
    CUDAQ_INFO("Running jitCode.");
    auto module = currentModule.clone();
    ExecutionEngineOptions opts;
    opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
    opts.enableObjectDump = true;
    opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
    SmallVector<StringRef, 4> sharedLibs;
    for (auto &lib : extraLibPaths) {
      CUDAQ_INFO("Extra library loaded: {}", lib);
      sharedLibs.push_back(lib);
    }
    opts.sharedLibPaths = sharedLibs;

    auto ctx = module.getContext();
    {
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

      CUDAQ_INFO("- Pass manager was applied.");
    }
    // Verify MLIR conforming to the NVQIR-spec (known runtime functions and/or
    // QIR functions)
    {
      // Collect all functions that are defined (and have non-empty bodies) in
      // this module Op.
      const std::vector<llvm::StringRef> allFunctionNames = [&]() {
        std::vector<llvm::StringRef> allFuncs;
        for (auto &op : *module.getBody())
          if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(op))
            if (!funcOp.getFunctionBody().empty())
              allFuncs.emplace_back(funcOp.getName());
        return allFuncs;
      }();
      // Note: run this verification as a standalone step to decouple IR
      // conversion and verification.
      // Verification condition: all function definitions can only make function
      // calls to:
      //  (1) NVQIR-compliance functions, or
      //  (2) other functions defined in this module.
      PassManager pm(ctx);
      pm.addNestedPass<LLVM::LLVMFuncOp>(
          cudaq::opt::createVerifyNVQIRCallOpsPass(allFunctionNames));
      if (failed(pm.run(module)))
        throw std::runtime_error(
            "Failed check to verify IR compliance for NVQIR runtime.");

      CUDAQ_INFO("- Finish IR input verification.");
    }

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

    CUDAQ_INFO("- Creating the MLIR ExecutionEngine");
    auto uniqueJit =
        getValueOrThrow(ExecutionEngine::create(module, opts),
                        "Failed to create MLIR JIT ExecutionEngine");
    CUDAQ_INFO("- MLIR ExecutionEngine created successfully.");
    return uniqueJit;
  }

  void
  invokeMlirKernel(cudaq::ExecutionContext &io_context,
                   std::unique_ptr<MLIRContext> &contextPtr,
                   std::string_view irString,
                   const std::vector<std::string> &passes,
                   const std::string &entryPointFn, std::size_t numTimes = 1,
                   std::function<void(std::size_t)> postExecCallback = {}) {
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBufferCopy(irString),
                                 llvm::SMLoc());
    auto module = parseSourceFile<ModuleOp>(sourceMgr, contextPtr.get());
    if (!module)
      throw std::runtime_error("Failed to parse the input MLIR code");
    auto engine = jitMlirCode(*module, passes);
    llvm::SmallVector<void *> returnArg;
    const std::string entryPointFunc =
        std::string(cudaq::runtime::cudaqGenPrefixName) + entryPointFn;
    if (auto funcOp = module->lookupSymbol<LLVM::LLVMFuncOp>(entryPointFunc)) {
      auto funcTy = funcOp.getFunctionType();
      auto returnTy = funcTy.getReturnType();
      // These are the returned types that we support.
      if (returnTy.isF32()) {
        io_context.invocationResultBuffer.resize(sizeof(float));
        returnArg.push_back(io_context.invocationResultBuffer.data());
      } else if (returnTy.isF64()) {
        io_context.invocationResultBuffer.resize(sizeof(double));
        returnArg.push_back(io_context.invocationResultBuffer.data());
      } else if (returnTy.isInteger(1)) {
        static_assert(sizeof(bool) == sizeof(char),
                      "Incompatible boolean data type. CUDA-Q kernels expect "
                      "sizeof(bool) == sizeof(char).");
        io_context.invocationResultBuffer.resize(sizeof(bool));
        returnArg.push_back(io_context.invocationResultBuffer.data());
      } else if (returnTy.isIntOrIndex()) {
        io_context.invocationResultBuffer.resize(
            (returnTy.getIntOrFloatBitWidth() + 7) / 8);
        returnArg.push_back(io_context.invocationResultBuffer.data());
      }
    }

    // Note: currently, we only return data from kernel on single-shot
    // execution. Once we enable arbitrary sample return type, we can run this
    // in a loop and return a vector of return type.
    if (numTimes == 1 && !returnArg.empty()) {
      simulationStart = std::chrono::high_resolution_clock::now();
      llvm::Error error = engine->invokePacked(entryPointFunc, returnArg);
      if (error)
        throw std::runtime_error("JIT invocation failed");
      if (postExecCallback)
        postExecCallback(0);
    } else {
      auto fnPtr =
          getValueOrThrow(engine->lookup(entryPointFunc),
                          "Failed to look up entry-point function symbol");
      if (!fnPtr)
        throw std::runtime_error("Failed to get entry function");

      auto fn = reinterpret_cast<void (*)()>(fnPtr);
      simulationStart = std::chrono::high_resolution_clock::now();
      for (std::size_t i = 0; i < numTimes; ++i) {
        // Invoke the kernel
        fn();
        if (postExecCallback)
          postExecCallback(i);
      }
    }
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
    CUDAQ_INFO("Request simulator {} at {}", simulatorName, simLibPath.c_str());
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

  virtual json processRequest(const std::string &reqBody,
                              bool forceLog = false) {
    // Create a watchdog thread to kill the process if the request is taking too
    // long.
    std::mutex watchdogMutex;
    std::condition_variable watchdogCV;
    bool processingComplete = false;
    std::future<void> watchdogResult = std::async(std::launch::async, [&]() {
      std::unique_lock<std::mutex> lock(watchdogMutex);
      std::chrono::seconds timeout(60 * 60 * 24 * 30); // default to 30 days
      if (auto timeoutStr = getenv("WATCHDOG_TIMEOUT_SEC"))
        timeout = std::chrono::seconds(atoi(timeoutStr));

      if (watchdogCV.wait_for(lock, timeout,
                              [&]() { return processingComplete; })) {
        // Succeeded. Gracefully return from the async.
        return;
      } else {
        // Timed out. Perform abort.
        fmt::print("Processing timed out after {} seconds! Aborting!\n",
                   timeout.count());
        exit(-1);
      }
    });

    // Notify watchdog thread of graceful completion at scope exit
    auto notifyWatchdog = llvm::make_scope_exit([&] {
      std::unique_lock<std::mutex> lock(watchdogMutex);
      processingComplete = true;
      lock.unlock();
      watchdogCV.notify_one();
      watchdogResult.get();
    });

    try {
      // IMPORTANT: This assumes the REST server handles incoming requests
      // sequentially.
      static std::size_t g_requestCounter = 0;
      auto requestJson = json::parse(reqBody);
      cudaq::RestRequest request(requestJson);

      std::ostringstream os;
      os << "[RemoteRestRuntimeServer] Incoming job request from client "
         << request.clientVersion;
      if (forceLog) {
        // Force the request to appear in the logs regardless of server log
        // level.
        cudaq::log(os.str());
      } else {
        CUDAQ_INFO(os.str());
      }

      // Verify REST API version of the incoming request
      // If the incoming JSON payload has a different version than the one this
      // server is expecting, throw an error. Note: we don't support
      // automatically versioning the payload (converting payload between
      // different versions) at the moment.
      if (static_cast<int>(request.version) != version().first)
        throw std::runtime_error(fmt::format(
            "Incompatible REST payload version detected: supported version {}, "
            "got version {}.",
            version().first, request.version));

      std::string validationMsg;
      const bool shouldHandle = filterRequest(request, validationMsg);
      if (!shouldHandle) {
        json resultJson;
        resultJson["status"] = "Invalid Request";
        resultJson["errorMessage"] = validationMsg;
        return resultJson;
      }

      const auto reqId = g_requestCounter++;
      m_codeTransform[reqId] =
          CodeTransformInfo(request.format, request.passes);
      json resultJson;
      std::vector<char> decodedCodeIr;
      auto errorCode = llvm::decodeBase64(request.code, decodedCodeIr);
      if (errorCode) {
        LLVMConsumeError(llvm::wrap(std::move(errorCode)));
        throw std::runtime_error("Failed to decode input IR");
      }
      std::string_view codeStr(decodedCodeIr.data(), decodedCodeIr.size());

      if (request.opt.has_value() && request.opt->optimizer) {
        if (!request.opt->optimizer_n_params.has_value())
          throw std::runtime_error(
              "Cannot run optimizer without providing optimizer_n_params");

        handleVQERequest(
            reqId, request.executionContext, request.simulator, codeStr,
            request.opt->gradient.get(), *request.opt->optimizer,
            *request.opt->optimizer_n_params, request.entryPoint, request.seed);
        resultJson["executionContext"] = request.executionContext;
      } else if (request.executionContext.name == "state-overlap") {
        if (!request.overlapKernel.has_value())
          throw std::runtime_error("Missing overlap kernel data.");
        std::vector<char> decodedCodeIr1, decodedCodeIr2;
        auto errorCode1 = llvm::decodeBase64(request.code, decodedCodeIr1);
        auto errorCode2 =
            llvm::decodeBase64(request.overlapKernel->ir, decodedCodeIr2);
        if (errorCode1) {
          LLVMConsumeError(llvm::wrap(std::move(errorCode1)));
          throw std::runtime_error("Failed to decode input IR (request.code)");
        }
        if (errorCode2) {
          LLVMConsumeError(llvm::wrap(std::move(errorCode2)));
          throw std::runtime_error(
              "Failed to decode input IR (request.overlapKernel->ir)");
        }
        std::string_view codeStr1(decodedCodeIr1.data(), decodedCodeIr1.size());
        cudaq::ExecutionContext stateContext1("extract-state");
        handleRequest(reqId, stateContext1, request.simulator, codeStr1,
                      request.entryPoint, request.args.data(),
                      request.args.size(), request.seed);
        std::string_view codeStr2(decodedCodeIr2.data(), decodedCodeIr2.size());
        cudaq::ExecutionContext stateContext2("extract-state");
        handleRequest(reqId, stateContext2, request.simulator, codeStr2,
                      request.overlapKernel->entryPoint,
                      request.overlapKernel->args.data(),
                      request.overlapKernel->args.size(), request.seed);
        request.executionContext.overlapResult =
            stateContext1.simulationState->overlap(
                *stateContext2.simulationState);
        resultJson["executionContext"] = request.executionContext;
      } else {
        if (request.overlapKernel.has_value())
          throw std::runtime_error("Unexpected data: overlap kernel is "
                                   "provided in non-overlap compute mode.");
        std::vector<char> decodedCodeIr;
        auto errorCode = llvm::decodeBase64(request.code, decodedCodeIr);
        if (errorCode) {
          LLVMConsumeError(llvm::wrap(std::move(errorCode)));
          throw std::runtime_error("Failed to decode input IR (request.code)");
        }
        std::string_view codeStr(decodedCodeIr.data(), decodedCodeIr.size());
        handleRequest(reqId, request.executionContext, request.simulator,
                      codeStr, request.entryPoint, request.args.data(),
                      request.args.size(), request.seed);

        // If specific amplitudes are requested.
        // Note: this could be the case whereby the state vector is too large
        // for full retrieval (determined by
        // `RemoteSimulationState::maxQubitCountForFullStateTransfer`).
        if (request.executionContext.name == "extract-state" &&
            request.executionContext.amplitudeMaps.has_value()) {
          // Acquire the state, no need to send the full state back
          auto serverState =
              std::move(request.executionContext.simulationState);
          for (auto &[key, val] :
               request.executionContext.amplitudeMaps.value()) {
            val = serverState->getAmplitude(key);
          }
        }

        resultJson["executionContext"] = request.executionContext;
      }
      m_codeTransform.erase(reqId);
      return resultJson;
    } catch (std::exception &e) {
      json resultJson;
      resultJson["status"] = "Failed to process incoming request";
      resultJson["errorMessage"] = e.what();
      return resultJson;
    } catch (...) {
      json resultJson;
      resultJson["status"] = "Failed to process incoming request";
      std::string exType = __cxxabiv1::__cxa_current_exception_type()->name();
      auto demangledPtr =
          __cxxabiv1::__cxa_demangle(exType.c_str(), nullptr, nullptr, nullptr);
      if (demangledPtr) {
        std::string demangledName(demangledPtr);
        resultJson["errorMessage"] = "Unknown error of type " + demangledName;
      } else {
        resultJson["errorMessage"] = "Unknown error";
      }
      return resultJson;
    }
  }
};

} // namespace

CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeServer, RemoteRestRuntimeServer, rest)
