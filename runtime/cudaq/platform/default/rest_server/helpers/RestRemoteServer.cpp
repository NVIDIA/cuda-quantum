/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "JsonConvert.h"
#include "common/JIT.h"
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
          std::string mutableReq;
          for (const auto &[k, v] : headers)
            cudaq::info("Request Header: {} : {}", k, v);
          // Checking if this request has its body sent on as NVCF assets.
          const auto dirIter = headers.find("NVCF-ASSET-DIR");
          const auto assetIdIter = headers.find("NVCF-FUNCTION-ASSET-IDS");
          if (dirIter != headers.end() && assetIdIter != headers.end()) {
            const std::string dir = dirIter->second;
            const auto ids = cudaq::split(assetIdIter->second, ',');
            if (ids.size() != 1) {
              json js;
              js["status"] =
                  fmt::format("Invalid asset Id data: {}", assetIdIter->second);
              return js;
            }
            // Load the asset file
            std::filesystem::path assetFile =
                std::filesystem::path(dir) / ids[0];
            if (!std::filesystem::exists(assetFile)) {
              json js;
              js["status"] = fmt::format("Unable to find the asset file {}",
                                         assetFile.string());
              return js;
            }
            std::ifstream t(assetFile);
            std::string requestFromFile((std::istreambuf_iterator<char>(t)),
                                        std::istreambuf_iterator<char>());
            mutableReq = requestFromFile;
          } else {
            mutableReq = reqBody;
          }

          if (m_hasMpi)
            cudaq::mpi::broadcast(mutableReq, 0);
          auto resultJs = processRequest(mutableReq);
          // Check whether we have a limit in terms of response size.
          if (headers.contains("NVCF-MAX-RESPONSE-SIZE-BYTES")) {
            const std::size_t maxResponseSizeBytes = std::stoll(
                headers.find("NVCF-MAX-RESPONSE-SIZE-BYTES")->second);
            if (resultJs.dump().size() > maxResponseSizeBytes) {
              // If the response size is larger than the limit, write it to the
              // large output directory rather than sending it back as an HTTP
              // response.
              const auto outputDirIter = headers.find("NVCF-LARGE-OUTPUT-DIR");
              const auto reqIdIter = headers.find("NVCF-REQID");
              if (outputDirIter == headers.end() ||
                  reqIdIter == headers.end()) {
                json js;
                js["status"] =
                    "Failed to locate output file location for large response.";
                return js;
              }

              const std::string outputDir = outputDirIter->second;
              const std::string fileName = reqIdIter->second + "_result.json";
              const std::filesystem::path outputFile =
                  std::filesystem::path(outputDir) / fileName;
              std::ofstream file(outputFile.string());
              file << resultJs.dump();
              file.flush();
              json js;
              js["resultFile"] = fileName;
              return js;
            }
          }

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
    if (requestInfo.format == cudaq::CodeFormat::LLVM) {
      if (io_context.name == "sample") {
        // In library mode (LLVM), check to see if we have mid-circuit measures
        // by tracing the kernel function.
        cudaq::ExecutionContext context("tracer");
        platform.set_exec_ctx(&context);
        cudaq::invokeWrappedKernel(ir, std::string(kernelName), kernelArgs,
                                   argsSize);
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
          // If it has conditionals, loop over individual circuit executions
          cudaq::invokeWrappedKernel(ir, std::string(kernelName), kernelArgs,
                                     argsSize, io_context.shots,
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
          platform.set_exec_ctx(&io_context);
        } else {
          // If no conditionals, nothing special to do for library mode
          platform.set_exec_ctx(&io_context);
          cudaq::invokeWrappedKernel(ir, std::string(kernelName), kernelArgs,
                                     argsSize);
        }
      } else {
        platform.set_exec_ctx(&io_context);
        cudaq::invokeWrappedKernel(ir, std::string(kernelName), kernelArgs,
                                   argsSize);
      }
    } else {
      platform.set_exec_ctx(&io_context);
      if (io_context.name == "sample" &&
          io_context.hasConditionalsOnMeasureResults) {
        // Need to run simulation shot-by-shot
        cudaq::sample_result counts;
        invokeMlirKernel(m_mlirContext, ir, requestInfo.passes,
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
        platform.set_exec_ctx(&io_context);
      } else {
        invokeMlirKernel(m_mlirContext, ir, requestInfo.passes,
                         std::string(kernelName));
      }
    }
    platform.reset_exec_ctx();
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

      cudaq::info("- Pass manager was applied.");
    }
    // Verify MLIR conforming to the NVQIR-spec (known runtime functions and/or
    // QIR functions)
    {
      // Note: run this verification as a standalone step to decouple IR
      // conversion and verfication.
      PassManager pm(ctx);
      pm.addNestedPass<LLVM::LLVMFuncOp>(
          cudaq::opt::createVerifyNVQIRCallOpsPass());
      if (failed(pm.run(module)))
        throw std::runtime_error(
            "Failed to IR compliance verification against NVQIR runtime.");

      cudaq::info("- Finish IR input verification.");
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

    cudaq::info("- Creating the MLIR ExecutionEngine");
    auto uniqueJit = llvm::cantFail(ExecutionEngine::create(module, opts));
    cudaq::info("- MLIR ExecutionEngine created successfully.");
    return uniqueJit;
  }

  void
  invokeMlirKernel(std::unique_ptr<MLIRContext> &contextPtr,
                   std::string_view irString,
                   const std::vector<std::string> &passes,
                   const std::string &entryPointFn, std::size_t numTimes = 1,
                   std::function<void(std::size_t)> postExecCallback = {}) {
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
    for (std::size_t i = 0; i < numTimes; ++i) {
      // Invoke the kernel
      fn();
      if (postExecCallback) {
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
    try {
      // IMPORTANT: This assumes the REST server handles incoming requests
      // sequentially.
      static std::size_t g_requestCounter = 0;
      auto requestJson = json::parse(reqBody);
      cudaq::RestRequest request(requestJson);
      cudaq::info(
          "[RemoteRestRuntimeServer] Incoming job request from client {}",
          request.clientVersion);
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
      std::vector<char> decodedCodeIr;
      if (llvm::decodeBase64(request.code, decodedCodeIr)) {
        throw std::runtime_error("Failed to decode input IR");
      }
      std::string_view codeStr(decodedCodeIr.data(), decodedCodeIr.size());
      handleRequest(reqId, request.executionContext, request.simulator, codeStr,
                    request.entryPoint, request.args.data(),
                    request.args.size(), request.seed);
      json resultJson;
      resultJson["executionContext"] = request.executionContext;
      m_codeTransform.erase(reqId);
      return resultJson;
    } catch (std::exception &e) {
      json resultJson;
      resultJson["status"] = "Failed to process incoming request";
      resultJson["errorMessage"] = e.what();
      return resultJson;
    }
  }
};

// Runtime server for NVCF
class NvcfRuntimeServer : public RemoteRestRuntimeServer {
protected:
  virtual bool filterRequest(const cudaq::RestRequest &in_request,
                             std::string &outValidationMessage) const override {
    // We only support MLIR payload on the NVCF server.
    if (in_request.format != cudaq::CodeFormat::MLIR) {
      outValidationMessage =
          "Unsupported input format: only CUDA Quantum MLIR data is allowed.";
      return false;
    }

    return true;
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeServer, RemoteRestRuntimeServer, rest)
CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeServer, NvcfRuntimeServer, nvcf)
