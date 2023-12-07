/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "JsonConvert.h"
#include "RestServer.h"
#include "common/Logger.h"
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
#include "llvm/IR/Module.h"
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
namespace nvqir {
CircuitSimulator *getCircuitSimulatorInternal();
}
using namespace mlir;
namespace {
std::unique_ptr<ExecutionEngine>
jitCode(ModuleOp currentModule, std::vector<std::string> extraLibPaths = {}) {
  cudaq::info("Running jitCode.");
  auto module = currentModule.clone();
  auto ctx = module.getContext();
  PassManager pm(ctx);
  OpPassManager &optPM = pm.nest<func::FuncOp>();

  optPM.addPass(cudaq::opt::createUnwindLoweringPass());
  cudaq::opt::addAggressiveEarlyInlining(pm);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createApplyOpSpecializationPass());
  optPM.addPass(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addPass(cudaq::opt::createLoopNormalize());
  pm.addPass(cudaq::opt::createLoopUnroll());
  pm.addPass(createCanonicalizerPass());
  optPM.addPass(cudaq::opt::createQuakeAddDeallocs());
  optPM.addPass(cudaq::opt::createQuakeAddMetadata());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (failed(pm.run(module)))
    throw std::runtime_error("Failed to JIT compile the Quake representation.");

  optPM.addPass(cudaq::opt::createLowerToCFGPass());
  optPM.addPass(cudaq::opt::createCombineQuantumAllocations());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(cudaq::opt::createConvertToQIRPass());

  if (failed(pm.run(module)))
    throw std::runtime_error("Failed to JIT compile the Quake representation.");

  cudaq::info("- Pass manager was applied.");
  ExecutionEngineOptions opts;
  opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
  opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
  SmallVector<StringRef, 4> sharedLibs;
  for (auto &lib : extraLibPaths) {
    cudaq::info("Extra library loaded: {}", lib);
    sharedLibs.push_back(lib);
  }
  opts.sharedLibPaths = sharedLibs;
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

  cudaq::info(" - Creating the MLIR ExecutionEngine");
  auto jitOrError = ExecutionEngine::create(module, opts);
  assert(!!jitOrError);

  auto uniqueJit = std::move(jitOrError.get());

  cudaq::info("- JIT Engine created successfully.");

  return uniqueJit;
}

void invokeKernel(cudaq::ExecutionContext &io_executionContext,
                  const std::string &irString,
                  const std::string &entryPointFn) {
  auto contextPtr = cudaq::initializeMLIR();
  auto fileBuf = llvm::MemoryBuffer::getMemBufferCopy(irString);
  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(fileBuf), llvm::SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, contextPtr.get());
  auto engine = jitCode(*module);
  //  Extract the entry point
  auto fnPtr = engine->lookup(entryPointFn);

  if (!fnPtr)
    throw std::runtime_error("Failed to get entry function");

  auto fn = reinterpret_cast<void (*)()>(*fnPtr);
  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();
  circuitSimulator->setExecutionContext(&io_executionContext);
  fn();
  circuitSimulator->resetExecutionContext();
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
  cudaq::info("Request simulator {} at {}", simulatorName, simLibPath.c_str());
  void *simLibHandle = dlopen(simLibPath.c_str(), RTLD_GLOBAL | RTLD_NOW);
  if (!simLibHandle) {
    char *error_msg = dlerror();
    throw std::runtime_error(fmt::format(
        "Failed to open simulator backend library: {}.",
        error_msg ? std::string(error_msg) : std::string("Unknown error")));
  }
  return simLibHandle;
}

json processRequest(const std::string &reqBody) {
  auto requestJson = json::parse(reqBody);
  cudaq::RestRequest request(requestJson);
  const std::string entryPointFunc =
      std::string(cudaq::runtime::cudaqGenPrefixName) + request.entryPoint;
  void *handle = loadNvqirSimLib(request.simulator);
  if (request.seed != 0)
    cudaq::set_random_seed(request.seed);
  invokeKernel(request.executionContext, request.code, entryPointFunc);
  json resultContextJs = request.executionContext;
  dlclose(handle);
  return resultContextJs;
}
} // namespace

int main(int argc, char **argv) {
  constexpr int DEFAULT_PORT = 3030;
  const int port = [&]() {
    for (int i = 1; i < argc - 1; ++i)
      if (std::string(argv[i]) == "-p" || std::string(argv[i]) == "-port" ||
          std::string(argv[i]) == "--port")
        return atoi(argv[i + 1]);

    return DEFAULT_PORT;
  }();
  cudaq::mpi::initialize();
  cudaq::RestServer server(port);
  server.addRoute(cudaq::RestServer::Method::GET, "/",
                  [](const std::string &reqBody) {
                    // Return an empty JSON string,
                    // e.g., for client to ping the server.
                    return json();
                  });

  // Request the server to shut down.
  server.addRoute(cudaq::RestServer::Method::POST, "/shutdown",
                  [&server](const std::string &reqBody) {
                    std::string shutDownMsg = "SHUTDOWN";
                    cudaq::mpi::broadcast(shutDownMsg, 0);
                    server.stop();
                    return json();
                  });

  // New simulation request.
  server.addRoute(cudaq::RestServer::Method::POST, "/job",
                  [](const std::string &reqBody) {
                    std::string mutableReq = reqBody;
                    cudaq::mpi::broadcast(mutableReq, 0);
                    return processRequest(reqBody);
                  });

  if (cudaq::mpi::rank() == 0) {
    // Only run this app on Rank 0;
    // the rest will wait for a broadcast.
    server.start();
  } else {
    for (;;) {
      std::string jsonRequestBody;
      cudaq::mpi::broadcast(jsonRequestBody, 0);
      if (jsonRequestBody == "SHUTDOWN")
        break;
      try {
        // All ranks need to join, e.g., MPI-capable backends.
        processRequest(jsonRequestBody);
      } catch (...) {
        // Doing nothing, the root rank will report the error
      }
    }
  }
  cudaq::mpi::finalize();
}
