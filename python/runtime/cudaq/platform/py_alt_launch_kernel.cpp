/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CAPI/Dialects.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/CodeGen/Pipelines.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/platform.h"
#include "cudaq/platform/qpu.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/ExecutionEngine.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include <fmt/core.h>
#include <pybind11/stl.h>

#include "JITExecutionCache.h"
#include "utils/OpaqueArguments.h"

namespace py = pybind11;
using namespace mlir;

namespace cudaq {
static std::unique_ptr<JITExecutionCache> jitCache;

std::tuple<ExecutionEngine *, void *, std::size_t>
jitAndCreateArgs(const std::string &name, MlirModule module,
                 cudaq::OpaqueArguments &runtimeArgs,
                 const std::vector<std::string> &names) {
  auto mod = unwrap(module);
  auto cloned = mod.clone();
  auto context = cloned.getContext();
  registerLLVMDialectTranslation(*context);
  // {
  //   static std::mutex g_mutex;
  //   static std::unordered_set<mlir::MLIRContext *> g_knownContexts;
  //   std::scoped_lock<std::mutex> lock(g_mutex);
  //   if (!g_knownContexts.contains(context)) {
  //     registerLLVMDialectTranslation(*context);
  //     g_knownContexts.emplace(context);
  //   }
  // }

  // Have we JIT compiled this before?
  std::string moduleString;
  {
    llvm::raw_string_ostream os(moduleString);
    cloned.print(os);
  }
  std::hash<std::string> hasher;
  auto hashKey = hasher(moduleString);

  ExecutionEngine *jit = nullptr;
  if (jitCache->hasJITEngine(hashKey))
    jit = jitCache->getJITEngine(hashKey);
  else {

    PassManager pm(context);
    pm.addNestedPass<func::FuncOp>(
        cudaq::opt::createPySynthCallableBlockArgs(names));
    pm.addPass(cudaq::opt::createGenerateDeviceCodeLoader(/*genAsQuake=*/true));
    pm.addPass(cudaq::opt::createGenerateKernelExecution());
    pm.addPass(cudaq::opt::createLambdaLiftingPass());
    cudaq::opt::addPipelineToQIR<>(pm);
    if (failed(pm.run(cloned)))
      throw std::runtime_error(
          "cudaq::builder failed to JIT compile the Quake representation.");

    ExecutionEngineOptions opts;
    opts.transformer = [](llvm::Module *m) { return llvm::ErrorSuccess(); };
    opts.jitCodeGenOptLevel = llvm::CodeGenOpt::None;
    SmallVector<StringRef, 4> sharedLibs;
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

    auto jitOrError = ExecutionEngine::create(cloned, opts);
    assert(!!jitOrError);
    auto uniqueJit = std::move(jitOrError.get());
    jit = uniqueJit.release();
    jitCache->cache(hashKey, jit);
  }

  void *rawArgs = nullptr;
  std::size_t size = 0;
  if (runtimeArgs.size()) {
    auto expectedPtr = jit->lookup(name + ".argsCreator");
    if (!expectedPtr) {
      throw std::runtime_error(
          "cudaq::builder failed to get argsCreator function.");
    }
    auto argsCreator =
        reinterpret_cast<std::size_t (*)(void **, void **)>(*expectedPtr);
    rawArgs = nullptr;
    size = argsCreator(runtimeArgs.data(), &rawArgs);
  }
  return std::make_tuple(jit, rawArgs, size);
}

void pyAltLaunchKernel(const std::string &name, MlirModule module,
                       cudaq::OpaqueArguments &runtimeArgs,
                       const std::vector<std::string> &names) {
  auto [jit, rawArgs, size] =
      jitAndCreateArgs(name, module, runtimeArgs, names);

  auto mod = unwrap(module);
  auto thunkName = name + ".thunk";
  auto thunkPtr = jit->lookup(thunkName);
  if (!thunkPtr)
    throw std::runtime_error("cudaq::builder failed to get thunk function");

  auto thunk = reinterpret_cast<void (*)(void *)>(*thunkPtr);

  std::string properName = name;

  // Need to first invoke the init_func()
  auto kernelInitFunc = properName + ".init_func";
  auto initFuncPtr = jit->lookup(kernelInitFunc);
  if (!initFuncPtr) {
    throw std::runtime_error(
        "cudaq::builder failed to get kernelReg function.");
  }
  auto kernelInit = reinterpret_cast<void (*)()>(*initFuncPtr);
  kernelInit();

  // Need to first invoke the kernelRegFunc()
  auto kernelRegFunc = properName + ".kernelRegFunc";
  auto regFuncPtr = jit->lookup(kernelRegFunc);
  if (!regFuncPtr) {
    throw std::runtime_error(
        "cudaq::builder failed to get kernelReg function.");
  }
  auto kernelReg = reinterpret_cast<void (*)()>(*regFuncPtr);
  kernelReg();

  auto &platform = cudaq::get_platform();
  if (platform.is_remote() || platform.is_emulated()) {
    struct ArgWrapper {
      ModuleOp mod;
      std::vector<std::string> callableNames;
      void *rawArgs = nullptr;
    };
    auto *wrapper = new ArgWrapper{mod, names, rawArgs};
    cudaq::altLaunchKernel(name.c_str(), thunk,
                           reinterpret_cast<void *>(wrapper), size, 0);
    delete wrapper;
  } else
    cudaq::altLaunchKernel(name.c_str(), thunk, rawArgs, size, 0);

  std::free(rawArgs);
}

MlirModule synthesizeKernel(const std::string &name, MlirModule module,
                            cudaq::OpaqueArguments &runtimeArgs) {
  auto [jit, rawArgs, size] = jitAndCreateArgs(name, module, runtimeArgs, {});
  auto cloned = unwrap(module).clone();
  auto context = cloned.getContext();
  registerLLVMDialectTranslation(*context);

  PassManager pm(context);
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createQuakeSynthesizer(name, rawArgs));
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createLoopNormalize());
  pm.addPass(cudaq::opt::createLoopUnroll());
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(cloned)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");
  std::free(rawArgs);
  return wrap(cloned);
}

std::string getQIRLL(const std::string &name, MlirModule module,
                     cudaq::OpaqueArguments &runtimeArgs,
                     std::string &profile) {
  auto [jit, rawArgs, size] = jitAndCreateArgs(name, module, runtimeArgs, {});
  auto cloned = unwrap(module).clone();
  auto context = cloned.getContext();
  registerLLVMDialectTranslation(*context);

  PassManager pm(context);
  if (profile.empty())
    cudaq::opt::addPipelineToQIR<>(pm);
  else
    cudaq::opt::addPipelineToQIR<true>(pm, profile);
  if (failed(pm.run(cloned)))
    throw std::runtime_error(
        "cudaq::builder failed to JIT compile the Quake representation.");
  std::free(rawArgs);

  llvm::LLVMContext llvmContext;
  llvmContext.setOpaquePointers(false);
  auto llvmModule = translateModuleToLLVMIR(cloned, llvmContext);
  auto optPipeline = makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  if (auto err = optPipeline(llvmModule.get()))
    throw std::runtime_error("Failed to optimize LLVM IR ");

  std::string str;
  {
    llvm::raw_string_ostream os(str);
    llvmModule->print(os, nullptr);
  }
  return str;
}

void bindAltLaunchKernel(py::module &mod) {
  jitCache = std::make_unique<JITExecutionCache>();

  auto callableArgHandler = [](cudaq::OpaqueArguments &argData,
                               py::object &arg) {
    if (py::hasattr(arg, "module")) {
      // Just give it some dummy data that will not be used.
      // We synthesize away all callables, the block argument
      // remains but it is not used, so just give argsCreator
      // something, and we'll make sure its cleaned up.
      long *ourAllocatedArg = new long();
      argData.emplace_back(ourAllocatedArg,
                           [](void *ptr) { delete static_cast<long *>(ptr); });
      return true;
    }
    return false;
  };

  mod.def(
      "pyAltLaunchKernel",
      [&](const std::string &kernelName, MlirModule module,
          py::args runtimeArgs, std::vector<std::string> callable_names) {
        cudaq::OpaqueArguments args;
        cudaq::packArgs(args, runtimeArgs, callableArgHandler);
        pyAltLaunchKernel(kernelName, module, args, callable_names);
      },
      py::arg("kernelName"), py::arg("module"), py::kw_only(),
      py::arg("callable_names") = std::vector<std::string>{}, "DOC STRING");

  mod.def("synthesize", [](py::object kernel, py::args runtimeArgs) {
    MlirModule module = kernel.attr("module").cast<MlirModule>();
    auto name = kernel.attr("name").cast<std::string>();
    cudaq::OpaqueArguments args;
    cudaq::packArgs(args, runtimeArgs);
    return synthesizeKernel(name, module, args);
  });

  mod.def(
      "get_qir",
      [](py::object kernel, py::args runtimeArgs, std::string profile) {
        MlirModule module = kernel.attr("module").cast<MlirModule>();
        auto name = kernel.attr("name").cast<std::string>();
        cudaq::OpaqueArguments args;
        cudaq::packArgs(args, runtimeArgs);
        return getQIRLL(name, module, args, profile);
      },
      py::arg("kernel"), py::kw_only(), py::arg("profile") = "");
}
} // namespace cudaq