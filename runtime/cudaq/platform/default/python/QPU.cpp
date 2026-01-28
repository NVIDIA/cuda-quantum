/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QPU.h"
#include "common/ArgumentConversion.h"
#include "common/Environment.h"
#include "common/RuntimeMLIR.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/AddMetadata.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

static void specializeKernel(const std::string &name, ModuleOp module,
                             const std::vector<void *> &rawArgs,
                             Type resultTy = {},
                             bool enablePythonCodegenDump = false) {
  PassManager pm(module.getContext());
  cudaq::opt::ArgumentConverter argCon(name, module);
  argCon.gen(name, module, rawArgs);
  SmallVector<std::string> kernels;
  SmallVector<std::string> substs;
  for (auto *kInfo : argCon.getKernelSubstitutions()) {
    std::string kernName =
        cudaq::runtime::cudaqGenPrefixName + kInfo->getKernelName().str();
    kernels.emplace_back(kernName);
    std::string substBuff;
    llvm::raw_string_ostream ss(substBuff);
    ss << kInfo->getSubstitutionModule();
    substs.emplace_back(substBuff);
  }

  // Collect references for the argument synthesis.
  SmallVector<StringRef> kernelRefs{kernels.begin(), kernels.end()};
  SmallVector<StringRef> substRefs{substs.begin(), substs.end()};

  // Run a pass manager to specialize & optimize the kernel to be launched.
  pm.addPass(cudaq::opt::createArgumentSynthesisPass(
      kernelRefs, substRefs, /*changeSemantics=*/false));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createLambdaLifting({.constantPropagation = true}));
  // We must inline these lambda calls before apply specialization as it does
  // not perform control/adjoint specialization across function call boundary.
  cudaq::opt::addAggressiveInlining(pm);
  pm.addPass(
      cudaq::opt::createApplySpecialization({.constantPropagation = true}));
  cudaq::opt::addAggressiveInlining(pm);
  pm.addPass(cudaq::opt::createDistributedDeviceCall());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  if (resultTy) {
    // If we're expecting a result, then we want to call the .thunk function so
    // that the result is properly marshaled. Add the GKE pass to generate the
    // .thunk. At this point, the kernel should have been specialized so it has
    // an arity of 0.
    pm.addPass(
        cudaq::opt::createGenerateKernelExecution({.positNullary = true}));
  }
  pm.addPass(createSymbolDCEPass());
  if (enablePythonCodegenDump) {
    module.getContext()->disableMultithreading();
    pm.enableIRPrinting();
  }
  if (failed(pm.run(module)))
    throw std::runtime_error("Could not successfully apply argument synth.");
}

/// Lowers \p module to LLVM code. The LLVM code will use "full QIR" as the
/// transport layer. If \p kernelName and \p args are provided, they will
/// specialize the selected entry-point kernel.
std::string cudaq::detail::lower_to_qir_llvm(const std::string &name,
                                             ModuleOp module,
                                             OpaqueArguments &args,
                                             const std::string &format) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getQIR", name);
  // Translate the module to QIR transport layer (as LLVM code).
  cudaq::detail::mergeAllCallableClosures(module, name, args.getArgs());
  specializeKernel(name, module, args.getArgs());
  PassManager pm(module.getContext());
  cudaq::opt::addAggressiveInlining(pm);
  cudaq::opt::createTargetFinalizePipeline(pm);
  cudaq::opt::addAOTPipelineConvertToQIR(pm, format);
  if (failed(pm.run(module)))
    throw std::runtime_error("Conversion to " + format + " failed.");
  llvm::LLVMContext llvmContext;
  llvmContext.setOpaquePointers(false);
  std::unique_ptr<llvm::Module> llvmModule =
      translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
    return "{translation failed}";
  std::string result;
  llvm::raw_string_ostream os(result);
  llvmModule->print(os, nullptr);
  os.flush();
  return result;
}

/// Lowers \p module to `Open QASM 2`. The output will be a string of `Open
/// QASM` code. \p kernelName and \p args should be provided, as they will
/// specialize the selected entry-point kernel.
std::string cudaq::detail::lower_to_openqasm(const std::string &name,
                                             ModuleOp module,
                                             OpaqueArguments &args) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getASM", name);
  // Translate module to OpenQASM2 transport layer.
  cudaq::detail::mergeAllCallableClosures(module, name, args.getArgs());
  specializeKernel(name, module, args.getArgs());
  auto *ctx = module.getContext();
  PassManager pm(ctx);
  cudaq::opt::createTargetFinalizePipeline(pm);
  cudaq::opt::createPipelineTransformsForPythonToOpenQASM(pm);
  cudaq::opt::addPipelineTranslateToOpenQASM(pm);
  const bool enablePrintMLIRBeforeAndAfterEachPass =
      cudaq::getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);
  if (enablePrintMLIRBeforeAndAfterEachPass) {
    ctx->disableMultithreading();
    pm.enableIRPrinting();
  }
  if (failed(pm.run(module)))
    throw std::runtime_error("Conversion to OpenQASM failed.");
  std::string result;
  llvm::raw_string_ostream os(result);
  if (failed(cudaq::translateToOpenQASM(module, os)))
    return "{translation failed}";
  os.flush();
  return result;
}

/// Scan \p module and set flags in the current platform context accordingly.
static void establishExecutionContext(ModuleOp module) {
  auto &plat = cudaq::get_platform();
  auto *currentExecCtx = plat.get_exec_ctx();
  if (!currentExecCtx)
    return;

  for (auto &artifact : module) {
    quake::detail::QuakeFunctionAnalysis analysis{&artifact};
    auto info = analysis.getAnalysisInfo();
    if (info.empty())
      continue;
    auto result = info[&artifact];
    if (result.hasConditionalsOnMeasure) {
      currentExecCtx->hasConditionalsOnMeasureResults = true;
      break;
    }
  }

  plat.set_exec_ctx(currentExecCtx);
}

static ExecutionEngine *alreadyBuiltJITCode() {
  auto *currentExecCtx = cudaq::get_platform().get_exec_ctx();
  if (!currentExecCtx || !currentExecCtx->allowJitEngineCaching)
    return {};
  return reinterpret_cast<ExecutionEngine *>(currentExecCtx->jitEng);
}

/// In a sample launch context, the (`JIT` compiled) execution engine may be
/// cached so that it can be called many times in a loop without being
/// recompiled. This exploits the fact that the arguments processed at the
/// sample callsite are invariant by the definition of a `CUDA-Q` kernel.
static bool cacheJITForPerformance(ExecutionEngine *jit) {
  auto *currentExecCtx = cudaq::get_platform().get_exec_ctx();
  if (currentExecCtx && currentExecCtx->allowJitEngineCaching) {
    if (!currentExecCtx->jitEng)
      currentExecCtx->jitEng = reinterpret_cast<void *>(jit);
    return true;
  }
  return false;
}

namespace {
struct PythonLauncher : public cudaq::ModuleLauncher {
  cudaq::KernelThunkResultType launchModule(const std::string &name,
                                            ModuleOp module,
                                            const std::vector<void *> &rawArgs,
                                            Type resultTy) override {
    // In this launch scenario, we have a ModuleOp that has the entry-point
    // kernel, but needs to be merged with anything else it may call. The
    // merging of modules mirrors the late binding and dynamic scoping of the
    // host language (Python).
    ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::launchModule");
    const bool enablePythonCodegenDump =
        cudaq::getEnvBool("CUDAQ_PYTHON_CODEGEN_DUMP", false);

    std::string fullName = cudaq::runtime::cudaqGenPrefixName + name;
    cudaq::KernelThunkResultType result{nullptr, 0};
    ExecutionEngine *jit = alreadyBuiltJITCode();
    if (!jit) {
      // 1. Check that this call is sane.
      if (enablePythonCodegenDump)
        module.dump();
      auto funcOp = module.lookupSymbol<func::FuncOp>(fullName);
      if (!funcOp)
        throw std::runtime_error("no kernel named " + name +
                                 " found in module");

      // 2. Merge other modules (e.g., if there are device kernel calls).
      cudaq::detail::mergeAllCallableClosures(module, name, rawArgs);

      // Mark all newly merged kernels private.
      for (auto &op : module)
        if (auto f = dyn_cast<func::FuncOp>(op))
          if (f != funcOp)
            f.setPrivate();

      establishExecutionContext(module);

      // 3. LLVM JIT the code so we can execute it.
      CUDAQ_INFO("Run Argument Synth.\n");
      if (enablePythonCodegenDump)
        module.dump();
      specializeKernel(name, module, rawArgs, resultTy,
                       enablePythonCodegenDump);

      // 4. Execute the code right here, right now.
      jit = cudaq::createQIRJITEngine(module, "qir:");
    }

    if (resultTy) {
      // Proceed to call the .thunk function so that the result value will be
      // properly marshaled into the buffer we allocated in
      // appendTheResultBuffer().
      // FIXME: Python ought to set up the call stack so that a legit C++ entry
      // point can be called instead of winging it and duplicating what the core
      // compiler already does.
      auto funcPtr = jit->lookup(name + ".thunk");
      if (!funcPtr)
        throw std::runtime_error(
            "kernel disappeared underneath execution engine");
      void *buff = const_cast<void *>(rawArgs.back());
      result = reinterpret_cast<cudaq::KernelThunkResultType (*)(void *, bool)>(
          *funcPtr)(buff, /*client_server=*/false);
    } else {
      auto funcPtr = jit->lookup(fullName);
      if (!funcPtr)
        throw std::runtime_error(
            "kernel disappeared underneath execution engine");
      reinterpret_cast<void (*)()>(*funcPtr)();
    }
    if (!cacheJITForPerformance(jit))
      delete jit;
    // FIXME: actually handle results
    return result;
  }

  void *specializeModule(const std::string &name, ModuleOp module,
                         const std::vector<void *> &rawArgs, Type resultTy,
                         void *cachedEngine) override {
    // In this launch scenario, we have a ModuleOp that has the entry-point
    // kernel, but needs to be merged with anything else it may call. The
    // merging of modules mirrors the late binding and dynamic scoping of the
    // host language (Python).
    ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::launchModule");
    const bool enablePythonCodegenDump =
        cudaq::getEnvBool("CUDAQ_PYTHON_CODEGEN_DUMP", false);

    std::string fullName = cudaq::runtime::cudaqGenPrefixName + name;
    // 1. Check that this call is sane.
    if (enablePythonCodegenDump)
      module.dump();
    auto funcOp = module.lookupSymbol<func::FuncOp>(fullName);
    if (!funcOp)
      throw std::runtime_error("no kernel named " + name + " found in module");

    // 2. Merge other modules (e.g., if there are device kernel calls).
    cudaq::detail::mergeAllCallableClosures(module, name, rawArgs);

    // Mark all newly merged kernels private.
    for (auto &op : module)
      if (auto f = dyn_cast<func::FuncOp>(op))
        if (f != funcOp)
          f.setPrivate();

    establishExecutionContext(module);

    // 3. LLVM JIT the code so we can execute it.
    CUDAQ_INFO("Run Argument Synth.\n");
    if (enablePythonCodegenDump)
      module.dump();
    specializeKernel(name, module, rawArgs, resultTy, enablePythonCodegenDump);

    // 4. Execute the code right here, right now.
    auto *jit = cudaq::createQIRJITEngine(module, "qir:");
    auto **cache = reinterpret_cast<ExecutionEngine **>(cachedEngine);
    if (*cache)
      throw std::runtime_error("cache must not be populated");
    *cache = jit;

    std::string entryName = resultTy ? name + ".thunk" : fullName;
    auto funcPtr = jit->lookup(entryName);
    if (!funcPtr)
      throw std::runtime_error(
          "kernel disappeared underneath execution engine");
    return *funcPtr;
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::ModuleLauncher, PythonLauncher, default)
