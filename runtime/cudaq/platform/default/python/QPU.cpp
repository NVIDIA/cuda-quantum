/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QPU.h"
#include "common/ArgumentWrapper.h"
#include "common/CompiledKernel.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/AddMetadata.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Verifier/QIRLLVMIRDialect.h"
#include "cudaq_internal/compiler/ArgumentConversion.h"
#include "cudaq_internal/compiler/JIT.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include <cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h>

using namespace mlir;
using namespace cudaq_internal::compiler;
using cudaq::JitEngine;

static void specializeKernel(const std::string &name, ModuleOp module,
                             const std::vector<void *> &rawArgs,
                             Type resultTy = {},
                             bool enablePythonCodegenDump = false,
                             bool isEntryPoint = true,
                             bool isFullySpecialized = true) {
  PassManager pm(module.getContext());
  cudaq_internal::compiler::ArgumentConverter argCon(name, module);
  // Look up the kernel's type signature.
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
  // Run GKE to generate `.thunk` / `.argsCreator` when the kernel has a result
  // or any unspecialized arguments so they can be properly marshaled
  if (isEntryPoint && (resultTy || !isFullySpecialized)) {
    pm.addPass(cudaq::opt::createGenerateKernelExecution(
        {.positNullary = isFullySpecialized, .ignoreHostFunction = true}));
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
  mergeAllCallableClosures(module, name, args.getArgs());
  specializeKernel(name, module, args.getArgs());
  PassManager pm(module.getContext());
  cudaq::opt::addAggressiveInlining(pm);
  cudaq::opt::createTargetFinalizePipeline(pm);
  cudaq::opt::addAOTPipelineConvertToQIR(pm, format);
  if (failed(pm.run(module)))
    throw std::runtime_error("Conversion to " + format + " failed.");
  if (failed(cudaq::verifier::checkQIRLLVMIRDialect(module, format)))
    throw std::runtime_error("QIR conformance failed.");
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
  mergeAllCallableClosures(module, name, args.getArgs());
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
static void updateExecutionContext(ModuleOp module) {
  auto *currentExecCtx = cudaq::getExecutionContext();
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
}

static std::optional<JitEngine>
alreadyBuiltJITCode(const std::string &name,
                    const std::vector<void *> &rawArgs) {
  auto *currentExecCtx = cudaq::getExecutionContext();
  if (currentExecCtx && currentExecCtx->allowJitEngineCaching) {
    auto jit = currentExecCtx->jitEng;
    if (jit && cudaq::compiler_artifact::isPersistingJITEngine()) {
      CUDAQ_INFO("Loading previously compiled JIT engine for {}. This will "
                 "re-run the previous job, discarding any changes to the "
                 "kernel, arguments or launch configuration.",
                 currentExecCtx->kernelName);
      cudaq::compiler_artifact::checkArtifactReuse(name, jit.value());
    }
    return jit;
  }

  // Fallback for callers without an ExecutionContext (e.g. direct kernel
  // calls): look up the artifact saved by a previous compilation.
  return cudaq::compiler_artifact::getArtifactJit(name);
}

/// In a sample launch context, the (`JIT` compiled) execution engine may be
/// cached so that it can be called many times in a loop without being
/// recompiled. This exploits the fact that the arguments processed at the
/// sample callsite are invariant by the definition of a `CUDA-Q` kernel.
static void cacheJITForPerformance(JitEngine jit) {
  auto *currentExecCtx = cudaq::getExecutionContext();
  if (currentExecCtx && currentExecCtx->allowJitEngineCaching) {
    if (!currentExecCtx->jitEng)
      currentExecCtx->jitEng = jit;
  }
}

namespace {
struct PythonLauncher : public cudaq::ModuleLauncher {
  cudaq::CompiledKernel compileModule(const std::string &name, ModuleOp module,
                                      const std::vector<void *> &rawArgs,
                                      bool isEntryPoint) override {

    ScopedTraceWithContext(cudaq::TIMING_LAUNCH,
                           "PythonLauncher::compileModule");
    const bool enablePythonCodegenDump =
        cudaq::getEnvBool("CUDAQ_PYTHON_CODEGEN_DUMP", false);

    std::string fullName = cudaq::runtime::cudaqGenPrefixName + name;

    auto funcOp = module.lookupSymbol<func::FuncOp>(fullName);
    if (!funcOp)
      throw std::runtime_error("no kernel named " + name + " found in module");
    Type resultTy = cudaq::runtime::getReturnType(funcOp);

    const bool hasResult = !!resultTy;
    auto resultInfo = createResultInfo(resultTy, isEntryPoint, module);

    // Determine whether the kernel needs argument packing (argsCreator) by
    // checking if any non-callable arguments are present. This must be done
    // before the cache lookup so the cached path uses the correct value.
    bool isFullySpecialized = true;
    FunctionType fromFuncTy = funcOp.getFunctionType();
    // Specialization for direct calls will take care of partial specialization
    // separately
    bool isLocalSimulator =
        !(cudaq::is_remote_platform() || cudaq::is_emulated_platform());

    std::vector<void *> closureArgs;

    // Special handling in case the arguments were already synthesized
    // TODO: should ensure args have no uses if this is the case?
    size_t numArgs = rawArgs.size() - (hasResult ? 1 : 0);
    if (isEntryPoint && isLocalSimulator &&
        numArgs == fromFuncTy.getNumInputs()) {
      // TODO: is copying even necessary here? Or can we just overwrite?
      closureArgs = rawArgs;
      for (auto [i, ty] : llvm::enumerate(fromFuncTy.getInputs())) {
        if (!isa<cudaq::cc::CallableType>(ty)) {
          isFullySpecialized = false;
          closureArgs[i] = nullptr;
        }
      }
    } else {
      // Avoid copying
      closureArgs = std::move(rawArgs);
    }

    if (auto jit = alreadyBuiltJITCode(name, rawArgs)) {
      cudaq::CompiledKernel ck(name, resultInfo);
      ck.attachJit(*jit, isFullySpecialized);
      return ck;
    }

    // 1. Check that this call is sane.
    if (enablePythonCodegenDump)
      module.dump();

    // 2. Merge other modules (e.g., if there are device kernel calls).
    mergeAllCallableClosures(module, name, rawArgs);

    // Mark all newly merged kernels private.
    for (auto &op : module)
      if (auto f = dyn_cast<func::FuncOp>(op))
        if (f != funcOp)
          f.setPrivate();

    updateExecutionContext(module);

    // 3. Specialize the kernel (argument synthesis, optimization).
    CUDAQ_INFO("Run Argument Synth.\n");
    if (enablePythonCodegenDump)
      module.dump();

    specializeKernel(name, module, closureArgs, resultTy,
                     enablePythonCodegenDump, isEntryPoint, isFullySpecialized);

    // 4. Lower to QIR and JIT compile.
    auto jit = createJITEngine(module, "qir:");
    cacheJITForPerformance(jit);
    cudaq::compiler_artifact::saveArtifact(name, jit);

    cudaq::CompiledKernel ck(name, resultInfo);
    ck.attachJit(jit, isFullySpecialized);
    return ck;
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::ModuleLauncher, PythonLauncher, default)
