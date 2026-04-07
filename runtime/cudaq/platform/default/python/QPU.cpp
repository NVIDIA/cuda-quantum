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
#include "common/RuntimeTarget.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/AddMetadata.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Optimizer/Transforms/ResourceCount.h"
#include "cudaq/Verifier/QIRLLVMIRDialect.h"
#include "cudaq/platform.h"
#include "cudaq_internal/compiler/ArgumentConversion.h"
#include "cudaq_internal/compiler/JIT.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include <unordered_set>

// Declared in runtime/cudaq/algorithms/resource_estimation.h (not included
// here to avoid pulling in cudaq/platform.h which creates circular deps).
namespace nvqir {
void setResourceCounts(cudaq::Resources &&);
}

using namespace mlir;
using namespace cudaq_internal::compiler;
using cudaq::JitEngine;

static void
specializeKernel(const std::string &name, ModuleOp module,
                 const std::vector<void *> &rawArgs, Type resultTy = {},
                 bool enablePythonCodegenDump = false, bool isEntryPoint = true,
                 const std::unordered_set<unsigned> &varArgIndices = {}) {
  PassManager pm(module.getContext());
  ArgumentConverter argCon(name, module);
  if (varArgIndices.empty())
    argCon.gen(name, module, rawArgs);
  else
    argCon.gen(rawArgs, varArgIndices);
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
  // If we're persisting the jit cache we need to run GKE to have access
  // to `.argsCreator` to serialize the arguments.
  if (!varArgIndices.empty()) {
    pm.addPass(
        cudaq::opt::createGenerateKernelExecution({.positNullary = false}));
  } else if ((resultTy && isEntryPoint) ||
             cudaq::compiler_artifact::isPersistingJITEngine()) {
    // If we're expecting a result, then we want to call the .thunk function so
    // that the result is properly marshaled. Add the GKE pass to generate the
    // .thunk. At this point, the kernel should have been specialized so it has
    // an arity of 0.
    auto nullary = true;
    for (auto arg : rawArgs)
      if (!arg) {
        nullary = false;
        break;
      }
    pm.addPass(cudaq::opt::createGenerateKernelExecution(
        {.positNullary = nullary, .ignoreHostFunction = true}));
  }
  pm.addPass(createSymbolDCEPass());
  if (enablePythonCodegenDump) {
    module.getContext()->disableMultithreading();
    pm.enableIRPrinting();
  }
  if (failed(pm.run(module)))
    throw std::runtime_error("Could not successfully apply argument synth.");
}

/// Replace %KEY% and %KEY:default% placeholders in a pipeline string with
/// values from the runtime config map. If the key is in runtimeConfig, use
/// that value. Otherwise use the inline default if provided (%KEY:val%).
/// Keys in the pipeline are uppercase; runtimeConfig keys are lowercase.
/// This is the Python JIT equivalent of ServerHelper::updatePassPipeline().
static void substitutePipelinePlaceholders(
    std::string &pipeline,
    const std::map<std::string, std::string> &runtimeConfig) {
  std::string::size_type pos = 0;
  while (pos < pipeline.size()) {
    auto start = pipeline.find('%', pos);
    if (start == std::string::npos)
      break;
    auto end = pipeline.find('%', start + 1);
    if (end == std::string::npos)
      break;
    auto token = pipeline.substr(start + 1, end - start - 1);
    auto colon = token.find(':');
    auto key = (colon != std::string::npos) ? token.substr(0, colon) : token;

    // Lowercase the key to match runtimeConfig convention.
    std::string lower;
    for (char c : key)
      lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    auto it = runtimeConfig.find(lower);

    if (it != runtimeConfig.end()) {
      pipeline.replace(start, end - start + 1, it->second);
      pos = start + it->second.size();
    } else if (colon != std::string::npos) {
      auto defaultVal = token.substr(colon + 1);
      pipeline.replace(start, end - start + 1, defaultVal);
      pos = start + defaultVal.size();
    } else {
      pos = end + 1;
    }
  }
}

/// Run target-specific passes if the active target config defines a pipeline.
/// Interleaves jit-deploy-pipeline between high and mid-level stages.
/// specializeKernel() covers what hw-jit-prep-pipeline and
/// jit-finalize-pipeline do (inlining, specialization, DistributedDeviceCall),
/// so those are not interleaved here. Targets needing passes from those stages
/// (e.g., apply-control-negations) should include them in their own config
/// fields. Only reads top-level config:, not configuration-matrix entries.
static void runTargetPassPipeline(ModuleOp module) {
  auto *rt = cudaq::get_platform().get_runtime_target();
  if (!rt)
    return;
  auto &cfg = rt->config;
  if (!cfg.BackendConfig.has_value() || !cfg.BackendConfig->hasPassPipeline())
    return;
  auto pipeline = cfg.BackendConfig->getPassPipeline("jit-deploy-pipeline", "");
  substitutePipelinePlaceholders(pipeline, rt->runtimeConfig);
  PassManager pm(module.getContext());
  std::string errMsg;
  llvm::raw_string_ostream errOS(errMsg);
  if (failed(parsePassPipeline(pipeline, pm, errOS)))
    throw std::runtime_error("Failed to parse target pipeline: " + errMsg);
  if (failed(pm.run(module)))
    throw std::runtime_error("Target pass pipeline failed.");
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
  runTargetPassPipeline(module);
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
  runTargetPassPipeline(module);
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
  if (!currentExecCtx || !currentExecCtx->allowJitEngineCaching)
    return std::nullopt;

  auto jit = currentExecCtx->jitEng;
  if (jit && cudaq::compiler_artifact::isPersistingJITEngine()) {
    CUDAQ_INFO("Loading previously compiled JIT engine for {}. This will "
               "re-run the previous job, discarding any changes to the kernel, "
               "arguments or launch configuration.",
               currentExecCtx->kernelName);

    // Ensure the arguments are the same as the previous launch.
    auto argsCreatorThunk = [&jit, &name]() {
      return (void *)jit->lookupRawNameOrFail(name + ".argsCreator");
    };
    cudaq::compiler_artifact::checkArtifactReuse(name, rawArgs, jit.value(),
                                                 argsCreatorThunk);
  }

  return jit;
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

/// When the execution context is "resource-count", extract gate counts and
/// depth metrics from the optimized MLIR IR. Pre-counted gates are erased
/// from the module, so the subsequent JIT compiles a near-empty module.
static void precountResources(ModuleOp module) {
  auto *ctx = cudaq::getExecutionContext();
  if (!ctx || ctx->name != "resource-count")
    return;
  auto counts = cudaq::opt::countResourcesFromIR(module);
  if (failed(counts))
    throw std::runtime_error("Resource count preprocessing failed.");
  nvqir::setResourceCounts(std::move(*counts));
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

    std::unordered_set<unsigned> varArgIndices;
    {
      auto mangledNameMap = module->getAttrOfType<mlir::DictionaryAttr>(
          cudaq::runtime::mangledNameMap);
      bool parametricCompatible = false;
      if (mangledNameMap)
        if (auto attr = mangledNameMap.getAs<mlir::StringAttr>(fullName)) {
          mlir::StringRef mn = attr.getValue();
          parametricCompatible = mn != "BuilderKernel.EntryPoint" &&
                                 !mn.contains("PyKernelFakeEntryPoint");
        }
      if (parametricCompatible)
        for (auto [idx, argTy] :
             llvm::enumerate(funcOp.getFunctionType().getInputs()))
          if (auto vecTy = dyn_cast<cudaq::cc::StdvecType>(argTy))
            if (isa<mlir::FloatType>(vecTy.getElementType()))
              varArgIndices.insert(idx);
    }
    {
      auto *execCtx = cudaq::getExecutionContext();
      if (!execCtx || !execCtx->useParametricJit)
        varArgIndices.clear();
    }
    const bool isFullySpecialized = varArgIndices.empty();
    auto resultInfo = createResultInfo(resultTy, isEntryPoint, module);

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
    specializeKernel(name, module, rawArgs, resultTy, enablePythonCodegenDump,
                     isEntryPoint, varArgIndices);

    // 3b. Run target-specific passes if configured.
    runTargetPassPipeline(module);

    // 3c. Pre-count resources from the optimized IR when resource-counting.
    precountResources(module);

    // 4. Lower to QIR and JIT compile.
    auto jit = createJITEngine(module, "qir:");
    cacheJITForPerformance(jit);
    auto argsCreatorThunk = [&jit, &name]() {
      return (void *)jit.lookupRawNameOrFail(name + ".argsCreator");
    };
    cudaq::compiler_artifact::saveArtifact(name, rawArgs, jit,
                                           argsCreatorThunk);

    cudaq::CompiledKernel ck(name, resultInfo);
    ck.attachJit(jit, isFullySpecialized);
    return ck;
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::ModuleLauncher, PythonLauncher, default)
