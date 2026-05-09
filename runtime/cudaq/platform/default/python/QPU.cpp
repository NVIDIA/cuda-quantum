/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QPU.h"
#include "common/ArgumentWrapper.h"
#include "common/CompiledModule.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "common/RuntimeTarget.h"
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
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq_internal/compiler/JIT.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "cudaq_internal/compiler/TracePassInstrumentation.h"
#include "runtime/cudaq/platform/PythonSignalCheck.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include <cudaq/Optimizer/CodeGen/QIROpaqueStructTypes.h>

// Declared in runtime/cudaq/algorithms/resource_estimation.h (not included
// here to avoid pulling in cudaq/platform.h which creates circular deps).
namespace nvqir {
void setResourceCounts(cudaq::Resources &&);
}

using namespace mlir;

static void specializeKernel(const std::string &name, ModuleOp module,
                             std::span<void *const> rawArgs, Type resultTy = {},
                             bool enablePythonCodegenDump = false,
                             bool isEntryPoint = true,
                             bool isFullySpecialized = true) {
  PassManager pm(module.getContext());
  cudaq::addPythonSignalInstrumentation(pm);
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
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
  llvm::SmallVector<llvm::StringRef> kernelRefs{kernels.begin(), kernels.end()};
  llvm::SmallVector<llvm::StringRef> substRefs{substs.begin(), substs.end()};

  // Run a pass manager to specialize & optimize the kernel to be launched.
  pm.addPass(cudaq::opt::createArgumentSynthesisPass(
      kernelRefs, substRefs, /*changeSemantics=*/false));
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
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
    pm.addPass(cudaq::opt::createRunSemanticsHackery());
  }
  pm.addPass(mlir::createSymbolDCEPass());
  if (enablePythonCodegenDump) {
    module.getContext()->disableMultithreading();
    pm.enableIRPrinting();
  }
  if (failed(cudaq::runPassManagerReleasingGIL(pm, module)))
    throw std::runtime_error("Pass pipeline failed.");
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
static void runTargetPassPipeline(mlir::ModuleOp module) {
  auto *rt = cudaq::get_platform().get_runtime_target();
  if (!rt)
    return;
  auto &cfg = rt->config;
  if (!cfg.BackendConfig.has_value() || !cfg.BackendConfig->hasPassPipeline())
    return;
  auto pipeline = cfg.BackendConfig->getPassPipeline("jit-deploy-pipeline", "");
  substitutePipelinePlaceholders(pipeline, rt->runtimeConfig);
  auto *ctx = module.getContext();
  PassManager pm(ctx);
  cudaq::addPythonSignalInstrumentation(pm);
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  std::string errMsg;
  llvm::raw_string_ostream errOS(errMsg);
  if (mlir::failed(mlir::parsePassPipeline(pipeline, pm, errOS)))
    throw std::runtime_error("Failed to parse target pipeline: " + errMsg);
  if (failed(cudaq::runPassManagerReleasingGIL(pm, module)))
    throw std::runtime_error("Pass pipeline failed.");
}

/// Lowers \p module to LLVM code. The LLVM code will use "full QIR" as the
/// transport layer. If \p kernelName and \p args are provided, they will
/// specialize the selected entry-point kernel.
std::string cudaq::detail::lower_to_qir_llvm(const std::string &name,
                                             mlir::ModuleOp module,
                                             OpaqueArguments &args,
                                             const std::string &format) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getQIR", name);
  // Translate the module to QIR transport layer (as LLVM code).
  cudaq_internal::compiler::mergeAllCallableClosures(module, name,
                                                     args.getArgs());
  specializeKernel(name, module, args.getArgs());
  runTargetPassPipeline(module);
  PassManager pm(module.getContext());
  cudaq::addPythonSignalInstrumentation(pm);
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  cudaq::opt::addAggressiveInlining(pm);
  cudaq::opt::createTargetFinalizePipeline(pm);
  cudaq::opt::addAOTPipelineConvertToQIR(pm, format);
  if (failed(cudaq::runPassManagerReleasingGIL(pm, module)))
    throw std::runtime_error("Pass pipeline failed.");
  if (failed(cudaq::verifier::checkQIRLLVMIRDialect(module, format)))
    throw std::runtime_error("QIR conformance failed.");
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext);
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
                                             mlir::ModuleOp module,
                                             OpaqueArguments &args) {
  ScopedTraceWithContext(cudaq::TIMING_JIT, "getASM", name);
  // Translate module to OpenQASM2 transport layer.
  cudaq_internal::compiler::mergeAllCallableClosures(module, name,
                                                     args.getArgs());
  specializeKernel(name, module, args.getArgs());
  runTargetPassPipeline(module);
  auto *ctx = module.getContext();
  PassManager pm(ctx);
  cudaq::addPythonSignalInstrumentation(pm);
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  cudaq::opt::createTargetFinalizePipeline(pm);
  cudaq::opt::createPipelineTransformsForPythonToOpenQASM(pm);
  cudaq::opt::addPipelineTranslateToOpenQASM(pm);
  const bool enablePrintMLIRBeforeAndAfterEachPass =
      cudaq::getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", false);
  if (enablePrintMLIRBeforeAndAfterEachPass) {
    ctx->disableMultithreading();
    pm.enableIRPrinting();
  }
  if (failed(cudaq::runPassManagerReleasingGIL(pm, module)))
    throw std::runtime_error("Pass pipeline failed.");
  std::string result;
  llvm::raw_string_ostream os(result);
  if (mlir::failed(cudaq::translateToOpenQASM(module, os)))
    return "{translation failed}";
  os.flush();
  return result;
}

/// Scan \p module and set flags in the current platform context accordingly.
static void updateExecutionContext(mlir::ModuleOp module) {
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

static std::optional<cudaq::JitEngine>
alreadyBuiltJITCode(const std::string &name) {
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
static void cacheJITForPerformance(cudaq::JitEngine jit) {
  auto *currentExecCtx = cudaq::getExecutionContext();
  if (currentExecCtx && currentExecCtx->allowJitEngineCaching) {
    if (!currentExecCtx->jitEng)
      currentExecCtx->jitEng = jit;
  }
}

/// When the execution context is "resource-count", extract gate counts and
/// depth metrics from the optimized MLIR IR. Pre-counted gates are erased
/// from the module, so the subsequent JIT compiles a near-empty module.
static void precountResources(mlir::ModuleOp module) {
  auto *ctx = cudaq::getExecutionContext();
  if (!ctx || ctx->name != "resource-count")
    return;
  auto counts = cudaq::opt::countResourcesFromIR(module);
  if (mlir::failed(counts))
    return;
  nvqir::setResourceCounts(std::move(*counts));
}

namespace {
struct PythonLauncher : public cudaq::ModuleLauncher {
  cudaq::CompiledModule compileModule(const cudaq::SourceModule &src,
                                      cudaq::KernelArgs args,
                                      bool isEntryPoint) override {

    ScopedTraceWithContext(cudaq::TIMING_LAUNCH,
                           "PythonLauncher::compileModule");
    const auto &name = src.getName();
    auto mlirArt = src.getMlir();
    if (!mlirArt)
      throw std::runtime_error(
          "PythonLauncher::compileModule requires an MLIR artifact on the "
          "SourceModule for kernel '" +
          name + "'.");
    ModuleOp module =
        cudaq_internal::compiler::CompiledModuleHelper::getMlirModuleOp(
            *mlirArt);
    const bool enablePythonCodegenDump =
        cudaq::getEnvBool("CUDAQ_PYTHON_CODEGEN_DUMP", false);

    std::string fullName = cudaq::runtime::cudaqGenPrefixName + name;

    auto funcOp = module.lookupSymbol<mlir::func::FuncOp>(fullName);
    if (!funcOp)
      throw std::runtime_error("no kernel named " + name + " found in module");
    mlir::Type resultTy = cudaq::runtime::getReturnType(funcOp);

    const bool hasResult = !!resultTy;
    auto resultInfo =
        cudaq_internal::compiler::CompiledModuleHelper::createResultInfo(
            resultTy, isEntryPoint, module);

    // Determine whether the kernel needs argument packing (argsCreator) by
    // checking if any non-callable arguments are present. This must be done
    // before the cache lookup so the cached path uses the correct value.
    bool isFullySpecialized = true;
    FunctionType fromFuncTy = funcOp.getFunctionType();
    // Specialization for direct calls will take care of partial specialization
    // separately
    bool isLocalSimulator =
        !(cudaq::is_remote_platform() || cudaq::is_emulated_platform());

    std::vector<void *> closureArgsVec;
    std::span<void *const> closureArgs;
    std::span<void *const> rawArgs =
        args.hasTypeErased() ? *args.getTypeErased() : std::span<void *const>();

    // Special handling in case the arguments were already synthesized
    size_t numArgs = rawArgs.size() - (hasResult ? 1 : 0);
    if (isEntryPoint && isLocalSimulator &&
        numArgs == fromFuncTy.getNumInputs()) {
      closureArgsVec = std::vector(rawArgs.begin(), rawArgs.end());
      for (auto [i, ty] : llvm::enumerate(fromFuncTy.getInputs())) {
        if (!isa<cudaq::cc::CallableType>(ty)) {
          isFullySpecialized = false;
          closureArgsVec[i] = nullptr;
        }
      }
      closureArgs = closureArgsVec;
    } else {
      // Avoid copying
      closureArgs = rawArgs;
    }

    if (auto jit = alreadyBuiltJITCode(name)) {
      auto jitArtifacts =
          cudaq_internal::compiler::CompiledModuleHelper::createJitArtifacts(
              name, *jit, resultInfo, isFullySpecialized);
      return cudaq_internal::compiler::CompiledModuleHelper::
          createCompiledModule(name, resultInfo, jitArtifacts);
    }

    // 1. Check that this call is sane.
    if (enablePythonCodegenDump)
      module.dump();

    // 2. Merge other modules (e.g., if there are device kernel calls).
    cudaq_internal::compiler::mergeAllCallableClosures(module, name, rawArgs);

    // Mark all newly merged kernels private.
    for (auto &op : module)
      if (auto f = mlir::dyn_cast<mlir::func::FuncOp>(op))
        if (f != funcOp)
          f.setPrivate();

    updateExecutionContext(module);

    // 3. Specialize the kernel (argument synthesis, optimization).
    CUDAQ_INFO("Run Argument Synth.\n");
    if (enablePythonCodegenDump)
      module.dump();

    specializeKernel(name, module, closureArgs, resultTy,
                     enablePythonCodegenDump, isEntryPoint, isFullySpecialized);

    // 3b. Run target-specific passes if configured.
    runTargetPassPipeline(module);

    // 3c. Pre-count resources from the optimized IR when resource-counting.
    precountResources(module);

    // 4. Lower to QIR and JIT compile.
    auto jit = cudaq_internal::compiler::createJITEngine(module, "qir:");
    cacheJITForPerformance(jit);
    cudaq::compiler_artifact::saveArtifact(name, jit);

    auto jitArtifacts =
        cudaq_internal::compiler::CompiledModuleHelper::createJitArtifacts(
            name, jit, resultInfo, isFullySpecialized);
    return cudaq_internal::compiler::CompiledModuleHelper::createCompiledModule(
        name, std::move(resultInfo), jitArtifacts);
  }
};
} // namespace

// PythonLauncher registration. This TU only builds into the Python extension
// (_quakeDialects.so), but `launchModule` / `specializeModule` live in
// libcudaq.so. CUDA-Q Registry uses `static inline Head/Tail`, so each DSO
// that instantiates the template gets its own copy — `CUDAQ_REGISTER_TYPE`
// would add the node to the extension's (unseen-by-libcudaq) registry. We
// instead call the `cudaq_add_module_launcher_node` bridge defined in
// libcudaq.so so the registration lands in the registry that `launchModule`
// actually reads. Mirrors the `cudaq_add_qpu_node` pattern used for QPUs.
extern "C" void cudaq_add_module_launcher_node(void *node_ptr);

namespace {
struct PythonLauncherRegistration {
  cudaq::RegistryEntry<cudaq::ModuleLauncher> entry;
  cudaq::Registry<cudaq::ModuleLauncher>::node node;
  PythonLauncherRegistration()
      : entry("default", &PythonLauncherRegistration::ctorFn), node(entry) {
    cudaq_add_module_launcher_node(&node);
  }
  static std::unique_ptr<cudaq::ModuleLauncher> ctorFn() {
    return std::make_unique<PythonLauncher>();
  }
};
static PythonLauncherRegistration s_pythonLauncherRegistration;
} // namespace

extern "C" void cudaq_ensure_default_launcher_linked(void) {}
