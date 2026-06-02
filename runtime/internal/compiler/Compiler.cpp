/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/Compiler.h"
#include "common/DeviceCodeRegistry.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "common/KernelExecution.h"
#include "common/Resources.h"
#include "common/Timing.h"
#include "cudaq_internal/compiler/ArgumentConversion.h"
#include "cudaq_internal/compiler/JIT.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "nlohmann/json.hpp"
#include "cudaq/Optimizer/Builder/Marshal.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Transforms/AddMetadata.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Optimizer/Transforms/ResourceCount.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/utils/cudaq_utils.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include <memory>
#include <mlir/IR/OwningOpRef.h>
#include <optional>
#include <regex>

using namespace mlir;

namespace {
/// Conditionally form an output_names JSON object if this was for QIR
nlohmann::json formOutputNames(const std::string &codegenTranslation,
                               mlir::ModuleOp moduleOp,
                               const std::string &codeStr) {
  // Form an output_names mapping from codeStr
  nlohmann::json output_names;
  std::vector<char> bitcode;
  if (codegenTranslation.starts_with("qir")) {
    // decodeBase64 will throw a runtime exception if it fails
    if (llvm::decodeBase64(codeStr, bitcode)) {
      CUDAQ_INFO("Could not decode codeStr {}", codeStr);
    } else {
      llvm::LLVMContext llvmContext;
      auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
          llvm::StringRef(bitcode.data(), bitcode.size()));
      auto moduleOrError =
          llvm::parseBitcodeFile(buffer->getMemBufferRef(), llvmContext);
      if (moduleOrError.takeError())
        throw std::runtime_error("Could not parse bitcode file");
      auto module = std::move(moduleOrError.get());
      for (llvm::Function &func : *module) {
        if (func.hasFnAttribute("entry_point") &&
            func.hasFnAttribute("output_names")) {
          output_names = nlohmann::json::parse(
              func.getFnAttribute("output_names").getValueAsString());
          break;
        }
      }
    }
  } else if (codegenTranslation.starts_with("qasm2")) {
    for (auto &op : moduleOp) {
      if (op.hasAttr(cudaq::entryPointAttrName) && op.hasAttr("output_names")) {
        if (auto strAttr = mlir::dyn_cast_if_present<mlir::StringAttr>(
                op.getAttr(cudaq::opt::QIROutputNamesAttrName))) {
          output_names = nlohmann::json::parse(strAttr.getValue());
          break;
        }
      }
    }
  }
  return output_names;
}
/// Extract qubit-mapping reorder indices from the entry-point attributes.
std::vector<std::size_t> extractMappingReorderIdx(mlir::ModuleOp moduleOp,
                                                  mlir::func::FuncOp epFunc) {
  assert(moduleOp.template lookupSymbol<mlir::func::FuncOp>(epFunc.getName()) &&
         "Entry point function must survive the lowering pipeline.");
  std::vector<std::size_t> mapping_reorder_idx;
  if (auto mappingAttr = dyn_cast_if_present<mlir::ArrayAttr>(
          epFunc->getAttr("mapping_reorder_idx"))) {
    mapping_reorder_idx.resize(mappingAttr.size());
    std::transform(mappingAttr.begin(), mappingAttr.end(),
                   mapping_reorder_idx.begin(), [](mlir::Attribute attr) {
                     return mlir::cast<mlir::IntegerAttr>(attr).getInt();
                   });
  }
  return mapping_reorder_idx;
}
} // namespace

std::pair<const void *, std::shared_ptr<mlir::MLIRContext>>
cudaq_internal::compiler::Compiler::loadQuakeCodeByName(
    const std::string &kernelName) {
  std::shared_ptr<mlir::MLIRContext> context(
      cudaq_internal::compiler::getOwningMLIRContext().release());

  // Get the quake representation of the kernel
  auto quakeCode = cudaq::get_quake_by_name(kernelName);
  auto m_module = parseSourceString<mlir::ModuleOp>(quakeCode, context.get());
  if (!m_module)
    throw std::runtime_error("module cannot be parsed");

  return std::make_pair(m_module.release().getAsOpaquePointer(), context);
}

cudaq_internal::compiler::Compiler::Compiler(
    std::unique_ptr<cudaq::CompileTarget> &&target_)
    : target(std::move(target_)) {
  assert(target && "target cannot be null");
  emulate = target->emulate;

  cudaq_internal::compiler::initializeMLIR();

  // Print the IR if requested
  printIR = cudaq::getEnvBool("CUDAQ_DUMP_JIT_IR", printIR);

  // Get additional debug values
  disableMLIRthreading =
      cudaq::getEnvBool("CUDAQ_MLIR_DISABLE_THREADING", disableMLIRthreading);
  enablePrintMLIREachPass =
      cudaq::getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", enablePrintMLIREachPass);
  enablePassStatistics =
      cudaq::getEnvBool("CUDAQ_MLIR_PASS_STATISTICS", enablePassStatistics);

  // If the very verbose enablePrintMLIREachPass flag is set, then
  // multi-threading must be disabled.
  if (enablePrintMLIREachPass) {
    disableMLIRthreading = true;
  }
}

cudaq_internal::compiler::Compiler::~Compiler() = default;

// =============================================================================
// Common helpers for policy-specific runPassPipeline overloads
// =============================================================================

void cudaq_internal::compiler::Compiler::applyPipeline(
    const std::string &pipeline, mlir::ModuleOp moduleOp,
    const std::string &kernelName) {
  auto *contextPtr = moduleOp.getContext();
  mlir::PassManager pm(contextPtr);
  std::string errMsg;
  llvm::raw_string_ostream os(errMsg);
  CUDAQ_INFO("Pass pipeline for {} = {}", kernelName, pipeline);
  if (failed(parsePassPipeline(pipeline, pm, os)))
    throw std::runtime_error(
        "Remote rest platform failed to add passes to pipeline (" + errMsg +
        ").");
  if (failed(cudaq_internal::compiler::runPassManager(pm,
                                                      moduleOp.getOperation())))
    throw std::runtime_error("Remote rest platform Quake lowering failed.");
}

static bool eraseNonCallableArguments(std::span<void *const> &rawArgs,
                                      std::vector<void *> &closureArgs,
                                      mlir::func::FuncOp funcOp) {
  bool isFullySpecialized = true;

  FunctionType fromFuncTy = funcOp.getFunctionType();
  closureArgs = std::vector(rawArgs.begin(), rawArgs.end());
  for (auto [i, ty] : llvm::enumerate(fromFuncTy.getInputs())) {
    if (!isa<cudaq::cc::CallableType>(ty)) {
      isFullySpecialized = false;
      closureArgs[i] = nullptr;
    }
  }
  rawArgs = closureArgs;
  return isFullySpecialized;
}

std::tuple<mlir::ModuleOp, mlir::func::FuncOp, bool>
cudaq_internal::compiler::Compiler::prepareModule(const std::string &kernelName,
                                                  mlir::ModuleOp m_module,
                                                  cudaq::KernelArgs args,
                                                  bool isEntryPoint) {
  auto *contextPtr = m_module.getContext();

  auto origFn = m_module.template lookupSymbol<mlir::func::FuncOp>(
      std::string(cudaq::runtime::cudaqGenPrefixName) + kernelName);

  auto moduleOp =
      lowerQuakeCodeBuildModule(kernelName, m_module, contextPtr, origFn);

  auto epFunc =
      moduleOp.template lookupSymbol<mlir::func::FuncOp>(origFn.getName());
  const bool isPython = moduleOp->hasAttr(cudaq::runtime::pythonUniqueAttrName);
  bool isFullySpecialized = true;
  auto rawArgs = args.getTypeErased();
  auto packed = args.getPacked();
  if (!args.empty()) {
    mlir::PassManager pm(contextPtr);
    if (isPython && rawArgs)
      cudaq_internal::compiler::mergeAllCallableClosures(moduleOp, kernelName,
                                                         *rawArgs);

    // Mark all newly merged kernels private, and leave the entry point alone.
    for (auto &op : moduleOp)
      if (auto f = dyn_cast<mlir::func::FuncOp>(op))
        if (f != epFunc)
          f.setPrivate();

    if (rawArgs) {
      CUDAQ_INFO("Run Argument Synth.\n");
      // For quantum devices, we generate a collection of `init` and
      // `num_qubits` functions and their substitutions created
      // from a kernel and arguments that generated a state argument.
      cudaq_internal::compiler::ArgumentConverter argCon(kernelName, moduleOp);
      // Must stay in scope as `eraseNonCallableArguments` may populate it
      std::vector<void *> closureArgs;
      if (cudaq::opt::marshal::isFullySynthesized(epFunc)) {
        // Already fully specialized, nothing to do.
        isFullySpecialized = true;
      } else if (isEntryPoint && !target->fullySpecialize) {
        // We disable specialization by erasing args that should not be inlined
        isFullySpecialized =
            eraseNonCallableArguments(*rawArgs, closureArgs, epFunc);
      }
      argCon.gen(*rawArgs);

      // Store kernel and substitution strings on the stack.
      // We pass string references to the `createArgumentSynthesisPass`.
      mlir::SmallVector<std::string> kernels;
      mlir::SmallVector<std::string> substs;
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
      mlir::SmallVector<mlir::StringRef> kernelRefs{kernels.begin(),
                                                    kernels.end()};
      mlir::SmallVector<mlir::StringRef> substRefs{substs.begin(),
                                                   substs.end()};
      pm.addPass(cudaq::opt::createArgumentSynthesisPass(
          kernelRefs, substRefs, target->argumentSynthChangeSemantics));
      pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
      pm.addPass(
          cudaq::opt::createLambdaLifting({.constantPropagation = true}));
      // We must inline these lambda calls before apply specialization as it
      // does not perform control/adjoint specialization across function call
      // boundary.
      cudaq::opt::addAggressiveInlining(pm);
      pm.addPass(
          cudaq::opt::createApplySpecialization({.constantPropagation = true}));
      cudaq::opt::addAggressiveInlining(pm);

      // Lower cc.device_calls
      if (target->supportDeviceCalls) {
        pm.addPass(cudaq::opt::createDistributedDeviceCall());
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
      }

      if (target->pipelineConfig.replaceStateWithKernel) {
        pm.addNestedPass<mlir::func::FuncOp>(
            cudaq::opt::createReplaceStateWithKernel());
        cudaq::opt::addAggressiveInlining(pm);
      }

      bool hasResult = !!cudaq::runtime::getReturnType(epFunc);
      // Run GKE to generate `.thunk` / `.argsCreator` when the kernel has a
      // result or any unspecialized arguments so they can be properly marshaled
      if (isEntryPoint && (hasResult || !isFullySpecialized)) {
        pm.addPass(cudaq::opt::createGenerateKernelExecution(
            {.positNullary = isFullySpecialized, .ignoreHostFunction = true}));
        pm.addPass(cudaq::opt::createRunSemanticsHackery());
      }
      pm.addPass(mlir::createSymbolDCEPass());
    } else if (packed) {
      CUDAQ_INFO("Run Quake Synth.\n");
      pm.addPass(
          cudaq::opt::createQuakeSynthesizer(kernelName, packed->data.data()));
    }
    pm.addPass(mlir::createCanonicalizerPass());
    if (failed(cudaq_internal::compiler::runPassManager(
            pm, moduleOp.getOperation())))
      throw std::runtime_error(
          "Could not successfully apply kernel specialization.");
  }

  return {moduleOp, epFunc, isFullySpecialized};
}

std::pair<bool, std::string>
cudaq_internal::compiler::Compiler::executeMainPipeline(
    mlir::ModuleOp moduleOp, const std::string &kernelName) {
  if (target->pipelineConfig.skipTargetLoweringPipeline) {
    return {false, ""};
  }
  auto passPipelineConfig = getPassPipeline(*target);
  auto combineMeasurements =
      passPipelineConfig.find("combine-measurements") != std::string::npos;
  if (emulate && combineMeasurements) {
    std::regex combine("(.*),([ ]*)combine-measurements(.*)");
    std::string replacement("$1$3");
    passPipelineConfig =
        std::regex_replace(passPipelineConfig, combine, replacement);
    CUDAQ_INFO("Delaying combine-measurements pass due to emulation. "
               "Updating pipeline to {}",
               passPipelineConfig);
  }
  applyPipeline(passPipelineConfig, moduleOp, kernelName);
  return {combineMeasurements, passPipelineConfig};
}

cudaq::CompiledModule
cudaq_internal::compiler::Compiler::assembleCompiledModule(
    const std::string &kernelName,
    std::vector<std::pair<std::string, mlir::ModuleOp>> &modules, bool needJit,
    bool isFullySpecialized, bool isEntryPoint, bool runCombineMeasurements,
    std::optional<cudaq::Resources> resourceCounts,
    cudaq::CompiledModule::CompilationMetadata metadata,
    std::shared_ptr<mlir::MLIRContext> context) {
  std::vector<
      cudaq_internal::compiler::CompiledModuleHelper::NamedCompiledArtifact>
      artifacts;
  cudaq::ResultInfo resultInfo;
  if (needJit) {
    for (auto &[name, module] : modules) {
      auto clonedModule = module.clone();
      std::string fullName = cudaq::runtime::cudaqGenPrefixName + kernelName;
      auto funcOp = module.template lookupSymbol<mlir::func::FuncOp>(fullName);
      mlir::Type resultTy = cudaq::runtime::getReturnType(funcOp);
      resultInfo =
          cudaq_internal::compiler::CompiledModuleHelper::createResultInfo(
              resultTy, isEntryPoint, module);
      auto jitArtifacts =
          cudaq_internal::compiler::CompiledModuleHelper::createJitArtifacts(
              kernelName,
              cudaq_internal::compiler::createJITEngine(
                  clonedModule, target->pipelineConfig.codegenTranslation),
              resultInfo, isFullySpecialized);
      // The first artifact is the kernel entry point; rename it to the
      // per-module name (relevant for the multi-module observe path where the
      // module name is a Pauli term id)
      assert(!jitArtifacts.empty());
      assert(jitArtifacts[0].first == kernelName);
      jitArtifacts[0].first = name;
      for (auto &jitArtifact : jitArtifacts)
        artifacts.push_back(std::move(jitArtifact));
      if (resourceCounts)
        artifacts.push_back(
            cudaq_internal::compiler::CompiledModuleHelper::
                createResourcesArtifact(name, std::move(*resourceCounts)));
    }
  }

  if (runCombineMeasurements)
    for (auto &[name, module] : modules)
      applyPipeline("func.func(combine-measurements)", module, kernelName);

  for (auto &[name, module] : modules) {
    artifacts.push_back(
        cudaq_internal::compiler::CompiledModuleHelper::createMlirArtifact(
            name, module, context));
  }

  return cudaq_internal::compiler::CompiledModuleHelper::createCompiledModule(
      kernelName, std::move(resultInfo), std::move(artifacts),
      std::move(metadata));
}

static bool hasConditionalsOnMeasureResults(mlir::ModuleOp moduleOp) {
  return std::any_of(moduleOp.begin(), moduleOp.end(), [](auto &innerOp) {
    cudaq::quake::detail::QuakeFunctionAnalysis analysis{&innerOp};
    auto info = analysis.getAnalysisInfo();
    if (info.empty())
      return false;
    auto result = info[&innerOp];
    return result.hasConditionalsOnMeasure;
  });
}

cudaq::CompiledModule cudaq_internal::compiler::Compiler::runPassPipeline(
    const std::string &kernelName, const void *modulePtr,
    cudaq::KernelArgs args, bool isEntryPoint,
    std::shared_ptr<mlir::MLIRContext> context) {
  ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "Compiler::runPassPipeline");
  mlir::ModuleOp m_module = mlir::ModuleOp::getFromOpaquePointer(modulePtr);
  assert(!context || context.get() == m_module.getContext());
  auto [moduleOp, epFunc, isFullySpecialized] =
      prepareModule(kernelName, m_module, args, isEntryPoint);

  bool hasConditionalsOnMeasRes = hasConditionalsOnMeasureResults(moduleOp);

  // Populate conditional measurement flag in the context.
  if (hasConditionalsOnMeasRes &&
      !target->supportConditionalsOnMeasureResults) {
    throw std::runtime_error(
        "`cudaq::sample` and `cudaq::sample_async` no longer support "
        "kernels "
        "that branch on measurement results. Kernel '" +
        kernelName +
        "' uses conditional feedback. Use `cudaq::run` or "
        "`cudaq::run_async` "
        "instead. See CUDA-Q documentation for migration guide.");
  }

  auto [combineMeasurements, passPipeline] =
      executeMainPipeline(moduleOp, kernelName);

  // We need to run resource counting preprocessing after the pass pipeline as
  // the pre-processing might change the IR structure (may interfere with
  // other passes).
  std::optional<cudaq::Resources> resourceCounts;
  if (target->emitResourceCounts) {
    auto result = cudaq::opt::countResourcesFromIR(moduleOp);
    if (succeeded(result))
      resourceCounts = std::move(*result);
  }

  std::vector<std::size_t> mapping_reorder_idx;
  if (target->storeReorderIdx)
    mapping_reorder_idx = extractMappingReorderIdx(moduleOp, epFunc);

  // Warn if kernel has named measurement registers (sub-registers).
  if (target->warnNamedMeasurements) {
    auto funcOp = moduleOp.template lookupSymbol<mlir::func::FuncOp>(
        std::string(cudaq::runtime::cudaqGenPrefixName) + kernelName);
    if (funcOp) {
      bool hasNamedMeasurements = false;
      funcOp.walk([&](cudaq::quake::MeasurementInterface meas) {
        if (meas.getOptionalRegisterName().has_value()) {
          hasNamedMeasurements = true;
          return mlir::WalkResult::interrupt();
        }
        return mlir::WalkResult::advance();
      });
      if (hasNamedMeasurements) {
        warnedNamedMeasurements = true;
        std::cerr << "WARNING: Kernel \"" << kernelName
                  << "\" uses named measurement results "
                  << "but is invoked in sampling mode. Support for "
                  << "sub-registers in `sample_result` is deprecated and will "
                  << "be removed in a future release. Use `run` to retrieve "
                  << "individual measurement results." << std::endl;
      }
    }
  }

  if (target->pipelineConfig.addMeasurements) {
    // No need to add measurements only to remove them eventually
    if (target->pipelineConfig.postCodeGenPasses.find("remove-measurements") ==
        std::string::npos)
      applyPipeline("func.func(add-measurements)", moduleOp, kernelName);
  }

  // Apply observations if necessary
  std::vector<std::pair<std::string, mlir::ModuleOp>> modules;
  if (target->pauliTermSplitObservable) {
    applyPipeline("canonicalize,cse", moduleOp, kernelName);
    for (const auto &term : *target->pauliTermSplitObservable) {
      if (term.is_identity())
        continue;

      // Get the ansatz
      [[maybe_unused]] auto ansatz =
          moduleOp.template lookupSymbol<mlir::func::FuncOp>(
              cudaq::runtime::cudaqGenPrefixName + kernelName);
      assert(ansatz && "could not find the ansatz kernel");

      // Create a new Module to clone the ansatz into it
      auto tmpModuleOp = moduleOp.clone();

      // Create the pass manager, add the quake observe ansatz pass and run it
      // followed by the canonicalizer
      auto *contextPtr = moduleOp.getContext();
      mlir::PassManager pm(contextPtr);
      pm.addNestedPass<mlir::func::FuncOp>(cudaq::opt::createObserveAnsatzPass(
          term.get_binary_symplectic_form()));
      if (failed(cudaq_internal::compiler::runPassManager(
              pm, tmpModuleOp.getOperation())))
        throw std::runtime_error("Could not apply measurements to ansatz.");
      // The full pass pipeline was run above, but the ansatz pass can
      // introduce gates that aren't supported by the backend, so we need to
      // re-run the gate set mapping if that existed in the original pass
      // pipeline.
      auto csvSplit = cudaq::split(passPipeline, ',');
      for (auto &pass : csvSplit)
        if (pass.ends_with("-gate-set-mapping"))
          applyPipeline(pass, tmpModuleOp, kernelName);
      if (!emulate && combineMeasurements)
        applyPipeline("func.func(combine-measurements)", tmpModuleOp,
                      kernelName);
      modules.emplace_back(term.get_term_id(), tmpModuleOp);
    }
  } else {
    modules.emplace_back(kernelName, moduleOp);
  }

  bool needJit = emulate || target->emitResourceCounts || target->emitJit;
  return assembleCompiledModule(
      kernelName, modules, needJit, isFullySpecialized, isEntryPoint,
      emulate && combineMeasurements, std::move(resourceCounts),
      {.reorderIdx = mapping_reorder_idx,
       .hasConditionalsOnMeasureResults = hasConditionalsOnMeasRes},
      context);
}

std::vector<cudaq::KernelExecution>
cudaq_internal::compiler::Compiler::emitKernelExecutions(
    const cudaq::CompiledModule &compiled) {
  const auto &codegenTranslation = target->pipelineConfig.codegenTranslation;
  const auto &postCodeGenPasses = target->pipelineConfig.postCodeGenPasses;

  // Apply user-specified codegen
  std::vector<cudaq::KernelExecution> codes;
  for (const auto &[name, mlirArtifact] : compiled.getMlirArtifacts()) {
    mlir::OwningOpRef<ModuleOp> compiled_module =
        cudaq_internal::compiler::CompiledModuleHelper::getMlirModuleOp(
            mlirArtifact)
            .clone();

    std::string codeStr;
    nlohmann::json j;
    if (target->emitTargetCode) {
      // Get the code gen translation
      auto translation =
          cudaq_internal::compiler::getTranslation(codegenTranslation);
      llvm::raw_string_ostream outStr(codeStr);
      if (disableMLIRthreading)
        compiled_module->getContext()->disableMultithreading();
      if (codegenTranslation.starts_with("qir")) {
        if (failed(translation(*compiled_module, codegenTranslation, outStr,
                               postCodeGenPasses, printIR,
                               enablePrintMLIREachPass, enablePassStatistics)))
          throw std::runtime_error("Could not successfully translate to " +
                                   codegenTranslation + ".");
      } else {
        if (failed(translation(*compiled_module, outStr, postCodeGenPasses,
                               printIR, enablePrintMLIREachPass,
                               enablePassStatistics)))
          throw std::runtime_error("Could not successfully translate to " +
                                   codegenTranslation + ".");
      }

      // Form an output_names mapping from codeStr
      j = formOutputNames(codegenTranslation, *compiled_module, codeStr);
    }

    // Retrieve pre-computed JIT engine and resource counts (if any).
    std::optional<cudaq::JitEngine> optionalJit;
    std::optional<cudaq::Resources> optionalResourceCounts;
    auto jit = compiled.getJit(name);
    if (jit)
      optionalJit = jit->getEngine();
    auto resourceCounts = compiled.getResources(name);
    if (resourceCounts)
      optionalResourceCounts = *resourceCounts;

    auto mapping_reorder_idx = compiled.getMetadata().reorderIdx;
    codes.emplace_back(name, codeStr, optionalJit, optionalResourceCounts, j,
                       mapping_reorder_idx);
  }

  return codes;
}

mlir::ModuleOp cudaq_internal::compiler::Compiler::lowerQuakeCodeBuildModule(
    const std::string &kernelName, mlir::ModuleOp m_module,
    mlir::MLIRContext *contextPtr, mlir::func::FuncOp func) {
  llvm::SmallVector<mlir::func::FuncOp> newFuncOpsWithDefinitions;
  llvm::SmallSet<std::string, 4> deviceCallCallees;
  // For every declaration without a definition, we need to try to find the
  // function in the Quake registry and copy the functions into this module.
  m_module.walk([&](mlir::func::FuncOp funcOp) {
    if (!funcOp.isDeclaration()) {
      // Skipping function because it already has a definition.
      return mlir::WalkResult::advance();
    }
    // Definition doesn't exist, so we need to find it in the Quake
    // registry.
    mlir::StringRef fullFuncName = funcOp.getName();
    mlir::StringRef kernelName = [fullFuncName]() {
      mlir::StringRef retVal = fullFuncName;
      // TODO - clean this up to not have to do this. Considering the
      // module's map, or cudaq::detail::getKernelName(). But make sure it
      // works for standard C++ functions.

      // Only get the portion before the first ".".
      if (auto ix = fullFuncName.find("."); ix != mlir::StringRef::npos)
        retVal = fullFuncName.substr(0, ix);
      // Also strip out __nvqpp_mlirgen__function_ from the beginning of the
      // function name.
      if (retVal.starts_with(cudaq::runtime::cudaqGenPrefixName)) {
        retVal = retVal.substr(cudaq::runtime::cudaqGenPrefixLength);
      }
      return retVal;
    }();
    std::string quakeCode =
        kernelName.empty() ? ""
                           : cudaq::get_quake_by_name(kernelName.str(),
                                                      /*throwException=*/false);
    if (quakeCode.empty()) {
      // Skipping function because it does not have a quake code.
      return mlir::WalkResult::advance();
    }
    auto tmp_module = parseSourceString<mlir::ModuleOp>(quakeCode, contextPtr);
    auto tmpFuncOpWithDefinition =
        tmp_module->lookupSymbol<mlir::func::FuncOp>(fullFuncName);
    auto newNameAttr = mlir::StringAttr::get(m_module.getContext(),
                                             fullFuncName.str() + ".stitch");
    auto clonedFunc = tmpFuncOpWithDefinition.clone();
    clonedFunc.setName(newNameAttr);
    mlir::SymbolTable symTable(m_module);
    symTable.insert(clonedFunc);
    newFuncOpsWithDefinitions.push_back(clonedFunc);

    if (failed(mlir::SymbolTable::replaceAllSymbolUses(
            funcOp.getOperation(), newNameAttr, m_module.getOperation()))) {
      throw std::runtime_error(fmt::format(
          "Failed to replace symbol uses for function {}", fullFuncName.str()));
    }
    return mlir::WalkResult::advance();
  });

  // For each one of the added functions, we need to traverse them to find
  // device calls (in order to create declarations for them)
  for (auto &funcOp : newFuncOpsWithDefinitions) {
    mlir::OpBuilder builder(m_module);
    builder.setInsertionPointToStart(m_module.getBody());
    funcOp.walk([&](cudaq::cc::DeviceCallOp deviceCall) {
      auto calleeName = deviceCall.getCallee();
      // If the callee is already in the symbol table, nothing to do.
      if (m_module.lookupSymbol<mlir::func::FuncOp>(calleeName))
        return;

      // Otherwise, we need to create a declaration for the callback
      // function.
      auto argTypes = deviceCall.getArgs().getTypes();
      auto resTypes = deviceCall.getResultTypes();
      auto funcType = builder.getFunctionType(argTypes, resTypes);

      // Create a *declaration* (no body) for the callback function.
      [[maybe_unused]] auto decl = mlir::func::FuncOp::create(
          builder, deviceCall.getLoc(), calleeName, funcType);
      decl.setPrivate();
      deviceCallCallees.insert(calleeName.str());
    });
  }

  // Create a new Module to clone the function into
  auto location = mlir::FileLineColLoc::get(contextPtr, "<builder>", 1, 1);
  mlir::ImplicitLocOpBuilder builder(location, contextPtr);

  // FIXME this should be added to the builder.
  if (!func->hasAttr(cudaq::entryPointAttrName))
    func->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
  auto moduleOp = mlir::ModuleOp::create(builder);
  moduleOp->setAttrs(m_module->getAttrDictionary());
  auto mangledNameMap = m_module->getAttrOfType<mlir::DictionaryAttr>(
      cudaq::runtime::mangledNameMap);

  for (auto &op : m_module.getOps()) {
    // Add any global symbols, including global constant arrays. Global
    // constant arrays can be created during compilation,
    // `lift-array-alloc`, `argument-synthesis`, `quake-synthesizer`, and
    // `get-concrete-matrix` passes.
    if (auto lfunc = dyn_cast<mlir::func::FuncOp>(op)) {
      bool skip = lfunc.getName().ends_with(".thunk");
      if (!skip && mangledNameMap &&
          !deviceCallCallees.contains(lfunc.getName().str()))
        for (auto &entry : mangledNameMap)
          if (lfunc.getName() ==
              cast<mlir::StringAttr>(entry.getValue()).getValue()) {
            skip = true;
            break;
          }
      if (!skip) {
        auto clonedFunc = lfunc.clone();
        if (clonedFunc.getName() != func.getName())
          clonedFunc.setPrivate();
        moduleOp.push_back(std::move(clonedFunc));
      }
    } else {
      moduleOp.push_back(op.clone());
    }
  }
  return moduleOp;
}
