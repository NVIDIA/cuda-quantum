/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "Compiler.h"
#include "ServerHelper.h"
#include "common/ArgumentConversion.h"
#include "common/CodeGenConfig.h"
#include "common/DeviceCodeRegistry.h"
#include "common/Environment.h"
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "common/NoiseModel.h"
#include "common/Resources.h"
#include "common/RuntimeMLIR.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "cudaq/Optimizer/Transforms/AddMetadata.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/TargetConfig.h"
#include "cudaq/runtime/logger/logger.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include <functional>
#include <memory>
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
        if (auto strAttr = op.getAttr(cudaq::opt::QIROutputNamesAttrName)
                               .dyn_cast_or_null<mlir::StringAttr>()) {
          output_names = nlohmann::json::parse(strAttr.getValue());
          break;
        }
      }
    }
  }
  return output_names;
}
} // namespace

namespace cudaq {

std::tuple<mlir::ModuleOp, std::unique_ptr<mlir::MLIRContext>, void *>
Compiler::extractQuakeCodeAndContext(const std::string &kernelName,
                                     void *data) {
  auto context = cudaq::getOwningMLIRContext();

  // Get the quake representation of the kernel
  auto quakeCode = cudaq::get_quake_by_name(kernelName);
  auto m_module = parseSourceString<mlir::ModuleOp>(quakeCode, context.get());
  if (!m_module)
    throw std::runtime_error("module cannot be parsed");

  return std::make_tuple(m_module.release(), std::move(context), data);
}

Compiler::Compiler(ServerHelper *serverHelper,
                   const std::map<std::string, std::string> &backendConfig,
                   config::TargetConfig &config, const noise_model *noiseModel,
                   bool emulate)
    : emulate(emulate) {

  cudaq::initializeMLIR();

  // Print the IR if requested
  printIR = getEnvBool("CUDAQ_DUMP_JIT_IR", printIR);

  // Get additional debug values
  disableMLIRthreading =
      getEnvBool("CUDAQ_MLIR_DISABLE_THREADING", disableMLIRthreading);
  enablePrintMLIREachPass =
      getEnvBool("CUDAQ_MLIR_PRINT_EACH_PASS", enablePrintMLIREachPass);
  enablePassStatistics =
      getEnvBool("CUDAQ_MLIR_PASS_STATISTICS", enablePassStatistics);

  // If the very verbose enablePrintMLIREachPass flag is set, then
  // multi-threading must be disabled.
  if (enablePrintMLIREachPass) {
    disableMLIRthreading = true;
  }

  if (config.BackendConfig.has_value()) {
    const auto codeGenSpec = config.getCodeGenSpec(backendConfig);
    if (!codeGenSpec.empty()) {
      CUDAQ_INFO("Set codegen translation: {}", codeGenSpec);
      codegenTranslation = codeGenSpec;
      // Validate codegen configuration.
      parseCodeGenTranslation(codegenTranslation);
    }

    const std::string allowEarlyExitSetting =
        codegenTranslation.starts_with("qir-adaptive") ? "true" : "false";

    // 1. Apply all the target-agnostic high-level passes. If this is an
    // emulation and a noise model has been set, do not erase the noise
    // callbacks.
    if (emulate)
      // FIXME: Noise should eventually be enabled for emulated hardware targets
      passPipelineConfig += ",emul-jit-prep-pipeline{erase-noise=true"
                            " allow-early-exit=" +
                            allowEarlyExitSetting + "}";
    else
      passPipelineConfig +=
          ",hw-jit-prep-pipeline{allow-early-exit=" + allowEarlyExitSetting +
          "}";

    // 2. Apply target-specific high-level passes from the .yml file, if any.
    if (!config.BackendConfig->JITHighLevelPipeline.empty()) {
      CUDAQ_INFO("Appending JIT high level pipeline: {}",
                 config.BackendConfig->JITHighLevelPipeline);
      passPipelineConfig += "," + config.BackendConfig->JITHighLevelPipeline;
    }

    // 3. Appply the target-agnostic deployment passes. Any additional
    // restructuring to get ready for decomposition.
    passPipelineConfig += ",jit-deploy-pipeline";

    // 4. Apply the target-specific mid-level passes. This decomposed quantum
    // gates for a specific target machine, etc.
    if (!config.BackendConfig->JITMidLevelPipeline.empty()) {
      CUDAQ_INFO("Appending JIT mid level pipeline: {}",
                 config.BackendConfig->JITMidLevelPipeline);
      passPipelineConfig += "," + config.BackendConfig->JITMidLevelPipeline;
    }

    // 5. Apply the target-agnostic finalization passes. This lowers the IR to
    // CFG form.
    passPipelineConfig += ",jit-finalize-pipeline";

    // 6. Apply the target-specific low-level passes.
    if (!config.BackendConfig->JITLowLevelPipeline.empty()) {
      CUDAQ_INFO("Appending JIT low level pipeline: {}",
                 config.BackendConfig->JITLowLevelPipeline);
      passPipelineConfig += "," + config.BackendConfig->JITLowLevelPipeline;
    }

    if (!config.BackendConfig->PostCodeGenPasses.empty()) {
      CUDAQ_INFO("Adding post-codegen lowering pipeline: {}",
                 config.BackendConfig->PostCodeGenPasses);
      postCodeGenPasses = config.BackendConfig->PostCodeGenPasses;
    }
  }

  auto disableQM = backendConfig.find("disable_qubit_mapping");
  if (disableQM != backendConfig.end() && disableQM->second == "true") {
    // Replace the qubit-mapping{device=<>} with
    // qubit-mapping{device=bypass} to effectively disable the qubit-mapping
    // pass. Use $1 - $4 to make sure any other pass options are left
    // untouched.
    std::regex qubitMapping(
        "(.*)qubit-mapping\\{(.*)device=[^,\\}]+(.*)\\}(.*)");
    std::string replacement("$1qubit-mapping{$2device=bypass$3}$4");
    passPipelineConfig =
        std::regex_replace(passPipelineConfig, qubitMapping, replacement);
    CUDAQ_INFO("disable_qubit_mapping option found, so updated lowering "
               "pipeline to {}",
               passPipelineConfig);
  }

  std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
  std::filesystem::path platformPath =
      cudaqLibPath.parent_path().parent_path() / "targets";

  serverHelper->updatePassPipeline(platformPath, passPipelineConfig);
}

Compiler::~Compiler() = default;

std::vector<cudaq::KernelExecution> Compiler::lowerQuakeCodePart2(
    ExecutionContext *executionContext, const std::string &kernelName,
    void *kernelArgs, const std::vector<void *> &rawArgs,
    mlir::ModuleOp m_module, mlir::MLIRContext *contextPtr, void *updatedArgs) {
  // Extract the kernel name
  auto origFn = m_module.template lookupSymbol<mlir::func::FuncOp>(
      std::string(cudaq::runtime::cudaqGenPrefixName) + kernelName);

  auto moduleOp =
      lowerQuakeCodeBuildModule(kernelName, m_module, contextPtr, origFn);

  // Lambda to apply a specific pipeline to the given ModuleOp
  auto runPassPipeline = [&](const std::string &pipeline,
                             mlir::ModuleOp moduleOpIn) {
    mlir::PassManager pm(contextPtr);
    std::string errMsg;
    llvm::raw_string_ostream os(errMsg);
    CUDAQ_INFO("Pass pipeline for {} = {}", kernelName, pipeline);
    if (failed(parsePassPipeline(pipeline, pm, os)))
      throw std::runtime_error(
          "Remote rest platform failed to add passes to pipeline (" + errMsg +
          ").");
    if (disableMLIRthreading || enablePrintMLIREachPass)
      moduleOpIn.getContext()->disableMultithreading();
    if (enablePrintMLIREachPass)
      pm.enableIRPrinting();
    if (failed(pm.run(moduleOpIn)))
      throw std::runtime_error("Remote rest platform Quake lowering failed.");
  };

  auto epFunc =
      moduleOp.template lookupSymbol<mlir::func::FuncOp>(origFn.getName());
  const bool isPython = moduleOp->hasAttr(runtime::pythonUniqueAttrName);
  if (!rawArgs.empty() || updatedArgs) {
    mlir::PassManager pm(contextPtr);
    if (isPython)
      detail::mergeAllCallableClosures(moduleOp, kernelName, rawArgs);

    // Mark all newly merged kernels private, and leave the entry point alone.
    for (auto &op : moduleOp)
      if (auto f = dyn_cast<mlir::func::FuncOp>(op))
        if (f != epFunc)
          f.setPrivate();

    if (!rawArgs.empty()) {
      CUDAQ_INFO("Run Argument Synth.\n");
      // For quantum devices, we generate a collection of `init` and
      // `num_qubits` functions and their substitutions created
      // from a kernel and arguments that generated a state argument.
      cudaq::opt::ArgumentConverter argCon(kernelName, moduleOp);
      argCon.gen(rawArgs);

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
      pm.addPass(opt::createArgumentSynthesisPass(kernelRefs, substRefs));
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
      pm.addNestedPass<mlir::func::FuncOp>(opt::createReplaceStateWithKernel());
      cudaq::opt::addAggressiveInlining(pm);
      pm.addPass(mlir::createSymbolDCEPass());
    } else if (updatedArgs) {
      CUDAQ_INFO("Run Quake Synth.\n");
      pm.addPass(cudaq::opt::createQuakeSynthesizer(kernelName, updatedArgs));
    }
    pm.addPass(mlir::createCanonicalizerPass());
    if (disableMLIRthreading || enablePrintMLIREachPass)
      moduleOp.getContext()->disableMultithreading();
    if (enablePrintMLIREachPass)
      pm.enableIRPrinting();
    if (failed(pm.run(moduleOp)))
      throw std::runtime_error("Could not successfully apply quake-synth.");
  }

  if (emulate && executionContext && executionContext->name == "sample") {
    // Populate conditional measurement flag in the context.
    for (auto &artifact : moduleOp) {
      quake::detail::QuakeFunctionAnalysis analysis{&artifact};
      auto info = analysis.getAnalysisInfo();
      if (info.empty())
        continue;
      auto result = info[&artifact];
      if (result.hasConditionalsOnMeasure) {
        throw std::runtime_error(
            "`cudaq::sample` and `cudaq::sample_async` no longer support "
            "kernels "
            "that branch on measurement results. Kernel '" +
            kernelName +
            "' uses conditional feedback. Use `cudaq::run` or "
            "`cudaq::run_async` "
            "instead. See CUDA-Q documentation for migration guide.");
      }
    }
  }
  // Delay combining measurements for backends that cannot handle
  // subveqs and multiple measurements until we created the emulation code.
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

  runPassPipeline(passPipelineConfig, moduleOp);
  // We need to run resource counting preprocessing after the pass pipeline as
  // the pre-processing might change the IR structure (may interfere with
  // other passes).
  std::unique_ptr<Resources> resourceCounts;
  if (executionContext && executionContext->name == "resource-count") {
    resourceCounts = std::make_unique<Resources>();
    // Each pass may run in a separate thread, so we have to make sure to
    // grab this reference in this thread
    std::function<void(std::string, size_t, size_t)> f =
        [&resourceCounts](std::string gate, size_t nControls, size_t count) {
          CUDAQ_INFO("Appending: {}", gate);
          resourceCounts->appendInstruction(gate, nControls, count);
        };
    cudaq::opt::ResourceCountPreprocessOptions opt{f};
    mlir::PassManager pm(moduleOp.getContext());
    pm.addNestedPass<mlir::func::FuncOp>(
        opt::createResourceCountPreprocess(opt));
    pm.addPass(mlir::createCanonicalizerPass());
    if (enablePrintMLIREachPass)
      pm.enableIRPrinting();
    if (failed(pm.run(moduleOp)))
      throw std::runtime_error(
          "Could not successfully apply resource count preprocess.");
  }

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

  if (executionContext) {
    if (executionContext->name == "sample") {
      executionContext->reorderIdx = mapping_reorder_idx;
      // Warn if kernel has named measurement registers (sub-registers).
      if (!executionContext->warnedNamedMeasurements) {
        auto funcOp = moduleOp.template lookupSymbol<mlir::func::FuncOp>(
            std::string(cudaq::runtime::cudaqGenPrefixName) + kernelName);
        if (funcOp) {
          bool hasNamedMeasurements = false;
          funcOp.walk([&](quake::MeasurementInterface meas) {
            if (meas.getOptionalRegisterName().has_value()) {
              hasNamedMeasurements = true;
              return mlir::WalkResult::interrupt();
            }
            return mlir::WalkResult::advance();
          });
          if (hasNamedMeasurements) {
            executionContext->warnedNamedMeasurements = true;
            std::cerr
                << "WARNING: Kernel \"" << kernelName
                << "\" uses named measurement results "
                << "but is invoked in sampling mode. Support for "
                << "sub-registers in `sample_result` is deprecated and will "
                << "be removed in a future release. Use `run` to retrieve "
                << "individual measurement results." << std::endl;
          }
        }
      }
      // No need to add measurements only to remove them eventually
      if (postCodeGenPasses.find("remove-measurements") == std::string::npos)
        runPassPipeline("func.func(add-measurements)", moduleOp);
    } else {
      executionContext->reorderIdx.clear();
    }
  }

  std::vector<std::pair<std::string, mlir::ModuleOp>> modules;
  // Apply observations if necessary
  if (executionContext && executionContext->name == "observe") {
    mapping_reorder_idx.clear();
    runPassPipeline("canonicalize,cse", moduleOp);
    cudaq::spin_op &spin = executionContext->spin.value();
    for (const auto &term : spin) {
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
      mlir::PassManager pm(contextPtr);
      pm.addNestedPass<mlir::func::FuncOp>(cudaq::opt::createObserveAnsatzPass(
          term.get_binary_symplectic_form()));
      if (disableMLIRthreading || enablePrintMLIREachPass)
        tmpModuleOp.getContext()->disableMultithreading();
      if (enablePrintMLIREachPass)
        pm.enableIRPrinting();
      if (failed(pm.run(tmpModuleOp)))
        throw std::runtime_error("Could not apply measurements to ansatz.");
      // The full pass pipeline was run above, but the ansatz pass can
      // introduce gates that aren't supported by the backend, so we need to
      // re-run the gate set mapping if that existed in the original pass
      // pipeline.
      auto csvSplit = cudaq::split(passPipelineConfig, ',');
      for (auto &pass : csvSplit)
        if (pass.ends_with("-gate-set-mapping"))
          runPassPipeline(pass, tmpModuleOp);
      if (!emulate && combineMeasurements)
        runPassPipeline("func.func(combine-measurements)", tmpModuleOp);
      modules.emplace_back(term.get_term_id(), tmpModuleOp);
    }
  } else {
    modules.emplace_back(kernelName, moduleOp);
  }

  if (emulate ||
      (executionContext && executionContext->name == "resource-count")) {
    // If we are in emulation mode, we need to first get a full QIR
    // representation of the code. Then we'll map to an LLVM Module, create a
    // JIT ExecutionEngine pointer and use that for execution
    for (auto &[name, module] : modules) {
      auto clonedModule = module.clone();
      jitEngines.emplace_back(
          cudaq::createQIRJITEngine(clonedModule, codegenTranslation));
    }
  }

  if (emulate && combineMeasurements)
    for (auto &[name, module] : modules)
      runPassPipeline("func.func(combine-measurements)", module);

  // Get the code gen translation
  auto translation = cudaq::getTranslation(codegenTranslation);

  // Apply user-specified codegen
  std::vector<cudaq::KernelExecution> codes;
  for (auto iter : llvm::enumerate(modules)) {
    auto &[name, moduleOpI] = iter.value();
    std::string codeStr;
    llvm::raw_string_ostream outStr(codeStr);
    if (disableMLIRthreading)
      moduleOpI.getContext()->disableMultithreading();
    if (codegenTranslation.starts_with("qir")) {
      if (failed(translation(moduleOpI, codegenTranslation, outStr,
                             postCodeGenPasses, printIR,
                             enablePrintMLIREachPass, enablePassStatistics)))
        throw std::runtime_error("Could not successfully translate to " +
                                 codegenTranslation + ".");
    } else {
      if (failed(translation(moduleOpI, outStr, postCodeGenPasses, printIR,
                             enablePrintMLIREachPass, enablePassStatistics)))
        throw std::runtime_error("Could not successfully translate to " +
                                 codegenTranslation + ".");
    }

    // Form an output_names mapping from codeStr
    nlohmann::json j = formOutputNames(codegenTranslation, moduleOpI, codeStr);

    auto optionalJit = jitEngines.size() > iter.index()
                           ? std::optional(jitEngines[iter.index()])
                           : std::nullopt;
    auto optionalResourceCounts = resourceCounts
                                      ? std::optional(*resourceCounts.release())
                                      : std::nullopt;
    codes.emplace_back(name, codeStr, optionalJit, optionalResourceCounts, j,
                       mapping_reorder_idx);
  }

  return codes;
}

/// @brief Extract the Quake representation for the given kernel name and
/// lower it to the code format required for the specific backend. The
/// lowering process is controllable via the configuration file in the
/// platform directory for the targeted backend.
std::vector<cudaq::KernelExecution>
Compiler::lowerQuakeCode(ExecutionContext *executionContext,
                         const std::string &kernelName, void *kernelArgs,
                         const std::vector<void *> &rawArgs) {

  auto [m_module, contextPtr, updatedArgs] =
      extractQuakeCodeAndContext(kernelName, kernelArgs);
  return lowerQuakeCodePart2(executionContext, kernelName, kernelArgs, rawArgs,
                             m_module, contextPtr.get(), updatedArgs);
}

std::vector<cudaq::KernelExecution>
Compiler::lowerQuakeCode(ExecutionContext *executionContext,
                         const std::string &kernelName, void *kernelArgs) {
  return lowerQuakeCode(executionContext, kernelName, kernelArgs, {});
}

std::vector<cudaq::KernelExecution>
Compiler::lowerQuakeCode(ExecutionContext *executionContext,
                         const std::string &kernelName,
                         const std::vector<void *> &rawArgs) {
  return lowerQuakeCode(executionContext, kernelName, nullptr, rawArgs);
}

std::vector<cudaq::KernelExecution>
Compiler::lowerQuakeCode(ExecutionContext *executionContext,
                         const std::string &kernelName, mlir::ModuleOp module,
                         const std::vector<void *> &rawArgs) {
  return lowerQuakeCodePart2(executionContext, kernelName, nullptr, rawArgs,
                             module, module.getContext(), nullptr);
}

mlir::ModuleOp Compiler::lowerQuakeCodeBuildModule(
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
      // module's map, or cudaq::details::getKernelName(). But make sure it
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
      [[maybe_unused]] auto decl = builder.create<mlir::func::FuncOp>(
          deviceCall.getLoc(), calleeName, funcType);
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
  auto moduleOp = builder.create<mlir::ModuleOp>();
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
      if (!skip && !deviceCallCallees.contains(lfunc.getName().str()))
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

} // namespace cudaq
