/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/JITTargetPipeline.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Target/CompilationStage.h"
#include "cudaq/runtime/logger/logger.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include <optional>
#include <regex>
#include <variant>
#include <vector>

using cudaq_internal::compiler::JITPassPipeline;
using cudaq_internal::compiler::JITPipelineLayout;
using cudaq_internal::compiler::StagePasses;

/// Apply the target's `disable_qubit_mapping` runtime option, which sets
/// `device=bypass` on every `qubit-mapping` pass in \p pipeline.
static std::string
applyDisableQubitMapping(std::string pipeline,
                         const cudaq::CompileTarget &target) {
  if (!target.disableQubitMapping)
    return pipeline;
  std::regex qubitMappingDevice(
      R"re((qubit-mapping\{(?:[^}]*\s)?device=)(?:'[^'}]*'|"[^"}]*"|[^\s{}])+)re");
  auto updated = std::regex_replace(pipeline, qubitMappingDevice, "$1bypass");
  if (updated != pipeline)
    CUDAQ_INFO("disable_qubit_mapping option found, so updated custom lowering "
               "pipeline to {}",
               updated);
  return updated;
}

/// Remove `combine-measurements` from \p pipeline, to defer it (see
/// `JITPassPipeline::deferCombineMeasurements`).
static std::string removeCombineMeasurements(const std::string &pipeline) {
  static const std::regex combine(
      R"re(\s*,\s*combine-measurements(\{[^}]*\})?|combine-measurements(\{[^}]*\})?\s*,?\s*)re");
  return std::regex_replace(pipeline, combine, "");
}

/// Prepare (and adjust) the string pipeline \p text for MLIR parsing.
///
/// Records metadata into \p layout, defers combine-measurements if required and
/// verify that the pipeline is valid.
///
/// Returns the validated pipeline text, or `nullopt` and a \p diagnostic.
static std::optional<std::string>
preprocessTextPipeline(const std::string &text,
                       const cudaq::CompileTarget &target,
                       const cudaq::CompileOptions &options,
                       JITPipelineLayout &layout, std::string &diagnostic) {
  // 1. Amend pipeline if target disables qubit mapping
  auto pipeline = applyDisableQubitMapping(text, target);

  // 2. Defer combine-measurements (if present) when emulate=true or computing
  // observables
  if ((options.emulate || options.measureObservable) &&
      pipeline.find("combine-measurements") != std::string::npos) {
    layout.deferCombineMeasurements = true;
    pipeline = removeCombineMeasurements(pipeline);
    CUDAQ_INFO("Deferring combine-measurements pass. Updating custom lowering "
               "pipeline to {}",
               pipeline);
  }

  llvm::raw_string_ostream errors(diagnostic);
  const auto fail = [&] {
    diagnostic = "invalid custom pipeline '" + pipeline + "': " + diagnostic;
    return std::nullopt;
  };

  // 3. Look for a gate mapping pass and save it separately
  llvm::SmallVector<llvm::StringRef> elements;
  llvm::StringRef(pipeline).split(elements, ',');
  for (auto element : elements) {
    element = element.trim();
    if (element.ends_with("-gate-set-mapping") &&
        mlir::failed(mlir::parsePassPipeline(
            element, layout.gateSetMappingPasses, errors)))
      return fail();
  }

  // 4. Check that the pipeline parses, so that adding it later cannot fail
  mlir::OpPassManager check(mlir::ModuleOp::getOperationName());
  if (mlir::failed(mlir::parsePassPipeline(pipeline, check, errors)))
    return fail();
  return pipeline;
}

/// Create a function that appends pipeline \p text to a pass manager.
static cudaq_internal::compiler::AddPassesFn
createAddPassesFn(std::string text) {
  return [text = std::move(text)](mlir::OpPassManager &pm) {
    (void)mlir::parsePassPipeline(text, pm);
  };
}

/// Name of the profile kind, for diagnostics.
static std::string_view getProfileName(const cudaq::config::Profile &profile) {
  return std::visit(
      [](const auto &profile) { return std::string_view(profile.name); },
      profile);
}

/// The output profile that an inferred stage lowers to.
///
/// Overrides the `output` profile if the target disables qubit mapping and
/// output is a mapped profile. Otherwise, returns `output`
static cudaq::config::Profile
getInferredOutputProfile(const cudaq::config::Profile &output,
                         const cudaq::CompileTarget &target) {
  const auto *mapped = std::get_if<cudaq::config::QuakeMappedProfile>(&output);
  if (!mapped || !target.disableQubitMapping || !mapped->device)
    return output;
  auto bypass = *mapped;
  bypass.device = "bypass";
  CUDAQ_INFO("disable_qubit_mapping option found, so mapping to the bypass "
             "device");
  return bypass;
}

/// Collect the passes of every stage of \p target into \p layout
///
/// Only the first stage may add passes before deploy and only the last stage
/// may add passes after finalize.
static bool collectStages(
    const cudaq::CompileTarget &target, const cudaq::CompileOptions &options,
    const std::vector<
        std::unique_ptr<cudaq_internal::compiler::LoweringHandler>> &handlers,
    JITPipelineLayout &layout, std::string &diagnostic) {
  cudaq::config::Profile inputProfile = cudaq::config::UncheckedProfile{};
  for (std::size_t i = 0; i < target.stages.size(); ++i) {
    const auto &stage = target.stages[i];
    const auto failStage = [&](std::string reason) {
      diagnostic = "compile target stage " + std::to_string(i + 1) + ": ";
      diagnostic += reason;
      return false;
    };

    StagePasses passes;
    if (const auto *custom = std::get_if<cudaq::config::CustomPipelineLowering>(
            &stage.lowering)) {
      if (!custom->overridePassPipeline.empty())
        return failStage(
            "an override pipeline must be the target's only stage");

      // Add custom pipelines to their corresponding stage
      auto passToStage = {
          std::pair{&custom->highLevelPipeline, &StagePasses::preDeployPasses},
          {&custom->midLevelPipeline, &StagePasses::postDeployPasses},
          {&custom->lowLevelPipeline, &StagePasses::postFinalizePasses}};
      for (auto [text, place] : passToStage) {
        if (text->empty())
          continue;
        auto pipeline =
            preprocessTextPipeline(*text, target, options, layout, diagnostic);
        if (!pipeline)
          return failStage(diagnostic);
        passes.*place = createAddPassesFn(std::move(*pipeline));
      }
    } else {
      const auto *handler =
          selectLoweringHandler(handlers, inputProfile, stage.irOutputProfile);
      if (!handler ||
          mlir::failed(handler->addStagePasses(
              inputProfile,
              getInferredOutputProfile(stage.irOutputProfile, target), options,
              passes)))
        return failStage(
            std::string(handler ? "the selected lowering handler failed to "
                                  "construct a pipeline from "
                                : "no lowering handler supports lowering "
                                  "from ") +
            std::string(getProfileName(inputProfile)) + " to " +
            std::string(getProfileName(stage.irOutputProfile)));
    }

    // A later stage's passes before deploy would run before the earlier
    // stages' passes, and an earlier stage's passes after finalize would run
    // after the later stages' passes, so either would break stage order.
    if (i > 0 && passes.preDeployPasses)
      return failStage(
          "only the first stage may add passes before jit-deploy-pipeline");
    if (i + 1 < target.stages.size() && passes.postFinalizePasses)
      return failStage(
          "only the last stage may add passes after jit-finalize-pipeline");
    layout.stages.push_back(std::move(passes));
    inputProfile = stage.irOutputProfile;
  }
  return true;
}

void JITPipelineLayout::addStagePasses(
    mlir::OpPassManager &pm,
    cudaq_internal::compiler::AddPassesFn StagePasses::*place) const {
  for (const auto &stage : stages)
    if (const auto &addPasses = stage.*place)
      addPasses(pm);
}

void JITPipelineLayout::addPasses(mlir::OpPassManager &pm) const {
  pm.addPass(mlir::createCanonicalizerPass());
  cudaq::opt::addJITPrepPipeline(pm, prep);
  addStagePasses(pm, &StagePasses::preDeployPasses);
  cudaq::opt::addJITDeployPipeline(pm, deploy);
  addStagePasses(pm, &StagePasses::postDeployPasses);
  cudaq::opt::addJITFinalizePipeline(pm, finalize);
  addStagePasses(pm, &StagePasses::postFinalizePasses);
}

std::optional<JITPipelineLayout> cudaq_internal::compiler::getPipelineLayout(
    const cudaq::CompileTarget &target, const cudaq::CompileOptions &options,
    const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
    std::string &diagnostic) {
  if (target.stages.empty()) {
    diagnostic = "the target has no stages, so its pipeline is empty";
    return std::nullopt;
  }
  if (target.getOverridePassPipeline()) {
    diagnostic = "an override pipeline replaces the whole pipeline";
    return std::nullopt;
  }

  // The emission format, `emulate`, and the target's loop, noise, and QEC
  // settings set the options of prep, deploy, and finalize.
  JITPipelineLayout layout;
  const auto &emissionFormat = target.codegen.emissionFormat;
  const bool unrollLoops = !target.preserveLoops && emissionFormat != "nop";
  layout.prep = {
      .eraseNoise = target.noiseSupport == cudaq::NoiseSupport::erase,
      .rejectNoise = target.noiseSupport == cudaq::NoiseSupport::unsupported,
      .eraseQEC = target.eraseQEC,
      .allowEarlyExit = emissionFormat.starts_with("qir-adaptive"),
      .disableLoopUnrolling = !unrollLoops};
  layout.deploy = {.disableLoopUnrolling = !unrollLoops};
  layout.finalize = {.lowerDeviceCalls =
                         emissionFormat != "nop" || options.emulate};

  if (!collectStages(target, options, handlers, layout, diagnostic))
    return std::nullopt;
  return layout;
}

std::optional<JITPassPipeline> cudaq_internal::compiler::getPassPipeline(
    const cudaq::CompileTarget &target, const cudaq::CompileOptions &options,
    const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
    std::string &diagnostic) {
  JITPassPipeline result;
  if (target.stages.empty())
    return result;

  JITPipelineLayout layout;
  if (const auto *override = target.getOverridePassPipeline()) {
    // An override pipeline replaces the whole pipeline, so it must be the
    // target's only stage.
    auto pipeline =
        preprocessTextPipeline(*override, target, options, layout, diagnostic);
    if (!pipeline) {
      diagnostic = "compile target stage 1: " + diagnostic;
      return std::nullopt;
    }
    createAddPassesFn(std::move(*pipeline))(result.passes);
  } else {
    auto inferred = getPipelineLayout(target, options, handlers, diagnostic);
    if (!inferred)
      return std::nullopt;
    layout = std::move(*inferred);
    layout.addPasses(result.passes);
  }

  result.deferCombineMeasurements = layout.deferCombineMeasurements;
  result.gateSetMappingPasses = std::move(layout.gateSetMappingPasses);
  return result;
}

bool cudaq_internal::compiler::specializesArguments(
    const cudaq::CompileTarget &target,
    const std::vector<std::unique_ptr<LoweringHandler>> &handlers) {
  if (target.fullySpecialize)
    return true;
  cudaq::config::Profile inputProfile = cudaq::config::UncheckedProfile{};
  for (const auto &stage : target.stages) {
    if (std::holds_alternative<cudaq::config::InferredLowering>(
            stage.lowering)) {
      const auto *handler =
          selectLoweringHandler(handlers, inputProfile, stage.irOutputProfile);
      if (handler &&
          handler->requiresArgumentValues(inputProfile, stage.irOutputProfile))
        return true;
    }
    inputProfile = stage.irOutputProfile;
  }
  return false;
}

bool cudaq_internal::compiler::specializesArguments(
    const cudaq::CompileTarget &target) {
  static const auto builtinHandlers = makeBuiltinLoweringHandlers();
  return specializesArguments(target, builtinHandlers);
}
