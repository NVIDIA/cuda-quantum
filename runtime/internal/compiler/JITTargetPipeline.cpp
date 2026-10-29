/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq/Target/CompilationStage.h"
#include "cudaq/runtime/logger/logger.h"
#include <regex>
#include <variant>

static void appendPipelineStage(std::string &pipeline,
                                const std::string &stage) {
  if (stage.empty())
    return;
  if (!pipeline.empty())
    pipeline += ",";
  pipeline += stage;
}

/// Apply the target's `disable_qubit_mapping` runtime option, which sets
/// `device=bypass` on every `qubit-mapping` pass in \p pipeline. As in MLIR's
/// pass option parser, a device value ends at the first space outside a pair
/// of single or double quotes, and an unpaired quote is an ordinary character.
/// It also ends at `}`, which closes the pass options. So `grid(3,3)`,
/// `file('/path/arch (1).txt')`, and `file(/home/o'brien/arch.txt)` are each
/// replaced whole, and the pass's other options are kept.
static std::string
applyDisableQubitMapping(std::string pipeline,
                         const cudaq::CompileTarget &target) {
  if (!target.disableQubitMapping)
    return pipeline;
  std::regex qubitMappingDevice(
      R"re((qubit-mapping\{(?:[^}]*\s)?device=)(?:'[^'}]*'|"[^"}]*"|[^\s{}])+)re");
  pipeline = std::regex_replace(pipeline, qubitMappingDevice, "$1bypass");
  CUDAQ_INFO("disable_qubit_mapping option found, so updated lowering "
             "pipeline to {}",
             pipeline);
  return pipeline;
}

/// Name of the profile kind, for diagnostics.
static std::string_view getProfileName(const cudaq::config::Profile &profile) {
  return std::visit(
      [](const auto &profile) { return std::string_view(profile.name); },
      profile);
}

/// The passes of the first handler in \p handlers that lowers \p input to
/// \p output. Sets \p diagnostic and returns `nullopt` when no handler supports
/// the lowering or the selected handler fails.
static std::optional<cudaq_internal::compiler::StagePasses>
getInferredStagePasses(
    const std::vector<
        std::unique_ptr<cudaq_internal::compiler::LoweringHandler>> &handlers,
    const cudaq::config::Profile &input, const cudaq::config::Profile &output,
    const cudaq::CompileOptions &options, std::string &diagnostic) {
  const auto *handler =
      cudaq_internal::compiler::selectLoweringHandler(handlers, input, output);
  std::optional<cudaq_internal::compiler::StagePasses> passes;
  if (handler)
    passes = handler->getStagePasses(input, output, options);
  if (!passes) {
    diagnostic = handler ? "the selected lowering handler failed to construct "
                           "a pipeline from "
                         : "no lowering handler supports lowering from ";
    diagnostic += getProfileName(input);
    diagnostic += " to ";
    diagnostic += getProfileName(output);
  }
  return passes;
}

/// Build the JIT pipeline from `canonicalize`, prep, deploy, and finalize,
/// with the stages' passes around them. The emission format and `emulate`
/// set the options of prep, deploy, and finalize.
static std::string
buildPassPipeline(const cudaq_internal::compiler::StagePasses &targetPasses,
                  const cudaq::CompileTarget &target,
                  const cudaq::CompileOptions &options) {
  const std::string lowerDeviceCalls =
      (target.codegen.emissionFormat == "nop" && !options.emulate) ? "false"
                                                                   : "true";
  const std::string allowEarlyExit =
      target.codegen.emissionFormat.starts_with("qir-adaptive") ? "true"
                                                                : "false";
  const std::string noLoopUnroll =
      target.codegen.emissionFormat == "nop" ? " no-loop-unroll=true" : "";

  std::string passPipeline = "canonicalize";
  if (options.emulate)
    appendPipelineStage(passPipeline, "emul-jit-prep-pipeline{erase-noise=true "
                                      "allow-early-exit=" +
                                          allowEarlyExit + noLoopUnroll + "}");
  else
    appendPipelineStage(passPipeline, "hw-jit-prep-pipeline{allow-early-exit=" +
                                          allowEarlyExit + noLoopUnroll + "}");

  const std::string deployStage =
      target.codegen.emissionFormat == "nop"
          ? "jit-deploy-pipeline{no-loop-unroll=true}"
          : "jit-deploy-pipeline";
  const std::string finalizeStage =
      "jit-finalize-pipeline{lower-device-calls=" + lowerDeviceCalls + "}";

  appendPipelineStage(passPipeline, targetPasses.preDeployPasses);
  appendPipelineStage(passPipeline, deployStage);
  appendPipelineStage(passPipeline, targetPasses.postDeployPasses);
  appendPipelineStage(passPipeline, finalizeStage);
  appendPipelineStage(passPipeline, targetPasses.postFinalizePasses);
  return passPipeline;
}

std::optional<std::string> cudaq_internal::compiler::getPassPipeline(
    const cudaq::CompileTarget &target, const cudaq::CompileOptions &options,
    const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
    std::string &diagnostic) {
  if (target.stages.empty())
    return std::string{};

  StagePasses targetPasses;
  cudaq::config::Profile inputProfile = cudaq::config::UncheckedProfile{};
  for (std::size_t i = 0; i < target.stages.size(); ++i) {
    const auto &stage = target.stages[i];
    const auto failStage = [&](std::string_view reason) {
      diagnostic = "compile target stage " + std::to_string(i + 1) + ": ";
      diagnostic += reason;
      return std::nullopt;
    };

    std::optional<StagePasses> stagePasses;
    std::string reason;
    if (const auto *custom = std::get_if<cudaq::config::CustomPipelineLowering>(
            &stage.lowering)) {
      // An override pipeline replaces the whole pipeline, so it must be the
      // target's only stage.
      if (!custom->overridePassPipeline.empty()) {
        if (target.stages.size() > 1)
          return failStage(
              "an override pipeline must be the target's only stage");
        return applyDisableQubitMapping(custom->overridePassPipeline, target);
      }
      // A custom stage's high-, mid-, and low-level pipelines run before
      // deploy, between deploy and finalize, and after finalize.
      stagePasses = StagePasses{.preDeployPasses = custom->highLevelPipeline,
                                .postDeployPasses = custom->midLevelPipeline,
                                .postFinalizePasses = custom->lowLevelPipeline};
    } else {
      stagePasses = getInferredStagePasses(
          handlers, inputProfile, stage.irOutputProfile, options, reason);
    }
    if (!stagePasses)
      return failStage(reason);

    // A later stage's passes before deploy would run before the earlier
    // stages' passes, and an earlier stage's passes after finalize would run
    // after the later stages' passes, so either would break stage order.
    if (i > 0 && !stagePasses->preDeployPasses.empty())
      return failStage(
          "only the first stage may add passes before jit-deploy-pipeline");
    if (i + 1 < target.stages.size() &&
        !stagePasses->postFinalizePasses.empty())
      return failStage(
          "only the last stage may add passes after jit-finalize-pipeline");
    appendPipelineStage(targetPasses.preDeployPasses,
                        stagePasses->preDeployPasses);
    appendPipelineStage(targetPasses.postDeployPasses,
                        stagePasses->postDeployPasses);
    appendPipelineStage(targetPasses.postFinalizePasses,
                        stagePasses->postFinalizePasses);
    inputProfile = stage.irOutputProfile;
  }

  return applyDisableQubitMapping(
      buildPassPipeline(targetPasses, target, options), target);
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
