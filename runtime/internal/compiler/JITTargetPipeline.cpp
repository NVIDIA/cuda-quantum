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

/// Apply the target's `disable_qubit_mapping` runtime option, which rewrites
/// `qubit-mapping{...device=...}` in \p pipeline to use `device=bypass`.
static std::string
applyDisableQubitMapping(std::string pipeline,
                         const cudaq::CompileTarget &target) {
  if (!target.disableQubitMapping)
    return pipeline;
  std::regex qubitMapping("(.*)qubit-mapping\\{(.*)device=[^,\\}]+(.*)\\}(.*)");
  std::string replacement("$1qubit-mapping{$2device=bypass$3}$4");
  pipeline = std::regex_replace(pipeline, qubitMapping, replacement);
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
  // TODO: remove this restriction. This exists as the pipelines inferred for
  // each stage don't compose nicely (e.g. it doesn't make sense to run
  // `emul-jit-prep-pipeline` more than once)
  if (target.stages.size() > 1) {
    diagnostic = "Multi-stage compilation is currently not supported";
    return std::nullopt;
  }
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
    if (const auto *custom = std::get_if<cudaq::config::CustomPipelineLowering>(
            &stage.lowering)) {
      // An override pipeline replaces the whole pipeline.
      if (!custom->overridePassPipeline.empty())
        return custom->overridePassPipeline;
      // A custom stage's high-, mid-, and low-level pipelines run before
      // deploy, between deploy and finalize, and after finalize.
      stagePasses = StagePasses{.preDeployPasses = custom->highLevelPipeline,
                                .postDeployPasses = custom->midLevelPipeline,
                                .postFinalizePasses = custom->lowLevelPipeline};
    } else {
      std::string profiles(getProfileName(inputProfile));
      profiles += " to ";
      profiles += getProfileName(stage.irOutputProfile);
      const auto *handler = selectLoweringHandler(
          handlers, inputProfile, stage.irOutputProfile, options);
      if (!handler)
        return failStage("no lowering handler supports lowering from " +
                         profiles);
      stagePasses =
          handler->getStagePasses(inputProfile, stage.irOutputProfile, options);
      if (!stagePasses)
        return failStage("the selected lowering handler failed to construct "
                         "a pipeline from " +
                         profiles);
    }

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
