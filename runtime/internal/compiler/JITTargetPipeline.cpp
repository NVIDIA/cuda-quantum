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

/// Rewrite `qubit-mapping{...device=...}` to use `device=bypass`.
static void setQubitMappingBypass(std::string &pipeline) {
  std::regex qubitMapping("(.*)qubit-mapping\\{(.*)device=[^,\\}]+(.*)\\}(.*)");
  std::string replacement("$1qubit-mapping{$2device=bypass$3}$4");
  pipeline = std::regex_replace(pipeline, qubitMapping, replacement);
}

namespace {
/// Inputs required to build the passes of a single compile target stage.
struct StageResolutionContext {
  const cudaq::CompileTarget &target;
  const cudaq::CompileOptions &options;
  const std::vector<std::unique_ptr<cudaq_internal::compiler::LoweringHandler>>
      &handlers;
  const cudaq::config::Profile &inputProfile;
  const cudaq::config::Profile &outputProfile;
};
} // namespace

/// Name of the profile kind, for diagnostics.
static std::string_view getProfileName(const cudaq::config::Profile &profile) {
  return std::visit(
      [](const auto &profile) { return std::string_view(profile.name); },
      profile);
}

/// Append the JIT pipeline built around the user-provided pipelines of a
/// `CustomPipelineLowering` to `passPipeline`.
static bool appendResolvedStagePipeline(
    std::string &passPipeline,
    const cudaq::config::CustomPipelineLowering &lowering,
    const StageResolutionContext &ctx, std::string &) {
  const auto &target = ctx.target;
  const auto &options = ctx.options;

  if (!lowering.overridePassPipeline.empty()) {
    appendPipelineStage(passPipeline, lowering.overridePassPipeline);
    return true;
  }

  const std::string lowerDeviceCalls =
      (target.codegen.emissionFormat == "nop" && !options.emulate) ? "false"
                                                                   : "true";
  const std::string allowEarlyExit =
      target.codegen.emissionFormat.starts_with("qir-adaptive") ? "true"
                                                                : "false";
  const std::string noLoopUnroll =
      target.codegen.emissionFormat == "nop" ? " no-loop-unroll=true" : "";

  if (passPipeline.empty())
    passPipeline = "canonicalize";

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

  appendPipelineStage(passPipeline, lowering.highLevelPipeline);
  appendPipelineStage(passPipeline, deployStage);
  appendPipelineStage(passPipeline, lowering.midLevelPipeline);
  appendPipelineStage(passPipeline, finalizeStage);
  appendPipelineStage(passPipeline, lowering.lowLevelPipeline);

  // Handle disable_qubit_mapping runtime option.
  if (target.disableQubitMapping) {
    setQubitMappingBypass(passPipeline);
    CUDAQ_INFO("disable_qubit_mapping option found, so updated lowering "
               "pipeline to {}",
               passPipeline);
  }
  return true;
}

/// Infer the pipeline using the provided lowering handlers and append it to
/// `passPipeline`. On failure, sets `diagnostic` and returns false.
static bool appendResolvedStagePipeline(std::string &passPipeline,
                                        const cudaq::config::InferredLowering &,
                                        const StageResolutionContext &ctx,
                                        std::string &diagnostic) {
  const auto *handler = cudaq_internal::compiler::selectLoweringHandler(
      ctx.handlers, ctx.inputProfile, ctx.outputProfile, ctx.options);
  std::optional<std::string> pipeline;
  const char *failure = "no lowering handler supports lowering from ";
  if (handler) {
    pipeline =
        handler->getPipeline(ctx.inputProfile, ctx.outputProfile, ctx.options);
    failure = "the selected lowering handler failed to construct a "
              "pipeline from ";
  }
  if (!pipeline) {
    diagnostic = failure;
    diagnostic += getProfileName(ctx.inputProfile);
    diagnostic += " to ";
    diagnostic += getProfileName(ctx.outputProfile);
    return false;
  }
  appendPipelineStage(passPipeline, *pipeline);
  return true;
}

std::optional<std::string> cudaq_internal::compiler::getPassPipeline(
    const cudaq::CompileTarget &target, const cudaq::CompileOptions &options,
    const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
    std::string &diagnostic) {
  std::string passPipeline;
  cudaq::config::Profile inputProfile = cudaq::config::UncheckedProfile{};

  // TODO: remove this restriction. This exists as the pipelines inferred for
  // each stage don't compose nicely (e.g. it doesn't make sense to run
  // `emul-jit-prep-pipeline` more than once)
  if (target.stages.size() > 1) {
    diagnostic = "Multi-stage compilation is currently not supported";
    return std::nullopt;
  }

  for (std::size_t i = 0; i < target.stages.size(); ++i) {
    const auto &stage = target.stages[i];
    const StageResolutionContext ctx{target, options, handlers, inputProfile,
                                     stage.irOutputProfile};
    std::string stageDiagnostic;
    const bool success = std::visit(
        [&](const auto &lowering) {
          return appendResolvedStagePipeline(passPipeline, lowering, ctx,
                                             stageDiagnostic);
        },
        stage.lowering);
    if (!success) {
      diagnostic = "compile target stage " + std::to_string(i + 1) + ": " +
                   stageDiagnostic;
      return std::nullopt;
    }
    inputProfile = stage.irOutputProfile;
  }

  return passPipeline;
}
