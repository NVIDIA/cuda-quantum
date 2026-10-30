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

/// The passes \p handler adds to lower \p input to \p output. Sets
/// \p diagnostic and returns `nullopt` when \p handler is null or fails.
static std::optional<cudaq_internal::compiler::StagePasses>
getInferredStagePasses(const cudaq_internal::compiler::LoweringHandler *handler,
                       const cudaq::config::Profile &input,
                       const cudaq::config::Profile &output,
                       const cudaq::CompileOptions &options,
                       std::string &diagnostic) {
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
/// with the stages' passes around them. The emission format, `emulate`, and
/// \p bookends set the options of prep, deploy, and finalize. An option from
/// \p bookends appears in the text only when it differs from its default, so
/// existing targets keep their pipeline text. Emulation's prep is the
/// exception: it has always spelled out `erase-noise`.
static std::string
buildPassPipeline(const cudaq_internal::compiler::StagePasses &targetPasses,
                  const cudaq_internal::compiler::BookendOptions &bookends,
                  const cudaq::CompileTarget &target,
                  const cudaq::CompileOptions &options) {
  const std::string lowerDeviceCalls =
      (target.codegen.emissionFormat == "nop" && !options.emulate) ? "false"
                                                                   : "true";
  const std::string allowEarlyExit =
      target.codegen.emissionFormat.starts_with("qir-adaptive") ? "true"
                                                                : "false";
  const bool unrollLoops =
      bookends.unrollLoops && target.codegen.emissionFormat != "nop";
  const std::string noLoopUnroll = unrollLoops ? "" : " no-loop-unroll=true";
  const std::string eraseQEC = bookends.eraseQEC ? "" : " erase-qec=false";

  std::string passPipeline = "canonicalize";
  if (options.emulate)
    appendPipelineStage(
        passPipeline, "emul-jit-prep-pipeline{erase-noise=" +
                          std::string(bookends.eraseNoise ? "true" : "false") +
                          " allow-early-exit=" + allowEarlyExit + noLoopUnroll +
                          eraseQEC + "}");
  else
    appendPipelineStage(
        passPipeline,
        "hw-jit-prep-pipeline{allow-early-exit=" + allowEarlyExit +
            noLoopUnroll + (bookends.eraseNoise ? "" : " erase-noise=false") +
            eraseQEC + "}");

  const std::string deployStage =
      unrollLoops ? "jit-deploy-pipeline"
                  : "jit-deploy-pipeline{no-loop-unroll=true}";
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
  BookendOptions bookends;
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
      const auto *handler =
          selectLoweringHandler(handlers, inputProfile, stage.irOutputProfile);
      stagePasses = getInferredStagePasses(
          handler, inputProfile, stage.irOutputProfile, options, reason);
      if (handler) {
        const auto needs =
            handler->getBookendOptions(inputProfile, stage.irOutputProfile);
        bookends.unrollLoops &= needs.unrollLoops;
        bookends.eraseNoise &= needs.eraseNoise;
        bookends.eraseQEC &= needs.eraseQEC;
      }
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
      buildPassPipeline(targetPasses, bookends, target, options), target);
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
