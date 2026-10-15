/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq/runtime/logger/logger.h"
#include <regex>

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

std::string cudaq_internal::compiler::getPassPipeline(
    const cudaq::CompileTarget &target, const cudaq::CompileOptions &options) {
  std::optional<cudaq::config::CustomPipelineLowering> pipelineConfig;
  if (!target.stages.empty())
    pipelineConfig = target.getCustomPipelineLowering();
  const std::string allowEarlyExit =
      target.codegenTranslation.starts_with("qir-adaptive") ? "true" : "false";
  const std::string noLoopUnroll =
      target.codegenTranslation == "nop" ? " no-loop-unroll=true" : "";

  std::string passPipeline = "canonicalize";

  if (pipelineConfig && !pipelineConfig->overridePassPipeline.empty()) {
    passPipeline = pipelineConfig->overridePassPipeline;
    return passPipeline;
  }

  if (options.emulate)
    appendPipelineStage(passPipeline, "emul-jit-prep-pipeline{erase-noise=true "
                                      "allow-early-exit=" +
                                          allowEarlyExit + noLoopUnroll + "}");
  else
    appendPipelineStage(passPipeline, "hw-jit-prep-pipeline{allow-early-exit=" +
                                          allowEarlyExit + noLoopUnroll + "}");

  const std::string lowerDeviceCalls =
      (target.codegenTranslation == "nop" && !options.emulate) ? "false"
                                                               : "true";
  const std::string deployStage =
      target.codegenTranslation == "nop"
          ? "jit-deploy-pipeline{no-loop-unroll=true}"
          : "jit-deploy-pipeline";
  const std::string finalizeStage =
      "jit-finalize-pipeline{lower-device-calls=" + lowerDeviceCalls + "}";

  if (pipelineConfig)
    appendPipelineStage(passPipeline, pipelineConfig->highLevelPipeline);
  appendPipelineStage(passPipeline, deployStage);
  if (pipelineConfig)
    appendPipelineStage(passPipeline, pipelineConfig->midLevelPipeline);
  appendPipelineStage(passPipeline, finalizeStage);
  if (pipelineConfig)
    appendPipelineStage(passPipeline, pipelineConfig->lowLevelPipeline);

  // Handle disable_qubit_mapping runtime option.
  if (target.disableQubitMapping) {
    setQubitMappingBypass(passPipeline);
    CUDAQ_INFO("disable_qubit_mapping option found, so updated lowering "
               "pipeline to {}",
               passPipeline);
  }

  return passPipeline;
}
