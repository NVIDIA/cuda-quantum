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

std::string
cudaq_internal::compiler::getPassPipeline(const cudaq::CompileTarget &target) {
  const auto &pipelineConfig = target.pipelineConfig;
  const std::string allowEarlyExit =
      pipelineConfig.codegenTranslation.starts_with("qir-adaptive") ? "true"
                                                                    : "false";
  const std::string noLoopUnroll =
      pipelineConfig.codegenTranslation == "nop" ? " no-loop-unroll=true" : "";

  std::string passPipeline = "canonicalize";

  if (!pipelineConfig.overridePassPipeline.empty()) {
    passPipeline = pipelineConfig.overridePassPipeline;
    target.updatePassPipeline(passPipeline);
    return passPipeline;
  }

  if (target.emulate)
    appendPipelineStage(passPipeline, "emul-jit-prep-pipeline{erase-noise=true "
                                      "allow-early-exit=" +
                                          allowEarlyExit + noLoopUnroll + "}");
  else
    appendPipelineStage(passPipeline, "hw-jit-prep-pipeline{allow-early-exit=" +
                                          allowEarlyExit + noLoopUnroll + "}");

  const std::string lowerDeviceCalls =
      (pipelineConfig.codegenTranslation == "nop" && !target.emulate) ? "false"
                                                                      : "true";
  const std::string deployStage =
      pipelineConfig.codegenTranslation == "nop"
          ? "jit-deploy-pipeline{no-loop-unroll=true}"
          : "jit-deploy-pipeline";
  const std::string finalizeStage =
      "jit-finalize-pipeline{lower-device-calls=" + lowerDeviceCalls + "}";

  appendPipelineStage(passPipeline, pipelineConfig.highLevelPipeline);
  appendPipelineStage(passPipeline, deployStage);
  appendPipelineStage(passPipeline, pipelineConfig.midLevelPipeline);
  appendPipelineStage(passPipeline, finalizeStage);
  appendPipelineStage(passPipeline, pipelineConfig.lowLevelPipeline);

  // Handle disable_qubit_mapping runtime option.
  if (pipelineConfig.disableQubitMapping) {
    setQubitMappingBypass(passPipeline);
    CUDAQ_INFO("disable_qubit_mapping option found, so updated lowering "
               "pipeline to {}",
               passPipeline);
  }

  target.updatePassPipeline(passPipeline);

  return passPipeline;
}
