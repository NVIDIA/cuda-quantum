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
  else {
    // Local simulators apply unitaries natively; skip synthesis (it drops
    // global phase, breaking controlled custom ops). Hardware keeps it.
    const std::string noUnitarySynthesis =
        pipelineConfig.nativeGateSet ? " no-unitary-synthesis=true" : "";
    appendPipelineStage(passPipeline, "hw-jit-prep-pipeline{allow-early-exit=" +
                                          allowEarlyExit + noLoopUnroll +
                                          noUnitarySynthesis + "}");
  }

  const std::string lowerDeviceCalls =
      (pipelineConfig.codegenTranslation == "nop" && !target.emulate) ? "false"
                                                                      : "true";
  // Local simulators support multi-controlled gates natively; skip
  // decomposition (its ancillas show up as extra measured qubits). Else keep
  // it.
  std::vector<std::string> deployOpts;
  if (pipelineConfig.codegenTranslation == "nop")
    deployOpts.push_back("no-loop-unroll=true");
  if (pipelineConfig.nativeGateSet)
    deployOpts.push_back("no-native-gate-decomposition=true");
  std::string deployStage = "jit-deploy-pipeline";
  if (!deployOpts.empty()) {
    deployStage += "{";
    for (std::size_t i = 0; i < deployOpts.size(); ++i)
      deployStage += (i ? " " : "") + deployOpts[i];
    deployStage += "}";
  }
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
