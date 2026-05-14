/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/JITTargetPipeline.h"
#include "cudaq/Support/TargetConfig.h"
#include <cctype>
#include <regex>

static void appendPipelineStage(std::string &pipeline,
                                const std::string &stage) {
  if (stage.empty())
    return;
  if (!pipeline.empty())
    pipeline += ",";
  pipeline += stage;
}

void cudaq_internal::compiler::substitutePipelinePlaceholders(
    std::string &pipeline,
    const std::map<std::string, std::string> &runtimeConfig) {
  std::string::size_type pos = 0;
  while (pos < pipeline.size()) {
    auto start = pipeline.find('%', pos);
    if (start == std::string::npos)
      break;
    auto end = pipeline.find('%', start + 1);
    if (end == std::string::npos)
      break;
    auto token = pipeline.substr(start + 1, end - start - 1);
    auto colon = token.find(':');
    auto key = (colon != std::string::npos) ? token.substr(0, colon) : token;

    std::string lower;
    for (char c : key)
      lower += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    auto it = runtimeConfig.find(lower);

    if (it != runtimeConfig.end()) {
      pipeline.replace(start, end - start + 1, it->second);
      pos = start + it->second.size();
    } else if (colon != std::string::npos) {
      auto defaultVal = token.substr(colon + 1);
      pipeline.replace(start, end - start + 1, defaultVal);
      pos = start + defaultVal.size();
    } else {
      pos = end + 1;
    }
  }
}

void cudaq_internal::compiler::setQubitMappingBypass(std::string &pipeline) {
  std::regex qubitMapping("(.*)qubit-mapping\\{(.*)device=[^,\\}]+(.*)\\}(.*)");
  std::string replacement("$1qubit-mapping{$2device=bypass$3}$4");
  pipeline = std::regex_replace(pipeline, qubitMapping, replacement);
}

cudaq_internal::compiler::JITTargetPipelineConfig
cudaq_internal::compiler::JITTargetPipelineConfig::createFromTargetConfig(
    const cudaq::config::TargetConfig &config,
    const std::map<std::string, std::string> &runtimeConfig, bool emulate) {
  JITTargetPipelineConfig pipelineConfig;
  if (!config.BackendConfig.has_value())
    return pipelineConfig;

  pipelineConfig.hasBackendConfig = true;
  const auto &backendConfig = *config.BackendConfig;
  pipelineConfig.hasConfiguredPassPipeline = backendConfig.hasPassPipeline();
  pipelineConfig.codegenTranslation = config.getCodeGenSpec(runtimeConfig);
  pipelineConfig.postCodeGenPasses = backendConfig.PostCodeGenPasses;

  if (!backendConfig.TargetPassPipeline.empty()) {
    pipelineConfig.passPipelineConfig = backendConfig.TargetPassPipeline;
    pipelineConfig.usesTargetPassPipelineOverride = true;
    substitutePipelinePlaceholders(pipelineConfig.passPipelineConfig,
                                   runtimeConfig);
    return pipelineConfig;
  }

  const std::string allowEarlyExit =
      pipelineConfig.codegenTranslation.starts_with("qir-adaptive") ? "true"
                                                                    : "false";

  if (emulate)
    appendPipelineStage(pipelineConfig.passPipelineConfig,
                        "emul-jit-prep-pipeline{erase-noise=true "
                        "allow-early-exit=" +
                            allowEarlyExit + "}");
  else
    appendPipelineStage(
        pipelineConfig.passPipelineConfig,
        "hw-jit-prep-pipeline{allow-early-exit=" + allowEarlyExit + "}");

  const std::string lowerDeviceCalls =
      (pipelineConfig.codegenTranslation == "nop" && !emulate) ? "false"
                                                               : "true";
  appendPipelineStage(
      pipelineConfig.passPipelineConfig,
      backendConfig.getPassPipeline(
          "jit-deploy-pipeline", "jit-finalize-pipeline{lower-device-calls=" +
                                     lowerDeviceCalls + "}"));
  substitutePipelinePlaceholders(pipelineConfig.passPipelineConfig,
                                 runtimeConfig);
  pipelineConfig.runsStandardFinalize = true;
  return pipelineConfig;
}
