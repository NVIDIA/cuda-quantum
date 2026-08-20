/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/CompileTarget.h"
#include "cudaq/Support/Hash.h"
#include "cudaq/runtime/logger/logger.h"
#include <cctype>

/// Replace `%KEY%` and `%KEY:default%` placeholders from runtime options.
static void substitutePipelinePlaceholders(
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

/// Replace literal placeholder keys in a pipeline stage string.
static void applyPipelineSubstitutions(
    std::string &pipeline,
    const std::map<std::string, std::string> &pipelineSubstitutions) {
  for (const auto &[key, value] : pipelineSubstitutions) {
    std::string::size_type pos = 0;
    while ((pos = pipeline.find(key, pos)) != std::string::npos) {
      pipeline.replace(pos, key.size(), value);
      pos += value.size();
    }
  }
}

cudaq::CompileTarget cudaq::CompileTarget::createFromConfig(
    config::TargetConfig targetConfig,
    std::map<std::string, std::string> runtimeConfig,
    std::map<std::string, std::string> pipelineSubstitutions) {
  cudaq::CompileTarget target;
  const config::BackendEndConfigEntry defaultConfig;

  const auto &backendConfig =
      targetConfig.BackendConfig.value_or(defaultConfig);
  auto prepPipeline = [&](const std::string &stage,
                          const std::string &stageName) {
    std::string pipeline = stage;
    if (!pipeline.empty()) {
      substitutePipelinePlaceholders(pipeline, runtimeConfig);
      applyPipelineSubstitutions(pipeline, pipelineSubstitutions);
      CUDAQ_INFO("{:<27} {}", stageName + ":", pipeline);
    }
    return pipeline;
  };

  if (!backendConfig.TargetPassPipeline.empty()) {
    target.pipelineConfig.overridePassPipeline = prepPipeline(
        backendConfig.TargetPassPipeline, "Pass pipeline (overridden)");
  } else {
    target.pipelineConfig.highLevelPipeline =
        prepPipeline(backendConfig.JITHighLevelPipeline, "JIT high level");
    target.pipelineConfig.midLevelPipeline =
        prepPipeline(backendConfig.JITMidLevelPipeline, "JIT mid level");
    target.pipelineConfig.lowLevelPipeline =
        prepPipeline(backendConfig.JITLowLevelPipeline, "JIT low level");
  }
  auto codegenTranslation = targetConfig.getCodeGenSpec(runtimeConfig);
  if (!codegenTranslation.empty()) {
    target.pipelineConfig.codegenTranslation = codegenTranslation;
    CUDAQ_INFO("{:<27} {}\n", "Codegen:", codegenTranslation);
  }
  if (!backendConfig.PostCodeGenPasses.empty()) {
    target.pipelineConfig.postCodeGenPasses = backendConfig.PostCodeGenPasses;
    CUDAQ_INFO("{:<27} {}\n",
               "Post-codegen:", target.pipelineConfig.postCodeGenPasses);
  }

  disableResourceCounting =
      backendConfig.DisableResourceCounting.value_or(false);

  // Handle disable_qubit_mapping runtime option.
  auto disableQM = runtimeConfig.find("disable_qubit_mapping");
  if (disableQM != runtimeConfig.end() && disableQM->second == "true") {
    target.pipelineConfig.disableQubitMapping = true;
    CUDAQ_INFO("{:<27} {}\n", "disable_qubit_mapping:", "true");
  }

  return target;
}

std::size_t std::hash<cudaq::CompileTarget>::operator()(
    const cudaq::CompileTarget &t) const noexcept {
  return cudaq::detail::hashVal(
      t.pipelineConfig, t.overrideAOTCompilation,
      t.supportConditionalsOnMeasureResults, t.supportDeviceCalls,
      t.supportExplicitMeasurements, t.supportObservableMeasurements,
      t.supportSampleWithoutMeasurements, t.fullySpecialize,
      t.argumentSynthChangeSemantics, t.disableResourceCounting);
}

std::size_t std::hash<cudaq::CompileTarget::PipelineConfig>::operator()(
    const cudaq::CompileTarget::PipelineConfig &pc) const noexcept {
  return cudaq::detail::hashVal(
      pc.overridePassPipeline, pc.highLevelPipeline, pc.midLevelPipeline,
      pc.lowLevelPipeline, pc.codegenTranslation, pc.postCodeGenPasses,
      pc.disableQubitMapping, pc.replaceStateWithKernel);
}
