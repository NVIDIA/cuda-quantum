/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/CompileTarget.h"
#include "cudaq/Support/Hash.h"
#include "cudaq/Target/TargetConfig.h"
#include <cctype>
#include <cstdlib>
#include <iostream>

/// Report an unsupported compilation stage and terminate.
[[noreturn]] static void reportUnsupportedStage() {
  std::cerr << "Unsupported compilation stage: only a single stage with a "
               "custom-pipeline lowering is currently supported\n";
  std::abort();
}

const cudaq::config::CustomPipelineLowering &
cudaq::CompileTarget::getCustomPipelineLowering() const {
  if (stages.size() != 1)
    reportUnsupportedStage();
  const auto *lowering =
      std::get_if<config::CustomPipelineLowering>(&stages[0].lowering);
  if (!lowering)
    reportUnsupportedStage();
  return *lowering;
}

cudaq::config::CustomPipelineLowering &
cudaq::CompileTarget::getCustomPipelineLowering() {
  if (stages.size() != 1)
    reportUnsupportedStage();
  auto *lowering =
      std::get_if<config::CustomPipelineLowering>(&stages[0].lowering);
  if (!lowering)
    reportUnsupportedStage();
  return *lowering;
}

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
  const config::BackendEndConfigEntry defaultConfig;

  const auto &backendConfig =
      targetConfig.BackendConfig.value_or(defaultConfig);
  cudaq::CompileTarget target = backendConfig.CompileTarget;
  auto prepPipeline = [&](const std::string &stage) {
    std::string pipeline = stage;
    if (!pipeline.empty()) {
      substitutePipelinePlaceholders(pipeline, runtimeConfig);
      applyPipelineSubstitutions(pipeline, pipelineSubstitutions);
    }
    return pipeline;
  };
  auto resolvePipeline = [&](config::CustomPipelineLowering &lowering) {
    if (!lowering.overridePassPipeline.empty()) {
      lowering.overridePassPipeline =
          prepPipeline(lowering.overridePassPipeline);
    } else {
      lowering.highLevelPipeline = prepPipeline(lowering.highLevelPipeline);
      lowering.midLevelPipeline = prepPipeline(lowering.midLevelPipeline);
      lowering.lowLevelPipeline = prepPipeline(lowering.lowLevelPipeline);
    }
  };

  // Handle disable_qubit_mapping runtime option.
  auto disableQM = runtimeConfig.find("disable_qubit_mapping");
  target.disableQubitMapping =
      disableQM != runtimeConfig.end() && disableQM->second == "true";
  for (auto &stage : target.stages) {
    auto *lowering =
        std::get_if<config::CustomPipelineLowering>(&stage.lowering);
    if (!lowering)
      continue;
    resolvePipeline(*lowering);
  }

  target.codegenTranslation = targetConfig.getCodeGenSpec(runtimeConfig);

  target.supportResourceCounts = backendConfig.SupportResourceCounts;

  return target;
}

std::size_t std::hash<cudaq::CompileTarget>::operator()(
    const cudaq::CompileTarget &t) const noexcept {
  return cudaq::detail::hashVal(
      t.overrideAOTCompilation, t.stages, t.codegenTranslation,
      t.postCodeGenPasses, t.supportConditionalsOnMeasureResults,
      t.supportDeviceCalls, t.supportResourceCounts,
      t.supportExplicitMeasurements, t.supportObservableMeasurements,
      t.supportSampleWithoutMeasurements, t.fullySpecialize,
      t.argumentSynthChangeSemantics, t.disableQubitMapping,
      t.replaceStateWithKernel);
}

std::size_t std::hash<cudaq::config::UncheckedProfile>::operator()(
    const cudaq::config::UncheckedProfile &) const noexcept {
  return 0;
}

std::size_t std::hash<cudaq::config::CustomPipelineLowering>::operator()(
    const cudaq::config::CustomPipelineLowering &lowering) const noexcept {
  return cudaq::detail::hashVal(
      lowering.overridePassPipeline, lowering.highLevelPipeline,
      lowering.midLevelPipeline, lowering.lowLevelPipeline);
}

std::size_t std::hash<cudaq::config::CompilationStage>::operator()(
    const cudaq::config::CompilationStage &stage) const noexcept {
  std::size_t seed = 0;
  std::visit([&](const auto &p) { cudaq::detail::hashCombine(seed, p); },
             stage.irOutputProfile);
  std::visit([&](const auto &l) { cudaq::detail::hashCombine(seed, l); },
             stage.lowering);
  return seed;
}

std::size_t std::hash<std::vector<cudaq::config::CompilationStage>>::operator()(
    const std::vector<cudaq::config::CompilationStage> &stages) const noexcept {
  std::size_t seed = 0;
  for (const auto &stage : stages)
    cudaq::detail::hashCombine(seed, stage);
  return seed;
}
