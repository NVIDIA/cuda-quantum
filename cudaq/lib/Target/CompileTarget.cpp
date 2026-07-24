/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/CompileTarget.h"
#include "cudaq/runtime/logger/logger.h"
#include <cctype>
#include <functional>

cudaq::CompileTarget::RuntimeEndpoint::RuntimeEndpoint(
    const std::string &name, const std::map<std::string, std::string> &options)
    : name(name), options(options) {}

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

cudaq::CompileTarget::CompileTarget(
    config::TargetConfig targetConfig,
    std::map<std::string, std::string> runtimeConfig, bool emulate_,
    std::map<std::string, std::string> pipelineSubstitutions)
    : emulate(emulate_) {
  if (!targetConfig.BackendConfig.has_value()) {
    pipelineConfig.skipTargetLoweringPipeline = true;
    return;
  }

  const auto &backendConfig = *targetConfig.BackendConfig;
  if (!backendConfig.hasPassPipeline()) {
    pipelineConfig.skipTargetLoweringPipeline = true;
  }

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
    pipelineConfig.overridePassPipeline = prepPipeline(
        backendConfig.TargetPassPipeline, "Pass pipeline (overridden)");
  } else {
    pipelineConfig.highLevelPipeline =
        prepPipeline(backendConfig.JITHighLevelPipeline, "JIT high level");
    pipelineConfig.midLevelPipeline =
        prepPipeline(backendConfig.JITMidLevelPipeline, "JIT mid level");
    pipelineConfig.lowLevelPipeline =
        prepPipeline(backendConfig.JITLowLevelPipeline, "JIT low level");
  }
  auto codegenTranslation = targetConfig.getCodeGenSpec(runtimeConfig);
  if (!codegenTranslation.empty()) {
    pipelineConfig.codegenTranslation = codegenTranslation;
    CUDAQ_INFO("{:<27} {}\n", "Codegen:", codegenTranslation);
  }
  if (!backendConfig.PostCodeGenPasses.empty()) {
    pipelineConfig.postCodeGenPasses = backendConfig.PostCodeGenPasses;
    CUDAQ_INFO("{:<27} {}\n",
               "Post-codegen:", pipelineConfig.postCodeGenPasses);
  }

  // Handle disable_qubit_mapping runtime option.
  auto disableQM = runtimeConfig.find("disable_qubit_mapping");
  if (disableQM != runtimeConfig.end() && disableQM->second == "true") {
    pipelineConfig.disableQubitMapping = true;
    CUDAQ_INFO("{:<27} {}\n", "disable_qubit_mapping:", "true");
  }
}

template <typename T>
inline void hash_combine(std::size_t &seed, const T &val) {
  std::hash<T> hasher;
  seed ^= hasher(val) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename... Args>
inline std::size_t hash_val(const Args &...args) {
  std::size_t seed = 0;
  (hash_combine(seed, args), ...);
  return seed;
}

std::size_t std::hash<cudaq::CompileTarget>::operator()(
    const cudaq::CompileTarget &t) const noexcept {
  auto pauliStr =
      t.pauliTermSplitObservable ? t.pauliTermSplitObservable->to_string() : "";
  return hash_val(t.pipelineConfig, t.overrideAOTCompilation, t.emulate,
                  t.warnNamedMeasurements,
                  t.supportConditionalsOnMeasureResults, t.supportDeviceCalls,
                  t.storeReorderIdx, t.emitResourceCounts, t.emitJit,
                  t.emitTargetCode, t.fullySpecialize, t.isLocalSimulator,
                  t.argumentSynthChangeSemantics, t.runtimeEndpoint, pauliStr);
}

std::size_t std::hash<cudaq::CompileTarget::PipelineConfig>::operator()(
    const cudaq::CompileTarget::PipelineConfig &pc) const noexcept {
  return hash_val(pc.overridePassPipeline, pc.highLevelPipeline,
                  pc.midLevelPipeline, pc.lowLevelPipeline,
                  pc.codegenTranslation, pc.postCodeGenPasses,
                  pc.skipTargetLoweringPipeline, pc.disableQubitMapping,
                  pc.replaceStateWithKernel, pc.addMeasurements);
}

std::size_t std::hash<cudaq::CompileTarget::RuntimeEndpoint>::operator()(
    const cudaq::CompileTarget::RuntimeEndpoint &re) const noexcept {
  std::size_t seed = 0;
  for (const auto &[key, value] : re.options) {
    hash_combine(seed, key);
    hash_combine(seed, value);
  }
  hash_combine(seed, re.name);
  return seed;
}
