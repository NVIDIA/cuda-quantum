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

/// Report an unsupported IR profile and terminate. This code is used in
/// contexts built without exception support, so errors abort instead of
/// throwing.
[[noreturn]] static void reportUnsupportedProfile() {
  std::cerr << "Unsupported IR profile: only ExplicitPipelineProfile is "
               "currently supported\n";
  std::abort();
}

const cudaq::config::ExplicitPipelineProfile &
cudaq::CompileTarget::getExplicitPipelineProfile() const {
  if (profiles.size() != 1)
    reportUnsupportedProfile();
  const auto *explicitProfile = profile<config::ExplicitPipelineProfile>();
  if (!explicitProfile)
    reportUnsupportedProfile();
  return *explicitProfile;
}

cudaq::config::ExplicitPipelineProfile &
cudaq::CompileTarget::getExplicitPipelineProfile() {
  if (profiles.size() != 1)
    reportUnsupportedProfile();
  auto *explicitProfile = profile<config::ExplicitPipelineProfile>();
  if (!explicitProfile)
    reportUnsupportedProfile();
  return *explicitProfile;
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
  auto resolvePipeline = [&](config::ExplicitPipelineProfile &profile) {
    if (!profile.overridePassPipeline.empty()) {
      profile.overridePassPipeline = prepPipeline(profile.overridePassPipeline);
    } else {
      profile.highLevelPipeline = prepPipeline(profile.highLevelPipeline);
      profile.midLevelPipeline = prepPipeline(profile.midLevelPipeline);
      profile.lowLevelPipeline = prepPipeline(profile.lowLevelPipeline);
    }
  };

  // Handle disable_qubit_mapping runtime option.
  auto disableQM = runtimeConfig.find("disable_qubit_mapping");
  auto disableQubitMapping =
      disableQM != runtimeConfig.end() && disableQM->second == "true";
  for (auto &stage : target.profiles) {
    auto *profile = std::get_if<config::ExplicitPipelineProfile>(&stage);
    if (!profile)
      continue;
    resolvePipeline(*profile);
    profile->disableQubitMapping |= disableQubitMapping;
  }

  target.codegenTranslation = targetConfig.getCodeGenSpec(runtimeConfig);

  target.supportResourceCounts = backendConfig.SupportResourceCounts;

  return target;
}

std::size_t std::hash<cudaq::CompileTarget>::operator()(
    const cudaq::CompileTarget &t) const noexcept {
  return cudaq::detail::hashVal(
      t.overrideAOTCompilation, t.profiles, t.codegenTranslation,
      t.supportConditionalsOnMeasureResults, t.supportDeviceCalls,
      t.supportResourceCounts, t.supportExplicitMeasurements,
      t.supportObservableMeasurements, t.supportSampleWithoutMeasurements,
      t.fullySpecialize, t.argumentSynthChangeSemantics);
}

std::size_t std::hash<cudaq::config::ExplicitPipelineProfile>::operator()(
    const cudaq::config::ExplicitPipelineProfile &profile) const noexcept {
  return cudaq::detail::hashVal(
      profile.overridePassPipeline, profile.highLevelPipeline,
      profile.midLevelPipeline, profile.lowLevelPipeline,
      profile.postCodeGenPasses, profile.disableQubitMapping,
      profile.replaceStateWithKernel);
}

std::size_t std::hash<std::vector<cudaq::config::Profile>>::operator()(
    const std::vector<cudaq::config::Profile> &profiles) const noexcept {
  std::size_t seed = 0;
  for (const auto &profile : profiles)
    cudaq::detail::hashCombine(seed, profile);
  return seed;
}
