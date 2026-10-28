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
#include <algorithm>
#include <cctype>
#include <cmath>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>

const std::string *cudaq::CompileTarget::getOverridePassPipeline() const {
  if (stages.size() != 1)
    return nullptr;
  const auto *lowering =
      std::get_if<config::CustomPipelineLowering>(&stages[0].lowering);
  if (!lowering || lowering->overridePassPipeline.empty())
    return nullptr;
  return &lowering->overridePassPipeline;
}

/// Replace `%KEY%` and `%KEY:default%` placeholders from runtime options.
/// Returns whether the pipeline contains a `%SEED%` placeholder.
static bool substitutePipelinePlaceholders(
    std::string &pipeline,
    const std::map<std::string, std::string> &runtimeConfig) {
  bool usesSeed = false;
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
    usesSeed |= lower == "seed";
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
  return usesSeed;
}

/// Resolve the placeholder in \p profile's `approximation-tolerance` and check
/// the value, so that a bad or missing tolerance fails when the target is
/// created rather than when a kernel compiles.
static void
resolveProfileOptions(cudaq::config::QuakeUnmappedProfile &profile,
                      const std::map<std::string, std::string> &runtimeConfig) {
  auto &tolerance = profile.approximationTolerance;
  substitutePipelinePlaceholders(tolerance, runtimeConfig);
  if (tolerance.empty()) {
    if (profile.requiresSynthesis())
      throw std::runtime_error(
          "quake-unmapped requires 'approximation-tolerance' when its cost "
          "model "
          "is 't-count' and 'basis-gates' has no rotation gate");
    return;
  }
  // `std::from_chars` for `double` is unavailable on older macOS deployment
  // targets, so parse with the classic locale and require the whole string.
  std::istringstream stream(tolerance);
  stream.imbue(std::locale::classic());
  double value = 0;
  stream >> value;
  if (stream.fail() || stream.peek() != std::char_traits<char>::eof() ||
      !std::isfinite(value) || value < 0)
    throw std::runtime_error(
        "quake-unmapped 'approximation-tolerance' must be a "
        "non-negative number, got '" +
        tolerance + "'");
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
      bool usesSeed = substitutePipelinePlaceholders(pipeline, runtimeConfig);
      target.resetOnRandomSeedSet |= usesSeed;
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
    if (auto *lowering =
            std::get_if<config::CustomPipelineLowering>(&stage.lowering))
      resolvePipeline(*lowering);
    if (auto *profile =
            std::get_if<config::QuakeUnmappedProfile>(&stage.irOutputProfile))
      resolveProfileOptions(*profile, runtimeConfig);
  }

  target.codegen.emissionFormat = targetConfig.getCodeGenSpec(runtimeConfig);

  target.supportResourceCounts = backendConfig.SupportResourceCounts;

  return target;
}

/// Whether lowering should be skipped altogether in this compilation stage
/// (true when an empty custom pipeline is specified).
static bool skipLoweringPipeline(const cudaq::config::CompilationStage &stage) {
  const auto *custom =
      std::get_if<cudaq::config::CustomPipelineLowering>(&stage.lowering);
  return custom && custom->empty();
}

bool cudaq::CompileTarget::hasPassPipeline() const {
  return !std::all_of(stages.begin(), stages.end(), skipLoweringPipeline);
}

std::size_t std::hash<cudaq::CompileTarget>::operator()(
    const cudaq::CompileTarget &t) const noexcept {
  return cudaq::detail::hashVal(
      t.overrideAOTCompilation, t.stages, t.codegen,
      t.supportConditionalsOnMeasureResults, t.supportDeviceCalls,
      t.supportResourceCounts, t.supportExplicitMeasurements,
      t.supportObservableMeasurements, t.supportSampleWithoutMeasurements,
      t.fullySpecialize, t.argumentSynthChangeSemantics, t.disableQubitMapping,
      t.replaceStateWithKernel);
}

std::size_t std::hash<cudaq::config::UncheckedProfile>::operator()(
    const cudaq::config::UncheckedProfile &) const noexcept {
  return 0;
}

std::size_t
std::hash<cudaq::config::QuakeReferenceSemanticsProfile>::operator()(
    const cudaq::config::QuakeReferenceSemanticsProfile &profile)
    const noexcept {
  std::size_t seed = 0;
  for (const auto &gate : profile.basisGates)
    cudaq::detail::hashCombine(seed, gate);
  cudaq::detail::hashCombine(seed, profile.measurementType);
  cudaq::detail::hashCombine(seed, profile.resetAfterMeasure);
  return seed;
}

std::size_t std::hash<cudaq::config::QuakeUnmappedProfile>::operator()(
    const cudaq::config::QuakeUnmappedProfile &profile) const noexcept {
  std::size_t seed = 0;
  for (const auto &gate : profile.basisGates)
    cudaq::detail::hashCombine(seed, gate);
  cudaq::detail::hashCombine(seed, profile.approximationTolerance);
  cudaq::detail::hashCombine(seed, profile.costModel);
  return seed;
}

bool cudaq::config::QuakeUnmappedProfile::requiresSynthesis() const {
  // A basis with any of these continuous rotations, uncontrolled or with any
  // number of controls, can express every rotation without approximating it.
  static constexpr std::string_view rotations[] = {"r1", "rx",        "ry",
                                                   "rz", "phased_rx", "u3"};
  const auto isRotation = [](std::string_view gate) {
    const auto controls = gate.find('(');
    if (controls != std::string_view::npos && gate.substr(controls) != "(n)")
      return false;
    gate = gate.substr(0, controls);
    return std::find(std::begin(rotations), std::end(rotations), gate) !=
           std::end(rotations);
  };
  return costModel == CostModel::t_count && !basisGates.empty() &&
         std::none_of(basisGates.begin(), basisGates.end(), isRotation);
}

std::size_t std::hash<cudaq::config::CustomPipelineLowering>::operator()(
    const cudaq::config::CustomPipelineLowering &lowering) const noexcept {
  return cudaq::detail::hashVal(
      lowering.overridePassPipeline, lowering.highLevelPipeline,
      lowering.midLevelPipeline, lowering.lowLevelPipeline);
}

std::size_t std::hash<cudaq::config::InferredLowering>::operator()(
    const cudaq::config::InferredLowering &) const noexcept {
  // Fixed tag: the lowering kind contributes to stage hashes, keeping an
  // inferred lowering distinct from any custom pipeline (even an empty one).
  return 0x5e1f9a3b;
}

std::size_t std::hash<cudaq::config::Codegen>::operator()(
    const cudaq::config::Codegen &codegen) const noexcept {
  return cudaq::detail::hashVal(codegen.emissionFormat,
                                codegen.postCodegenPasses);
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
