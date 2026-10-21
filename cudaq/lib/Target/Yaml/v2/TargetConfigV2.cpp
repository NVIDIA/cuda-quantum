/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// Convert parsed YAML to canonical configuration types.

#include "Yaml/v2/TargetConfigV2.h"
#include "Yaml/CanonicalConversion.h"

static cudaq::config::Profile
toCanonicalProfile(const cudaq::config::v2::UncheckedProfileV2 &profile) {
  return cudaq::config::UncheckedProfile{};
}
static cudaq::config::Profile toCanonicalProfile(
    const cudaq::config::v2::QuakeReferenceSemanticsProfileV2 &profile) {
  return cudaq::config::QuakeReferenceSemanticsProfile{
      .basisGates = profile.options.value().basisGates.value(),
      .measurementType = profile.options.value().measurementType.value(),
      .resetAfterMeasure = profile.options.value().resetAfterMeasure.value()};
}

static cudaq::config::Profile
toCanonicalProfile(const cudaq::config::v2::IrOutputProfileV2 &profile) {
  return profile.visit([](const auto &p) { return toCanonicalProfile(p); });
}

static cudaq::config::Lowering
toCanonicalLowering(const std::string &lowering) {
  return cudaq::config::CustomPipelineLowering{.overridePassPipeline =
                                                   lowering};
}

static cudaq::config::Lowering
toCanonicalLowering(const cudaq::config::v2::StagedPipelinesV2 &pipelines) {
  return cudaq::config::CustomPipelineLowering{
      .highLevelPipeline = pipelines.highLevel.value(),
      .midLevelPipeline = pipelines.midLevel.value(),
      .lowLevelPipeline = pipelines.lowLevel.value()};
}

static cudaq::config::Lowering toCanonicalLowering(
    const cudaq::config::v2::CustomPipelineLoweringV2 &lowering) {
  return lowering.customPipeline.visit(
      [](const auto &l) { return toCanonicalLowering(l); });
}

static cudaq::config::CompilationStage
toCanonicalStage(const cudaq::config::v2::CompilationStageV2 &stage) {
  cudaq::config::Profile profile = toCanonicalProfile(stage.irOutputProfile);
  cudaq::config::Lowering lowering = cudaq::config::InferredLowering{};
  if (stage.lowering)
    lowering = toCanonicalLowering(*stage.lowering);
  return cudaq::config::CompilationStage{.irOutputProfile = profile,
                                         .lowering = std::move(lowering)};
}

static cudaq::config::BackendEndConfigEntry
toCanonicalEntry(const cudaq::config::v2::BackendEndConfigEntryV2 &cfg,
                 const cudaq::config::v2::CompileTargetV2 *compileTarget) {
  cudaq::config::BackendEndConfigEntry out;
  cudaq::config::detail::toCanonicalEntryBase(cfg, out);
  if (compileTarget) {
    for (const auto &stage : compileTarget->stages)
      out.CompileTarget.stages.push_back(toCanonicalStage(stage));
    compileTarget->codegen.value().visit([&](const auto &codegen) {
      using T = std::decay_t<decltype(codegen)>;
      if constexpr (std::is_same_v<T, std::string>)
        out.CompileTarget.codegen.emissionFormat = codegen;
      else
        out.CompileTarget.codegen = {
            .emissionFormat = codegen.emissionFormat.value(),
            .postCodegenPasses = codegen.postCodegenPasses.value()};
    });
  }
  return out;
}

cudaq::config::TargetConfig
cudaq::config::v2::toCanonical(const TargetConfigV2 &cfg) {
  if (cfg.compileTarget && !cfg.configurationMatrix.value().empty())
    throw std::runtime_error(
        "'compile-target' cannot be combined with 'configuration-matrix'.");
  if (cfg.compileTarget && !cfg.config)
    throw std::runtime_error(
        "'compile-target' requires a 'config' section to attach to.");

  TargetConfig out;
  out.Name = cfg.name;
  out.Description = cfg.description;
  out.CudaqVersion = cfg.cudaqVersion.value();
  out.WarningMsg = cfg.warning.value();
  out.GpuRequired = cfg.gpuRequirements.value();
  detail::appendCanonicalArguments(cfg.targetArguments.value(), out);
  if (cfg.config)
    out.BackendConfig = toCanonicalEntry(
        *cfg.config, cfg.compileTarget ? &*cfg.compileTarget : nullptr);
  detail::appendCanonicalConfigMap(cfg.configurationMatrix.value(), out,
                                   [](const BackendEndConfigEntryV2 &entry) {
                                     return toCanonicalEntry(entry, nullptr);
                                   });
  detail::flattenPluginLibraries(out);
  return out;
}
