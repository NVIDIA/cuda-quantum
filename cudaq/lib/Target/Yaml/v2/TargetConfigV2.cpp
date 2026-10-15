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
toCanonicalProfile(const cudaq::config::v2::ProfileV2 &profile) {
  return profile.visit([](const auto &p) -> cudaq::config::Profile {
    using T = std::decay_t<decltype(p)>;
    if constexpr (std::is_same_v<
                      T, cudaq::config::v2::ExplicitPipelineProfileV2>) {
      cudaq::config::ExplicitPipelineProfile out;
      p.pipelines.visit([&](const auto &pipelines) {
        using C = std::decay_t<decltype(pipelines)>;
        if constexpr (std::is_same_v<
                          C, cudaq::config::v2::ExplicitPipelineOverrideV2>) {
          out.overridePassPipeline = pipelines.overridePassPipeline;
        } else {
          out.highLevelPipeline = pipelines.highLevelPipeline.value();
          out.midLevelPipeline = pipelines.midLevelPipeline.value();
          out.lowLevelPipeline = pipelines.lowLevelPipeline.value();
        }
      });
      return out;
    }
  });
}

static cudaq::config::BackendEndConfigEntry
toCanonicalEntry(const cudaq::config::v2::BackendEndConfigEntryV2 &cfg,
                 const cudaq::config::v2::CompileTargetV2 *compileTarget) {
  cudaq::config::BackendEndConfigEntry out;
  cudaq::config::detail::toCanonicalEntryBase(cfg, out);
  if (compileTarget) {
    for (const auto &stage : compileTarget->stages)
      out.CompileTarget.profiles.push_back(toCanonicalProfile(stage));
    out.CompileTarget.codegenTranslation = compileTarget->codegen.value();
  }
  return out;
}

cudaq::config::TargetConfig
cudaq::config::v2::toCanonical(const TargetConfigV2 &cfg) {
  if (cfg.compileTarget && cfg.compileTarget->stages.empty())
    throw std::runtime_error(
        "'compile-target.stages' must contain at least one stage.");
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
