/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Yaml/v1/TargetConfigV1.h"
#include "Yaml/CanonicalConversion.h"
#include <utility>

static cudaq::config::BackendEndConfigEntry
toCanonicalEntry(const cudaq::config::v1::BackendEndConfigEntryV1 &cfg) {
  cudaq::config::BackendEndConfigEntry out;
  cudaq::config::detail::toCanonicalEntryBase(cfg, out);
  // v1 compilation fields map onto a single explicit-pipeline profile.
  cudaq::config::ExplicitPipelineProfile profile;
  profile.overridePassPipeline = cfg.targetPassPipeline.value();
  profile.highLevelPipeline = cfg.jitHighLevelPipeline.value();
  profile.midLevelPipeline = cfg.jitMidLevelPipeline.value();
  profile.lowLevelPipeline = cfg.jitLowLevelPipeline.value();
  profile.postCodeGenPasses = cfg.postCodegenPasses.value();
  out.CompileTarget.profiles.push_back(std::move(profile));
  out.CompileTarget.codegenTranslation = cfg.codegenEmission.value();
  return out;
}

cudaq::config::TargetConfig
cudaq::config::v1::toCanonical(const TargetConfigV1 &cfg) {
  TargetConfig out;
  out.Name = cfg.name;
  out.Description = cfg.description;
  out.CudaqVersion = cfg.cudaqVersion.value();
  out.WarningMsg = cfg.warning.value();
  out.GpuRequired = cfg.gpuRequirements.value();
  detail::appendCanonicalArguments(cfg.targetArguments.value(), out);
  if (cfg.config)
    out.BackendConfig = toCanonicalEntry(*cfg.config);
  detail::appendCanonicalConfigMap(cfg.configurationMatrix.value(), out,
                                   [](const BackendEndConfigEntryV1 &entry) {
                                     return toCanonicalEntry(entry);
                                   });
  detail::flattenPluginLibraries(out);
  return out;
}
