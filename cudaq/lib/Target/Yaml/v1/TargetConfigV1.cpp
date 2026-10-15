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
  // v1 compilation fields map onto a single stage with a custom-pipeline
  // lowering.
  cudaq::config::CustomPipelineLowering lowering;
  lowering.overridePassPipeline = cfg.targetPassPipeline.value();
  lowering.highLevelPipeline = cfg.jitHighLevelPipeline.value();
  lowering.midLevelPipeline = cfg.jitMidLevelPipeline.value();
  lowering.lowLevelPipeline = cfg.jitLowLevelPipeline.value();
  out.CompileTarget.stages.push_back(cudaq::config::CompilationStage{
      .irOutputProfile = cudaq::config::UncheckedProfile{},
      .lowering = std::move(lowering)});
  out.CompileTarget.postCodeGenPasses = cfg.postCodegenPasses.value();
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
