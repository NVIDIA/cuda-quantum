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

/// The longest loop, in iterations, that the inferred simulator stage
/// unrolls. It equals the default of `cc-loop-unroll`, so the stage unrolls
/// the same loops as codegen does when it normalizes a kernel.
// TODO: Raise this toward the profile's default only after measuring loops of
// 50 to 1024 iterations.
static constexpr unsigned simulatorUnrollLimit = 50;

/// Give \p target the stage of a simulator that declares no pipelines: a
/// `quake-unmapped` stage that optimizes for gate count and accepts any gate.
/// Prep keeps noise and QEC operations, as simulators always have.
// TODO: Once version 2 lets configuration-matrix entries declare stages,
// write the in-tree simulators' stage into their YAML.
static void addSimulatorStage(cudaq::CompileTarget &target) {
  target.stages.push_back(cudaq::config::CompilationStage{
      .irOutputProfile =
          cudaq::config::QuakeUnmappedProfile{
              .costModel = cudaq::config::CostModel::gate_count,
              .maxUnrollIterations = simulatorUnrollLimit},
      .lowering = cudaq::config::InferredLowering{}});
  target.noiseSupport = cudaq::NoiseSupport::supported;
  target.eraseQEC = false;
}

static cudaq::config::BackendEndConfigEntry
toCanonicalEntry(const cudaq::config::v1::BackendEndConfigEntryV1 &cfg) {
  cudaq::config::BackendEndConfigEntry out;
  cudaq::config::detail::toCanonicalEntryBase(cfg, out);
  cudaq::config::CustomPipelineLowering lowering;
  lowering.overridePassPipeline = cfg.targetPassPipeline.value();
  lowering.highLevelPipeline = cfg.jitHighLevelPipeline.value();
  lowering.midLevelPipeline = cfg.jitMidLevelPipeline.value();
  lowering.lowLevelPipeline = cfg.jitLowLevelPipeline.value();
  // A simulator that declares no pipelines gets the simulator stage. Any
  // other entry's version 1 compilation fields map onto a single stage with a
  // custom-pipeline lowering.
  if (lowering.empty() && !out.SimulationBackend.values.empty())
    addSimulatorStage(out.CompileTarget);
  else
    out.CompileTarget.stages.push_back(cudaq::config::CompilationStage{
        .irOutputProfile = cudaq::config::UncheckedProfile{},
        .lowering = std::move(lowering)});
  out.CompileTarget.codegen = {.emissionFormat = cfg.codegenEmission.value(),
                               .postCodegenPasses =
                                   cfg.postCodegenPasses.value()};
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
