/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// YAML v2 schema for target configuration.
//
// v2 moves the compilation options out of `config` into a dedicated
// top-level `compile-target` block and introduces support for declarative
// IR profiles.

#include "Yaml/v1/TargetConfigV1.h"
#include <optional>
#include <string>
#include <vector>

namespace cudaq::config::v2 {

// Schema fragments unchanged from v1.
using ConditionalBuildConfigV2 = v1::ConditionalBuildConfigV1;
using TargetArchitectureSettingsV2 = v1::TargetArchitectureSettingsV1;
using MachineArchitectureConfigV2 = v1::MachineArchitectureConfigV1;
using TargetArgumentV2 = v1::TargetArgumentV1;
using SimulationBackendInputV2 = v1::SimulationBackendInput;

/// Backend (non-compilation) target configuration. Omits v1's compilation
/// fields (`jit-*-pipeline`, `target-pass-pipeline`, `codegen`-emission,
/// `post-codegen`-passes).
struct BackendEndConfigEntryV2 {
  rfl::DefaultVal<bool> genTargetBackend;
  rfl::DefaultVal<bool> libraryMode;
  // Default preserves the existing runtime default (true).
  rfl::DefaultVal<bool> supportResourceCounts = true;
  rfl::DefaultVal<std::string> platformLibrary;
  rfl::DefaultVal<std::string> libraryModeExecutionManager;
  rfl::DefaultVal<std::string> platformQpu;
  rfl::DefaultVal<std::vector<std::string>> preprocessorDefines;
  rfl::DefaultVal<std::vector<std::string>> compilerFlags;
  rfl::DefaultVal<std::vector<std::string>> linkLibs;
  rfl::DefaultVal<std::vector<std::string>> pluginLibraries;
  rfl::DefaultVal<std::vector<std::string>> linkerFlags;
  rfl::DefaultVal<SimulationBackendInputV2> nvqirSimulationBackend;
  rfl::DefaultVal<std::vector<ConditionalBuildConfigV2>> rules;
};

struct BackendFeatureMapV2 {
  std::string name;
  std::vector<TargetFeatureFlag> optionFlags;
  /// `default` must be renamed to avoid C++ keyword conflicts.
  rfl::Rename<"default", rfl::DefaultVal<bool>> isDefault;
  BackendEndConfigEntryV2 config;
};

/// `override-pass-pipeline` prescribes the whole compilation pipeline; it
/// cannot be combined with the staged pipelines.
struct ExplicitPipelineOverrideV2 {
  std::string overridePassPipeline;
};

/// Staged pipelines inserted at the appropriate compilation stages.
struct ExplicitPipelineStagesV2 {
  rfl::DefaultVal<std::string> highLevelPipeline;
  rfl::DefaultVal<std::string> midLevelPipeline;
  rfl::DefaultVal<std::string> lowLevelPipeline;
};

/// Transitional profile prescribing an explicit MLIR pass pipeline instead
/// of describing the target IR in a declarative way. The `pipelines` variant
/// makes `override-pass-pipeline` mutually exclusive with the staged pipelines:
/// the override struct requires its key, the staged struct accepts only the
/// three staged keys, so a mapping mixing both matches neither.
struct ExplicitPipelineProfileV2 {
  using Tag = rfl::Literal<"explicit-pipeline">;
  rfl::Variant<ExplicitPipelineOverrideV2, ExplicitPipelineStagesV2> pipelines;
};

/// One compilation stage, defined by its target IR profile. Add new IR profiles
/// to this union.
using ProfileV2 = rfl::TaggedUnion<"ir-profile", ExplicitPipelineProfileV2>;

struct CompileTargetV2 {
  std::vector<ProfileV2> stages;
  rfl::DefaultVal<std::string> codegen;
};

struct TargetConfigV2 {
  using Tag = rfl::Literal<"2">;
  std::string name;
  std::string description;
  rfl::DefaultVal<std::string> cudaqVersion;
  rfl::DefaultVal<std::string> warning;
  rfl::DefaultVal<std::vector<TargetArgumentV2>> targetArguments;
  rfl::DefaultVal<bool> gpuRequirements;
  // Structural: a target without a direct backend omits `config`.
  std::optional<BackendEndConfigEntryV2> config;
  std::optional<CompileTargetV2> compileTarget;
  rfl::DefaultVal<std::vector<BackendFeatureMapV2>> configurationMatrix;
};

/// Converts a parsed v2 configuration to the canonical model, applying
/// defaults, normalization, and cross-field validation.
/// Throws std::runtime_error on validation failures.
TargetConfig toCanonical(const TargetConfigV2 &config);

} // namespace cudaq::config::v2
