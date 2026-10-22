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
#include "cudaq/Target/Profile/QuakeReferenceSemanticsProfile.h"
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
/// `post-codegen-passes`).
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

/// (Dummy) IR profile that accepts any IR.
struct UncheckedProfileV2 {
  using Tag = rfl::Literal<UncheckedProfile::name>;
};

/// IR profile for reference-semantics Quake IR.
struct QuakeReferenceSemanticsProfileV2 {
  using Tag = rfl::Literal<"quake-reference-semantics">;

  /// Valid options.
  struct QuakeReferenceSemanticsProfileOptionsV2 {
    rfl::DefaultVal<std::vector<std::string>> basisGates;
    rfl::DefaultVal<MeasurementForm> measurementType = MeasurementForm::any;
    rfl::DefaultVal<bool> resetAfterMeasure = false;
  };

  rfl::DefaultVal<QuakeReferenceSemanticsProfileOptionsV2> options;
};

/// Union of all supported IR profiles, selected by its `name`.
using IrOutputProfileV2 = rfl::TaggedUnion<"name", UncheckedProfileV2,
                                           QuakeReferenceSemanticsProfileV2>;

/// Staged pipelines inserted at the appropriate compilation stages.
struct StagedPipelinesV2 {
  rfl::DefaultVal<std::string> highLevel;
  rfl::DefaultVal<std::string> midLevel;
  rfl::DefaultVal<std::string> lowLevel;
};

/// Lowering described as a custom MLIR pass pipeline. Either a string or a
/// combination of high, mid, and low level pipelines.
struct CustomPipelineLoweringV2 {
  rfl::Variant<std::string, StagedPipelinesV2> customPipeline;
};

/// One compilation stage: the IR output profile to lower to and the lowering
/// to get there.
struct CompilationStageV2 {
  IrOutputProfileV2 irOutputProfile;
  CustomPipelineLoweringV2 lowering;
};

/// Code generation: emission format and optional `post-codegen-passes`
/// pipeline.
struct CodegenV2 {
  rfl::DefaultVal<std::string> emissionFormat;
  rfl::DefaultVal<std::string> postCodegenPasses;
};

struct CompileTargetV2 {
  std::vector<CompilationStageV2> stages;
  /// Either a plain emission-format string or the full `codegen` mapping.
  rfl::DefaultVal<rfl::Variant<std::string, CodegenV2>> codegen;
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
