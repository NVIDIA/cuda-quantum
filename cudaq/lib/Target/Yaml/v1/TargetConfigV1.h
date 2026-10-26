/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// YAML v1 schema for target configuration

#include "Yaml/TargetConfigYamlAdapters.h"
#include <optional>
#include <string>
#include <vector>

namespace cudaq::config::v1 {

struct ConditionalBuildConfigV1 {
  /// `if` must be renamed to avoid C++ keyword conflicts.
  rfl::Rename<"if", std::string> condition;
  rfl::DefaultVal<std::string> compilerFlag;
  rfl::DefaultVal<std::string> linkFlag;
};

struct TargetArchitectureSettingsV1 {
  rfl::DefaultVal<std::string> codegenEmission;
};

struct MachineArchitectureConfigV1 {
  rfl::DefaultVal<std::string> archName;
  rfl::DefaultVal<std::vector<std::string>> machineNames;
  rfl::DefaultVal<std::string> pattern;
  TargetArchitectureSettingsV1 config;
};

struct TargetArgumentV1 {
  std::string key;
  rfl::DefaultVal<bool> required;
  rfl::DefaultVal<std::string> platformArg;
  rfl::DefaultVal<std::string> helpString;
  // Omitted type means `ArgumentType::string` (enumerator value 0).
  rfl::DefaultVal<ArgumentType> type;
  rfl::DefaultVal<std::vector<MachineArchitectureConfigV1>> machineConfig;
};

/// `nvqir-simulation-backend` accepts a comma-separated scalar or a list.
using SimulationBackendInput =
    rfl::Variant<std::string, std::vector<std::string>>;

struct BackendEndConfigEntryV1 {
  rfl::DefaultVal<bool> genTargetBackend;
  rfl::DefaultVal<bool> libraryMode;
  // Default preserves the existing runtime default (true).
  rfl::DefaultVal<bool> supportResourceCounts = true;
  rfl::DefaultVal<std::string> jitHighLevelPipeline;
  rfl::DefaultVal<std::string> jitMidLevelPipeline;
  rfl::DefaultVal<std::string> jitLowLevelPipeline;
  rfl::DefaultVal<std::string> targetPassPipeline;
  rfl::DefaultVal<std::string> codegenEmission;
  rfl::DefaultVal<std::string> postCodegenPasses;
  rfl::DefaultVal<std::string> platformLibrary;
  rfl::DefaultVal<std::string> libraryModeExecutionManager;
  rfl::DefaultVal<std::string> platformQpu;
  rfl::DefaultVal<std::vector<std::string>> preprocessorDefines;
  rfl::DefaultVal<std::vector<std::string>> compilerFlags;
  rfl::DefaultVal<std::vector<std::string>> linkLibs;
  rfl::DefaultVal<std::vector<std::string>> pluginLibraries;
  rfl::DefaultVal<std::vector<std::string>> linkerFlags;
  rfl::DefaultVal<SimulationBackendInput> nvqirSimulationBackend;
  rfl::DefaultVal<std::vector<ConditionalBuildConfigV1>> rules;
};

struct BackendFeatureMapV1 {
  std::string name;
  std::vector<TargetFeatureFlag> optionFlags;
  /// `default` must be renamed to avoid C++ keyword conflicts.
  rfl::Rename<"default", rfl::DefaultVal<bool>> isDefault;
  BackendEndConfigEntryV1 config;
};

struct TargetConfigV1 {
  using Tag = rfl::Literal<"1">;
  std::string name;
  std::string description;
  rfl::DefaultVal<std::string> cudaqVersion;
  rfl::DefaultVal<std::string> warning;
  rfl::DefaultVal<std::vector<TargetArgumentV1>> targetArguments;
  rfl::DefaultVal<bool> gpuRequirements;
  // Structural: a target without a direct backend omits `config`.
  std::optional<BackendEndConfigEntryV1> config;
  rfl::DefaultVal<std::vector<BackendFeatureMapV1>> configurationMatrix;
};

/// Converts a parsed v1 configuration to the canonical model, applying
/// defaults, normalization, and cross-field validation.
/// Throws std::runtime_error on validation failures.
TargetConfig toCanonical(const TargetConfigV1 &config);

} // namespace cudaq::config::v1
