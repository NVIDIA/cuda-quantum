/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

// Shared helpers for converting parsed YAML schema structs (any schema
// version) into the canonical target model. The per-version structs are
// structurally identical for the non-compilation fields, so these helpers are
// templated on the schema struct type; each version only adds its own
// compilation-state conversion on top.

#include "cudaq/Target/TargetConfig.h"
#include <rfl.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::config::detail {

std::string trimScalar(const std::string &segment);

/// Scalar input is split on commas; each segment is trimmed and empty
/// segments are discarded after trimming. List input passes through.
std::vector<std::string> normalizeSimulationBackends(
    const rfl::Variant<std::string, std::vector<std::string>> &input);

void validatePattern(const std::string &pattern);

template <typename MachineArchitectureConfigV>
MachineArchitectureConfig
toCanonicalArch(const MachineArchitectureConfigV &cfg) {
  if (cfg.machineNames.value().empty() && cfg.pattern.value().empty())
    throw std::runtime_error(
        "Either 'machine-names' or 'pattern' must be specified.");
  if (!cfg.pattern.value().empty())
    validatePattern(cfg.pattern.value());
  MachineArchitectureConfig out;
  out.Name = cfg.archName.value();
  out.MachineNames = cfg.machineNames.value();
  out.MachinePattern = cfg.pattern.value();
  out.Configuration.CodegenEmission = cfg.config.codegenEmission.value();
  return out;
}

/// Fills everything except the compilation state, which each schema version
/// expresses differently.
template <typename BackendEndConfigEntryV>
void toCanonicalEntryBase(const BackendEndConfigEntryV &cfg,
                          BackendEndConfigEntry &out) {
  out.GenTargetBackend = cfg.genTargetBackend.value();
  out.LibraryMode = cfg.libraryMode.value();
  out.SupportResourceCounts = cfg.supportResourceCounts.value();
  out.NeedsReorderIdx = cfg.needsReorderIdx.value();
  out.NeedsImplicitMeasurements = cfg.needsImplicitMeasurements.value();
  out.PlatformLibrary = cfg.platformLibrary.value();
  out.LibraryModeExecutionManager = cfg.libraryModeExecutionManager.value();
  out.PlatformQpu = cfg.platformQpu.value();
  out.PreprocessorDefines = cfg.preprocessorDefines.value();
  out.CompilerFlags = cfg.compilerFlags.value();
  out.LinkLibs = cfg.linkLibs.value();
  out.PluginLibraries = cfg.pluginLibraries.value();
  out.LinkerFlags = cfg.linkerFlags.value();
  out.SimulationBackend.values =
      normalizeSimulationBackends(cfg.nvqirSimulationBackend.value());
  for (const auto &rule : cfg.rules.value())
    out.ConditionalBuildConfigs.push_back({rule.condition.value(),
                                           rule.compilerFlag.value(),
                                           rule.linkFlag.value()});
}

/// Validates machine-config usage and appends the converted target arguments.
template <typename TargetArgumentV>
void appendCanonicalArguments(const std::vector<TargetArgumentV> &args,
                              TargetConfig &out) {
  unsigned machineConfigArgs = 0;
  for (const auto &arg : args) {
    if (!arg.machineConfig.value().empty() &&
        arg.type.value() != ArgumentType::machine_config)
      throw std::runtime_error(
          "If 'machine-config' is provided, 'type' must be 'machine-config'.");
    if (arg.type.value() == ArgumentType::machine_config)
      ++machineConfigArgs;
  }
  if (machineConfigArgs > 1)
    throw std::runtime_error("There should only ever be 1 "
                             "machine-configuration entry in the target "
                             "arguments.");
  for (const auto &arg : args) {
    TargetArgument outArg;
    outArg.KeyName = arg.key;
    outArg.IsRequired = arg.required.value();
    outArg.PlatformArgKey = arg.platformArg.value();
    outArg.HelpString = arg.helpString.value();
    outArg.Type = arg.type.value();
    for (const auto &machineConfig : arg.machineConfig.value())
      outArg.MachineConfigs.push_back(toCanonicalArch(machineConfig));
    out.TargetArguments.push_back(std::move(outArg));
  }
}

/// Converts every configuration-matrix entry with `toEntry`, the per-version
/// backend-entry converter.
template <typename BackendFeatureMapV, typename ToEntry>
void appendCanonicalConfigMap(const std::vector<BackendFeatureMapV> &matrix,
                              TargetConfig &out, ToEntry &&toEntry) {
  for (const auto &entry : matrix) {
    BackendFeatureMap outEntry;
    outEntry.Name = entry.name;
    outEntry.Flags =
        static_cast<TargetFeatureFlag>(combineFeatureFlags(entry.optionFlags));
    outEntry.Default = entry.isDefault.value().value();
    outEntry.Config = toEntry(entry.config);
    out.ConfigMap.push_back(std::move(outEntry));
  }
}

/// Flattens config.plugin-libraries for callers that only need the
/// target-level runtime payload list.
inline void flattenPluginLibraries(TargetConfig &out) {
  if (out.PluginLibraries.empty() && out.BackendConfig &&
      !out.BackendConfig->PluginLibraries.empty())
    out.PluginLibraries = out.BackendConfig->PluginLibraries;
}

} // namespace cudaq::config::detail
