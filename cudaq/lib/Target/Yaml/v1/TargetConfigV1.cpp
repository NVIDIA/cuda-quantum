/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Yaml/v1/TargetConfigV1.h"
#include <algorithm>
#include <cctype>
#include <regex>
#include <utility>

static std::string trim(const std::string &segment) {
  const auto isSpace = [](unsigned char c) { return std::isspace(c); };
  const auto begin = std::find_if_not(segment.begin(), segment.end(), isSpace);
  const auto end =
      std::find_if_not(segment.rbegin(), segment.rend(), isSpace).base();
  if (begin >= end)
    return {};
  return std::string(begin, end);
}

/// Scalar input is split on commas; each segment is trimmed and empty
/// segments are discarded after trimming. List input passes through.
static std::vector<std::string> normalizeSimulationBackends(
    const cudaq::config::v1::SimulationBackendInput &input) {
  return input.visit([](const auto &value) -> std::vector<std::string> {
    using T = std::decay_t<decltype(value)>;
    if constexpr (std::is_same_v<T, std::vector<std::string>>) {
      return value;
    } else {
      std::vector<std::string> result;
      std::size_t begin = 0;
      while (begin <= value.size()) {
        const auto end = value.find(',', begin);
        auto segment = trim(value.substr(begin, end - begin));
        if (!segment.empty())
          result.push_back(std::move(segment));
        if (end == std::string::npos)
          break;
        begin = end + 1;
      }
      return result;
    }
  });
}

static void validatePattern(const std::string &pattern) {
  try {
    std::regex re(pattern);
  } catch (const std::regex_error &e) {
    throw std::runtime_error("'" + pattern +
                             "' is not a valid regex: " + e.what());
  }
}

static cudaq::config::MachineArchitectureConfig
toCanonicalArch(const cudaq::config::v1::MachineArchitectureConfigV1 &cfg) {
  if (cfg.machineNames.value().empty() && cfg.pattern.value().empty())
    throw std::runtime_error(
        "Either 'machine-names' or 'pattern' must be specified.");
  if (!cfg.pattern.value().empty())
    validatePattern(cfg.pattern.value());
  cudaq::config::MachineArchitectureConfig out;
  out.Name = cfg.archName.value();
  out.MachineNames = cfg.machineNames.value();
  out.MachinePattern = cfg.pattern.value();
  out.Configuration.CodegenEmission = cfg.config.codegenEmission.value();
  return out;
}

static cudaq::config::BackendEndConfigEntry
toCanonicalEntry(const cudaq::config::v1::BackendEndConfigEntryV1 &cfg) {
  cudaq::config::BackendEndConfigEntry out;
  out.GenTargetBackend = cfg.genTargetBackend.value();
  out.LibraryMode = cfg.libraryMode.value();
  out.SupportResourceCounts = cfg.supportResourceCounts.value();
  out.JITHighLevelPipeline = cfg.jitHighLevelPipeline.value();
  out.JITMidLevelPipeline = cfg.jitMidLevelPipeline.value();
  out.JITLowLevelPipeline = cfg.jitLowLevelPipeline.value();
  out.TargetPassPipeline = cfg.targetPassPipeline.value();
  out.CodegenEmission = cfg.codegenEmission.value();
  out.PostCodeGenPasses = cfg.postCodegenPasses.value();
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
  return out;
}

cudaq::config::TargetConfig
cudaq::config::v1::toCanonical(const TargetConfigV1 &cfg) {
  unsigned machineConfigArgs = 0;
  for (const auto &arg : cfg.targetArguments.value()) {
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

  TargetConfig out;
  out.Name = cfg.name;
  out.Description = cfg.description;
  out.CudaqVersion = cfg.cudaqVersion.value();
  out.WarningMsg = cfg.warning.value();
  out.GpuRequired = cfg.gpuRequirements.value();
  for (const auto &arg : cfg.targetArguments.value()) {
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
  if (cfg.config)
    out.BackendConfig = toCanonicalEntry(*cfg.config);
  for (const auto &entry : cfg.configurationMatrix.value()) {
    BackendFeatureMap outEntry;
    outEntry.Name = entry.name;
    outEntry.Flags =
        static_cast<TargetFeatureFlag>(combineFeatureFlags(entry.optionFlags));
    outEntry.Default = entry.isDefault.value().value();
    outEntry.Config = toCanonicalEntry(entry.config);
    out.ConfigMap.push_back(std::move(outEntry));
  }
  // Flatten config.plugin-libraries for callers that only need the
  // target-level runtime payload list.
  if (out.PluginLibraries.empty() && out.BackendConfig &&
      !out.BackendConfig->PluginLibraries.empty())
    out.PluginLibraries = out.BackendConfig->PluginLibraries;
  return out;
}
