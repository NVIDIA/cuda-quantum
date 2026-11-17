/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/TargetConfig.h"
#include <algorithm>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// CLI spellings of the `option-flags` feature bits. Keep in sync with the
// YAML traits table in `Yaml/TargetConfigYamlAdapters.h`.
static const std::unordered_map<std::string, cudaq::config::TargetFeatureFlag>
    stringToFeatureFlag{
        {"fp32", cudaq::config::TargetFeatureFlag::fp32},
        {"fp64", cudaq::config::TargetFeatureFlag::fp64},
        {"mgpu", cudaq::config::TargetFeatureFlag::mgpu},
        {"mqpu", cudaq::config::TargetFeatureFlag::mqpu},
        {"dep-analysis", cudaq::config::TargetFeatureFlag::dep_analysis},
        {"qpp", cudaq::config::TargetFeatureFlag::qpp}};

/// Split `value` on commas, discarding empty segments.
static std::vector<std::string> splitFeatureFlags(const std::string &value) {
  std::vector<std::string> flags;
  std::size_t pos = 0;
  while (pos <= value.size()) {
    const auto comma = value.find(',', pos);
    const auto end = comma == std::string::npos ? value.size() : comma;
    if (end > pos)
      flags.emplace_back(value.substr(pos, end - pos));
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return flags;
}

namespace {
std::string configLocation(const std::filesystem::path &configPath) {
  return configPath.empty() ? std::string{} : " in " + configPath.string();
}
} // namespace

std::string cudaq::config::TargetConfig::getCodeGenSpec(
    const std::map<std::string, std::string> &targetArgs) const {
  // Check whether we have a per-machine config
  const auto machineConfigIter = std::find_if(
      TargetArguments.begin(), TargetArguments.end(),
      [&](const cudaq::config::TargetArgument &argConfig) {
        return argConfig.Type == cudaq::config::ArgumentType::machine_config;
      });
  // Without a machine-specific rule, the emission format is that of the
  // backend configuration that the arguments select.
  const auto selectedEmissionFormat = [&]() -> std::string {
    const auto *backend = selectBackend(targetArgs);
    return backend ? backend->CompileTarget.codegen.emissionFormat : "";
  };
  if (machineConfigIter == TargetArguments.end())
    return selectedEmissionFormat();

  // Get the machine name from the CLI argument
  std::string machineName;
  for (const auto &[argKey, argVal] : targetArgs) {
    if (argKey == machineConfigIter->PlatformArgKey) {
      machineName = argVal;
      break;
    }
  }

  if (!machineName.empty()) {
    // Check for match
    for (auto &archConfig : machineConfigIter->MachineConfigs) {
      // Check names first
      if (std::find(archConfig.MachineNames.begin(),
                    archConfig.MachineNames.end(),
                    machineName) != archConfig.MachineNames.end()) {
        return archConfig.Configuration.CodegenEmission;
      }
      // Check pattern if provided
      if (!archConfig.MachinePattern.empty()) {
        std::regex re(archConfig.MachinePattern);
        if (std::regex_search(machineName, re)) {
          return archConfig.Configuration.CodegenEmission;
        }
      }
    }
  }

  return selectedEmissionFormat();
}

const cudaq::config::TargetArgument *
cudaq::config::TargetConfig::findArgument(const std::string &argKey) const {
  const auto iter =
      std::find_if(TargetArguments.begin(), TargetArguments.end(),
                   [&](const cudaq::config::TargetArgument &argConfig) {
                     return argKey == argConfig.KeyName ||
                            argKey == "--target-" + argConfig.KeyName ||
                            argKey == "--" + Name + "-" + argConfig.KeyName;
                   });
  return iter == TargetArguments.end() ? nullptr : &*iter;
}

const cudaq::config::BackendFeatureMap *
cudaq::config::TargetConfig::selectFeatureConfig(
    const std::map<std::string, std::string> &targetArgs) const {
  // Collect the feature flags that the `option-flags` arguments name.
  unsigned featureFlag = 0;
  for (const auto &[argKey, argVal] : targetArgs) {
    const auto *argument = findArgument(argKey);
    if (!argument ||
        argument->Type != cudaq::config::ArgumentType::option_flags)
      continue;
    for (const auto &flag : splitFeatureFlags(argVal)) {
      const auto iter = stringToFeatureFlag.find(flag);
      if (iter == stringToFeatureFlag.end())
        throw std::runtime_error("Unknown feature flag '" + flag + "'");
      featureFlag |= static_cast<unsigned>(iter->second);
    }
  }

  const auto defaultEntry = std::find_if(
      ConfigMap.begin(), ConfigMap.end(),
      [](const BackendFeatureMap &entry) { return entry.Default; });
  // Without an option flag, the default entry is selected.
  if (featureFlag == 0)
    return defaultEntry == ConfigMap.end() ? nullptr : &*defaultEntry;

  // The flags select the entry they match exactly, or the entry they match
  // once the default entry's flags are added. For example, when the default
  // is `fp32`, `option=mqpu` selects the `fp32,mqpu` entry.
  const unsigned defaultFlag = defaultEntry == ConfigMap.end()
                                   ? 0
                                   : static_cast<unsigned>(defaultEntry->Flags);
  const auto selected = std::find_if(
      ConfigMap.begin(), ConfigMap.end(), [&](const BackendFeatureMap &entry) {
        const auto entryFlags = static_cast<unsigned>(entry.Flags);
        return featureFlag == entryFlags ||
               (featureFlag | defaultFlag) == entryFlags;
      });
  return selected == ConfigMap.end() ? nullptr : &*selected;
}

const cudaq::config::BackendEndConfigEntry *
cudaq::config::TargetConfig::selectBackend(
    const std::map<std::string, std::string> &targetArgs) const {
  if (ConfigMap.empty())
    return BackendConfig.has_value() ? &*BackendConfig : nullptr;
  const auto *featureConfig = selectFeatureConfig(targetArgs);
  return featureConfig ? &featureConfig->Config : nullptr;
}

bool cudaq::config::BackendEndConfigEntry::hasPassPipeline() const {
  return CompileTarget.hasPassPipeline();
}

cudaq::config::TargetVersionCompatibilityResult
cudaq::config::checkExternalTargetVersion(
    const TargetConfig &config, std::string_view currentVersion,
    const std::filesystem::path &configPath) {
  if (config.CudaqVersion == currentVersion)
    return {};

  const auto pluginStr = config.CudaqVersion.empty() ? std::string("(unknown)")
                                                     : config.CudaqVersion;
  const auto currentStr = currentVersion.empty() ? std::string("(unknown)")
                                                 : std::string(currentVersion);
  return {TargetVersionCompatibility::Warning,
          "warning: target '" + config.Name + "' was built for CUDA-Q " +
              pluginStr + ", but the current CUDA-Q version is " + currentStr +
              "; compatibility is not guaranteed" + configLocation(configPath)};
}
