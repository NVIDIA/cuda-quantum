/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/TargetConfig.h"
#include <charconv>
#include <cstdint>
#include <regex>
#include <tuple>

namespace {
struct NumericVersion {
  std::uint64_t Major = 0;
  std::uint64_t Minor = 0;
  std::uint64_t Patch = 0;
};

std::optional<NumericVersion> parseNumericVersion(std::string_view value) {
  NumericVersion version;
  const char *cursor = value.data();
  const char *end = cursor + value.size();

  auto parseComponent = [&](std::uint64_t &component) {
    const auto result = std::from_chars(cursor, end, component);
    if (result.ec != std::errc{} || result.ptr == cursor)
      return false;
    cursor = result.ptr;
    return true;
  };

  if (!parseComponent(version.Major) || cursor == end || *cursor++ != '.' ||
      !parseComponent(version.Minor) || cursor == end || *cursor++ != '.' ||
      !parseComponent(version.Patch))
    return std::nullopt;

  return version;
}

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
        return argConfig.Type == cudaq::config::ArgumentType::MachineConfig;
      });
  if (machineConfigIter == TargetArguments.end()) {
    // No machine specific config
    return BackendConfig.has_value() ? BackendConfig->CodegenEmission : "";
  }

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

  // No machine specific config rule matches, fallback to the default backend
  // config
  return BackendConfig.has_value() ? BackendConfig->CodegenEmission : "";
}

bool cudaq::config::BackendEndConfigEntry::hasPassPipeline() const {
  return !TargetPassPipeline.empty() || !JITHighLevelPipeline.empty() ||
         !JITMidLevelPipeline.empty() || !JITLowLevelPipeline.empty();
}

cudaq::config::TargetVersionCompatibilityResult
cudaq::config::checkExternalTargetVersion(
    const TargetConfig &config, std::string_view currentVersion,
    const std::filesystem::path &configPath) {
  const auto current = parseNumericVersion(currentVersion);

  // If the current CUDA-Q version is non-numeric (e.g. a dev or CI build),
  // semver comparison is impossible. Fall back to string equality: warn if
  // the versions differ, but carry on either way.
  if (!current) {
    if (config.CudaqVersion != std::string(currentVersion)) {
      const auto pluginStr = config.CudaqVersion.empty()
                                 ? std::string("(unknown)")
                                 : config.CudaqVersion;
      const auto currentStr = currentVersion.empty()
                                  ? std::string("(unknown)")
                                  : std::string(currentVersion);
      return {TargetVersionCompatibility::Warning,
              "warning: target '" + config.Name + "' was built for CUDA-Q " +
                  pluginStr + ", but the current CUDA-Q version is " +
                  currentStr +
                  "; versions are non-numeric so compatibility cannot be "
                  "verified" +
                  configLocation(configPath)};
    }
    return {};
  }

  // Current version is numeric: the plugin must also declare a valid numeric
  // version so we can make a meaningful compatibility decision.
  const auto pluginVersion = parseNumericVersion(config.CudaqVersion);
  if (!pluginVersion) {
    return {TargetVersionCompatibility::Error,
            "error: target '" + config.Name +
                "' has missing or malformed cudaq-version metadata" +
                configLocation(configPath)};
  }

  const auto pluginTuple = std::tie(pluginVersion->Major, pluginVersion->Minor,
                                    pluginVersion->Patch);
  const auto currentTuple =
      std::tie(current->Major, current->Minor, current->Patch);
  if (currentTuple < pluginTuple) {
    return {TargetVersionCompatibility::Error,
            "error: target '" + config.Name + "' was built for CUDA-Q " +
                config.CudaqVersion + ", but the current CUDA-Q version is " +
                std::string(currentVersion) + configLocation(configPath)};
  }

  if (current->Major > pluginVersion->Major ||
      (current->Major == pluginVersion->Major &&
       current->Minor > pluginVersion->Minor)) {
    return {TargetVersionCompatibility::Warning,
            "warning: target '" + config.Name +
                "' was built and tested with CUDA-Q " + config.CudaqVersion +
                "; the current CUDA-Q version is newer (" +
                std::string(currentVersion) +
                "), so compatibility is not guaranteed" +
                configLocation(configPath)};
  }

  return {};
}
