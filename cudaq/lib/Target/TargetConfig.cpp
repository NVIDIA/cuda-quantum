/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/TargetConfig.h"
#include <regex>

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
