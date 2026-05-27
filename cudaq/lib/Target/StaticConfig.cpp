/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/StaticConfig.h"
#include <regex>

std::string cudaq::config::StaticConfig::getCodeGenSpec(
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

std::string cudaq::config::BackendEndConfigEntry::getPassPipeline(
    std::string_view deployStage, std::string_view finalizeStage) const {
  if (!TargetPassPipeline.empty())
    return TargetPassPipeline;

  std::string pipeline;
  auto append = [&](std::string_view stage) {
    if (stage.empty())
      return;
    if (!pipeline.empty())
      pipeline += ",";
    pipeline += stage;
  };

  append(JITHighLevelPipeline);
  append(deployStage);
  append(JITMidLevelPipeline);
  append(finalizeStage);
  append(JITLowLevelPipeline);
  return pipeline;
}
