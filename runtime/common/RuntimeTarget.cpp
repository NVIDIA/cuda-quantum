/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RuntimeTarget.h"
#include <sstream>

#include <iostream>

namespace cudaq {

simulation_precision RuntimeTarget::get_precision() const { return precision; }

std::string RuntimeTarget::get_target_args_help_string() const {
  std::stringstream ss;
  for (const auto &argConfig : config.TargetArguments) {
    ss << "  - " << argConfig.KeyName;
    if (!argConfig.HelpString.empty()) {
      ss << " (" << argConfig.HelpString << ")";
    }
    ss << "\n";
  }
  return ss.str();
}

/// Retrieve a named backend config entry from the target config.
static cudaq::config::BackendEndConfigEntry
getBackendEndConfigEntry(const cudaq::config::TargetConfig &config,
                         std::string &argName) {

  const auto &configMap = config.ConfigMap;
  const auto it = std::find_if(configMap.begin(), configMap.end(),
                               [&](auto m) { return m.Name == argName; });
  return it->Config;
}

/// Get the config entry for the target feature.
config::BackendEndConfigEntry
getRuntimeTargetConfigBackendEntry(RuntimeTarget &runtimeTarget,
                                   std::string &featureName) {
  // TODO: the maps show empty here, do we need to run some initialization
  // first?
  std::cout << "Config map:" << std::endl;
  for (auto &p : runtimeTarget.config.ConfigMap) {
    std::cout << p.Name << std::endl;
  }
  std::cout << "RuntimeTarget:" << std::endl;
  for (auto &p : runtimeTarget.runtimeConfig) {
    std::cout << p.first << "-> " << p.second << std::endl;
  }
  if (runtimeTarget.runtimeConfig.find(featureName) ==
      runtimeTarget.runtimeConfig.end())
    throw std::runtime_error(
        "Unknown runtime target configuration feature name " + featureName);

  return getBackendEndConfigEntry(runtimeTarget.config,
                                  runtimeTarget.runtimeConfig[featureName]);
}
} // namespace cudaq
