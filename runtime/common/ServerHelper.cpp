/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ServerHelper.h"

namespace cudaq {
void ServerHelper::parseConfigForCommonParams(const BackendConfig &config) {
  // Parse common parameters for each job and place into member variables
  for (auto &[key, val] : config) {
    // First Form a newKey with just the portion after the "." (i.e. jobId)
    auto ix = key.find_first_of('.');
    std::string newKey;
    if (ix != key.npos)
      newKey = key.substr(ix + 1);

    if (key.starts_with("output_names.")) {
      // Parse `val` into jobOutputNames.
      // Note: See `FunctionAnalysisData::resultQubitVals` of
      // LowerToQIRProfile.cpp for an example of how this was populated.
      OutputNamesType jobOutputNames;
      nlohmann::json outputNamesJSON = nlohmann::json::parse(val);
      for (const auto &el : outputNamesJSON[0]) {
        auto result = el[0].get<std::size_t>();
        auto qubitNum = el[1][0].get<std::size_t>();
        auto registerName = el[1][1].get<std::string>();
        jobOutputNames[result] = {qubitNum, registerName};
      }

      this->outputNames[newKey] = jobOutputNames;
    } else if (key.starts_with("reorderIdx.")) {
      nlohmann::json tmp = nlohmann::json::parse(val);
      this->reorderIdx[newKey] = tmp.get<std::vector<std::size_t>>();
    }
  }
}

std::optional<config::ArgumentType>
ServerHelper::getArgumentType(const std::string &key) const {
  for (const auto &arg : runtimeTarget.config.TargetArguments)
    // Check both the key name and platform-arg key
    if (arg.KeyName == key || arg.PlatformArgKey == key)
      return arg.Type;
  return std::nullopt;
}

nlohmann::json ServerHelper::getTypedConfigValue(const std::string &key) const {
  auto it = backendConfig.find(key);
  if (it == backendConfig.end())
    return nlohmann::json(); // null

  const std::string &value = it->second;
  auto argType = getArgumentType(key);

  // If no type info available, return as string
  if (!argType.has_value())
    return value;

  switch (argType.value()) {
  case config::ArgumentType::Bool:
    // Handle common boolean string representations
    if (value == "true" || value == "True" || value == "TRUE" || value == "1")
      return true;
    if (value == "false" || value == "False" || value == "FALSE" ||
        value == "0")
      return false;
    throw std::runtime_error("Invalid boolean value for '" + key + "': '" +
                             value + "'. Expected true/false/1/0.");
  case config::ArgumentType::Int: {
    try {
      return std::stoll(value);
    } catch (...) {
      // If parsing fails, return as string
      return value;
    }
  }
  case config::ArgumentType::String:
  case config::ArgumentType::UUID:
  case config::ArgumentType::FeatureFlag:
  case config::ArgumentType::MachineConfig:
  default:
    return value;
  }
  __builtin_unreachable();
}
} // namespace cudaq

LLVM_INSTANTIATE_REGISTRY(cudaq::ServerHelper::RegistryType)
