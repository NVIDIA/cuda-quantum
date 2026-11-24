/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace cudaq {

/// @brief Retrieve the value of an environment variable.
///
/// A variable that is set to the empty string counts as set and its (empty)
/// value is returned unchanged. The value is not trimmed.
/// @param key Name of the environment variable.
/// @param defaultVal Value returned when the variable is not set and not
/// required.
/// @param isRequired If true and the variable is not set, a
/// `std::runtime_error` is thrown.
/// @param missingMessage Optional message for the exception. If empty, the
/// message is "<key> environment variable is not set."
inline std::string getEnvVar(const std::string &key,
                             const std::string &defaultVal, bool isRequired,
                             const std::string &missingMessage = "") {
  const char *envVar = std::getenv(key.c_str());
  if (envVar == nullptr) {
    if (isRequired)
      throw std::runtime_error(missingMessage.empty()
                                   ? key + " environment variable is not set."
                                   : missingMessage);
    return defaultVal;
  }
  return std::string(envVar);
}

/// @brief Return `config[key]` if `key` is present (even if its value is
/// empty), otherwise `defaultValue`. `ConfigMap` is any map from string to
/// string (e.g. `BackendConfig`).
template <typename ConfigMap>
std::string getValueOrDefault(const ConfigMap &config, const std::string &key,
                              const std::string &defaultValue) {
  auto it = config.find(key);
  return it != config.end() ? it->second : defaultValue;
}

} // namespace cudaq
