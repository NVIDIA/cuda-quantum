/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Environment.h"
#include <algorithm>
#include <string>

static std::string toLower(const std::string &str) {
  std::string tmp(str);
  std::transform(tmp.begin(), tmp.end(), tmp.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return tmp;
}

static bool isTruthy(const std::string &str) {
  return str == "1" || str == "on" || str == "true" || str[0] == 'y';
}

/// @brief Helper function to get boolean environment variable
bool cudaq::getEnvBool(const char *envName, bool defaultVal = false) {
  if (auto envVal = std::getenv(envName)) {
    return isTruthy(toLower(envVal));
  }
  return defaultVal;
}

cudaq::PrintEachPassMode cudaq::getEnvPrintEachPassMode(const char *envName) {
  if (auto envVal = std::getenv(envName)) {
    auto tmp = toLower(envVal);
    if (tmp == "specialize" || tmp.starts_with("arg-synth") ||
        tmp.starts_with("argsynth"))
      return PrintEachPassMode::ArgSynthesis;
    if (isTruthy(tmp))
      return PrintEachPassMode::All;
  }
  return PrintEachPassMode::None;
}
