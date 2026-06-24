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

namespace cudaq {

/// @brief Helper function to get boolean environment variable
bool getEnvBool(const char *envName, bool defaultVal = false) {
  if (auto envVal = std::getenv(envName)) {
    std::string tmp(envVal);
    std::transform(tmp.begin(), tmp.end(), tmp.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return (tmp == "1" || tmp == "on" || tmp == "true" || tmp == "y" ||
            tmp == "yes");
  }
  return defaultVal;
}

} // namespace cudaq
