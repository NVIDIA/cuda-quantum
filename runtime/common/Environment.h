/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq {

/// @brief Helper function to get boolean environment variable
bool getEnvBool(const char *envName, bool defaultVal);

/// Valid options for `CUDAQ_MLIR_PRINT_EACH_PASS` environment variable.
enum class PrintEachPassMode { None = 0, Specialize = 1, All = 2 };

/// @brief Parse the environment variable into a PrintEachPassMode.
PrintEachPassMode
getEnvPrintEachPassMode(const char *envName = "CUDAQ_MLIR_PRINT_EACH_PASS");

} // namespace cudaq
