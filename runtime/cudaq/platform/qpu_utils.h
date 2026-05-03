/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file qpu_utils.h
/// @brief Utility functions for the CUDA-Q platforms to aimed at reducing
/// header file dependencies.
#include <string>

namespace cudaq {
namespace config {
class TargetConfig;
} // namespace config

namespace detail {
/// @brief Parses @p yamlContent as a target backend YAML configuration and
/// deserializes the result into @p targetConfig.
void parseTargetConfigYml(const std::string &yamlContent,
                          config::TargetConfig &targetConfig);

/// @brief Decodes the base64-encoded string @p encoded and returns the
/// decoded result.  Throws std::runtime_error on malformed input.
std::string decodeBase64(const std::string &encoded);

/// @brief Returns true if @p kernelName has the analog Hamiltonian kernel
/// prefix.
bool isAnalogHamiltonianKernel(const std::string &kernelName);
} // namespace detail

} // namespace cudaq
