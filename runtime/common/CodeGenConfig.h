/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace cudaq {

enum struct QirVersion { version_0_1, version_0_2 };

/// @brief Codegen configuration.
/// Note: it is currently flattened to contain all possible options
/// for all possible platforms.s
struct CodeGenConfig {
  std::string profile;
  bool isQIRProfile;
  QirVersion version;
  std::uint32_t qir_major_version;
  std::uint32_t qir_minor_version;
  bool isAdaptiveProfile;
  bool isBaseProfile;
  bool integerComputations;
  bool floatComputations;
  bool outputLog;
  bool eraseStackBounding;
  bool eraseRecordCalls;
  bool allowAllInstructions;
};

/// @brief Helper to parse `codegen` translation, with optional feature
/// annotation.
/// e.g., "qir-adaptive:0.2:int_computations,float_computations".
/// Handles errors and returns structured configuration.
CodeGenConfig parseCodeGenTranslation(const std::string &codegenTranslation);
} // namespace cudaq
