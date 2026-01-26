/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace cudaq {

enum struct QirVersion { version_0_1, version_1_0 };

/// @brief `Codegen` configuration.
/// Note: it is currently flattened to contain all possible options
/// for all possible platforms.
struct CodeGenConfig {
  // Profile name
  std::string profile;
  // True if this is a QIR profile.
  bool isQIRProfile;
  // QIR profile version enum
  QirVersion version;
  // QIR profile major version
  std::uint32_t qir_major_version;
  // QIR profile minor version
  std::uint32_t qir_minor_version;
  // True if this is an adaptive QIR profile.
  bool isAdaptiveProfile;
  // True if this is a base QIR profile.
  bool isBaseProfile;
  // True if integer computation is enabled.
  bool integerComputations = false;
  // True if floating-point computation is enabled.
  bool floatComputations = false;
  // True if QIR output to log is enabled.
  bool outputLog = false;
  // True if we should erase stacksave/stackrestore instructions.
  bool eraseStackBounding = false;
  // True if we should erase measurement result recording functions.
  bool eraseRecordCalls = false;
  // True if we should bypass instruction validation, i.e., allow all
  // instructions.
  bool allowAllInstructions = false;
};

/// @brief Helper to parse `codegen` translation, with optional feature
/// annotation.
/// e.g., `qir-adaptive:1.0:int_computations,float_computations`.
/// Handles errors and returns structured configuration.
CodeGenConfig parseCodeGenTranslation(const std::string &codegenTranslation);
} // namespace cudaq
