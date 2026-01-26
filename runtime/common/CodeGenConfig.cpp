/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CodeGenConfig.h"
#include "FmtCore.h"
#include "Logger.h"
#include <stdexcept>

namespace {

std::vector<std::string> splitString(const std::string &s,
                                     const char delimiter) {
  std::vector<std::string> tokens;
  size_t pos_start = 0, pos_end;
  std::string token;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + 1;
    tokens.push_back(token);
  }
  tokens.push_back(s.substr(pos_start)); // Add the last token
  return tokens;
}

/// @brief Helper to parse `codegen` translation, with optional feature
/// annotation.
// e.g., "qir-adaptive:1.0:int_computations,float_computations"
static std::tuple<std::string, std::string, std::vector<std::string>>
parseCodeGenTranslationString(const std::string &settingStr) {
  auto transportFields = splitString(settingStr, ':');
  auto size = transportFields.size();
  if (size == 1)
    return {transportFields[0], {}, {}};
  if (size == 2)
    return {transportFields[0], transportFields[1], {}};
  if (size == 3) {
    auto options = splitString(transportFields[2], ',');
    return {transportFields[0], transportFields[1], options};
  }
  throw std::runtime_error(
      fmt::format("Invalid codegen-emission string '{}'.", settingStr));
}
} // namespace

cudaq::CodeGenConfig
cudaq::parseCodeGenTranslation(const std::string &codegenTranslation) {
  auto [codeGenName, codeGenVersion, codeGenOptions] =
      parseCodeGenTranslationString(codegenTranslation);

  if (codeGenName.find("qir") == codeGenName.npos)
    return {.profile = codeGenName};

  CodeGenConfig config = {
      .profile = codeGenName,
      .isQIRProfile = true,
      .isAdaptiveProfile = codeGenName == "qir-adaptive",
      .isBaseProfile = codeGenName == "qir-base",
  };

  // Default version for base profile is 1.0
  if (config.isBaseProfile) {
    config.version = QirVersion::version_1_0;
    config.qir_major_version = 1;
    config.qir_minor_version = 0;
  }

  if (config.isAdaptiveProfile) {
    for (auto option : codeGenOptions) {
      if (option == "int_computations") {
        cudaq::info("Enable int_computations extension");
        config.integerComputations = true;
      } else if (option == "float_computations") {
        cudaq::info("Enable float_computations extension");
        config.floatComputations = true;
      } else if (option == "output_log") {
        cudaq::info("Enable output log support");
        config.outputLog = true;
      } else if (option == "erase_stack_bounding") {
        cudaq::info("Enable erasing stack bounding");
        config.eraseStackBounding = true;
      } else if (option == "erase_record_calls") {
        cudaq::info("Enable erasing record calls");
        config.eraseRecordCalls = true;
      } else if (option == "allow_all_instructions") {
        cudaq::info("Enable all instructions");
        config.allowAllInstructions = true;
      } else {
        throw std::runtime_error(fmt::format(
            "Invalid option '{}' for '{}' codegen.", option, codeGenName));
      }
    }
  } else {
    if (!codeGenOptions.empty())
      throw std::runtime_error(
          fmt::format("Invalid codegen-emission '{}'. Extra options are not "
                      "supported for '{}' codegen.",
                      codegenTranslation, codeGenName));
  }

  if (config.isAdaptiveProfile) {
    // If no version is specified, using the lowest version
    if (codeGenVersion.empty() || codeGenVersion == "0.1") {
      config.version = QirVersion::version_0_1;
      config.qir_major_version = 0;
      config.qir_minor_version = 1;
    } else if (codeGenVersion == "1.0") {
      config.version = QirVersion::version_1_0;
      config.qir_major_version = 1;
      config.qir_minor_version = 0;
    } else {
      throw std::runtime_error(
          fmt::format("Unsupported QIR version '{}', codegen setting: {}",
                      codeGenVersion, codegenTranslation));
    }
  }

  return config;
}
