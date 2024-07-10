/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/Support/YAMLTraits.h"
#include <optional>
#include <string>
#include <vector>

namespace cudaq {
namespace config {
enum TargetFeatureFlag : unsigned {
  flagsFP32 = 0x0001,
  flagsFP64 = 0x0002,
  flagsMgpu = 0x0004,
  flagsMqpu = 0x0008,
};

enum ArgumentType { String, Int, UUID, FeatureFlag };

struct TargetArgument {
  std::string KeyName;
  bool IsRequired = false;
  std::string PlatformArgKey;
  std::string HelpString;
  ArgumentType Type = ArgumentType::String;
  std::string DefaultValue;
  std::vector<std::string> ValidValues;
};

struct SimulationBackendSetting {
  std::vector<std::string> values;
};

struct ConditionalBuildConfig {
  std::string Condition;
  std::string CompileFlag;
  std::string LinkFlag;
};

struct BackendEndConfigEntry {
  std::optional<bool> GenTargetBackend;
  std::optional<bool> LibraryMode;
  std::string PlatformLoweringConfig;
  std::string CodegenEmission;
  std::string PostCodeGenPasses;
  std::string PlatformLibrary;
  std::string LibraryModeExecutionManager;
  std::string PlatformQpu;
  std::vector<std::string> PreprocessorDefines;
  std::vector<std::string> CompilerFlags;
  std::vector<std::string> LinkLibs;
  std::vector<std::string> LinkerFlags;
  SimulationBackendSetting SimulationBackend;
  std::vector<ConditionalBuildConfig> ConditionalBuildConfigs;
};

struct BackendFeatureMap {
  std::string Name;
  TargetFeatureFlag Flags;
  std::optional<bool> Default;
  BackendEndConfigEntry Config;
};

struct TargetConfig {
  std::string Name;
  std::string Description;
  std::string WarningMsg;
  std::vector<TargetArgument> TargetArguments;
  bool GpuRequired = false;

  std::optional<BackendEndConfigEntry> BackendConfig;
  std::vector<BackendFeatureMap> ConfigMap;
};

std::string processSimBackendConfig(const std::string &targetName,
                                    const BackendEndConfigEntry &configValue);

std::string processRuntimeArgs(const TargetConfig &config,
                               const std::vector<std::string> &targetArgv);
} // namespace config
} // namespace cudaq

LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::TargetFeatureFlag)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::TargetArgument)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::ConditionalBuildConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::BackendFeatureMap)

namespace llvm {
namespace yaml {
template <>
struct ScalarBitSetTraits<cudaq::config::TargetFeatureFlag> {
  static void bitset(IO &io, cudaq::config::TargetFeatureFlag &value);
};

template <>
struct ScalarEnumerationTraits<cudaq::config::ArgumentType> {
  static void enumeration(IO &io, cudaq::config::ArgumentType &value);
};

template <>
struct MappingTraits<cudaq::config::TargetArgument> {
  static void mapping(IO &io, cudaq::config::TargetArgument &info);
};

template <>
struct BlockScalarTraits<cudaq::config::SimulationBackendSetting> {
  static void output(const cudaq::config::SimulationBackendSetting &Value,
                     void *Ctxt, llvm::raw_ostream &OS);
  static StringRef input(StringRef Scalar, void *Ctxt,
                         cudaq::config::SimulationBackendSetting &Value);
};
template <>
struct MappingTraits<cudaq::config::ConditionalBuildConfig> {
  static void mapping(IO &io, cudaq::config::ConditionalBuildConfig &info);
};

template <>
struct MappingTraits<cudaq::config::BackendEndConfigEntry> {
  static void mapping(IO &io, cudaq::config::BackendEndConfigEntry &info);
};
template <>
struct MappingTraits<cudaq::config::BackendFeatureMap> {
  static void mapping(IO &io, cudaq::config::BackendFeatureMap &info);
};

template <>
struct MappingTraits<cudaq::config::TargetConfig> {
  static void mapping(IO &io, cudaq::config::TargetConfig &info);
};
} // namespace yaml
} // namespace llvm