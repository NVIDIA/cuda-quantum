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
/// Flag to enable feature(s) of the unified NVIDIA target.
// Use bitset so that we can combine different options, e.g., multi-gpu with
// fp32/64.
// Use raw enum since we need to use it as a bit set.
enum TargetFeatureFlag : unsigned {
  flagsFP32 = 0x0001,
  flagsFP64 = 0x0002,
  flagsMgpu = 0x0004,
  flagsMqpu = 0x0008,
};

/// @brief Configuration argument type annotation
// e.g., to support type validation.
enum class ArgumentType { String, Int, UUID, FeatureFlag };

/// @brief Encapsulates target-specific arguments
struct TargetArgument {
  /// CLI argument key name.
  // Note: target name is prepended automatically for nvq++ driver.
  std::string KeyName;
  bool IsRequired = false;
  /// Platform argument that this option should be mapped to.
  // If provided, value provided will be forwarded to the platform via this key.
  std::string PlatformArgKey;
  /// Help string for this argument.
  std::string HelpString;
  /// Type of the expected input value.
  ArgumentType Type = ArgumentType::String;
};

/// NVQIR simulator backend setting.
// Use a vector of strings to support single/multiple simulator libs associated
// with a single target (e.g., nvidia).
struct SimulationBackendSetting {
  std::vector<std::string> values;
};

/// Encapsulates conditional build configurations.
// e.g., the target may customize build/link flags depending on some specific
// conditions.
struct ConditionalBuildConfig {
  std::string Condition;
  std::string CompileFlag;
  std::string LinkFlag;
};

/// Top-level backend target configuration.
struct BackendEndConfigEntry {
  /// Set the `GEN_TARGET_BACKEND` var if provided.
  std::optional<bool> GenTargetBackend;
  /// Enable/disable the library mode if provide.
  std::optional<bool> LibraryMode;
  /// IR lowering configuration (hardware REST QPU)
  std::string PlatformLoweringConfig;
  /// Codegen emission configuration (hardware REST QPU)
  std::string CodegenEmission;
  /// Post code generation IR passes configuration (hardware REST QPU)
  std::string PostCodeGenPasses;
  /// Name of the platform library to use
  std::string PlatformLibrary;
  /// Name of the execution manager to use in library mode
  std::string LibraryModeExecutionManager;
  /// Name of the platform QPU implementation
  std::string PlatformQpu;
  /// Preprocessor defines for this target
  std::vector<std::string> PreprocessorDefines;
  /// Compiler flags for this target
  std::vector<std::string> CompilerFlags;
  /// Extra libraries to be linked in if any
  std::vector<std::string> LinkLibs;
  /// Extra linker flags for this target if any
  std::vector<std::string> LinkerFlags;
  /// Name of the NVQIR simulator backend(s)
  // If more than one are provided, the preceding one will be used if its .so
  // file can be located.
  SimulationBackendSetting SimulationBackend;
  /// Any conditional compile/link flags configurations.
  std::vector<ConditionalBuildConfig> ConditionalBuildConfigs;
};

/// Feature option mapping for NVIDIA target.
// For the unified `nvidia` target, users can use feature option to specify
// precision, mgpu/mqpu distribution. e.g., `--target-option fp32,mgpu` or
// `set_target('nvidia', option="fp32,mgpu")` would configure the C++ or Python
// execution to the single-precision, multi-GPU configuration.
struct BackendFeatureMap {
  /// Readable name for this configuration.
  std::string Name;
  /// The feature flag which trigger this configuration.
  TargetFeatureFlag Flags;
  /// Is it the default one when no option is present?
  std::optional<bool> Default;
  /// The full configuration for this option.
  BackendEndConfigEntry Config;
};

/// Schema of the target configuration file.
struct TargetConfig {
  /// Target name
  std::string Name;
  /// Target description
  std::string Description;
  /// Any warning messages to be printed when this target is selected
  // This could be used for deprecating messages.
  std::string WarningMsg;
  /// List of arguments that this target supports.
  std::vector<TargetArgument> TargetArguments;
  /// Does this target require GPU(s)?
  // If set, GPU presence will be checked at compile time.
  bool GpuRequired = false;
  /// Target configuration
  std::optional<BackendEndConfigEntry> BackendConfig;
  /// Additional configuration mapping (if this is a multi-configuration target)
  std::vector<BackendFeatureMap> ConfigMap;
};

/// Process the target configuration into a `nvq++` compatible script according
/// to the provided compile time (C++)/runtime (Python) target arguments.
std::string processRuntimeArgs(const TargetConfig &config,
                               const std::vector<std::string> &targetArgv);
} // namespace config
} // namespace cudaq

// These structs can be used in a vector.
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::TargetFeatureFlag)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::TargetArgument)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::ConditionalBuildConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::BackendFeatureMap)

namespace llvm {
namespace yaml {
// YML serialization declarations.
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
