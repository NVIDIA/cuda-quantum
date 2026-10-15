/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cudaq::config {
/// Flag to enable feature(s) of the unified NVIDIA target.
// Bit values, so that different options can be combined, e.g., multi-gpu with
// fp32/64.
enum class TargetFeatureFlag : unsigned {
  fp32 = 0x0001,
  fp64 = 0x0002,
  mgpu = 0x0004,
  mqpu = 0x0008,
  dep_analysis = 0x0010,
  qpp = 0x0020,
};

/// OR the underlying values of `flags` into a single feature mask.
inline unsigned combineFeatureFlags(std::span<const TargetFeatureFlag> flags) {
  unsigned mask = 0;
  for (const auto flag : flags)
    mask |= static_cast<unsigned>(flag);
  return mask;
}

/// @brief Configuration argument type annotation
// e.g., to support type validation.
enum class ArgumentType { string, integer, uuid, option_flags, machine_config };

/// @brief Architecture-specific compilation settings
// Different device architectures of a target may require customization.
// This is a subset of `BackendEndConfigEntry` that we provide
// architecture-specific customization.
struct TargetArchitectureSettings {
  /// Codegen emission specification
  std::string CodegenEmission;
};

/// @brief Specify architecture matching and customization
// This data can be attached to whatever CLI target argument field for
// device/machine name.
struct MachineArchitectureConfig {
  // Each architecture can be given a name, e.g., gen 1, gen 2
  std::string Name;
  // A list of device/machine names to be matched to this architecture
  std::vector<std::string> MachineNames;
  // A regex pattern for matching.
  // If both machine names and regex are provided, we match the list first
  // before using regex pattern.
  std::string MachinePattern;
  // Architecture-specific settings.
  TargetArchitectureSettings Configuration;
};

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
  ArgumentType Type = ArgumentType::string;
  /// Machine configuration (optional, valid if this argument is for a machine
  /// configuration specification)
  std::vector<MachineArchitectureConfig> MachineConfigs;
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
  /// Set the `GEN_TARGET_BACKEND` var.
  bool GenTargetBackend = false;
  /// Enable/disable the library mode.
  bool LibraryMode = false;
  /// Whether the target supports pre-computing resource counts for faster
  /// resource estimation.
  bool SupportResourceCounts = true;
  /// IR lowering configuration (hardware REST QPU)
  std::string JITHighLevelPipeline;
  std::string JITMidLevelPipeline;
  std::string JITLowLevelPipeline;
  /// Exact cudaq-opt passes for pseudo-targets
  std::string TargetPassPipeline;
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
  /// List of plugin shared libraries to dlopen at runtime. Names may omit the
  /// platform suffix (e.g. `"libcudaq-rest-qpu"i`); it is appended for the
  /// current platform when the library is resolved. Relative names are looked
  /// up in the CUDA-Q library directory and the target plugin's lib directory.
  std::vector<std::string> PluginLibraries;
  /// Extra linker flags for this target if any
  std::vector<std::string> LinkerFlags;
  /// Name of the NVQIR simulator backend(s)
  // If more than one are provided, the preceding one will be used if its .so
  // file can be located.
  SimulationBackendSetting SimulationBackend;
  /// Any conditional compile/link flags configurations.
  std::vector<ConditionalBuildConfig> ConditionalBuildConfigs;

  /// Returns true if any pass pipeline fields are configured.
  bool hasPassPipeline() const;
};

/// Feature option mapping for NVIDIA target.
// For the unified `nvidia` target, users can use feature option to specify
// precision, mgpu/mqpu distribution. e.g., `--target-option fp32,mgpu` or
// `set_target('nvidia', option="fp32,mgpu")` would configure the C++ or Python
// execution to the single-precision, multi-GPU configuration.
struct BackendFeatureMap {
  /// Readable name for this configuration.
  std::string Name;
  /// The feature flags (combined mask) which trigger this configuration.
  TargetFeatureFlag Flags;
  /// Is it the default one when no option is present?
  bool Default = false;
  /// The full configuration for this option.
  BackendEndConfigEntry Config;
};

/// Schema of the target configuration file.
class TargetConfig {
public:
  /// CUDA-Q version against which an external plugin target was built and
  /// tested. This is optional for in-tree targets and required when selecting
  /// an external plugin target.
  std::string CudaqVersion;
  /// All plugin libraries for this target (flattened from BackendConfig if
  /// present)
  std::vector<std::string> PluginLibraries;
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

  // Helper to determine the codegen config based on CLI arguments
  std::string
  getCodeGenSpec(const std::map<std::string, std::string> &targetArgs) const;
};

/// Result of checking an external target's CUDA-Q compatibility metadata.
enum class TargetVersionCompatibility { Compatible, Warning, Error };

struct TargetVersionCompatibilityResult {
  TargetVersionCompatibility Status = TargetVersionCompatibility::Compatible;
  std::string Diagnostic;
};

/// Validate an external plugin target against the current CUDA-Q version.
/// Exact matches are compatible; any version difference produces a warning.
TargetVersionCompatibilityResult
checkExternalTargetVersion(const TargetConfig &config,
                           std::string_view currentVersion,
                           const std::filesystem::path &configPath = {});

} // namespace cudaq::config
