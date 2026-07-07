/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Target/TargetConfigYaml.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/PrettyStackTrace.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <sstream>

#define DEBUG_TYPE "target-config"

using namespace llvm;

namespace {
static std::unordered_map<std::string, cudaq::config::TargetFeatureFlag>
    stringToFeatureFlag{{"fp32", cudaq::config::flagsFP32},
                        {"fp64", cudaq::config::flagsFP64},
                        {"mgpu", cudaq::config::flagsMgpu},
                        {"mqpu", cudaq::config::flagsMqpu},
                        {"dep-analysis", cudaq::config::flagsDepAnalysis},
                        {"qpp", cudaq::config::flagsQPP}};
}

/// @brief Convert the backend config entry into nvq++ compatible script.
static std::string processSimBackendConfig(
    const std::string &targetName,
    const cudaq::config::BackendEndConfigEntry &configValue) {
  std::stringstream output;
  if (configValue.GenTargetBackend.has_value())
    output << "GEN_TARGET_BACKEND="
           << (configValue.GenTargetBackend.value() ? "true" : "false") << "\n";

  if (configValue.LibraryMode.has_value())
    output << "LIBRARY_MODE="
           << (configValue.LibraryMode.value() ? "true" : "false") << "\n";

  if (!configValue.JITHighLevelPipeline.empty())
    output << "JIT_HIGH_LEVEL_PIPELINE=\"" << configValue.JITHighLevelPipeline
           << "\"\n";

  if (!configValue.JITMidLevelPipeline.empty())
    output << "JIT_MID_LEVEL_PIPELINE=\"" << configValue.JITMidLevelPipeline
           << "\"\n";

  if (!configValue.JITLowLevelPipeline.empty())
    output << "JIT_LOW_LEVEL_PIPELINE=\"" << configValue.JITLowLevelPipeline
           << "\"\n";

  if (!configValue.TargetPassPipeline.empty())
    output << "TARGET_PASS_PIPELINE=\"" << configValue.TargetPassPipeline
           << "\"\n";

  if (!configValue.CodegenEmission.empty())
    output << "CODEGEN_EMISSION=" << configValue.CodegenEmission << "\n";

  if (!configValue.PostCodeGenPasses.empty())
    output << "POST_CODEGEN_PASSES=\"" << configValue.PostCodeGenPasses
           << "\"\n";

  if (configValue.SupportsNegatedControls)
    output << "CUDAQ_TRANSLATE_ARGS=\"${CUDAQ_TRANSLATE_ARGS} "
              "--preserve-gate-control-polarity\"\n";

  if (!configValue.PlatformLibrary.empty())
    output << "PLATFORM_LIBRARY=" << configValue.PlatformLibrary << "\n";

  if (!configValue.LibraryModeExecutionManager.empty())
    output << "LIBRARY_MODE_EXECUTION_MANAGER="
           << configValue.LibraryModeExecutionManager << "\n";

  if (!configValue.PlatformQpu.empty())
    output << "PLATFORM_QPU=" << configValue.PlatformQpu << "\n";

  if (!configValue.PreprocessorDefines.empty()) {
    output << "PREPROCESSOR_DEFINES=\"${PREPROCESSOR_DEFINES}";

    for (const auto &def : configValue.PreprocessorDefines)
      output << " " << def;

    output << "\"\n";
  }

  if (!configValue.CompilerFlags.empty()) {
    output << "COMPILER_FLAGS=\"${COMPILER_FLAGS}";

    for (const auto &def : configValue.CompilerFlags)
      output << " " << def;

    output << "\"\n";
  }

  if (!configValue.LinkLibs.empty()) {
    output << "LINKLIBS=\"${LINKLIBS}";

    for (const auto &lib : configValue.LinkLibs)
      output << " " << lib;

    output << "\"\n";
  }

  if (!configValue.LinkerFlags.empty()) {
    output << "LINKER_FLAGS=\"${LINKER_FLAGS}";
    for (const auto &def : configValue.LinkerFlags)
      output << " " << def;

    output << "\"\n";
  }

  if (!configValue.SimulationBackend.values.empty()) {
    // Use platform-appropriate shared library extension
#ifdef __APPLE__
    constexpr const char *libExt = ".dylib";
#else
    constexpr const char *libExt = ".so";
#endif
    output << "if [ -f \"${install_dir}/lib/libnvqir-"
           << configValue.SimulationBackend.values.front() << libExt
           << "\" ]; then\n";
    output << "  NVQIR_SIMULATION_BACKEND=\""
           << configValue.SimulationBackend.values.front() << "\"\n";
    // If there are more than one simulator libs, create the `else` paths to
    // check their library files.
    for (std::size_t i = 1; i < configValue.SimulationBackend.values.size();
         ++i) {
      output << "elif [ -f \"${install_dir}/lib/libnvqir-"
             << configValue.SimulationBackend.values[i] << libExt
             << "\" ]; then\n";
      output << "  NVQIR_SIMULATION_BACKEND=\""
             << configValue.SimulationBackend.values[i] << "\"\n";
    }
    output << "else\n";
    output << "  error_exit=\"Unable to find NVQIR simulator lib for target "
           << targetName << ". Please check your installation.\"\n";
    output << "fi\n";
  }

  for (const auto &rule : configValue.ConditionalBuildConfigs) {
    output << "if [[ " << rule.Condition << " ]]; then\n";
    if (!rule.CompileFlag.empty())
      output << "  COMPILER_FLAGS=\"${COMPILER_FLAGS} " << rule.CompileFlag
             << "\"\n";

    if (!rule.LinkFlag.empty())
      output << "  LINKER_FLAGS=\"${LINKER_FLAGS} " << rule.LinkFlag << "\"\n";

    output << "fi\n";
  }

  return output.str();
}

// Compute the combined feature-flag bitset from the runtime/CLI args, using the
// same arg-key forms processRuntimeArgs accepts. Shared with
// `getActiveBackendConfig` so configuration-matrix selection stays consistent.
static unsigned
computeFeatureFlag(const cudaq::config::TargetConfig &config,
                   const std::map<std::string, std::string> &args) {
  unsigned featureFlag = 0;
  for (const auto &[argKey, argVal] : args) {
    const auto iter = std::find_if(
        config.TargetArguments.begin(), config.TargetArguments.end(),
        [&](const cudaq::config::TargetArgument &argConfig) {
          const std::string nvqppArgKey =
              "--" + config.Name + "-" + argConfig.KeyName;
          const std::string targetPrefixArgKey =
              "--target-" + argConfig.KeyName;
          return llvm::is_contained<llvm::StringRef>(
              {nvqppArgKey, targetPrefixArgKey, argConfig.KeyName}, argKey);
        });
    if (iter == config.TargetArguments.end() ||
        iter->Type != cudaq::config::ArgumentType::FeatureFlag)
      continue;
    llvm::SmallVector<llvm::StringRef> flagStrs;
    llvm::StringRef(argVal).split(flagStrs, ',', -1, false);
    for (const auto &flag : flagStrs) {
      const auto flagIter = stringToFeatureFlag.find(flag.str());
      if (flagIter == stringToFeatureFlag.end()) {
        llvm::errs() << "Unknown  feature flag '" << flag << "'\n";
        abort();
      }
      featureFlag += flagIter->second;
    }
  }
  return featureFlag;
}

// Select the configuration-matrix entry activated by `featureFlag` (exact match
// or default-implied match), or null if none / no matrix. Shared selection.
static const cudaq::config::BackendFeatureMap *
selectConfigMapEntry(const cudaq::config::TargetConfig &config,
                     unsigned featureFlag) {
  if (config.ConfigMap.empty())
    return nullptr;
  const auto defaultFeatureIter =
      std::find_if(config.ConfigMap.begin(), config.ConfigMap.end(),
                   [&](const cudaq::config::BackendFeatureMap &entry) {
                     return entry.Default.has_value() && entry.Default.value();
                   });
  const uint64_t defaultFlag =
      (defaultFeatureIter != config.ConfigMap.end())
          ? static_cast<uint64_t>(defaultFeatureIter->Flags)
          : 0;
  // Exact match + implicit-default match: with default fp32, `option=mqpu`
  // resolves the same as `option=fp32,mqpu`. With no option flag, pick default.
  const auto iter =
      featureFlag > 0
          ? std::find_if(config.ConfigMap.begin(), config.ConfigMap.end(),
                         [&](const cudaq::config::BackendFeatureMap &entry) {
                           return featureFlag == entry.Flags ||
                                  (featureFlag | defaultFlag) == entry.Flags;
                         })
          : std::find_if(config.ConfigMap.begin(), config.ConfigMap.end(),
                         [&](const cudaq::config::BackendFeatureMap &entry) {
                           return entry.Default.has_value() &&
                                  entry.Default.value();
                         });
  return iter != config.ConfigMap.end() ? &*iter : nullptr;
}

std::optional<cudaq::config::BackendEndConfigEntry>
cudaq::config::TargetConfig::getActiveBackendConfig(
    const std::map<std::string, std::string> &args) const {
  if (const auto *entry =
          selectConfigMapEntry(*this, computeFeatureFlag(*this, args)))
    return entry->Config;
  if (BackendConfig.has_value())
    return BackendConfig;
  return std::nullopt;
}

std::string cudaq::config::processRuntimeArgs(
    const cudaq::config::TargetConfig &config,
    const std::map<std::string, std::string> &args) {
  std::stringstream output;
  if (config.BackendConfig.has_value())
    output << processSimBackendConfig(config.Name,
                                      config.BackendConfig.value());

  const unsigned featureFlag = computeFeatureFlag(config, args);
  std::stringstream platformExtraArgs;
  for (const auto &[argKey, argVal] : args) {
    const auto iter = std::find_if(
        config.TargetArguments.begin(), config.TargetArguments.end(),
        [&](const cudaq::config::TargetArgument &argConfig) {
          // Here, we handle both cases: the config key as is (python kwargs)
          // or prefixed with the target name or "target".
          const std::string nvqppArgKey =
              "--" + config.Name + "-" + argConfig.KeyName;
          const std::string targetPrefixArgKey =
              "--target-" + argConfig.KeyName;
          return llvm::is_contained<llvm::StringRef>(
              {nvqppArgKey, targetPrefixArgKey, argConfig.KeyName}, argKey);
        });
    // Feature flags drive configuration-matrix selection (computeFeatureFlag,
    // applied below). Forward any other platform option to the platform.
    if (iter != config.TargetArguments.end() &&
        iter->Type != cudaq::config::ArgumentType::FeatureFlag &&
        !iter->PlatformArgKey.empty())
      platformExtraArgs << ";" << iter->PlatformArgKey << ";" << argVal;
  }

  if (!config.ConfigMap.empty()) {
    const auto *entry = selectConfigMapEntry(config, featureFlag);
    if (!entry) {
      llvm::errs() << "Unable to find a config entry for feature flag value "
                   << featureFlag << ".\n";
      llvm::errs() << "This indicates the requested combination of features "
                      "is not supported.\n";
      abort();
    }
    output << processSimBackendConfig(config.Name, entry->Config);
  }
  const auto platformExtraArgsStr = platformExtraArgs.str();
  if (!platformExtraArgsStr.empty())
    output << "PLATFORM_EXTRA_ARGS=\"${PLATFORM_EXTRA_ARGS}"
           << platformExtraArgsStr << "\"\n";

  return output.str();
}

namespace llvm {
namespace yaml {
void ScalarBitSetTraits<cudaq::config::TargetFeatureFlag>::bitset(
    IO &io, cudaq::config::TargetFeatureFlag &value) {
  for (const auto &[k, v] : stringToFeatureFlag)
    io.bitSetCase(value, k.c_str(), v);
}

void ScalarEnumerationTraits<cudaq::config::ArgumentType>::enumeration(
    IO &io, cudaq::config::ArgumentType &value) {
  io.enumCase(value, "string", cudaq::config::ArgumentType::String);
  io.enumCase(value, "integer", cudaq::config::ArgumentType::Int);
  io.enumCase(value, "uuid", cudaq::config::ArgumentType::UUID);
  io.enumCase(value, "option-flags", cudaq::config::ArgumentType::FeatureFlag);
  io.enumCase(value, "machine-config",
              cudaq::config::ArgumentType::MachineConfig);
}

void MappingTraits<cudaq::config::TargetArgument>::mapping(
    IO &io, cudaq::config::TargetArgument &info) {
  io.mapRequired("key", info.KeyName);
  io.mapOptional("required", info.IsRequired);
  io.mapOptional("platform-arg", info.PlatformArgKey);
  io.mapOptional("help-string", info.HelpString);
  io.mapOptional("type", info.Type);
  io.mapOptional("machine-config", info.MachineConfigs);
}

std::string MappingTraits<cudaq::config::TargetArgument>::validate(
    IO &io, cudaq::config::TargetArgument &info) {
  if (!info.MachineConfigs.empty() &&
      info.Type != cudaq::config::ArgumentType::MachineConfig)
    return "If 'machine-config' is provided, 'type' must be 'machine-config'.";
  return "";
}

void BlockScalarTraits<cudaq::config::SimulationBackendSetting>::output(
    const cudaq::config::SimulationBackendSetting &Value, void *Ctxt,
    llvm::raw_ostream &OS) {
  std::size_t idx = 0;
  for (const auto &val : Value.values) {
    OS << val;
    if (idx != Value.values.size() - 1)
      OS << ", ";

    ++idx;
  }
}

StringRef BlockScalarTraits<cudaq::config::SimulationBackendSetting>::input(
    StringRef Scalar, void *Ctxt,
    cudaq::config::SimulationBackendSetting &Value) {
  llvm::SmallVector<llvm::StringRef> values;
  Scalar.split(values, ',', -1, false);
  for (const auto &val : values)
    Value.values.emplace_back(val.trim().str());

  return StringRef();
}

void MappingTraits<cudaq::config::ConditionalBuildConfig>::mapping(
    IO &io, cudaq::config::ConditionalBuildConfig &info) {
  io.mapRequired("if", info.Condition);
  io.mapOptional("compiler-flag", info.CompileFlag);
  io.mapOptional("link-flag", info.LinkFlag);
}

void MappingTraits<cudaq::config::BackendEndConfigEntry>::mapping(
    IO &io, cudaq::config::BackendEndConfigEntry &info) {
  io.mapOptional("gen-target-backend", info.GenTargetBackend);
  io.mapOptional("library-mode", info.LibraryMode);
  io.mapOptional("jit-high-level-pipeline", info.JITHighLevelPipeline);
  io.mapOptional("jit-mid-level-pipeline", info.JITMidLevelPipeline);
  io.mapOptional("jit-low-level-pipeline", info.JITLowLevelPipeline);
  io.mapOptional("target-pass-pipeline", info.TargetPassPipeline);
  io.mapOptional("codegen-emission", info.CodegenEmission);
  io.mapOptional("post-codegen-passes", info.PostCodeGenPasses);
  io.mapOptional("supports-negated-controls", info.SupportsNegatedControls);
  io.mapOptional("platform-library", info.PlatformLibrary);
  io.mapOptional("library-mode-execution-manager",
                 info.LibraryModeExecutionManager);
  io.mapOptional("platform-qpu", info.PlatformQpu);
  io.mapOptional("preprocessor-defines", info.PreprocessorDefines);
  io.mapOptional("compiler-flags", info.CompilerFlags);
  io.mapOptional("link-libs", info.LinkLibs);
  io.mapOptional("linker-flags", info.LinkerFlags);
  io.mapOptional("nvqir-simulation-backend", info.SimulationBackend);
  io.mapOptional("rules", info.ConditionalBuildConfigs);
}

void MappingTraits<cudaq::config::BackendFeatureMap>::mapping(
    IO &io, cudaq::config::BackendFeatureMap &info) {
  io.mapRequired("name", info.Name);
  io.mapRequired("option-flags", info.Flags);
  io.mapOptional("default", info.Default);
  io.mapRequired("config", info.Config);
}

void MappingTraits<cudaq::config::TargetConfig>::mapping(
    IO &io, cudaq::config::TargetConfig &info) {
  io.mapRequired("name", info.Name);
  io.mapRequired("description", info.Description);
  io.mapOptional("warning", info.WarningMsg);
  io.mapOptional("target-arguments", info.TargetArguments);
  io.mapOptional("gpu-requirements", info.GpuRequired);
  io.mapOptional("config", info.BackendConfig);
  io.mapOptional("configuration-matrix", info.ConfigMap);
}

std::string MappingTraits<cudaq::config::TargetConfig>::validate(
    IO &io, cudaq::config::TargetConfig &info) {
  // There should only ever be 1 machine-configuration entry in the target
  // arguments.
  unsigned count = 0;
  for (const auto &targetArg : info.TargetArguments) {
    if (targetArg.Type == cudaq::config::ArgumentType::MachineConfig)
      count++;
  }
  if (count > 1)
    return "There should only ever be 1 machine-configuration entry in the "
           "target arguments.";
  return std::string();
}

void MappingTraits<cudaq::config::TargetArchitectureSettings>::mapping(
    IO &io, cudaq::config::TargetArchitectureSettings &info) {
  io.mapOptional("codegen-emission", info.CodegenEmission);
}

void MappingTraits<cudaq::config::MachineArchitectureConfig>::mapping(
    IO &io, cudaq::config::MachineArchitectureConfig &info) {
  io.mapOptional("arch-name", info.Name);
  io.mapOptional("machine-names", info.MachineNames);
  io.mapOptional("pattern", info.MachinePattern);
  io.mapRequired("config", info.Configuration);
}

std::string MappingTraits<cudaq::config::MachineArchitectureConfig>::validate(
    IO &io, cudaq::config::MachineArchitectureConfig &info) {
  if (info.MachineNames.empty() && info.MachinePattern.empty())
    return "Either 'machine-names' or 'pattern' must be specified.";

  if (!info.MachinePattern.empty()) {
    // Check if this is a valid regex
    llvm::Regex re(info.MachinePattern);
    std::string errorIfAny;
    if (!re.isValid(errorIfAny)) {
      std::stringstream ss;
      ss << "'" << info.MachinePattern
         << "' is not a valid regex: " << errorIfAny;
      return ss.str();
    }
  }
  return std::string();
}
} // namespace yaml
} // namespace llvm
