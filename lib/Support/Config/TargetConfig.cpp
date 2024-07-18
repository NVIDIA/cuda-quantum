/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Support/TargetConfig.h"
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
                        {"mqpu", cudaq::config::flagsMqpu}};
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

  if (!configValue.PlatformLoweringConfig.empty())
    output << "PLATFORM_LOWERING_CONFIG=\""
           << configValue.PlatformLoweringConfig << "\"\n";

  if (!configValue.CodegenEmission.empty())
    output << "CODEGEN_EMISSION=" << configValue.CodegenEmission << "\n";

  if (!configValue.PostCodeGenPasses.empty())
    output << "POST_CODEGEN_PASSES=\"" << configValue.PostCodeGenPasses
           << "\"\n";

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
    output << "if [ -f \"${install_dir}/lib/libnvqir-"
           << configValue.SimulationBackend.values.front() << ".so\" ]; then\n";
    output << "  NVQIR_SIMULATION_BACKEND=\""
           << configValue.SimulationBackend.values.front() << "\"\n";
    // If there are more than one simulator libs, create the `else` paths to
    // check their .so files.
    for (std::size_t i = 1; i < configValue.SimulationBackend.values.size();
         ++i) {
      output << "elif [ -f \"${install_dir}/lib/libnvqir-"
             << configValue.SimulationBackend.values[i] << ".so\" ]; then\n";
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

std::string
cudaq::config::processRuntimeArgs(const cudaq::config::TargetConfig &config,
                                  const std::vector<std::string> &targetArgv) {
  std::stringstream output;
  if (config.BackendConfig.has_value())
    output << processSimBackendConfig(config.Name,
                                      config.BackendConfig.value());

  unsigned featureFlag = 0;
  std::stringstream platformExtraArgs;
  for (std::size_t idx = 0; idx < targetArgv.size();) {
    const auto argsStr = targetArgv[idx];
    const auto iter = std::find_if(
        config.TargetArguments.begin(), config.TargetArguments.end(),
        [&](const cudaq::config::TargetArgument &argConfig) {
          // Here, we handle both cases: the config key as is (python kwargs)
          // or prefixed with the target name or "target".
          const std::string nvqppArgKey =
              "--" + config.Name + "-" + argConfig.KeyName;
          const std::string targetPrefixArgKey =
              "--target-" + argConfig.KeyName;
          return (nvqppArgKey == argsStr) || (targetPrefixArgKey == argsStr) ||
                 (argsStr == argConfig.KeyName);
        });
    if (iter != config.TargetArguments.end()) {
      if (iter->Type != cudaq::config::ArgumentType::FeatureFlag) {
        // If this is a platform option (platform argument key is provide),
        // forward the value to the platform extra arguments.
        if (!iter->PlatformArgKey.empty() && idx + 1 < targetArgv.size())
          platformExtraArgs << ";" << iter->PlatformArgKey << ";"
                            << targetArgv[idx + 1];
      } else if (idx + 1 < targetArgv.size()) {
        // This is an option flag, construct the value for mapping selection.
        const auto featureFlags = targetArgv[idx + 1];
        llvm::SmallVector<llvm::StringRef> flagStrs;
        llvm::StringRef(featureFlags).split(flagStrs, ',', -1, false);
        for (const auto &flag : flagStrs) {
          const auto iter = stringToFeatureFlag.find(flag.str());
          if (iter == stringToFeatureFlag.end()) {
            llvm::errs() << "Unknown  feature flag '" << flag << "'\n";
            abort();
          }
          featureFlag += iter->second;
        }
      }
    }
    // We assume the arguments are given as '<key> <value>' pairs.
    idx += 2;
  }

  if (!config.ConfigMap.empty()) {
    const auto defaultFeatureIter = std::find_if(
        config.ConfigMap.begin(), config.ConfigMap.end(),
        [&](const cudaq::config::BackendFeatureMap &entry) {
          return entry.Default.has_value() && entry.Default.value();
        });

    const uint64_t defaultFlag =
        (defaultFeatureIter != config.ConfigMap.end())
            ? static_cast<uint64_t>(defaultFeatureIter->Flags)
            : 0;

    const auto iter = [&]() {
      // If the command line set the feature flag, find it in the config map.
      // Otherwise, find the default.
      return featureFlag > 0
                 ? std::find_if(
                       config.ConfigMap.begin(), config.ConfigMap.end(),
                       [&](const cudaq::config::BackendFeatureMap &entry) {
                         // Mapping selection: exact match + implicit default
                         // match. e.g., if the default is fp32, `option=mqpu`
                         // is the same as `option=fp32,mqpu`. The config map
                         // entry associated with 'mqpu,fp32' will be activated.
                         return featureFlag == entry.Flags ||
                                (featureFlag | defaultFlag) == entry.Flags;
                       })
                 : std::find_if(
                       config.ConfigMap.begin(), config.ConfigMap.end(),
                       [&](const cudaq::config::BackendFeatureMap &entry) {
                         // No option flag was provided, find the default
                         // config.
                         return entry.Default.has_value() &&
                                entry.Default.value();
                       });
    }();
    if (iter == config.ConfigMap.end()) {
      llvm::errs() << "Unable to find a config entry for feature flag value "
                   << featureFlag << ".\n";
      llvm::errs() << "This indicates the requested combination of features "
                      "is not supported.\n";
      abort();
    }
    output << processSimBackendConfig(config.Name, iter->Config);
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
}

void MappingTraits<cudaq::config::TargetArgument>::mapping(
    IO &io, cudaq::config::TargetArgument &info) {
  io.mapRequired("key", info.KeyName);
  io.mapOptional("required", info.IsRequired);
  io.mapOptional("platform-arg", info.PlatformArgKey);
  io.mapOptional("help-string", info.HelpString);
  io.mapOptional("type", info.Type);
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
  io.mapOptional("platform-lowering-config", info.PlatformLoweringConfig);
  io.mapOptional("codegen-emission", info.CodegenEmission);
  io.mapOptional("post-codegen-passes", info.PostCodeGenPasses);
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

} // namespace yaml
} // namespace llvm
