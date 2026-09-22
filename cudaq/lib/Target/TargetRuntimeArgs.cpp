/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Runtime-argument processing and nvq++ backend-script emission for target
// configurations, extracted from the legacy YAML implementation
// (Yaml/TargetConfigYaml.cpp). Standard C++ only; no LLVM dependencies.

#include "TargetConfigHelper.h"
#include "cudaq/Target/TargetPluginLibrary.h"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace {
// CLI spellings of the `option-flags` feature bits. Keep in sync with the
// legacy YAML traits table in Yaml/TargetConfigYaml.cpp.
static std::unordered_map<std::string, cudaq::config::TargetFeatureFlag>
    stringToFeatureFlag{
        {"fp32", cudaq::config::TargetFeatureFlag::fp32},
        {"fp64", cudaq::config::TargetFeatureFlag::fp64},
        {"mgpu", cudaq::config::TargetFeatureFlag::mgpu},
        {"mqpu", cudaq::config::TargetFeatureFlag::mqpu},
        {"dep-analysis", cudaq::config::TargetFeatureFlag::dep_analysis},
        {"qpp", cudaq::config::TargetFeatureFlag::qpp}};

/// Split `value` on commas, discarding empty segments.
std::vector<std::string> splitFeatureFlags(const std::string &value) {
  std::vector<std::string> flags;
  std::size_t pos = 0;
  while (pos <= value.size()) {
    const auto comma = value.find(',', pos);
    const auto end = comma == std::string::npos ? value.size() : comma;
    if (end > pos)
      flags.emplace_back(value.substr(pos, end - pos));
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return flags;
}
} // namespace

/// Convert the backend config entry into nvq++ compatible script.
/// `pipelineName` is the name `TargetPassPipeline` (if any) was registered
/// under by `cudaq-opt`. The script emits that *name*, never the pipeline's own
/// raw text. This eliminates nvq++ carrying raw pass-pipeline text through its
/// environment.
static std::string processSimBackendConfig(
    const std::string &targetName, const std::string &pipelineName,
    const cudaq::config::BackendEndConfigEntry &configValue) {
  std::stringstream output;
  // These default to false in nvq++; only an explicit `true` is emitted.
  if (configValue.GenTargetBackend)
    output << "GEN_TARGET_BACKEND=true\n";

  if (configValue.LibraryMode)
    output << "LIBRARY_MODE=true\n";

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
    output << "TARGET_PASS_PIPELINE_NAME=" << pipelineName << "\n";

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
    // Use platform-appropriate shared library extension
    const std::string libExt{cudaq::config::kSharedLibraryExtension};
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

namespace {
struct ParsedTargetArgs {
  std::string platformExtraArgs;
  const cudaq::config::BackendFeatureMap *featureConfig = nullptr;
  const cudaq::config::BackendEndConfigEntry *backend = nullptr;
};
} // namespace

static ParsedTargetArgs
parseTargetArgs(const cudaq::config::TargetConfig &config,
                const std::map<std::string, std::string> &args) {
  unsigned featureFlag = 0;
  ParsedTargetArgs parsed;
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
          return argKey == nvqppArgKey || argKey == targetPrefixArgKey ||
                 argKey == argConfig.KeyName;
        });
    if (iter != config.TargetArguments.end()) {
      if (iter->Type != cudaq::config::ArgumentType::option_flags) {
        // If this is a platform option (platform argument key is provide),
        // forward the value to the platform extra arguments.
        if (!iter->PlatformArgKey.empty())
          parsed.platformExtraArgs += ";" + iter->PlatformArgKey + ";" + argVal;
      } else {
        // This is an option flag, construct the value for mapping selection.
        for (const auto &flag : splitFeatureFlags(argVal)) {
          const auto iter = stringToFeatureFlag.find(flag);
          if (iter == stringToFeatureFlag.end()) {
            std::cerr << "Unknown feature flag '" << flag << "'\n";
            std::abort();
          }
          featureFlag |= static_cast<unsigned>(iter->second);
        }
      }
    }
  }

  if (!config.ConfigMap.empty()) {
    const auto defaultFeatureIter =
        std::find_if(config.ConfigMap.begin(), config.ConfigMap.end(),
                     [&](const cudaq::config::BackendFeatureMap &entry) {
                       return entry.Default;
                     });

    const unsigned defaultFlag =
        (defaultFeatureIter != config.ConfigMap.end())
            ? static_cast<unsigned>(defaultFeatureIter->Flags)
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
                         return featureFlag ==
                                    static_cast<unsigned>(entry.Flags) ||
                                (featureFlag | defaultFlag) ==
                                    static_cast<unsigned>(entry.Flags);
                       })
                 : std::find_if(
                       config.ConfigMap.begin(), config.ConfigMap.end(),
                       [&](const cudaq::config::BackendFeatureMap &entry) {
                         // No option flag was provided, find the default
                         // config.
                         return entry.Default;
                       });
    }();
    if (iter != config.ConfigMap.end()) {
      parsed.featureConfig = &*iter;
      parsed.backend = &iter->Config;
    }
  } else if (config.BackendConfig.has_value()) {
    parsed.backend = &*config.BackendConfig;
  }

  return parsed;
}

const cudaq::config::BackendEndConfigEntry *
cudaq::config::selectBackend(const cudaq::config::TargetConfig &config,
                             const std::map<std::string, std::string> &args) {
  return parseTargetArgs(config, args).backend;
}

std::string cudaq::config::processRuntimeArgs(
    const cudaq::config::TargetConfig &config,
    const std::map<std::string, std::string> &args) {
  std::stringstream output;
  const auto parsed = parseTargetArgs(config, args);

  if (parsed.backend) {
    std::string pipelineName = "target-pass-pipeline-" + config.Name;
    if (parsed.featureConfig)
      pipelineName += "." + parsed.featureConfig->Name;
    output << processSimBackendConfig(config.Name, pipelineName,
                                      *parsed.backend);
  } else if (!config.ConfigMap.empty()) {
    std::cerr << "Unable to find a config entry for the requested feature "
                 "flags.\n";
    std::cerr << "This indicates the requested combination of features "
                 "is not supported.\n";
    std::abort();
  }

  if (!parsed.platformExtraArgs.empty())
    output << "PLATFORM_EXTRA_ARGS=\"${PLATFORM_EXTRA_ARGS}"
           << parsed.platformExtraArgs << "\"\n";

  return output.str();
}
