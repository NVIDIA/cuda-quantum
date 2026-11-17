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
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>

/// Convert the backend config entry into nvq++ compatible script.
/// `pipelineName` is the name the lowering's override pass pipeline (if any)
/// was registered under by `cudaq-opt`. The script emits that *name*, never
/// the pipeline's own raw text. This eliminates nvq++ carrying raw
/// pass-pipeline text through its environment.
static std::string processSimBackendConfig(
    const std::string &targetName, const std::string &pipelineName,
    const cudaq::config::BackendEndConfigEntry &configValue) {
  std::stringstream output;
  // These default to false in nvq++; only an explicit `true` is emitted.
  if (configValue.GenTargetBackend)
    output << "GEN_TARGET_BACKEND=true\n";

  if (configValue.LibraryMode)
    output << "LIBRARY_MODE=true\n";

  if (configValue.CompileTarget.getOverridePassPipeline() != nullptr)
    output << "TARGET_PASS_PIPELINE_NAME=" << pipelineName << "\n";

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

/// Returns the platform extra arguments that \p args give: the value of each
/// argument that forwards to a platform key.
static std::string
platformExtraArgs(const cudaq::config::TargetConfig &config,
                  const std::map<std::string, std::string> &args) {
  std::string extraArgs;
  for (const auto &[argKey, argVal] : args) {
    const auto *argument = config.findArgument(argKey);
    if (argument &&
        argument->Type != cudaq::config::ArgumentType::option_flags &&
        !argument->PlatformArgKey.empty())
      extraArgs += ";" + argument->PlatformArgKey + ";" + argVal;
  }
  return extraArgs;
}

std::string cudaq::config::processRuntimeArgs(
    const cudaq::config::TargetConfig &config,
    const std::map<std::string, std::string> &args) {
  std::stringstream output;
  // This function writes the script that the `nvq++` driver sources, and its
  // callers do not handle exceptions, so a selection error ends the process
  // as it always has.
  const cudaq::config::BackendFeatureMap *featureConfig = nullptr;
  try {
    featureConfig = config.selectFeatureConfig(args);
  } catch (const std::runtime_error &error) {
    std::cerr << error.what() << "\n";
    std::abort();
  }
  const cudaq::config::BackendEndConfigEntry *backend = nullptr;
  if (featureConfig)
    backend = &featureConfig->Config;
  else if (config.ConfigMap.empty() && config.BackendConfig)
    backend = &*config.BackendConfig;

  if (backend) {
    std::string pipelineName = "target-pass-pipeline-" + config.Name;
    if (featureConfig)
      pipelineName += "." + featureConfig->Name;
    output << processSimBackendConfig(config.Name, pipelineName, *backend);
  } else if (!config.ConfigMap.empty()) {
    std::cerr << "Target '" << config.Name
              << "' has no configuration for the requested options.\n";
    std::abort();
  }

  const auto extraArgs = platformExtraArgs(config, args);
  if (!extraArgs.empty())
    output << "PLATFORM_EXTRA_ARGS=\"${PLATFORM_EXTRA_ARGS}" << extraArgs
           << "\"\n";

  return output.str();
}
