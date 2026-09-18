/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file qpu_utils.h
/// @brief Utility functions for the CUDA-Q platforms to aimed at reducing
/// header file dependencies.
#include "cudaq/Target/TargetRegistry.h"
#include "cudaq/utils/owning_ptr.h"
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace cudaq {
class Executor;
class QuantumExecutionQueue;
class ServerHelper;
namespace detail {
/// @brief Decodes the base64-encoded string @p encoded and returns the
/// decoded result.  Throws std::runtime_error on malformed input.
std::string decodeBase64(const std::string &encoded);

/// @brief Retrieve a key-value option from a semicolon-delimited backend
/// configuration string. Values prefixed with `base64_` are decoded.
std::optional<std::string> getBackendConfigOption(const std::string &backend,
                                                  std::string_view key);

/// Return the explicitly configured target config path (a compiled plugin
/// library or YAML file for external targets), or @p fallback when the backend
/// configuration does not provide `__target_config_path`.
std::filesystem::path
getTargetConfigPath(const std::string &backend,
                    const std::filesystem::path &fallback);

/// Host facts for the current process (GPU count, CUDA-Q version, library
/// search path).
config::HostEnvironment currentHostEnvironment();

/// Target configuration resolved through `TargetRegistry` for a backend
/// string of the form `name[;key;value...]`.
struct ResolvedTargetConfig {
  std::string name;
  config::TargetConfig config;
  std::filesystem::path configPath;
  std::filesystem::path pluginLibDir;
  std::string simulatorName;
  std::string platformName;
  bool fp64Simulation;
};

/// Discover, load, and availability-check the target named by @p backend.
/// Throws `std::runtime_error` if the name is unknown or not available.
ResolvedTargetConfig resolveTargetConfig(const std::string &backend);

/// @brief Load runtime libraries owned by a target plugin. This loads every
/// YAML-declared plugin library and, when present, the conventional
/// `libcudaq-serverhelper-<target>` library.
void loadTargetPluginLibraries(const std::string &targetName,
                               const std::filesystem::path &configPath,
                               const config::TargetConfig &targetConfig);

/// Returns true if @p kernelName has the analog Hamiltonian kernel prefix.
bool isAnalogHamiltonianKernel(const std::string &kernelName);

/// @brief Look up the @c ServerHelper and @c Executor registered under
/// @p qpuName, initialize the server helper with @p backendConfig, wire it
/// into the executor, and populate the server helper's runtime target from
/// @p targetConfig and @p backendConfig. Throws @c std::runtime_error if no
/// @c ServerHelper is registered for @p qpuName.
void initServerHelperAndExecutor(
    const std::string &qpuName,
    const std::map<std::string, std::string> &backendConfig,
    const config::TargetConfig &targetConfig,
    owning_ptr<ServerHelper> &serverHelper,
    std::unique_ptr<Executor> &executor);

/// @brief Add an execution queue to the process-wide registry.
void registerExecutionQueue(QuantumExecutionQueue &queue);

/// @brief Remove an execution queue from the process-wide registry.
void unregisterExecutionQueue(QuantumExecutionQueue &queue);

/// @brief Shut down every execution queue alive in this process.
///
/// Note: Queues are reached directly rather than through the current platform:
/// changing targets leaves previous platforms alive with their queues loaded.
void shutdownExecutionQueues();
} // namespace detail

} // namespace cudaq
