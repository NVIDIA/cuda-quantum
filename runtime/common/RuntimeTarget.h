/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Support/TargetConfig.h"
#include "cudaq/host_config.h"
#include <filesystem>
#include <map>
#include <string>

namespace cudaq {

class ServerHelper;

/// @brief A RuntimeTarget encapsulates an available
/// backend simulator and quantum_platform for CUDA-Q
/// kernel execution.
struct RuntimeTarget {
  // Target name
  std::string name;
  // Simulator name (if this is a simulator target)
  std::string simulatorName;
  // Platform name
  std::string platformName;
  // Description
  std::string description;
  // Simulation precision
  simulation_precision precision;
  // Backend configuration (as specified in the target config YAML file)
  config::TargetConfig config;
  // The backend configuration map, used to store additional
  // key-value pairs for the backend configuration specified in the command-line
  // (C++) or the set_target call (Python).
  std::map<std::string, std::string> runtimeConfig;
  // Directory containing this target's plugin shared libraries (e.g.
  // `libcudaq-serverhelper-<name>.so`). For an external plugin this is
  // `<pkgRoot>/lib`; for an in-tree target it stays empty and the runtime
  // falls back to the default CUDA-Q library directory.
  std::string pluginLibDir;
  // Helper to generate the help string for the extra target arguments
  // (specified in the target config YAML file).
  std::string get_target_args_help_string() const;
  // Return target precision
  simulation_precision get_precision() const;

  /// @brief Reconstruct the path to this target's YAML config relative to
  /// `pluginLibDir`. For an external plugin package laid out as
  ///   `<pkgRoot>/lib/<plugin libs>`
  ///   `<pkgRoot>/targets/<name>.yml`
  /// this returns `<pkgRoot>/targets/<name>.yml`. Returns an empty path
  /// when either `pluginLibDir` or `name` is empty (i.e. this is an
  /// in-tree target whose YAML lives in the default install location).
  std::filesystem::path pluginYamlPath() const {
    if (pluginLibDir.empty() || name.empty())
      return {};
    return std::filesystem::path(pluginLibDir).parent_path() / "targets" /
           (name + ".yml");
  }
};
} // namespace cudaq
