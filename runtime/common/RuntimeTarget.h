/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/TargetConfig.h"
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
  /// Path to the target's config artifact. Empty for built-in targets resolved
  /// from the pre-compiled database.
  std::filesystem::path configPath;
  /// Non-empty when this target is known but not available on the current
  /// host (missing GPU, simulator, platform library, etc.).
  std::string availabilityDiagnostic;
  // Helper to generate the help string for the extra target arguments
  // (specified in the target config).
  std::string get_target_args_help_string() const;
  // Return target precision
  simulation_precision get_precision() const;
  /// Whether this target is available on the current host.
  bool isAvailable() const { return availabilityDiagnostic.empty(); }
};
} // namespace cudaq
