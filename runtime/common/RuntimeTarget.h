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
  // Helper to generate the help string for the extra target arguments
  // (specified in the target config YAML file).
  std::string get_target_args_help_string() const;
  // Return target precision
  simulation_precision get_precision() const;
};
} // namespace cudaq
