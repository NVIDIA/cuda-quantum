/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Support/TargetConfig.h"
#include "cudaq/host_config.h"
#include <string>

namespace cudaq {

/// @brief A RuntimeTarget encapsulates an available
/// backend simulator and quantum_platform for CUDA-Q
/// kernel execution.
struct RuntimeTarget {
  std::string name;
  std::string simulatorName;
  std::string platformName;
  std::string description;
  simulation_precision precision;
  config::TargetConfig config;

  /// @brief Return the number of QPUs this target exposes.
  std::size_t num_qpus();
  bool is_remote();
  bool is_remote_simulator();
  bool is_emulated();
  simulation_precision get_precision();
  std::string get_target_args_help_string() const;
};

} // namespace cudaq
