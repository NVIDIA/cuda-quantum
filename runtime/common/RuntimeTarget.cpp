/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RuntimeTarget.h"
#include "cudaq/platform.h"
#include <sstream>

namespace cudaq {

std::size_t RuntimeTarget::num_qpus() {
  auto &platform = cudaq::get_platform();
  return platform.num_qpus();
}

bool RuntimeTarget::is_remote() {
  auto &platform = cudaq::get_platform();
  return platform.is_remote();
}

bool RuntimeTarget::is_remote_simulator() {
  auto &platform = cudaq::get_platform();
  return platform.get_remote_capabilities().isRemoteSimulator;
}

bool RuntimeTarget::is_emulated() {
  auto &platform = cudaq::get_platform();
  return platform.is_emulated();
}

simulation_precision RuntimeTarget::get_precision() { return precision; }

std::string RuntimeTarget::get_target_args_help_string() const {
  std::stringstream ss;
  for (const auto &argConfig : config.TargetArguments) {
    ss << "  - " << argConfig.KeyName;
    if (!argConfig.HelpString.empty()) {
      ss << " (" << argConfig.HelpString << ")";
    }
    ss << "\n";
  }
  return ss.str();
}

} // namespace cudaq
                     