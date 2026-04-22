/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RuntimeTarget.h"
#include <sstream>

namespace cudaq {

simulation_precision RuntimeTarget::get_precision() const { return precision; }

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
