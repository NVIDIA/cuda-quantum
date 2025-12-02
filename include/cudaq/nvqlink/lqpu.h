/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <string>

namespace cudaq::nvqlink {

/// @brief LQPU configuration
/// @details Logical Quantum Processing Unit configuration.
/// In the new architecture, LQPU is just configuration data,
/// not a device container. Actual runtime components (Daemon, QCSDevice, Channel)
/// are managed explicitly.
struct LQPUConfig {
  std::string name;
  // Future: add connection/deployment info as needed
};

} // namespace cudaq::nvqlink
