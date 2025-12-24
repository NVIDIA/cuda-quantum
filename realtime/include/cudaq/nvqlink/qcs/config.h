/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <string>

namespace cudaq::nvqlink {

/// Configuration for QCS (Quantum Control System) device connection
/// QCS control plane uses UDP for out-of-band communication:
/// - RDMA parameter exchange (QPN, GID, vaddr, rkey)
/// - Program upload
/// - Execution control (START/STOP/ABORT commands)
struct QCSDeviceConfig {
  std::string name;
  uint16_t control_port{9999}; // UDP port for control plane

  bool is_valid() const { return !name.empty() && control_port > 0; }
};

} // namespace cudaq::nvqlink
