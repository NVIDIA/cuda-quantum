/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>

namespace cudaq::nvqlink {

/// @brief Abstract interface for traffic steering/flow control
///
/// FlowSwitch coordinates packet routing to different channels based on
/// flow rules (e.g., UDP port matching). Implementations use hardware
/// flow steering when available (e.g., ibv_create_flow for InfiniBand).
///
class FlowSwitch {
public:
  virtual ~FlowSwitch() = default;

  /// @brief Add a steering rule to route traffic to a specific channel
  ///
  /// @param udp_port UDP destination port to match
  /// @param channel_handle Opaque handle identifying the target channel
  ///                       (implementation-specific, e.g., ibv_qp* for Verbs)
  ///
  virtual void add_steering_rule(uint16_t udp_port, void *channel_handle) = 0;

  /// @brief Remove a steering rule
  ///
  /// @param udp_port UDP destination port to stop matching
  ///
  virtual void remove_steering_rule(uint16_t udp_port) = 0;
};

} // namespace cudaq::nvqlink
