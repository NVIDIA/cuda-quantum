/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/steering/flow_switch.h"

#include <cstdint>
#include <infiniband/verbs.h>
#include <map>
#include <mutex>

namespace cudaq::nvqlink {

/// @brief InfiniBand Verbs implementation of FlowSwitch
///
/// Uses `ibv_create_flow()` to program hardware flow steering rules.
/// Does not own any InfiniBand context - uses the QP's context.
///
class VerbsFlowSwitch : public FlowSwitch {
public:
  VerbsFlowSwitch() = default;
  ~VerbsFlowSwitch() override;

  void add_steering_rule(uint16_t udp_port, void *channel_handle) override;
  void remove_steering_rule(uint16_t udp_port) override;

private:
  // Map: udp_port -> ibv_flow*
  std::map<uint16_t, struct ibv_flow *> flow_handles_;
  std::mutex mutex_;
};

} // namespace cudaq::nvqlink
