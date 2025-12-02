/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/steering/verbs_flow_switch.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"
#include "cudaq/nvqlink/utils/instrumentation/profiler.h"

#include <arpa/inet.h>
#include <cstring>
#include <stdexcept>

namespace cudaq::nvqlink {

VerbsFlowSwitch::~VerbsFlowSwitch() {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "VerbsFlowSwitch::~VerbsFlowSwitch");

  std::lock_guard<std::mutex> lock(mutex_);

  // Destroy all flow rules
  for (auto &[port, flow] : flow_handles_) {
    if (flow) {
      ibv_destroy_flow(flow);
      NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                       "VerbsFlowSwitch: Destroyed flow rule for UDP port {}",
                       port);
    }
  }
  flow_handles_.clear();
}

void VerbsFlowSwitch::add_steering_rule(uint16_t udp_port,
                                        void *channel_handle) {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "VerbsFlowSwitch::add_steering_rule");

  if (!channel_handle) {
    throw std::invalid_argument("channel_handle cannot be null");
  }

  struct ibv_qp *qp = static_cast<struct ibv_qp *>(channel_handle);

  std::lock_guard<std::mutex> lock(mutex_);

  // Check if rule already exists
  if (flow_handles_.find(udp_port) != flow_handles_.end()) {
    throw std::runtime_error("Flow steering rule for UDP port " +
                             std::to_string(udp_port) + " already exists");
  }

  // Query QP attributes to get port number
  struct ibv_qp_attr qp_attr;
  struct ibv_qp_init_attr init_attr;

  int ret = ibv_query_qp(qp, &qp_attr, IBV_QP_PORT, &init_attr);
  if (ret != 0) {
    throw std::runtime_error("Failed to query QP attributes");
  }

  uint8_t port_num = qp_attr.port_num;

  // Create flow specification for UDP port matching
  // Structure: Ethernet -> IPv4 -> UDP
  struct {
    struct ibv_flow_attr attr;
    struct ibv_flow_spec_eth eth;
    struct ibv_flow_spec_ipv4 ipv4;
    struct ibv_flow_spec_tcp_udp udp;
  } __attribute__((packed)) flow_rule;

  std::memset(&flow_rule, 0, sizeof(flow_rule));

  // Flow attributes
  flow_rule.attr.type = IBV_FLOW_ATTR_NORMAL;
  flow_rule.attr.size = sizeof(flow_rule);
  flow_rule.attr.priority = 0;
  flow_rule.attr.num_of_specs = 3; // eth + ipv4 + udp
  flow_rule.attr.port = port_num;
  flow_rule.attr.flags = 0;

  // Ethernet spec (match all)
  flow_rule.eth.type = IBV_FLOW_SPEC_ETH;
  flow_rule.eth.size = sizeof(struct ibv_flow_spec_eth);

  // IPv4 spec (match all)
  flow_rule.ipv4.type = IBV_FLOW_SPEC_IPV4;
  flow_rule.ipv4.size = sizeof(struct ibv_flow_spec_ipv4);

  // UDP spec (match destination port)
  flow_rule.udp.type = IBV_FLOW_SPEC_UDP;
  flow_rule.udp.size = sizeof(struct ibv_flow_spec_tcp_udp);
  flow_rule.udp.val.dst_port = htons(udp_port); // Network byte order
  flow_rule.udp.mask.dst_port = 0xFFFF;         // Match all 16 bits

  // Create flow rule
  struct ibv_flow *flow = ibv_create_flow(qp, &flow_rule.attr);

  if (!flow) {
    NVQLINK_LOG_WARNING(
        DOMAIN_CHANNEL,
        "VerbsFlowSwitch: Failed to create hardware flow rule for UDP port {} "
        "(hardware steering may not be supported, traffic will use default QP)",
        udp_port);
    // Don't throw - hardware flow steering may not be supported
    // Packets will be delivered to default QP and software can filter
    return;
  }

  flow_handles_[udp_port] = flow;

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "VerbsFlowSwitch: Created flow rule: UDP port {} -> QP",
                   udp_port);
}

void VerbsFlowSwitch::remove_steering_rule(uint16_t udp_port) {
  NVQLINK_TRACE_FULL(DOMAIN_CHANNEL, "VerbsFlowSwitch::remove_steering_rule");

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = flow_handles_.find(udp_port);
  if (it == flow_handles_.end()) {
    NVQLINK_LOG_WARNING(DOMAIN_CHANNEL,
                        "VerbsFlowSwitch: No flow rule found for UDP port {}",
                        udp_port);
    return;
  }

  if (it->second) {
    ibv_destroy_flow(it->second);
    NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                     "VerbsFlowSwitch: Removed flow rule for UDP port {}",
                     udp_port);
  }

  flow_handles_.erase(it);
}

} // namespace cudaq::nvqlink
