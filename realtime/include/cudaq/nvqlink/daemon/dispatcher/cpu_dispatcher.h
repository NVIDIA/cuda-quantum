/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/daemon/dispatcher/dispatcher.h"
#include "cudaq/nvqlink/network/channel.h"

#include <atomic>
#include <thread>
#include <vector>

namespace cudaq::nvqlink {

/// CPU-mode dispatcher supporting both polling and event-driven backends.
/// - Polling backends: Calls receive_burst()
/// - Event-driven backends: Registers callback, calls process_events()
///
class CPUDispatcher : public Dispatcher {
public:
  CPUDispatcher(Channel *channel, FunctionRegistry *registry,
                const ComputeConfig &config);
  ~CPUDispatcher() override;

  void start() override;
  void stop() override;
  std::uint64_t get_packets_processed() const override;
  std::uint64_t get_packets_sent() const override;

private:
  void polling_worker_thread(std::uint32_t core_id);

  void event_driven_worker_thread(std::uint32_t core_id);

  /// Packet processing (common to both models)
  ///
  /// @param buffer Packet buffer
  ///
  void process_packet(Buffer *buffer);

  /// Callback for event-driven backends
  ///
  /// @param buffers Array of packet buffers
  /// @param count Number of buffers
  /// @param user_data User data
  ///
  static void packet_received_callback(Buffer **buffers, std::uint32_t count,
                                       void *user_data);

  Channel *channel_;
  FunctionRegistry *registry_;
  ComputeConfig config_;

  ChannelModel model_;

  std::atomic<bool> running_{false};
  std::atomic<std::uint64_t> packets_processed_{0};
  std::atomic<std::uint64_t> packets_sent_{0};
  std::vector<std::thread> threads_;
};

} // namespace cudaq::nvqlink
