/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/channel.h"
#include "cudaq/nvqlink/network/memory/buffer.h"

#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace cudaq::nvqlink::test {

/// @brief In-memory channel for unit testing without hardware
///
/// Simulates a network channel using in-memory queues.
/// Useful for testing Daemon, InputStream/OutputStream, and other
/// components without requiring RDMA hardware.
///
/// Features:
/// - inject_rx_packet(): Simulate incoming packets
/// - pop_tx_packet(): Read outgoing packets
/// - Thread-safe queue operations
class LoopbackChannel : public Channel {
public:
  explicit LoopbackChannel(size_t buffer_size = 2048, size_t num_buffers = 16);
  ~LoopbackChannel() override;

  // Channel interface
  void initialize() override;
  void cleanup() override;
  Buffer *acquire_buffer() override;
  void release_buffer(Buffer *buffer) override;
  uint32_t receive_burst(Buffer **buffers, uint32_t max) override;
  uint32_t send_burst(Buffer **buffers, uint32_t count) override;
  void register_memory(void *addr, size_t size) override {
  } // No-op for loopback
  ChannelModel get_execution_model() const override {
    return ChannelModel::POLLING;
  }
  void configure_queues(const std::vector<uint32_t> &queue_ids) override {
  } // No-op for loopback

  // Test helpers
  /// @brief Inject a packet into the RX queue (simulates incoming packet)
  void inject_rx_packet(const void *data, size_t len);

  /// @brief Pop a packet from the TX queue (reads outgoing packet)
  /// @return Packet data (empty if no packets available)
  std::vector<uint8_t> pop_tx_packet();

  /// @brief Check if there are TX packets available
  bool has_tx_packets() const;

  /// @brief Get number of RX packets waiting
  size_t rx_queue_size() const;

  /// @brief Get number of TX packets sent
  size_t tx_queue_size() const;

private:
  size_t buffer_size_;
  size_t num_buffers_;

  // Buffer pool
  std::vector<uint8_t> memory_pool_;
  std::vector<std::unique_ptr<Buffer>> buffers_;
  std::queue<Buffer *> free_buffers_;
  mutable std::mutex buffer_mutex_;

  // Packet queues (simulate network)
  std::queue<std::vector<uint8_t>> rx_queue_;
  std::queue<std::vector<uint8_t>> tx_queue_;
  mutable std::mutex rx_mutex_;
  mutable std::mutex tx_mutex_;

  bool initialized_{false};
};

} // namespace cudaq::nvqlink::test
