/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/channel.h"
#include "cudaq/nvqlink/network/roce/roce_buffer_pool.h"
#include "cudaq/nvqlink/network/roce/roce_ring_buffer.h"
#include "cudaq/nvqlink/network/roce/verbs_context.h"
#include "cudaq/nvqlink/network/steering/flow_switch.h"

#include <infiniband/verbs.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace cudaq::nvqlink {

/// RoCE channel for RDMA-based packet I/O.
///
/// Implements Channel interface using InfiniBand Verbs for zero-copy RDMA.
/// Only supports UC (Unreliable Connection) mode with memory polling.
///
/// UC Mode: Connection-oriented, requires QP pairing with remote client.
/// Uses RDMA WRITE for incoming packets (memory polling) and RDMA SEND for
/// responses.
///
/// Each RoCEChannel is self-sufficient and manages its own Queue Pairs.
/// Can operate independently or share resources via VerbsContext.
///
class RoCEChannel : public Channel {
public:
  /// Create an independent RoCE channel (no memory sharing).
  ///
  /// The channel creates its own `ibv_context` and `ibv_pd`, fully isolated
  /// from other channels. Use this mode when channels don't need to share
  /// memory registrations.
  ///
  /// @param device_name Device name (e.g., "mlx5_0") or index (e.g., "0")
  /// @param listen_port UDP port for receiving packets (used by FlowSwitch)
  /// @param flow_switch Traffic coordinator for steering packets to this
  /// channel
  ///
  RoCEChannel(const std::string &device_name, std::uint16_t listen_port,
              std::shared_ptr<FlowSwitch> flow_switch);

  /// Create a RoCE channel with shared memory resources.
  ///
  /// The channel shares `ibv_context` and `ibv_pd` with other channels via
  /// `VerbsContext`. Use this mode when multiple channels need to access
  /// the same memory registrations for zero-copy data sharing.
  ///
  /// @param shared_ctx Shared verbs context for memory sharing
  /// @param listen_port UDP port for receiving packets (used by FlowSwitch)
  /// @param flow_switch Traffic coordinator for steering packets to this
  /// channel
  ///
  RoCEChannel(std::shared_ptr<VerbsContext> shared_ctx,
              std::uint16_t listen_port,
              std::shared_ptr<FlowSwitch> flow_switch);

  ~RoCEChannel() override;

  /// Connection parameters
  struct ConnectionParams {
    std::uint32_t qpn;   // Queue Pair Number
    std::uint32_t psn;   // Packet Sequence Number
    union ibv_gid gid;   // Global ID (16 bytes)
    std::uint64_t vaddr; // Virtual address of ring buffer base
    std::uint32_t rkey;  // Remote key for RDMA WRITE
    std::uint16_t lid;   // Local ID (for IB, 0 for RoCE)
    // Ring buffer configuration
    std::uint32_t num_slots; // Number of ring buffer slots
    std::uint32_t slot_size; // Size of each slot in bytes
  };

  /// Export connection parameters for lite clients
  ConnectionParams get_connection_params() const;

  /// Set remote QP info (must be called before QP transitions to RTR)
  void set_remote_qp(std::uint32_t remote_qpn, const union ibv_gid &remote_gid);

  //===--------------------------------------------------------------------===//
  // Channel interface
  //===--------------------------------------------------------------------===//

  void initialize() override;
  void cleanup() override;
  ChannelModel get_execution_model() const override {
    return ChannelModel::POLLING;
  }

  void register_memory(void *addr, std::size_t size) override;

  // Buffer management - RoCE uses RoCEBufferPool
  Buffer *acquire_buffer() override;
  void release_buffer(Buffer *buffer) override;

  std::uint32_t receive_burst(Buffer **buffers,
                              std::uint32_t max_count) override;
  std::uint32_t send_burst(Buffer **buffers, std::uint32_t count) override;

  void configure_queues(const std::vector<std::uint32_t> &queue_ids) override;

private:
  // Pre-allocated receive buffer with registered memory
  struct RecvBuffer {
    void *addr{nullptr};
    struct ibv_mr *mr{nullptr};
    std::uint64_t buffer_id{0}; // Unique ID used as WR ID
  };

  //===--------------------------------------------------------------------===//
  // Initialization helpers
  //===--------------------------------------------------------------------===//

  void init_independent(const std::string &device_name);
  void init_shared(std::shared_ptr<VerbsContext> shared_ctx);

  //===--------------------------------------------------------------------===//
  // Resource access (handles both independent and shared modes)
  //===--------------------------------------------------------------------===//

  struct ibv_context *get_context() const;
  struct ibv_pd *get_pd() const;
  std::uint8_t get_port_num() const;

  //===--------------------------------------------------------------------===//
  // Queue Pair management
  //===--------------------------------------------------------------------===//

  void create_queue_pair();
  void transition_qp_to_rts(struct ibv_qp *qp);
  void create_completion_queues();

  //===--------------------------------------------------------------------===//
  // Buffer management
  //===--------------------------------------------------------------------===//

  void preallocate_recv_buffers(std::uint32_t count);
  void initial_post_recv_buffers(struct ibv_qp *qp);
  void repost_recv_buffer(struct ibv_qp *qp, RecvBuffer *rb);

  // Poll completion queue
  std::uint32_t poll_completions(struct ibv_cq *cq, struct ibv_wc *wc_array,
                                 std::uint32_t max_count);

  //===--------------------------------------------------------------------===//
  // Receive implementation
  //===--------------------------------------------------------------------===//

  // Initialize ring buffers
  void initialize_rx_ring_buffer(); // For receiving (RDMA WRITE target)
  void initialize_tx_ring_buffer(); // For sending (pre-allocated TX slots)

  // Common ring buffer allocation helper
  void *allocate_and_register_ring_buffer(std::size_t size, int access_flags,
                                          struct ibv_mr **out_mr,
                                          const char *buffer_name);

  //===--------------------------------------------------------------------===//
  // Member variables
  //===--------------------------------------------------------------------===//

  // Configuration
  std::uint16_t listen_port_;
  std::shared_ptr<FlowSwitch> flow_switch_;

  // Shared context (null if independent mode)
  std::shared_ptr<VerbsContext> shared_ctx_;

  // Independent mode resources (owned if shared_ctx_ is null)
  struct ibv_context *owned_context_{nullptr};
  struct ibv_pd *owned_pd_{nullptr};
  std::uint8_t port_num_{1};
  bool owns_context_{false};

  // Verbs resources per channel
  struct ibv_cq *send_cq_{nullptr};
  struct ibv_cq *recv_cq_{nullptr};
  struct ibv_qp *qp_{nullptr};

  // Remote QP info
  std::uint32_t remote_qpn_{0};
  union ibv_gid remote_gid_{};
  bool remote_qp_set_{false};

  // Buffer management (pre-allocated, NEVER allocated in hot path)
  std::unique_ptr<RoCEBufferPool> buffer_pool_;
  // Pre-allocated receive buffers with MRs
  std::vector<RecvBuffer> recv_buffers_;
  // Fast lookup by WR ID
  std::unordered_map<std::uint64_t, RecvBuffer *> buffer_id_map_;

  // RX Ring buffer for RDMA WRITE memory polling (client → server)
  std::unique_ptr<RingBufferPoller> rx_ring_buffer_poller_;
  void *rx_ring_buffer_base_{nullptr};
  struct ibv_mr *rx_ring_buffer_mr_{nullptr};
  RingBufferConfig rx_ring_buffer_config_;

  // TX Ring buffer for sending responses (server → client)
  // Pre-allocated slots, NO allocation on hot path
  void *tx_ring_buffer_base_{nullptr};
  struct ibv_mr *tx_ring_buffer_mr_{nullptr};
  std::uint32_t tx_num_slots_{1024};
  std::uint32_t tx_slot_size_{2048};
  std::vector<bool> tx_slot_free_; // Pre-allocated slot tracking
  std::size_t tx_next_slot_{0};    // Round-robin hint for allocation

  // Configuration constants
  static constexpr std::uint32_t RECV_RING_SIZE = 1024;
  static constexpr std::uint32_t SEND_RING_SIZE = 1024;
  static constexpr std::uint32_t CQ_SIZE = 2048;
  static constexpr std::uint32_t RECV_BUFFER_SIZE = 2048;
  static constexpr std::uint32_t MAX_INLINE_DATA = 64;
};

} // namespace cudaq::nvqlink
