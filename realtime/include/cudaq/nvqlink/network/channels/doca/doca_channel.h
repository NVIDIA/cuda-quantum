/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/channel.h"
#include "cudaq/nvqlink/network/channels/doca/doca_channel_config.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <doca_dev.h>
#include <doca_gpunetio.h>
#include <doca_verbs.h>
#include <infiniband/verbs.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace cudaq::nvqlink {

// Forward declarations
class DOCACompletionQueue;
class DOCAQueuePair;
class DOCABufferPool;

/// @brief DOCA-based channel with GPU-controlled NIC access
///
/// This channel uses NVIDIA DOCA GPUNetIO to give the GPU direct control
/// over NIC queue pairs, enabling ultra-low latency RPC processing where
/// the GPU polls CQs, processes requests, and submits responses without
/// CPU involvement in the hot path.
///
/// Key differences from RoCEChannel:
/// - CQ polling happens on GPU, not CPU
/// - Work requests submitted via GPU SM doorbell
/// - Zero CPU involvement in steady-state operation
///
class DOCAChannel : public Channel {
public:
  explicit DOCAChannel(const DOCAChannelConfig &config);
  ~DOCAChannel() override;

  // Disable copy/move (owns hardware resources)
  DOCAChannel(const DOCAChannel &) = delete;
  DOCAChannel &operator=(const DOCAChannel &) = delete;
  DOCAChannel(DOCAChannel &&) = delete;
  DOCAChannel &operator=(DOCAChannel &&) = delete;

  // Channel interface implementation

  /// @brief Initialize DOCA resources
  void initialize() override;

  /// @brief Cleanup DOCA resources
  void cleanup() override;

  // Note: These are NOT used in GPU mode - GPU kernel handles I/O directly
  // Kept for interface compatibility and potential fallback/testing modes
  std::uint32_t receive_burst(Buffer **buffers, std::uint32_t max) override;
  std::uint32_t send_burst(Buffer **buffers, std::uint32_t count) override;

  Buffer *acquire_buffer() override;
  void release_buffer(Buffer *buffer) override;

  void register_memory(void *addr, std::size_t size) override;
  void configure_queues(const std::vector<std::uint32_t> &queue_ids) override;

  // GPU-based polling is still polling (just done by GPU)
  ChannelModel get_execution_model() const override {
    return ChannelModel::POLLING;
  }

  /// @brief Get GPU memory handles for GPUDispatcher integration
  ///
  /// Returns DOCA-specific handles including:
  /// - rx_queue_addr: doca_gpu_dev_verbs_qp* (device-side QP)
  /// - cq_rq_addr: doca_gpu_dev_verbs_cq* (receive CQ)
  /// - buffer_mkey: Memory key for RDMA
  /// - exit_flag: GPU-accessible exit flag
  ///
  GPUMemoryHandles get_gpu_memory_handles() override;

  // Connection management (adapted from DocaRoceReceiver methods)

  /// @brief Get connection parameters for control plane exchange
  DOCAConnectionParams get_connection_params() const;

  /// @brief Configure remote peer after control plane exchange
  void set_remote_qp(std::uint32_t remote_qpn,
                     const std::array<std::uint8_t, 16> &remote_gid);

  /// @brief Signal kernel to exit gracefully
  void signal_exit();

  /// @brief Check if channel is in RTS (Ready to Send) state
  bool is_connected() const { return connected_; }

private:
  void init_doca_device();
  void init_gpu_device();
  void init_protection_domain();
  void init_uar();
  void init_completion_queues();
  void init_queue_pair();
  void init_buffer_pool();
  void init_exit_flag();
  void find_roce_gid();

  // Configuration
  DOCAChannelConfig config_;

  // DOCA core resources
  doca_dev *doca_device_ = nullptr;
  doca_gpu *doca_gpu_device_ = nullptr;
  doca_verbs_context *verbs_ctx_ = nullptr;
  doca_verbs_pd *verbs_pd_ = nullptr;
  doca_uar *uar_ = nullptr;
  ibv_pd *ibv_pd_ = nullptr; // Needed for buffer registration

  // Queue resources
  std::unique_ptr<DOCACompletionQueue> cq_rq_;
  std::unique_ptr<DOCACompletionQueue> cq_sq_;
  std::unique_ptr<DOCAQueuePair> qp_;

  // Buffer management
  std::unique_ptr<DOCABufferPool> buffer_pool_;

  // GPU resources
  std::uint32_t *gpu_exit_flag_ = nullptr;
  std::uint32_t *cpu_exit_flag_ = nullptr;
  cudaStream_t cuda_stream_ = nullptr;

  // Connection state
  std::uint32_t gid_index_ = 0;
  bool connected_ = false;
};

} // namespace cudaq::nvqlink
