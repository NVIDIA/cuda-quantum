/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace cudaq::nvqlink {

/// @brief Number of Work Queue Entries
/// 
/// Must be power of 2. Higher values increase throughput capacity
/// but also memory usage. 64 is a good balance for RPC workloads.
/// 
constexpr std::uint32_t DOCA_WQE_NUM = 64;

/// @brief Maximum inline send size in WQE
///
/// Payloads smaller than this can be embedded directly in the WQE,
/// avoiding an extra DMA read. Optimizes small RPC responses.
///
constexpr std::size_t DOCA_MAX_INLINE_SIZE = 44;

/// @brief Configuration for DOCA Channel
///
struct DOCAChannelConfig {
  // Network device (from ibv_name, ibv_port parameters)
  std::string nic_device{"mlx5_0"};
  std::uint32_t nic_port{1};

  // GPU configuration
  int gpu_device_id{0};

  // Buffer configuration (from cu_buffer_size, cu_page_size, pages)
  std::size_t buffer_size{64 * 1024 * 1024};
  std::size_t page_size{4096};
  unsigned num_pages{1024};

  // Queue configuration
  std::uint32_t wqe_num{DOCA_WQE_NUM};

  // RPC configuration (from cu_frame_size)
  std::size_t max_rpc_size{2048};

  // Remote peer (from peer_ip parameter)
  std::string peer_ip;
  std::uint32_t remote_qpn{0};

  bool is_valid() const {
    return !nic_device.empty() && buffer_size > 0 && page_size > 0 &&
           num_pages > 0 && wqe_num > 0;
  }
};

/// @brief Connection parameters exchanged via control plane
///
/// Uses std::array for GID to avoid DOCA header dependency in public API.
/// Can be converted to/from doca_verbs_gid internally.
///
struct DOCAConnectionParams {
  std::uint32_t qpn;                // Queue pair number
  std::array<std::uint8_t, 16> gid; // Global identifier (128-bit)
  std::uint64_t buffer_addr;        // Virtual address of buffer
  std::uint32_t rkey;               // Remote memory key
  std::uint32_t num_slots;          // Number of buffer slots
  std::size_t slot_size;            // Size of each slot
};

} // namespace cudaq::nvqlink
