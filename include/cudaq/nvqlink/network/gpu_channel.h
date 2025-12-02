/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/channel.h"

#include <cuda_runtime.h>

namespace cudaq::nvqlink {


/// Device-side packet queue handles for GPU direct access.
///
/// Contains raw pointers to NIC queue data structures that
/// the GPU can directly access for packet polling and sending.
struct GPUQueueHandles {
  void *rx_queue_base;       ///< RX queue descriptor ring (device ptr)
  void *tx_queue_base;       ///< TX queue descriptor ring (device ptr)
  std::uint32_t *rx_head;    ///< RX queue head index (device ptr)
  std::uint32_t *rx_tail;    ///< RX queue tail index (device ptr)
  std::uint32_t *tx_head;    ///< TX queue head index (device ptr)
  std::uint32_t *tx_tail;    ///< TX queue tail index (device ptr)
  std::uint32_t queue_size;  ///< Queue size (number of descriptors)
  void **packet_buffers;     ///< Array of packet buffer pointers (device ptr)
  std::size_t *packet_sizes; ///< Array of packet sizes (device ptr)
  volatile bool *running;    ///< Kernel running flag (device ptr)
};

/// GPU Channel for device-side NIC control.
///
/// Enables CUDA kernels to directly poll NIC queues and send packets
/// without CPU involvement. Requires GPUDirect RDMA or unified memory.
///
/// @example Persistent GPU kernel:
/// ```cuda
/// __global__ void gpu_network_kernel(GPUChannel* channel) {
///   GPUInputStream in(*channel);
///   GPUOutputStream out(*channel);
///
///   while (channel->is_running()) {
///     if (in.available()) {
///       int a = in.read<int>();
///       int b = in.read<int>();
///       out.write<int>(a + b);
///       out.flush();
///     }
///   }
/// }
/// ```
///
/// @note This is an advanced feature requiring hardware support:
///       - GPUDirect RDMA for NIC memory access
///       - Unified memory for queue synchronization
///       - Compatible NIC (DPDK with GPU support, DOCA, etc.)
///
class GPUChannel {
public:
  /// Construct GPUChannel from backend (wraps for GPU kernel use).
  ///
  /// @param channel Channel providing NIC access
  GPUChannel(Channel *channel);

  ~GPUChannel();

  /// Start persistent GPU kernel.
  ///
  /// Launches a long-running CUDA kernel that continuously polls
  /// the NIC and processes packets.
  ///
  /// @param kernel_func User kernel function pointer
  /// @param blocks Number of blocks
  /// @param threads Threads per block
  void start_kernel(void (*kernel_func)(GPUChannel *), dim3 blocks,
                    dim3 threads);

  /// Stop persistent GPU kernel.
  void stop_kernel();

  /// Check if kernel is running.
  ///
  /// @return true if kernel should continue running
  __host__ __device__ bool is_running() const;

  /// Get device-side queue handles.
  ///
  /// @return Pointer to GPU queue handles (device memory)
  __host__ __device__ GPUQueueHandles *get_queue_handles() {
    return d_queue_handles_;
  }

  /// Device-side: Receive a packet (non-blocking).
  ///
  /// Polls the RX queue for a new packet. Returns nullptr if no packet.
  ///
  /// @param packet_data Output: pointer to packet data (device memory)
  /// @param packet_size Output: size of packet
  /// @return true if packet received
  __device__ bool receive_packet(void **packet_data, std::size_t *packet_size);

  /// Device-side: Send a packet.
  ///
  /// @param packet_data Pointer to packet data (device memory)
  /// @param packet_size Size of packet
  /// @return true if packet sent successfully
  __device__ bool send_packet(void *packet_data, std::size_t packet_size);

  /// Device-side: Allocate a buffer for sending.
  ///
  /// @param size Requested buffer size
  /// @return Pointer to buffer (device memory), or nullptr on failure
  __device__ void *allocate_buffer(std::size_t size);

  /// Device-side: Release a received packet buffer.
  ///
  /// @param packet_data Pointer to packet buffer
  __device__ void release_buffer(void *packet_data);

  /// Get backend pointer (host-side only).
  Channel *channel() { return channel_; }

private:
  Channel *channel_;                 ///< Channel for NIC access
  GPUQueueHandles *d_queue_handles_; ///< Device-side queue handles
  GPUQueueHandles *h_queue_handles_; ///< Host-side queue handles
  volatile bool *d_running_;         ///< Device flag for kernel control
  bool *h_running_;                  ///< Host flag for kernel control
  cudaStream_t stream_;              ///< CUDA stream for kernel

  /// Initialize device-side queue handles
  void initialize_queue_handles();

  /// Cleanup device resources
  void cleanup();
};

} // namespace cudaq::nvqlink
