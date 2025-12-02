/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/utils/instrumentation/logger.h"
#include "cudaq/nvqlink/utils/instrumentation/profiler.h"
#include "cudaq/nvqlink/network/gpu_channel.h"

#include <cuda_runtime.h>
#include <stdexcept>

namespace cudaq::nvqlink {

GPUChannel::GPUChannel(Channel *channel)
    : channel_(channel), d_queue_handles_(nullptr), h_queue_handles_(nullptr),
      d_running_(nullptr), h_running_(nullptr), stream_(nullptr) {
  NVQLINK_TRACE_FULL(DOMAIN_GPU, "GPUChannel::constructor");

  if (!channel_) {
    throw std::invalid_argument("GPUChannel: channel cannot be null");
  }

  // Create CUDA stream for kernel
  cudaStreamCreate(&stream_);

  // Allocate device-side queue handles
  cudaMalloc(&d_queue_handles_, sizeof(GPUQueueHandles));

  // Allocate host-side queue handles
  h_queue_handles_ = new GPUQueueHandles();

  // Allocate running flag (managed memory for easy access from both sides)
  volatile bool *temp_running;
  cudaMallocManaged(&temp_running, sizeof(bool));
  d_running_ = temp_running;
  h_running_ =
      const_cast<bool *>(temp_running); // Remove volatile for host access
  *h_running_ = false;

  // Initialize queue handles from backend
  initialize_queue_handles();
}

GPUChannel::~GPUChannel() {
  NVQLINK_TRACE_FULL(DOMAIN_GPU, "GPUChannel::destructor");
  cleanup();
}

void GPUChannel::start_kernel(void (*kernel_func)(GPUChannel *), dim3 blocks,
                              dim3 threads) {
  NVQLINK_TRACE_FULL(DOMAIN_GPU, "GPUChannel::start_kernel");

  *h_running_ = true;

  // Launch persistent kernel
  kernel_func<<<blocks, threads, 0, stream_>>>(this);

  // Check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "GPUChannel: Kernel launch failed: {}",
                      cudaGetErrorString(err));
    *h_running_ = false;
    throw std::runtime_error("Failed to launch GPU kernel");
  }

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "[GPUChannel] Persistent kernel started");
}

void GPUChannel::stop_kernel() {
  NVQLINK_TRACE_FULL(DOMAIN_GPU, "GPUChannel::stop_kernel");

  if (*h_running_) {
    NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                     "[GPUChannel] Stopping persistent kernel...");

    // Signal kernel to stop
    *h_running_ = false;

    // Wait for kernel to finish
    cudaStreamSynchronize(stream_);

    NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "[GPUChannel] Persistent kernel stopped");
  }
}

__host__ __device__ bool GPUChannel::is_running() const {
#ifdef __CUDA_ARCH__
  // Device code
  return d_queue_handles_->running ? *d_queue_handles_->running : false;
#else
  // Host code
  return h_running_ ? *h_running_ : false;
#endif
}

__device__ bool GPUChannel::receive_packet(void **packet_data,
                                           std::size_t *packet_size) {
  GPUQueueHandles *handles = d_queue_handles_;

  // Load queue head/tail (volatile to prevent caching)
  std::uint32_t head = *handles->rx_head;
  std::uint32_t tail = *handles->rx_tail;

  // Check if queue has packets
  if (head == tail) {
    return false; // No packets available
  }

  // Get packet index
  std::uint32_t idx = head % handles->queue_size;

  // Get packet data and size
  *packet_data = handles->packet_buffers[idx];
  *packet_size = handles->packet_sizes[idx];

  // Advance head (only one GPU thread should do this!)
  // For simplicity, we assume single-threaded or proper synchronization
  atomicAdd((unsigned int *)handles->rx_head, 1);

  return true;
}

__device__ bool GPUChannel::send_packet(void *packet_data,
                                        std::size_t packet_size) {
  GPUQueueHandles *handles = d_queue_handles_;

  // Load queue head/tail
  std::uint32_t head = *handles->tx_head;
  std::uint32_t tail = *handles->tx_tail;

  // Check if queue has space
  if ((tail + 1) % handles->queue_size == head) {
    return false; // Queue full
  }

  // Get slot index
  std::uint32_t idx = tail % handles->queue_size;

  // Set packet data and size
  handles->packet_buffers[idx] = packet_data;
  handles->packet_sizes[idx] = packet_size;

  // Advance tail
  atomicAdd((unsigned int *)handles->tx_tail, 1);

  return true;
}

__device__ void *GPUChannel::allocate_buffer(std::size_t size) {
  // Simplified allocation - in real implementation, this would
  // come from a pre-allocated pool or use device-side malloc
  // For now, return nullptr to indicate feature not yet implemented
  return nullptr;
}

__device__ void GPUChannel::release_buffer(void *packet_data) {
  // Release packet buffer back to pool
  // Implementation depends on memory management strategy
}

void GPUChannel::initialize_queue_handles() {
  // NOTE: This is a simplified initialization.
  // In a real implementation, the backend would provide
  // device-accessible pointers to NIC queue structures.

  // For now, initialize with nullptr - backends need to
  // implement get_gpu_queue_handles() method
  h_queue_handles_->rx_queue_base = nullptr;
  h_queue_handles_->tx_queue_base = nullptr;
  h_queue_handles_->rx_head = nullptr;
  h_queue_handles_->rx_tail = nullptr;
  h_queue_handles_->tx_head = nullptr;
  h_queue_handles_->tx_tail = nullptr;
  h_queue_handles_->queue_size = 1024;
  h_queue_handles_->packet_buffers = nullptr;
  h_queue_handles_->packet_sizes = nullptr;
  h_queue_handles_->running = d_running_;

  // Copy to device
  cudaMemcpy(d_queue_handles_, h_queue_handles_, sizeof(GPUQueueHandles),
             cudaMemcpyHostToDevice);

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "[GPUChannel] Queue handles initialized");
  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "[GPUChannel] Note: Backend must provide GPU-accessible "
                   "queue pointers for full functionality");
}

void GPUChannel::cleanup() {
  // Stop kernel if running
  stop_kernel();

  // Free CUDA resources
  if (d_queue_handles_) {
    cudaFree(d_queue_handles_);
    d_queue_handles_ = nullptr;
  }

  if (d_running_) {
    cudaFree(const_cast<bool *>(d_running_));
    d_running_ = nullptr;
    h_running_ = nullptr;
  }

  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }

  // Free host resources
  if (h_queue_handles_) {
    delete h_queue_handles_;
    h_queue_handles_ = nullptr;
  }
}

} // namespace cudaq::nvqlink
