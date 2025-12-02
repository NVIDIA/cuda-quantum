/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 *                                                                             *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/dispatcher/gpu_dispatcher.h"
#include "cudaq/nvqlink/daemon/registry/function_registry.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"
#include "cudaq/nvqlink/utils/instrumentation/profiler.h"

#include <cstdint>
#include <cuda_runtime.h>

using namespace cudaq::nvqlink;

// RPC protocol structures (same as CPU)
struct __attribute__((packed)) RPCHeader {
  std::uint32_t function_id;
  std::uint32_t arg_len;
};

struct __attribute__((packed)) RPCResponse {
  std::int32_t status;
  std::uint32_t result_len;
};

// Device function type signature
using DeviceRPCFunction = int (*)(void *, std::uint32_t, std::uint32_t,
                                  std::uint32_t *);

// Persistent kernel implementation - MUST NOT BLOCK INDEFINITELY
__global__ void
daemon_persistent_kernel(void *rx_queue, void *tx_queue, void *buffer_pool,
                         void **function_table, std::uint32_t *function_ids,
                         std::size_t func_count, volatile int *shutdown_flag,
                         std::uint64_t *stats) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  std::uint64_t local_packet_count = 0;

  // For PoC: Don't run infinite loop, just do limited iterations
  // In production: this would be while(!*shutdown_flag)
  const std::uint64_t MAX_ITERATIONS = 100000; // Limit iterations for PoC

  for (std::uint64_t iter = 0; iter < MAX_ITERATIONS && !(*shutdown_flag);
       iter++) {
    // In production: call backend->poll_rx_queue()
    // For PoC: simulate packet processing

    local_packet_count++;

    // Periodic cooperative yielding
    if (iter % 1000 == 0) {
      __threadfence_system(); // Ensure memory visibility
    }
  }

  // Write stats back (only thread 0)
  if (tid == 0) {
    atomicAdd((unsigned long long *)stats, local_packet_count);
  }
}

GPUDispatcher::GPUDispatcher(Channel *channel, FunctionRegistry *registry,
                             const ComputeConfig &config)
    : channel_(channel), registry_(registry), config_(config) {}

GPUDispatcher::~GPUDispatcher() {
  stop();

  if (device_shutdown_flag_)
    cudaFree(device_shutdown_flag_);
  if (device_stats_)
    cudaFree(device_stats_);
  if (stream_)
    cudaStreamDestroy(stream_);
}

void GPUDispatcher::start() {
  NVQLINK_TRACE_FULL(DOMAIN_GPU, "GPUDispatcher::start");
  NVQLINK_TRACE_NAME_THREAD("GPU-Host");

  if (running_.exchange(true)) {
    return;
  }

  NVQLINK_LOG_INFO(DOMAIN_DISPATCHER, "Starting GPU dispatcher on device {}",
                   config_.gpu_device_id.value_or(0));

  // Set GPU device
  cudaSetDevice(config_.gpu_device_id.value_or(0));

  // Create stream
  cudaStreamCreate(&stream_);

  // Allocate device memory for control
  cudaMalloc(&device_shutdown_flag_, sizeof(int));
  cudaMalloc(&device_stats_, sizeof(std::uint64_t));

  int zero = 0;
  cudaMemcpy(device_shutdown_flag_, &zero, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(device_stats_, 0, sizeof(std::uint64_t));

  // Launch persistent kernel
  launch_persistent_kernel();

  // Check that launch succeeded (non-blocking check)
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Kernel launch failed: ") +
                             cudaGetErrorString(err));
  }

  NVQLINK_LOG_INFO(DOMAIN_GPU,
                   "GPU persistent kernel launched (running asynchronously)");
}

void GPUDispatcher::stop() {
  if (!running_.exchange(false)) {
    return;
  }

  NVQLINK_LOG_INFO(DOMAIN_DISPATCHER, "Signaling GPU kernel to shutdown...");

  // Signal kernel to exit
  int shutdown = 1;
  cudaMemcpy(device_shutdown_flag_, &shutdown, sizeof(int),
             cudaMemcpyHostToDevice);

  NVQLINK_LOG_INFO(DOMAIN_DISPATCHER, "Waiting for GPU kernel to complete...");

  // Wait for kernel completion
  cudaError_t err = cudaStreamSynchronize(stream_);
  if (err != cudaSuccess) {
    NVQLINK_LOG_WARNING(DOMAIN_DISPATCHER, "Warning: GPU kernel error: {}",
                        cudaGetErrorString(err));
  }

  // Read stats
  std::uint64_t total_packets = 0;
  cudaMemcpy(&total_packets, device_stats_, sizeof(std::uint64_t),
             cudaMemcpyDeviceToHost);

  NVQLINK_LOG_INFO(
      DOMAIN_DISPATCHER,
      "GPU dispatcher stopped. Processed {} iterations (PoC simulation)",
      total_packets);
}

std::uint64_t GPUDispatcher::get_packets_processed() const {
  if (!device_stats_ || !running_.load())
    return 0;

  std::uint64_t packets = 0;
  cudaError_t err = cudaMemcpy(&packets, device_stats_, sizeof(std::uint64_t),
                               cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return 0;
  }
  return packets;
}

std::uint64_t GPUDispatcher::get_packets_sent() const {
  // TODO: Implement GPU packet sent tracking
  return 0;
}

void GPUDispatcher::launch_persistent_kernel() {
  NVQLINK_TRACE_FULL(DOMAIN_GPU, "launch_persistent_kernel");

  // Get GPU memory handles from backend
  auto gpu_handles = channel_->get_gpu_memory_handles();

  // Get function table from registry
  auto func_table = registry_->get_gpu_function_table();

  // Launch kernel with minimal resources for PoC
  std::uint32_t blocks = config_.gpu_blocks > 0 ? config_.gpu_blocks : 1;
  std::uint32_t threads =
      config_.gpu_threads_per_block > 0 ? config_.gpu_threads_per_block : 32;

  NVQLINK_LOG_INFO(DOMAIN_DISPATCHER,
                   "Launching kernel with {} blocks, {} threads per block",
                   blocks, threads);

  // Launch kernel ASYNCHRONOUSLY on the stream
  daemon_persistent_kernel<<<blocks, threads, 0, stream_>>>(
      gpu_handles.rx_queue_addr, gpu_handles.tx_queue_addr,
      gpu_handles.buffer_pool_addr, func_table.device_function_ptrs,
      func_table.function_ids, func_table.count,
      (volatile int *)device_shutdown_flag_, (std::uint64_t *)device_stats_);

  // DON'T synchronize here - let kernel run asynchronously
  // Just check for launch errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("Kernel launch failed: ") +
                             cudaGetErrorString(err));
  }
}

void launch_daemon_kernel(void *rx_queue, void *tx_queue, void *buffer_pool,
                          void **function_table, std::uint32_t *function_ids,
                          std::size_t func_count, volatile int *shutdown_flag,
                          std::uint64_t *stats, std::uint32_t num_blocks,
                          std::uint32_t threads_per_block,
                          cudaStream_t stream) {
  daemon_persistent_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
      rx_queue, tx_queue, buffer_pool, function_table, function_ids, func_count,
      shutdown_flag, stats);
}
