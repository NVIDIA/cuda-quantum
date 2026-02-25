/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file dispatch_kernel.cu
/// @brief Simple dispatch kernel for testing libcudaq-realtime.
///
/// This example demonstrates a simple dispatch kernel that processes RPC
/// requests.
///

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <unistd.h>
#include <vector>

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel.cuh"
#include "cudaq/realtime/daemon/dispatcher/dispatch_modes.h"
#include "cudaq/realtime/daemon/dispatcher/kernel_types.h"

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define CUDAQ_CHECK(call)                                                      \
  do {                                                                         \
    auto err = call;                                                           \
    if (err != CUDAQ_OK) {                                                     \
      std::cerr << "CUDAQ error in " << __FILE__ << ":" << __LINE__ << ": "    \
                << err << std::endl;                                           \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

//==============================================================================
// Test Handler: Simple noop that copies input to output and adds 1 to each byte
//==============================================================================

/// @brief Test handler that adds 1 to each byte.
__device__ int increment_handler(const void *input, void *output,
                                 std::uint32_t arg_len,
                                 std::uint32_t max_result_len,
                                 std::uint32_t *result_len) {
  const std::uint8_t *in_data = static_cast<const std::uint8_t *>(input);
  std::uint8_t *out_data = static_cast<std::uint8_t *>(output);
  for (std::uint32_t i = 0; i < arg_len && i < max_result_len; ++i) {
    out_data[i] = in_data[i] + 1;
  }
  *result_len = arg_len;
  return 0;
}

//==============================================================================
// Host API Dispatch Kernel Test Helpers
//==============================================================================

constexpr std::uint32_t RPC_INCREMENT_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("rpc_increment");

__device__ int rpc_increment_handler(const void *input, void *output,
                                     std::uint32_t arg_len,
                                     std::uint32_t max_result_len,
                                     std::uint32_t *result_len) {
  const std::uint8_t *in_data = static_cast<const std::uint8_t *>(input);
  std::uint8_t *out_data = static_cast<std::uint8_t *>(output);
  for (std::uint32_t i = 0; i < arg_len && i < max_result_len; ++i) {
    out_data[i] = static_cast<std::uint8_t>(in_data[i] + 1);
  }
  *result_len = arg_len;
  return 0;
}

__global__ void init_rpc_function_table(cudaq_function_entry_t *entries) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[0].handler.device_fn_ptr =
        reinterpret_cast<void *>(&rpc_increment_handler);
    entries[0].function_id = RPC_INCREMENT_FUNCTION_ID;
    entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    entries[0].reserved[0] = 0;
    entries[0].reserved[1] = 0;
    entries[0].reserved[2] = 0;

    // Schema: 1 array argument (uint8), 1 array result (uint8)
    entries[0].schema.num_args = 1;
    entries[0].schema.num_results = 1;
    entries[0].schema.reserved = 0;
    entries[0].schema.args[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
    entries[0].schema.args[0].reserved[0] = 0;
    entries[0].schema.args[0].reserved[1] = 0;
    entries[0].schema.args[0].reserved[2] = 0;
    entries[0].schema.args[0].size_bytes = 0;   // Variable size
    entries[0].schema.args[0].num_elements = 0; // Variable size
    entries[0].schema.results[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
    entries[0].schema.results[0].reserved[0] = 0;
    entries[0].schema.results[0].reserved[1] = 0;
    entries[0].schema.results[0].reserved[2] = 0;
    entries[0].schema.results[0].size_bytes = 0;   // Variable size
    entries[0].schema.results[0].num_elements = 0; // Variable size
  }
}

bool allocate_ring_buffer(std::size_t num_slots, std::size_t slot_size,
                          volatile uint64_t **host_flags_out,
                          volatile uint64_t **device_flags_out,
                          std::uint8_t **host_data_out,
                          std::uint8_t **device_data_out) {
  void *host_flags_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_flags_ptr, num_slots * sizeof(uint64_t),
                                  cudaHostAllocMapped);
  if (err != cudaSuccess)
    return false;

  void *device_flags_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_flags_ptr, host_flags_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    return false;
  }

  void *host_data_ptr = nullptr;
  err =
      cudaHostAlloc(&host_data_ptr, num_slots * slot_size, cudaHostAllocMapped);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    return false;
  }

  void *device_data_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_data_ptr, host_data_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    cudaFreeHost(host_data_ptr);
    return false;
  }

  memset(host_flags_ptr, 0, num_slots * sizeof(uint64_t));

  *host_flags_out = static_cast<volatile uint64_t *>(host_flags_ptr);
  *device_flags_out = static_cast<volatile uint64_t *>(device_flags_ptr);
  *host_data_out = static_cast<std::uint8_t *>(host_data_ptr);
  *device_data_out = static_cast<std::uint8_t *>(device_data_ptr);
  return true;
}

void free_ring_buffer(volatile uint64_t *host_flags, std::uint8_t *host_data) {
  if (host_flags)
    cudaFreeHost(const_cast<uint64_t *>(host_flags));
  if (host_data)
    cudaFreeHost(host_data);
}

extern "C" void launch_dispatch_kernel_wrapper(
    volatile std::uint64_t *rx_flags, volatile std::uint64_t *tx_flags,
    std::uint8_t *rx_data, std::uint8_t *tx_data, std::size_t rx_stride_sz,
    std::size_t tx_stride_sz, cudaq_function_entry_t *function_table,
    std::size_t func_count, volatile int *shutdown_flag, std::uint64_t *stats,
    std::size_t num_slots, std::uint32_t num_blocks,
    std::uint32_t threads_per_block, cudaStream_t stream) {
  cudaq_launch_dispatch_kernel_regular(
      rx_flags, tx_flags, rx_data, tx_data, rx_stride_sz, tx_stride_sz,
      function_table, func_count, shutdown_flag, stats, num_slots, num_blocks,
      threads_per_block, stream);
}

//==============================================================================
// Test Kernel for DeviceCallMode
//==============================================================================

using HandlerFunc = int (*)(const void *, void *, std::uint32_t, std::uint32_t,
                            std::uint32_t *);

__device__ HandlerFunc d_increment_handler = increment_handler;

/// @brief Test kernel that dispatches to a handler using DeviceCallMode.
template <typename KernelType>
__global__ void test_dispatch_kernel(HandlerFunc handler, const void *input,
                                     void *output, std::uint32_t arg_len,
                                     std::uint32_t max_result_len,
                                     std::uint32_t *result_len, int *status) {

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *status = handler(input, output, arg_len, max_result_len, result_len);
  }

  KernelType::sync();
}

//==============================================================================
// Main
//==============================================================================
int main() {
  volatile uint64_t *rx_flags_host;
  volatile uint64_t *rx_flags_device;
  std::uint8_t *rx_data_host;
  std::uint8_t *rx_data_device;
  volatile uint64_t *tx_flags_host;
  volatile uint64_t *tx_flags_device;
  std::uint8_t *tx_data_host;
  std::uint8_t *tx_data_device;
  cudaq_function_entry_t *d_function_entries_ = nullptr;
  uint64_t *d_stats_ = nullptr;
  cudaq_dispatch_manager_t *manager_ = nullptr;
  cudaq_dispatcher_t *dispatcher_ = nullptr;
  constexpr std::size_t num_slots_ = 2;
  constexpr std::size_t slot_size_ = 256;

  allocate_ring_buffer(num_slots_, slot_size_, &rx_flags_host, &rx_flags_device,
                       &rx_data_host, &rx_data_device);
  allocate_ring_buffer(num_slots_, slot_size_, &tx_flags_host, &tx_flags_device,
                       &tx_data_host, &tx_data_device);

  const auto write_rpc_request = [&](std::size_t slot,
                                     const std::vector<std::uint8_t> &payload) {
    std::uint8_t *slot_data =
        const_cast<std::uint8_t *>(rx_data_host) + slot * slot_size_;
    auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(slot_data);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = RPC_INCREMENT_FUNCTION_ID;
    header->arg_len = static_cast<std::uint32_t>(payload.size());
    memcpy(slot_data + sizeof(cudaq::realtime::RPCHeader), payload.data(),
           payload.size());
  };

  const auto read_rpc_response = [&](std::size_t slot,
                                     std::vector<std::uint8_t> &payload,
                                     std::int32_t *status_out = nullptr,
                                     std::uint32_t *result_len_out = nullptr) {
    __sync_synchronize();
    // Read from TX buffer (dispatch kernel writes response to symmetric TX)
    const std::uint8_t *slot_data =
        const_cast<std::uint8_t *>(tx_data_host) + slot * slot_size_;
    auto *response =
        reinterpret_cast<const cudaq::realtime::RPCResponse *>(slot_data);

    if (response->magic != cudaq::realtime::RPC_MAGIC_RESPONSE)
      return false;
    if (status_out)
      *status_out = response->status;
    if (result_len_out)
      *result_len_out = response->result_len;
    if (response->status != 0)
      return false;

    payload.resize(response->result_len);
    memcpy(payload.data(), slot_data + sizeof(cudaq::realtime::RPCResponse),
           response->result_len);
    return true;
  };

  void *tmp_shutdown = nullptr;
  CUDA_CHECK(cudaHostAlloc(&tmp_shutdown, sizeof(int), cudaHostAllocMapped));
  volatile int *shutdown_flag_ = static_cast<volatile int *>(tmp_shutdown);
  void *tmp_d_shutdown = nullptr;
  CUDA_CHECK(cudaHostGetDevicePointer(&tmp_d_shutdown, tmp_shutdown, 0));
  volatile int *d_shutdown_flag_ = static_cast<volatile int *>(tmp_d_shutdown);
  *shutdown_flag_ = 0;
  int zero = 0;
  CUDA_CHECK(cudaMemcpy(const_cast<int *>(d_shutdown_flag_), &zero, sizeof(int),
                        cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(&d_stats_, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_stats_, 0, sizeof(uint64_t)));

  CUDA_CHECK(cudaMalloc(&d_function_entries_, sizeof(cudaq_function_entry_t)));
  init_rpc_function_table<<<1, 1>>>(d_function_entries_);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::size_t func_count_ = 1;

  CUDAQ_CHECK(cudaq_dispatch_manager_create(&manager_));
  cudaq_dispatcher_config_t config{};
  config.device_id = 0;
  config.num_blocks = 1;
  config.threads_per_block = 64;
  config.num_slots = static_cast<uint32_t>(num_slots_);
  config.slot_size = static_cast<uint32_t>(slot_size_);
  config.vp_id = 0;
  config.kernel_type = CUDAQ_KERNEL_REGULAR;
  config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  CUDAQ_CHECK(cudaq_dispatcher_create(manager_, &config, &dispatcher_));

  cudaq_ringbuffer_t ringbuffer{};
  ringbuffer.rx_flags = rx_flags_device;
  ringbuffer.tx_flags = tx_flags_device;
  ringbuffer.rx_data = rx_data_device;
  ringbuffer.tx_data = tx_data_device;
  ringbuffer.rx_stride_sz = slot_size_;
  ringbuffer.tx_stride_sz = slot_size_;
  CUDAQ_CHECK(cudaq_dispatcher_set_ringbuffer(dispatcher_, &ringbuffer));

  cudaq_function_table_t table{};
  table.entries = d_function_entries_;
  table.count = func_count_;
  CUDAQ_CHECK(cudaq_dispatcher_set_function_table(dispatcher_, &table));

  CUDAQ_CHECK(
      cudaq_dispatcher_set_control(dispatcher_, d_shutdown_flag_, d_stats_));
  CUDAQ_CHECK(cudaq_dispatcher_set_launch_fn(dispatcher_,
                                             &launch_dispatch_kernel_wrapper));
  CUDAQ_CHECK(cudaq_dispatcher_start(dispatcher_));

  std::vector<std::uint8_t> payload = {0, 1, 2, 3};
  write_rpc_request(0, payload);

  __sync_synchronize();
  const_cast<volatile uint64_t *>(rx_flags_host)[0] =
      reinterpret_cast<std::uint64_t>(rx_data_device);

  int timeout = 50;
  while (tx_flags_host[0] == 0 && timeout-- > 0) {
    usleep(1000);
  }

  if (timeout <= 0) {
    std::cerr << "Timeout waiting for RPC response" << std::endl;
    return 1;
  }

  std::vector<std::uint8_t> response;
  std::int32_t status = -1;
  std::uint32_t result_len = 0;
  read_rpc_response(0, response, &status, &result_len);

  for (std::size_t i = 0; i < result_len; ++i) {
    std::cout << "Response byte " << i << ": " << static_cast<int>(response[i])
              << std::endl;
  }

  return 0;
}
