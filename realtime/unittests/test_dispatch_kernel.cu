/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstring>
#include <unistd.h>

#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/nvqlink/daemon/dispatcher/kernel_types.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_modes.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel.cuh"

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

//==============================================================================
// Test Handler: Simple noop that copies input to output
//==============================================================================

/// @brief Test handler that adds 1 to each byte.
__device__ int increment_handler(void* buffer, std::uint32_t arg_len,
                                  std::uint32_t max_result_len,
                                  std::uint32_t* result_len) {
  std::uint8_t* data = static_cast<std::uint8_t*>(buffer);
  for (std::uint32_t i = 0; i < arg_len && i < max_result_len; ++i) {
    data[i] = data[i] + 1;
  }
  *result_len = arg_len;
  return 0;
}

//==============================================================================
// Host API Dispatch Kernel Test Helpers
//==============================================================================

constexpr std::uint32_t RPC_INCREMENT_FUNCTION_ID =
    cudaq::nvqlink::fnv1a_hash("rpc_increment");

__device__ int rpc_increment_handler(void* buffer, std::uint32_t arg_len,
                                     std::uint32_t max_result_len,
                                     std::uint32_t* result_len) {
  std::uint8_t* data = static_cast<std::uint8_t*>(buffer);
  for (std::uint32_t i = 0; i < arg_len && i < max_result_len; ++i) {
    data[i] = static_cast<std::uint8_t>(data[i] + 1);
  }
  *result_len = arg_len;
  return 0;
}

__global__ void init_rpc_function_table(void** table, std::uint32_t* ids) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    table[0] = reinterpret_cast<void*>(&rpc_increment_handler);
    ids[0] = RPC_INCREMENT_FUNCTION_ID;
  }
}

bool allocate_ring_buffer(std::size_t num_slots, std::size_t slot_size,
                          volatile uint64_t** host_flags_out,
                          volatile uint64_t** device_flags_out,
                          std::uint8_t** host_data_out,
                          std::uint8_t** device_data_out) {
  void* host_flags_ptr = nullptr;
  cudaError_t err = cudaHostAlloc(&host_flags_ptr,
                                  num_slots * sizeof(uint64_t),
                                  cudaHostAllocMapped);
  if (err != cudaSuccess)
    return false;

  void* device_flags_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_flags_ptr, host_flags_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    return false;
  }

  void* host_data_ptr = nullptr;
  err = cudaHostAlloc(&host_data_ptr,
                      num_slots * slot_size,
                      cudaHostAllocMapped);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    return false;
  }

  void* device_data_ptr = nullptr;
  err = cudaHostGetDevicePointer(&device_data_ptr, host_data_ptr, 0);
  if (err != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    cudaFreeHost(host_data_ptr);
    return false;
  }

  memset(host_flags_ptr, 0, num_slots * sizeof(uint64_t));

  *host_flags_out = static_cast<volatile uint64_t*>(host_flags_ptr);
  *device_flags_out = static_cast<volatile uint64_t*>(device_flags_ptr);
  *host_data_out = static_cast<std::uint8_t*>(host_data_ptr);
  *device_data_out = static_cast<std::uint8_t*>(device_data_ptr);
  return true;
}

void free_ring_buffer(volatile uint64_t* host_flags,
                      std::uint8_t* host_data) {
  if (host_flags)
    cudaFreeHost(const_cast<uint64_t*>(host_flags));
  if (host_data)
    cudaFreeHost(host_data);
}

extern "C" void launch_dispatch_kernel_wrapper(
    volatile std::uint64_t* rx_flags,
    volatile std::uint64_t* tx_flags,
    void** function_table,
    std::uint32_t* function_ids,
    std::size_t func_count,
    volatile int* shutdown_flag,
    std::uint64_t* stats,
    std::size_t num_slots,
    std::uint32_t num_blocks,
    std::uint32_t threads_per_block,
    cudaStream_t stream) {
  cudaq::nvqlink::launch_dispatch_kernel_regular_inline(
      rx_flags, tx_flags, function_table, function_ids, func_count,
      shutdown_flag, stats, num_slots, num_blocks, threads_per_block, stream);
}

//==============================================================================
// Test Kernel for DeviceCallMode
//==============================================================================

using HandlerFunc = int (*)(void*, std::uint32_t, std::uint32_t, std::uint32_t*);

__device__ HandlerFunc d_increment_handler = increment_handler;

/// @brief Test kernel that dispatches to a handler using DeviceCallMode.
template <typename KernelType>
__global__ void test_dispatch_kernel(
    HandlerFunc handler,
    void* buffer,
    std::uint32_t arg_len,
    std::uint32_t max_result_len,
    std::uint32_t* result_len,
    int* status) {
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *status = handler(buffer, arg_len, max_result_len, result_len);
  }
  
  KernelType::sync();
}

//==============================================================================
// Test Fixture
//==============================================================================

class DispatchKernelTest : public ::testing::Test {
protected:
  void SetUp() override {
    CUDA_CHECK(cudaMalloc(&d_buffer_, 1024));
    CUDA_CHECK(cudaMalloc(&d_result_len_, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_status_, sizeof(int)));
  }
  
  void TearDown() override {
    if (d_buffer_) cudaFree(d_buffer_);
    if (d_result_len_) cudaFree(d_result_len_);
    if (d_status_) cudaFree(d_status_);
  }
  
  void* d_buffer_ = nullptr;
  std::uint32_t* d_result_len_ = nullptr;
  int* d_status_ = nullptr;
};

//==============================================================================
// Tests
//==============================================================================

TEST_F(DispatchKernelTest, IncrementHandlerBasic) {
  // Prepare test data
  std::vector<uint8_t> input = {0, 1, 2, 3, 4};
  std::vector<uint8_t> expected = {1, 2, 3, 4, 5};
  CUDA_CHECK(cudaMemcpy(d_buffer_, input.data(), input.size(), 
                        cudaMemcpyHostToDevice));
  
  // Get device function pointer
  HandlerFunc h_handler;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_handler, d_increment_handler, 
                                   sizeof(HandlerFunc)));
  
  // Launch kernel
  test_dispatch_kernel<cudaq::realtime::RegularKernel><<<1, 32>>>(
      h_handler, d_buffer_, input.size(), 1024, d_result_len_, d_status_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Check results
  int status;
  std::uint32_t result_len;
  CUDA_CHECK(cudaMemcpy(&status, d_status_, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result_len, d_result_len_, sizeof(std::uint32_t), 
                        cudaMemcpyDeviceToHost));
  
  EXPECT_EQ(status, 0) << "Handler should return success";
  EXPECT_EQ(result_len, input.size()) << "Result length should match input";
  
  // Verify data incremented
  std::vector<uint8_t> output(input.size());
  CUDA_CHECK(cudaMemcpy(output.data(), d_buffer_, output.size(), 
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(expected, output) << "Increment handler should add 1 to each byte";
}

TEST_F(DispatchKernelTest, LargeBuffer) {
  // Test with larger data
  const std::size_t size = 512;
  std::vector<uint8_t> input(size);
  for (std::size_t i = 0; i < size; ++i) {
    input[i] = static_cast<uint8_t>(i & 0xFF);
  }
  
  CUDA_CHECK(cudaMemcpy(d_buffer_, input.data(), input.size(), 
                        cudaMemcpyHostToDevice));
  
  HandlerFunc h_handler;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_handler, d_increment_handler, 
                                   sizeof(HandlerFunc)));
  
  test_dispatch_kernel<cudaq::realtime::RegularKernel><<<1, 256>>>(
      h_handler, d_buffer_, input.size(), 1024, d_result_len_, d_status_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  std::uint32_t result_len;
  CUDA_CHECK(cudaMemcpy(&result_len, d_result_len_, sizeof(std::uint32_t), 
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(result_len, size) << "Should process all bytes";
  
  // Verify all bytes incremented
  std::vector<uint8_t> output(size);
  CUDA_CHECK(cudaMemcpy(output.data(), d_buffer_, output.size(), 
                        cudaMemcpyDeviceToHost));
  
  for (std::size_t i = 0; i < size; ++i) {
    uint8_t expected = static_cast<uint8_t>((i + 1) & 0xFF);
    EXPECT_EQ(output[i], expected) << "Mismatch at index " << i;
  }
}

class HostApiDispatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &rx_flags_host_,
                                     &rx_flags_, &rx_data_host_, &rx_data_));
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &tx_flags_host_,
                                     &tx_flags_, &tx_data_host_, &tx_data_));

    void* tmp_shutdown = nullptr;
    CUDA_CHECK(cudaHostAlloc(&tmp_shutdown, sizeof(int), cudaHostAllocMapped));
    shutdown_flag_ = static_cast<volatile int*>(tmp_shutdown);
    void* tmp_d_shutdown = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&tmp_d_shutdown, tmp_shutdown, 0));
    d_shutdown_flag_ = static_cast<volatile int*>(tmp_d_shutdown);
    *shutdown_flag_ = 0;
    int zero = 0;
    CUDA_CHECK(cudaMemcpy(const_cast<int*>(d_shutdown_flag_), &zero,
                          sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc(&d_stats_, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(d_stats_, 0, sizeof(uint64_t)));

    CUDA_CHECK(cudaMalloc(&d_function_table_, sizeof(void*)));
    CUDA_CHECK(cudaMalloc(&d_function_ids_, sizeof(std::uint32_t)));
    init_rpc_function_table<<<1, 1>>>(d_function_table_, d_function_ids_);
    CUDA_CHECK(cudaDeviceSynchronize());
    func_count_ = 1;

    ASSERT_EQ(cudaq_dispatch_manager_create(&manager_), CUDAQ_OK);
    cudaq_dispatcher_config_t config{};
    config.device_id = 0;
    config.num_blocks = 1;
    config.threads_per_block = 64;
    config.num_slots = static_cast<uint32_t>(num_slots_);
    config.slot_size = static_cast<uint32_t>(slot_size_);
    config.vp_id = 0;
    config.kernel_type = CUDAQ_KERNEL_REGULAR;
    config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    ASSERT_EQ(cudaq_dispatcher_create(manager_, &config, &dispatcher_), CUDAQ_OK);

    cudaq_ringbuffer_t ringbuffer{};
    ringbuffer.rx_flags = rx_flags_;
    ringbuffer.tx_flags = tx_flags_;
    ASSERT_EQ(cudaq_dispatcher_set_ringbuffer(dispatcher_, &ringbuffer), CUDAQ_OK);

    cudaq_function_table_t table{};
    table.device_function_ptrs = d_function_table_;
    table.function_ids = d_function_ids_;
    table.count = func_count_;
    ASSERT_EQ(cudaq_dispatcher_set_function_table(dispatcher_, &table), CUDAQ_OK);

    ASSERT_EQ(
        cudaq_dispatcher_set_control(dispatcher_, d_shutdown_flag_, d_stats_),
        CUDAQ_OK);
    ASSERT_EQ(cudaq_dispatcher_set_launch_fn(dispatcher_,
                                             &launch_dispatch_kernel_wrapper),
              CUDAQ_OK);
    ASSERT_EQ(cudaq_dispatcher_start(dispatcher_), CUDAQ_OK);
  }

  void TearDown() override {
    if (shutdown_flag_) {
      *shutdown_flag_ = 1;
      __sync_synchronize();
    }
    if (dispatcher_) {
      cudaq_dispatcher_stop(dispatcher_);
      cudaq_dispatcher_destroy(dispatcher_);
      dispatcher_ = nullptr;
    }
    if (manager_) {
      cudaq_dispatch_manager_destroy(manager_);
      manager_ = nullptr;
    }
    free_ring_buffer(rx_flags_host_, rx_data_host_);
    free_ring_buffer(tx_flags_host_, tx_data_host_);

    if (shutdown_flag_)
      cudaFreeHost(const_cast<int*>(shutdown_flag_));
    if (d_stats_)
      cudaFree(d_stats_);
    if (d_function_table_)
      cudaFree(d_function_table_);
    if (d_function_ids_)
      cudaFree(d_function_ids_);
  }

  void write_rpc_request(std::size_t slot,
                         const std::vector<std::uint8_t>& payload) {
    std::uint8_t* slot_data =
        const_cast<std::uint8_t*>(rx_data_host_) + slot * slot_size_;
    auto* header = reinterpret_cast<cudaq::nvqlink::RPCHeader*>(slot_data);
    header->function_id = RPC_INCREMENT_FUNCTION_ID;
    header->arg_len = static_cast<std::uint32_t>(payload.size());
    memcpy(slot_data + sizeof(cudaq::nvqlink::RPCHeader), payload.data(),
           payload.size());
  }

  bool read_rpc_response(std::size_t slot,
                         std::vector<std::uint8_t>& payload,
                         std::int32_t* status_out = nullptr,
                         std::uint32_t* result_len_out = nullptr) {
    __sync_synchronize();
    const std::uint8_t* slot_data =
        const_cast<std::uint8_t*>(rx_data_host_) + slot * slot_size_;
    auto* response =
        reinterpret_cast<const cudaq::nvqlink::RPCResponse*>(slot_data);

    if (status_out)
      *status_out = response->status;
    if (result_len_out)
      *result_len_out = response->result_len;
    if (response->status != 0)
      return false;

    payload.resize(response->result_len);
    memcpy(payload.data(),
           slot_data + sizeof(cudaq::nvqlink::RPCResponse),
           response->result_len);
    return true;
  }

  static constexpr std::size_t num_slots_ = 2;
  std::size_t slot_size_ = 256;
  volatile uint64_t* rx_flags_host_ = nullptr;
  volatile uint64_t* tx_flags_host_ = nullptr;
  volatile uint64_t* rx_flags_ = nullptr;
  volatile uint64_t* tx_flags_ = nullptr;
  std::uint8_t* rx_data_host_ = nullptr;
  std::uint8_t* tx_data_host_ = nullptr;
  std::uint8_t* rx_data_ = nullptr;
  std::uint8_t* tx_data_ = nullptr;

  volatile int* shutdown_flag_ = nullptr;
  volatile int* d_shutdown_flag_ = nullptr;
  uint64_t* d_stats_ = nullptr;

  void** d_function_table_ = nullptr;
  std::uint32_t* d_function_ids_ = nullptr;
  std::size_t func_count_ = 0;

  cudaq_dispatch_manager_t* manager_ = nullptr;
  cudaq_dispatcher_t* dispatcher_ = nullptr;
};

TEST_F(HostApiDispatchTest, RpcIncrementHandler) {
  std::vector<std::uint8_t> payload = {0, 1, 2, 3};
  write_rpc_request(0, payload);

  __sync_synchronize();
  const_cast<volatile uint64_t*>(rx_flags_host_)[0] =
      reinterpret_cast<std::uint64_t>(rx_data_);

  int timeout = 50;
  while (tx_flags_host_[0] == 0 && timeout-- > 0) {
    usleep(1000);
  }
  ASSERT_GT(timeout, 0) << "Timeout waiting for dispatch kernel response";

  std::vector<std::uint8_t> response;
  std::int32_t status = -1;
  std::uint32_t result_len = 0;
  ASSERT_TRUE(read_rpc_response(0, response, &status, &result_len));
  EXPECT_EQ(status, 0);
  ASSERT_EQ(result_len, payload.size());

  std::vector<std::uint8_t> expected = {1, 2, 3, 4};
  EXPECT_EQ(response, expected);
}

} // namespace
