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
#include <iostream>

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/kernel_types.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_modes.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel.cuh"

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
__device__ int increment_handler(const void* input, void* output,
                                  std::uint32_t arg_len,
                                  std::uint32_t max_result_len,
                                  std::uint32_t* result_len) {
  const std::uint8_t* in_data = static_cast<const std::uint8_t*>(input);
  std::uint8_t* out_data = static_cast<std::uint8_t*>(output);
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

__device__ int rpc_increment_handler(const void* input, void* output,
                                     std::uint32_t arg_len,
                                     std::uint32_t max_result_len,
                                     std::uint32_t* result_len) {
  const std::uint8_t* in_data = static_cast<const std::uint8_t*>(input);
  std::uint8_t* out_data = static_cast<std::uint8_t*>(output);
  for (std::uint32_t i = 0; i < arg_len && i < max_result_len; ++i) {
    out_data[i] = static_cast<std::uint8_t>(in_data[i] + 1);
  }
  *result_len = arg_len;
  return 0;
}

__global__ void init_rpc_function_table(cudaq_function_entry_t* entries) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[0].handler.device_fn_ptr = reinterpret_cast<void*>(&rpc_increment_handler);
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
    entries[0].schema.args[0].size_bytes = 0;  // Variable size
    entries[0].schema.args[0].num_elements = 0; // Variable size
    entries[0].schema.results[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
    entries[0].schema.results[0].reserved[0] = 0;
    entries[0].schema.results[0].reserved[1] = 0;
    entries[0].schema.results[0].reserved[2] = 0;
    entries[0].schema.results[0].size_bytes = 0;  // Variable size
    entries[0].schema.results[0].num_elements = 0; // Variable size
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
    std::uint8_t* rx_data,
    std::uint8_t* tx_data,
    std::size_t rx_stride_sz,
    std::size_t tx_stride_sz,
    cudaq_function_entry_t* function_table,
    std::size_t func_count,
    volatile int* shutdown_flag,
    std::uint64_t* stats,
    std::size_t num_slots,
    std::uint32_t num_blocks,
    std::uint32_t threads_per_block,
    cudaStream_t stream) {
  cudaq_launch_dispatch_kernel_regular(
      rx_flags, tx_flags, rx_data, tx_data, rx_stride_sz, tx_stride_sz,
      function_table, func_count,
      shutdown_flag, stats, num_slots, num_blocks, threads_per_block, stream);
}

//==============================================================================
// Test Kernel for DeviceCallMode
//==============================================================================

using HandlerFunc = int (*)(const void*, void*, std::uint32_t, std::uint32_t, std::uint32_t*);

__device__ HandlerFunc d_increment_handler = increment_handler;

/// @brief Test kernel that dispatches to a handler using DeviceCallMode.
template <typename KernelType>
__global__ void test_dispatch_kernel(
    HandlerFunc handler,
    const void* input,
    void* output,
    std::uint32_t arg_len,
    std::uint32_t max_result_len,
    std::uint32_t* result_len,
    int* status) {
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *status = handler(input, output, arg_len, max_result_len, result_len);
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
  // Prepare test data - separate input and output buffers
  std::vector<uint8_t> input = {0, 1, 2, 3, 4};
  std::vector<uint8_t> expected = {1, 2, 3, 4, 5};

  void* d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, 1024));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size(), 
                        cudaMemcpyHostToDevice));
  
  // Get device function pointer
  HandlerFunc h_handler;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_handler, d_increment_handler, 
                                   sizeof(HandlerFunc)));
  
  // Launch kernel with separate input/output buffers
  test_dispatch_kernel<cudaq::realtime::RegularKernel><<<1, 32>>>(
      h_handler, d_input, d_buffer_, input.size(), 1024, d_result_len_, d_status_);
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
  
  // Verify output buffer has incremented data
  std::vector<uint8_t> output(input.size());
  CUDA_CHECK(cudaMemcpy(output.data(), d_buffer_, output.size(), 
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(expected, output) << "Increment handler should add 1 to each byte";

  // Verify input buffer is unchanged
  std::vector<uint8_t> input_readback(input.size());
  CUDA_CHECK(cudaMemcpy(input_readback.data(), d_input, input.size(),
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(input, input_readback) << "Input buffer should be unchanged";

  cudaFree(d_input);
}

TEST_F(DispatchKernelTest, LargeBuffer) {
  // Test with larger data - separate input/output buffers
  const std::size_t size = 512;
  std::vector<uint8_t> input(size);
  for (std::size_t i = 0; i < size; ++i) {
    input[i] = static_cast<uint8_t>(i & 0xFF);
  }
  
  void* d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, 1024));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size(), 
                        cudaMemcpyHostToDevice));
  
  HandlerFunc h_handler;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_handler, d_increment_handler, 
                                   sizeof(HandlerFunc)));
  
  test_dispatch_kernel<cudaq::realtime::RegularKernel><<<1, 256>>>(
      h_handler, d_input, d_buffer_, input.size(), 1024, d_result_len_, d_status_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  std::uint32_t result_len;
  CUDA_CHECK(cudaMemcpy(&result_len, d_result_len_, sizeof(std::uint32_t), 
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(result_len, size) << "Should process all bytes";
  
  // Verify all bytes incremented in output buffer
  std::vector<uint8_t> output(size);
  CUDA_CHECK(cudaMemcpy(output.data(), d_buffer_, output.size(), 
                        cudaMemcpyDeviceToHost));
  
  for (std::size_t i = 0; i < size; ++i) {
    uint8_t expected = static_cast<uint8_t>((i + 1) & 0xFF);
    EXPECT_EQ(output[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_input);
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

    CUDA_CHECK(cudaMalloc(&d_function_entries_, sizeof(cudaq_function_entry_t)));
    init_rpc_function_table<<<1, 1>>>(d_function_entries_);
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
    ringbuffer.rx_data = rx_data_;
    ringbuffer.tx_data = tx_data_;
    ringbuffer.rx_stride_sz = slot_size_;
    ringbuffer.tx_stride_sz = slot_size_;
    ASSERT_EQ(cudaq_dispatcher_set_ringbuffer(dispatcher_, &ringbuffer), CUDAQ_OK);

    cudaq_function_table_t table{};
    table.entries = d_function_entries_;
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
    if (d_function_entries_)
      cudaFree(d_function_entries_);
  }

  void write_rpc_request(std::size_t slot,
                         const std::vector<std::uint8_t>& payload) {
    std::uint8_t* slot_data =
        const_cast<std::uint8_t*>(rx_data_host_) + slot * slot_size_;
    auto* header = reinterpret_cast<cudaq::realtime::RPCHeader*>(slot_data);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = RPC_INCREMENT_FUNCTION_ID;
    header->arg_len = static_cast<std::uint32_t>(payload.size());
    memcpy(slot_data + sizeof(cudaq::realtime::RPCHeader), payload.data(),
           payload.size());
  }

  bool read_rpc_response(std::size_t slot,
                         std::vector<std::uint8_t>& payload,
                         std::int32_t* status_out = nullptr,
                         std::uint32_t* result_len_out = nullptr) {
    __sync_synchronize();
    // Read from TX buffer (dispatch kernel writes response to symmetric TX)
    const std::uint8_t* slot_data =
        const_cast<std::uint8_t*>(tx_data_host_) + slot * slot_size_;
    auto* response =
        reinterpret_cast<const cudaq::realtime::RPCResponse*>(slot_data);

    if (response->magic != cudaq::realtime::RPC_MAGIC_RESPONSE)
      return false;
    if (status_out)
      *status_out = response->status;
    if (result_len_out)
      *result_len_out = response->result_len;
    if (response->status != 0)
      return false;

    payload.resize(response->result_len);
    memcpy(payload.data(),
           slot_data + sizeof(cudaq::realtime::RPCResponse),
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

  cudaq_function_entry_t* d_function_entries_ = nullptr;
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

//==============================================================================
// Graph Launch Test
//==============================================================================

// Graph kernel that processes RPC buffer via pointer indirection
__global__ void graph_increment_kernel(void** buffer_ptr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    void* buffer = *buffer_ptr;
    cudaq::realtime::RPCHeader* header = static_cast<cudaq::realtime::RPCHeader*>(buffer);
    
    std::uint32_t arg_len = header->arg_len;
    void* arg_buffer = static_cast<void*>(header + 1);
    std::uint8_t* data = static_cast<std::uint8_t*>(arg_buffer);
    
    // Increment each byte
    for (std::uint32_t i = 0; i < arg_len; ++i) {
      data[i] = data[i] + 1;
    }
    
    // Write response
    cudaq::realtime::RPCResponse* response = static_cast<cudaq::realtime::RPCResponse*>(buffer);
    response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
    response->status = 0;
    response->result_len = arg_len;
  }
}

constexpr std::uint32_t RPC_GRAPH_INCREMENT_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("rpc_graph_increment");

__global__ void init_graph_function_table(cudaq_function_entry_t* entries, 
                                          cudaGraphExec_t graph_exec) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[0].handler.graph_exec = graph_exec;
    entries[0].function_id = RPC_GRAPH_INCREMENT_FUNCTION_ID;
    entries[0].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
    entries[0].reserved[0] = 0;
    entries[0].reserved[1] = 0;
    entries[0].reserved[2] = 0;
  }
}

TEST(GraphLaunchTest, DispatchKernelGraphLaunch) {
  // Check compute capability
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  
  if (prop.major < 9) {
    GTEST_SKIP() << "Graph device launch requires compute capability 9.0+, found " 
                 << prop.major << "." << prop.minor;
  }
  
  // Allocate graph buffer pointer (for pointer indirection pattern)
  void** d_graph_buffer_ptr;
  CUDA_CHECK(cudaMalloc(&d_graph_buffer_ptr, sizeof(void*)));
  CUDA_CHECK(cudaMemset(d_graph_buffer_ptr, 0, sizeof(void*)));
  
  // Allocate test buffer
  constexpr size_t buffer_size = 1024;
  void* d_buffer;
  CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
  
  // Create the child graph (the one that will be launched from device)
  cudaGraph_t child_graph;
  cudaGraphExec_t child_graph_exec;
  
  CUDA_CHECK(cudaGraphCreate(&child_graph, 0));
  
  // Add kernel node to child graph
  cudaKernelNodeParams kernel_params = {};
  void* kernel_args[] = {&d_graph_buffer_ptr};
  kernel_params.func = reinterpret_cast<void*>(&graph_increment_kernel);
  kernel_params.gridDim = dim3(1, 1, 1);
  kernel_params.blockDim = dim3(32, 1, 1);
  kernel_params.sharedMemBytes = 0;
  kernel_params.kernelParams = kernel_args;
  kernel_params.extra = nullptr;
  
  cudaGraphNode_t kernel_node;
  CUDA_CHECK(cudaGraphAddKernelNode(&kernel_node, child_graph, nullptr, 0, &kernel_params));
  
  // Instantiate CHILD graph with DEVICE LAUNCH FLAG
  CUDA_CHECK(cudaGraphInstantiate(&child_graph_exec, child_graph,  
                                   cudaGraphInstantiateFlagDeviceLaunch));
  
  // Create stream for operations
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  
  // Upload the child graph to device
  CUDA_CHECK(cudaGraphUpload(child_graph_exec, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  
  // Set up function table with graph launch entry
  cudaq_function_entry_t* d_function_entries;
  CUDA_CHECK(cudaMalloc(&d_function_entries, sizeof(cudaq_function_entry_t)));
  init_graph_function_table<<<1, 1>>>(d_function_entries, child_graph_exec);
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Set up RPC buffer on host
  std::uint8_t* h_buffer = new std::uint8_t[buffer_size];
  cudaq::realtime::RPCHeader* h_header = reinterpret_cast<cudaq::realtime::RPCHeader*>(h_buffer);
  h_header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  h_header->function_id = RPC_GRAPH_INCREMENT_FUNCTION_ID;
  h_header->arg_len = 4;
  
  std::uint8_t* h_data = h_buffer + sizeof(cudaq::realtime::RPCHeader);
  h_data[0] = 0;
  h_data[1] = 1;
  h_data[2] = 2;
  h_data[3] = 3;
  
  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_buffer, h_buffer, buffer_size, cudaMemcpyHostToDevice));
  
  // Set up fake RX/TX flags for single-shot test
  volatile uint64_t* d_rx_flags;
  volatile uint64_t* d_tx_flags;
  CUDA_CHECK(cudaMalloc(&d_rx_flags, sizeof(uint64_t)));
  CUDA_CHECK(cudaMalloc(&d_tx_flags, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset((void*)d_rx_flags, 0, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset((void*)d_tx_flags, 0, sizeof(uint64_t)));
  
  // Set RX flag to point to our buffer (simulating incoming RPC)
  uint64_t buffer_addr = reinterpret_cast<uint64_t>(d_buffer);
  CUDA_CHECK(cudaMemcpy((void*)d_rx_flags, &buffer_addr, sizeof(uint64_t), cudaMemcpyHostToDevice));
  
  // Set up shutdown flag using pinned mapped memory so the dispatch kernel
  // can see host updates immediately
  volatile int* h_shutdown;
  volatile int* d_shutdown;
  {
    void* tmp_shutdown;
    CUDA_CHECK(cudaHostAlloc(&tmp_shutdown, sizeof(int), cudaHostAllocMapped));
    h_shutdown = static_cast<volatile int*>(tmp_shutdown);
    *h_shutdown = 0;
    
    void* tmp_d_shutdown;
    CUDA_CHECK(cudaHostGetDevicePointer(&tmp_d_shutdown, tmp_shutdown, 0));
    d_shutdown = static_cast<volatile int*>(tmp_d_shutdown);
  }
  
  // Set up stats
  uint64_t* d_stats;
  CUDA_CHECK(cudaMalloc(&d_stats, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(uint64_t)));
  
  // Create dispatch graph context - THIS WRAPS THE DISPATCH KERNEL IN A GRAPH
  // so that device-side cudaGraphLaunch() can work!
  cudaq_dispatch_graph_context* dispatch_ctx = nullptr;
  cudaError_t err = cudaq_create_dispatch_graph_regular(
      d_rx_flags, d_tx_flags,
      reinterpret_cast<std::uint8_t*>(d_buffer),  // rx_data
      reinterpret_cast<std::uint8_t*>(d_buffer),  // tx_data (same buffer for single-slot test)
      buffer_size,  // rx_stride_sz
      buffer_size,  // tx_stride_sz
      d_function_entries, 1,
      d_graph_buffer_ptr, d_shutdown, d_stats, 1,
      1, 32, stream, &dispatch_ctx);
  
  if (err != cudaSuccess) {
    GTEST_SKIP() << "Device-side graph launch not supported: " 
                 << cudaGetErrorString(err) << " (" << err << ")";
  }
  
  // Launch dispatch graph - now device-side cudaGraphLaunch will work!
  CUDA_CHECK(cudaq_launch_dispatch_graph(dispatch_ctx, stream));
  
  // Poll for the response using pinned memory and async operations
  // The child graph runs asynchronously (fire-and-forget) so we need to poll
  std::uint8_t* h_poll_buffer;
  CUDA_CHECK(cudaHostAlloc(&h_poll_buffer, sizeof(cudaq::realtime::RPCResponse), cudaHostAllocDefault));
  memset(h_poll_buffer, 0, sizeof(cudaq::realtime::RPCResponse));
  
  cudaStream_t poll_stream;
  CUDA_CHECK(cudaStreamCreate(&poll_stream));
  
  int timeout_ms = 5000;
  int poll_interval_ms = 100;
  bool got_response = false;
  
  for (int elapsed = 0; elapsed < timeout_ms; elapsed += poll_interval_ms) {
    CUDA_CHECK(cudaMemcpyAsync(h_poll_buffer, d_buffer, sizeof(cudaq::realtime::RPCResponse), 
                                cudaMemcpyDeviceToHost, poll_stream));
    CUDA_CHECK(cudaStreamSynchronize(poll_stream));
    
    cudaq::realtime::RPCResponse* peek = reinterpret_cast<cudaq::realtime::RPCResponse*>(h_poll_buffer);
    if (peek->magic == cudaq::realtime::RPC_MAGIC_RESPONSE) {
      got_response = true;
      break;
    }
    
    usleep(poll_interval_ms * 1000);
  }
  
  // Signal shutdown to allow kernel to exit
  *h_shutdown = 1;
  __sync_synchronize();
  usleep(100000); // Give kernel time to see shutdown flag
  
  // Copy final results
  CUDA_CHECK(cudaMemcpyAsync(h_buffer, d_buffer, buffer_size, cudaMemcpyDeviceToHost, poll_stream));
  CUDA_CHECK(cudaStreamSynchronize(poll_stream));
  
  // Clean up poll resources  
  CUDA_CHECK(cudaStreamDestroy(poll_stream));
  cudaFreeHost(h_poll_buffer);
  
  // Sync main stream (dispatch kernel should have exited)
  CUDA_CHECK(cudaStreamSynchronize(stream));
  
  ASSERT_TRUE(got_response) << "Timeout waiting for device-side graph launch response";
  
  // Verify response
  cudaq::realtime::RPCResponse* h_response = reinterpret_cast<cudaq::realtime::RPCResponse*>(h_buffer);
  EXPECT_EQ(h_response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE) 
      << "Expected RPC_MAGIC_RESPONSE, got 0x" << std::hex << h_response->magic;
  EXPECT_EQ(h_response->status, 0) << "Handler returned error status";
  EXPECT_EQ(h_response->result_len, 4u) << "Unexpected result length";
  
  // Verify data was incremented by graph kernel launched from dispatch kernel
  std::uint8_t* h_result = h_buffer + sizeof(cudaq::realtime::RPCResponse);
  EXPECT_EQ(h_result[0], 1) << "Expected h_result[0]=1";
  EXPECT_EQ(h_result[1], 2) << "Expected h_result[1]=2";
  EXPECT_EQ(h_result[2], 3) << "Expected h_result[2]=3";
  EXPECT_EQ(h_result[3], 4) << "Expected h_result[3]=4";
  
  // Cleanup
  delete[] h_buffer;
  CUDA_CHECK(cudaq_destroy_dispatch_graph(dispatch_ctx));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_stats));
  CUDA_CHECK(cudaFreeHost(const_cast<int*>(h_shutdown)));  // Free mapped memory
  CUDA_CHECK(cudaFree((void*)d_tx_flags));
  CUDA_CHECK(cudaFree((void*)d_rx_flags));
  CUDA_CHECK(cudaFree(d_function_entries));
  CUDA_CHECK(cudaGraphExecDestroy(child_graph_exec));
  CUDA_CHECK(cudaGraphDestroy(child_graph));
  CUDA_CHECK(cudaFree(d_graph_buffer_ptr));
  CUDA_CHECK(cudaFree(d_buffer));
}

} // namespace
