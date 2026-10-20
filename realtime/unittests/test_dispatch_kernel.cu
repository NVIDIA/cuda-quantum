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
#include <chrono>
#include <iostream>

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel.cuh"
#include "cudaq/realtime/daemon/dispatcher/dispatch_modes.h"
#include "cudaq/realtime/daemon/dispatcher/unified_device_transport.cuh"

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
///
/// Same two-pointer form as a host-call handler: the request (RPCHeader +
/// payload) is in rx_slot, the result goes after the RPCResponse in tx_slot,
/// and the handler sets result_len.  The dispatch kernel fills the other
/// response header fields.
__device__ int increment_handler(const void* rx_slot, void* tx_slot,
                                 std::size_t slot_size) {
  const auto* header = static_cast<const cudaq::realtime::RPCHeader*>(rx_slot);
  const std::uint8_t* in_data =
      reinterpret_cast<const std::uint8_t*>(header + 1);
  auto* response = static_cast<cudaq::realtime::RPCResponse*>(tx_slot);
  std::uint8_t* out_data = reinterpret_cast<std::uint8_t*>(response + 1);
  const std::uint32_t arg_len = header->arg_len;
  const std::uint32_t max_result_len =
      slot_size - sizeof(cudaq::realtime::RPCResponse);
  for (std::uint32_t i = 0; i < arg_len && i < max_result_len; ++i) {
    out_data[i] = in_data[i] + 1;
  }
  response->result_len = arg_len;
  return 0;
}

//==============================================================================
// Host API Dispatch Kernel Test Helpers
//==============================================================================

constexpr std::uint32_t RPC_INCREMENT_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("rpc_increment");

__device__ int rpc_increment_handler(const void* rx_slot, void* tx_slot,
                                     std::size_t slot_size) {
  const auto* header = static_cast<const cudaq::realtime::RPCHeader*>(rx_slot);
  const std::uint8_t* in_data =
      reinterpret_cast<const std::uint8_t*>(header + 1);
  auto* response = static_cast<cudaq::realtime::RPCResponse*>(tx_slot);
  std::uint8_t* out_data = reinterpret_cast<std::uint8_t*>(response + 1);
  const std::uint32_t arg_len = header->arg_len;
  const std::uint32_t max_result_len =
      slot_size - sizeof(cudaq::realtime::RPCResponse);
  for (std::uint32_t i = 0; i < arg_len && i < max_result_len; ++i) {
    out_data[i] = static_cast<std::uint8_t>(in_data[i] + 1);
  }
  response->result_len = arg_len;
  return 0;
}

constexpr std::uint32_t ALL_THREADS_COUNT_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("all_threads_count");

/// Counts handler invocations: the dispatch kernels call the handler on every
/// thread of their block, so one request adds threads_per_block.
__device__ unsigned long long g_all_threads_count = 0;

__device__ int all_threads_count_handler(const void* /*rx_slot*/,
                                         void* tx_slot,
                                         std::size_t /*slot_size*/) {
  atomicAdd(&g_all_threads_count, 1ull);
  if (threadIdx.x == 0)
    static_cast<cudaq::realtime::RPCResponse*>(tx_slot)->result_len = 0;
  return 0;
}

/// Entries in the table init_rpc_function_table fills: rpc_increment and
/// all_threads_count.
constexpr std::size_t kRpcFuncCount = 2;

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

    entries[1] = cudaq_function_entry_t{};
    entries[1].handler.device_fn_ptr =
        reinterpret_cast<void*>(&all_threads_count_handler);
    entries[1].function_id = ALL_THREADS_COUNT_FUNCTION_ID;
    entries[1].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
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

using HandlerFunc = int (*)(const void*, void*, std::size_t);

__device__ HandlerFunc d_increment_handler = increment_handler;

/// @brief Test kernel that dispatches to a handler using DeviceCallMode.
///
/// Mirrors the dispatch kernel's two-pointer call: the handler gets the RX
/// slot (RPCHeader + payload), the TX slot (RPCResponse + result) and the
/// slot size, and returns a status.
__global__ void test_dispatch_kernel(
    HandlerFunc handler,
    const void* rx_slot,
    void* tx_slot,
    std::size_t slot_size,
    int* status) {
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *status = handler(rx_slot, tx_slot, slot_size);
  }

  __syncthreads();
}

//==============================================================================
// Test Fixture
//==============================================================================

class DispatchKernelTest : public ::testing::Test {
protected:
  void SetUp() override {
    CUDA_CHECK(cudaMalloc(&d_buffer_, kSlotSize));
    CUDA_CHECK(cudaMalloc(&d_status_, sizeof(int)));
  }
  
  void TearDown() override {
    if (d_buffer_) cudaFree(d_buffer_);
    if (d_status_) cudaFree(d_status_);
  }

  // Stage an RPC request (RPCHeader + payload) into a host RX-slot image.
  static std::vector<uint8_t> make_rx_slot(const std::vector<uint8_t>& payload) {
    std::vector<uint8_t> rx_slot(kSlotSize, 0);
    auto* header =
        reinterpret_cast<cudaq::realtime::RPCHeader*>(rx_slot.data());
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = RPC_INCREMENT_FUNCTION_ID;
    header->arg_len = static_cast<std::uint32_t>(payload.size());
    std::memcpy(rx_slot.data() + sizeof(cudaq::realtime::RPCHeader),
                payload.data(), payload.size());
    return rx_slot;
  }

  static constexpr std::size_t kSlotSize = 1024;
  void* d_buffer_ = nullptr; // TX slot: RPCResponse + result bytes
  int* d_status_ = nullptr;
};

//==============================================================================
// Tests
//==============================================================================

TEST_F(DispatchKernelTest, IncrementHandlerBasic) {
  // Prepare test data - separate RX (request) and TX (response) slots
  std::vector<uint8_t> input = {0, 1, 2, 3, 4};
  std::vector<uint8_t> expected = {1, 2, 3, 4, 5};
  const std::vector<uint8_t> rx_slot = make_rx_slot(input);

  void* d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, kSlotSize));
  CUDA_CHECK(cudaMemcpy(d_input, rx_slot.data(), rx_slot.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_buffer_, 0, kSlotSize));

  // Get device function pointer
  HandlerFunc h_handler;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_handler, d_increment_handler,
                                   sizeof(HandlerFunc)));

  // Launch kernel with separate RX/TX slots (the dispatcher's two-pointer call)
  test_dispatch_kernel<<<1, 32>>>(h_handler, d_input, d_buffer_, kSlotSize,
                                  d_status_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check results: the status comes back as the return value, result_len
  // from the RPCResponse the handler filled in the TX slot
  int status;
  CUDA_CHECK(cudaMemcpy(&status, d_status_, sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<uint8_t> tx_slot(kSlotSize);
  CUDA_CHECK(cudaMemcpy(tx_slot.data(), d_buffer_, tx_slot.size(),
                        cudaMemcpyDeviceToHost));
  const auto* response =
      reinterpret_cast<const cudaq::realtime::RPCResponse*>(tx_slot.data());

  EXPECT_EQ(status, 0) << "Handler should return success";
  EXPECT_EQ(response->result_len, input.size())
      << "Result length should match input";

  // Verify the result bytes after the RPCResponse header are incremented
  const auto result_begin =
      tx_slot.begin() + sizeof(cudaq::realtime::RPCResponse);
  std::vector<uint8_t> output(result_begin, result_begin + input.size());
  EXPECT_EQ(expected, output) << "Increment handler should add 1 to each byte";

  // Verify the RX slot is unchanged
  std::vector<uint8_t> input_readback(rx_slot.size());
  CUDA_CHECK(cudaMemcpy(input_readback.data(), d_input, rx_slot.size(),
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(rx_slot, input_readback) << "RX slot should be unchanged";

  cudaFree(d_input);
}

TEST_F(DispatchKernelTest, LargeBuffer) {
  // Test with larger data - separate RX/TX slots
  const std::size_t size = 512;
  std::vector<uint8_t> input(size);
  for (std::size_t i = 0; i < size; ++i) {
    input[i] = static_cast<uint8_t>(i & 0xFF);
  }
  const std::vector<uint8_t> rx_slot = make_rx_slot(input);

  void* d_input = nullptr;
  CUDA_CHECK(cudaMalloc(&d_input, kSlotSize));
  CUDA_CHECK(cudaMemcpy(d_input, rx_slot.data(), rx_slot.size(),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_buffer_, 0, kSlotSize));

  HandlerFunc h_handler;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_handler, d_increment_handler,
                                   sizeof(HandlerFunc)));

  test_dispatch_kernel<<<1, 256>>>(h_handler, d_input, d_buffer_, kSlotSize,
                                   d_status_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<uint8_t> tx_slot(kSlotSize);
  CUDA_CHECK(cudaMemcpy(tx_slot.data(), d_buffer_, tx_slot.size(),
                        cudaMemcpyDeviceToHost));
  const auto* response =
      reinterpret_cast<const cudaq::realtime::RPCResponse*>(tx_slot.data());
  EXPECT_EQ(response->result_len, size) << "Should process all bytes";

  // Verify all bytes after the RPCResponse header are incremented
  const uint8_t* output =
      tx_slot.data() + sizeof(cudaq::realtime::RPCResponse);
  for (std::size_t i = 0; i < size; ++i) {
    uint8_t expected = static_cast<uint8_t>((i + 1) & 0xFF);
    EXPECT_EQ(output[i], expected) << "Mismatch at index " << i;
  }

  cudaFree(d_input);
}

class HostApiDispatchTest : public ::testing::Test {
protected:
  // The default-launcher fixtures below override these: they leave the
  // launcher to the dispatcher, which picks the archive's for the kernel type.
  virtual cudaq_kernel_type_t kernelType() const {
    return CUDAQ_KERNEL_REGULAR;
  }
  virtual bool setLaunchFn() const { return true; }

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

    CUDA_CHECK(cudaMalloc(&d_function_entries_,
                          kRpcFuncCount * sizeof(cudaq_function_entry_t)));
    init_rpc_function_table<<<1, 1>>>(d_function_entries_);
    CUDA_CHECK(cudaDeviceSynchronize());
    func_count_ = kRpcFuncCount;
    // No kernel is running yet, so the symbol copy cannot race a handler.
    const unsigned long long zero_count = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(g_all_threads_count, &zero_count,
                                  sizeof(zero_count)));

    ASSERT_EQ(cudaq_dispatch_manager_create(&manager_), CUDAQ_OK);
    cudaq_dispatcher_config_t config{};
    config.device_id = 0;
    config.num_blocks = 1;
    config.threads_per_block = kThreadsPerBlock;
    config.num_slots = static_cast<uint32_t>(num_slots_);
    config.slot_size = static_cast<uint32_t>(slot_size_);
    config.vp_id = 0;
    config.kernel_type = kernelType();
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
    if (setLaunchFn()) {
      ASSERT_EQ(cudaq_dispatcher_set_launch_fn(
                    dispatcher_, &launch_dispatch_kernel_wrapper),
                CUDAQ_OK);
    }
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

  void write_rpc_request(
      std::size_t slot, const std::vector<std::uint8_t>& payload,
      std::uint32_t function_id = RPC_INCREMENT_FUNCTION_ID) {
    std::uint8_t* slot_data =
        const_cast<std::uint8_t*>(rx_data_host_) + slot * slot_size_;
    auto* header = reinterpret_cast<cudaq::realtime::RPCHeader*>(slot_data);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = function_id;
    header->arg_len = static_cast<std::uint32_t>(payload.size());
    if (!payload.empty())
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

  // One increment request through slot 0, answered by the running kernel.
  void expectIncrementRoundTrip() {
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

  static constexpr std::size_t num_slots_ = 2;
  static constexpr std::uint32_t kThreadsPerBlock = 64;
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

TEST_F(HostApiDispatchTest, RpcIncrementHandler) { expectIncrementRoundTrip(); }

// Every thread of the dispatcher's block calls the handler once per request.
TEST_F(HostApiDispatchTest, AllThreadsCallHandler) {
  write_rpc_request(0, {}, ALL_THREADS_COUNT_FUNCTION_ID);
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
  std::uint32_t result_len = 1;
  ASSERT_TRUE(read_rpc_response(0, response, &status, &result_len));
  EXPECT_EQ(status, 0);
  EXPECT_EQ(result_len, 0u);

  // The dispatcher's stream is blocking: stop the kernel before reading a
  // device symbol.  TearDown's second stop is a no-op.
  *shutdown_flag_ = 1;
  __sync_synchronize();
  ASSERT_EQ(cudaq_dispatcher_stop(dispatcher_), CUDAQ_OK);
  unsigned long long count = 0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&count, g_all_threads_count, sizeof(count)));
  EXPECT_EQ(count, static_cast<unsigned long long>(kThreadsPerBlock))
      << "every thread of the 1 x " << kThreadsPerBlock
      << " block must call the handler exactly once";
  std::uint64_t processed = 0;
  ASSERT_EQ(cudaq_dispatcher_get_processed(dispatcher_, &processed), CUDAQ_OK);
  EXPECT_EQ(processed, 1u);
}

// No cudaq_dispatcher_set_launch_fn: start() takes the archive's launcher for
// the configured kernel type, linked into this binary.
class DefaultLauncherDispatchTest : public HostApiDispatchTest {
protected:
  bool setLaunchFn() const override { return false; }
};

TEST_F(DefaultLauncherDispatchTest, RegularRpcIncrementHandler) {
  expectIncrementRoundTrip();
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
      1, 32, /*triggered_graph_exec=*/nullptr, stream, &dispatch_ctx);
  
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

//==============================================================================
// Device-graph self-relaunch test: a DEVICE_CALL handler returns
// CUDAQ_DISPATCH_STATUS_TRIGGER_GRAPH on every call, so the scheduler fires a
// trivial "triggered" graph fire-and-forget and tail-self-relaunches via
// cudaGetCurrentGraphExec().  Feeding far more than 120 messages proves the
// self-relaunch resets the 120-fire-and-forget-launch-per-parent-execution
// budget, so the device-graph dispatch path can trigger an unbounded number of
// follow-up graphs (a single long-running parent graph would stall at 120).
//==============================================================================

constexpr std::uint32_t APPEND_TRIGGER_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("append_trigger");

// Handler invocations: every thread of the scheduler block calls the handler,
// while the triggered graph must fire exactly once per request.
__device__ unsigned long long g_trigger_handler_calls = 0;

// DEVICE_CALL handler that always requests the configured triggered graph fire.
__device__ int append_trigger_handler(const void* rx_slot, void* tx_slot,
                                      std::size_t slot_size) {
  (void)rx_slot;
  (void)slot_size;
  atomicAdd(&g_trigger_handler_calls, 1ull);
  if (threadIdx.x == 0)
    static_cast<cudaq::realtime::RPCResponse*>(tx_slot)->result_len = 0;
  return CUDAQ_DISPATCH_STATUS_TRIGGER_GRAPH;
}

__global__ void init_append_trigger_function_table(
    cudaq_function_entry_t* entries) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[0].handler.device_fn_ptr =
        reinterpret_cast<void*>(&append_trigger_handler);
    entries[0].function_id = APPEND_TRIGGER_FUNCTION_ID;
    entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    entries[0].reserved[0] = 0;
    entries[0].reserved[1] = 0;
    entries[0].reserved[2] = 0;
    entries[0].schema.num_args = 0;
    entries[0].schema.num_results = 0;
    entries[0].schema.reserved = 0;
  }
}

// Trivial body of the triggered (follow-up) graph: count invocations.
__global__ void triggered_counter_kernel(int* counter) {
  if (threadIdx.x == 0 && blockIdx.x == 0)
    atomicAdd(counter, 1);
}

TEST(DeviceGraphSelfRelaunchTest, Exceeds120) {
  // Device-side graph launch (cudaStreamGraphTailLaunch / device-launchable
  // graphs) requires compute capability 9.0+ (Hopper). On older GPUs the
  // instantiate/create calls below still succeed, but the device-side launches
  // silently no-op (the counter never advances), so guard explicitly here --
  // matching GraphLaunchTest.DispatchKernelGraphLaunch.
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  if (prop.major < 9) {
    GTEST_SKIP() << "Graph device launch requires compute capability 9.0+, "
                    "found "
                 << prop.major << "." << prop.minor;
  }

  // One triggered launch per message; >> 120 to prove the budget reset.
  constexpr int kNumMessages = 200;
  constexpr std::size_t kNumSlots = kNumMessages; // each slot used once
  constexpr std::size_t kSlotSize = 64;

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  // --- Counter + triggered graph (device-launchable) ---
  int* d_counter = nullptr;
  CUDA_CHECK(cudaMalloc(&d_counter, sizeof(int)));
  CUDA_CHECK(cudaMemset(d_counter, 0, sizeof(int)));

  cudaGraph_t trig_graph;
  CUDA_CHECK(cudaGraphCreate(&trig_graph, 0));
  cudaKernelNodeParams trig_params = {};
  void* trig_args[] = {&d_counter};
  trig_params.func = reinterpret_cast<void*>(triggered_counter_kernel);
  trig_params.gridDim = dim3(1, 1, 1);
  trig_params.blockDim = dim3(1, 1, 1);
  trig_params.sharedMemBytes = 0;
  trig_params.kernelParams = trig_args;
  trig_params.extra = nullptr;
  cudaGraphNode_t trig_node;
  CUDA_CHECK(
      cudaGraphAddKernelNode(&trig_node, trig_graph, nullptr, 0, &trig_params));
  cudaGraphExec_t trig_exec = nullptr;
  cudaError_t inst_err = cudaGraphInstantiate(
      &trig_exec, trig_graph, cudaGraphInstantiateFlagDeviceLaunch);
  if (inst_err != cudaSuccess) {
    GTEST_SKIP() << "Device-launchable graph not supported: "
                 << cudaGetErrorString(inst_err);
  }
  CUDA_CHECK(cudaGraphUpload(trig_exec, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // --- Function table: 1 DEVICE_CALL append-trigger handler ---
  cudaq_function_entry_t* d_function_entries;
  CUDA_CHECK(cudaMalloc(&d_function_entries, sizeof(cudaq_function_entry_t)));
  init_append_trigger_function_table<<<1, 1>>>(d_function_entries);
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Rings (pinned-mapped) ---
  volatile uint64_t *h_rx_flags, *d_rx_flags, *h_tx_flags, *d_tx_flags;
  std::uint8_t *h_rx_data, *d_rx_data, *h_tx_data, *d_tx_data;
  ASSERT_TRUE(allocate_ring_buffer(kNumSlots, kSlotSize, &h_rx_flags,
                                   &d_rx_flags, &h_rx_data, &d_rx_data));
  ASSERT_TRUE(allocate_ring_buffer(kNumSlots, kSlotSize, &h_tx_flags,
                                   &d_tx_flags, &h_tx_data, &d_tx_data));

  // Pre-lay all messages: RPCHeader per rx slot; rx_flag = device slot addr.
  for (int i = 0; i < kNumMessages; ++i) {
    std::uint8_t* slot = h_rx_data + static_cast<std::size_t>(i) * kSlotSize;
    auto* hdr = reinterpret_cast<cudaq::realtime::RPCHeader*>(slot);
    hdr->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    hdr->function_id = APPEND_TRIGGER_FUNCTION_ID;
    hdr->arg_len = 0;
    hdr->request_id = static_cast<std::uint32_t>(i);
    hdr->ptp_timestamp = 0;
    h_rx_flags[i] = reinterpret_cast<uint64_t>(
        d_rx_data + static_cast<std::size_t>(i) * kSlotSize);
  }
  __sync_synchronize();

  // --- Shutdown flag (pinned-mapped) ---
  volatile int* h_shutdown;
  volatile int* d_shutdown;
  {
    void* tmp;
    CUDA_CHECK(cudaHostAlloc(&tmp, sizeof(int), cudaHostAllocMapped));
    h_shutdown = static_cast<volatile int*>(tmp);
    *h_shutdown = 0;
    void* tmpd;
    CUDA_CHECK(cudaHostGetDevicePointer(&tmpd, tmp, 0));
    d_shutdown = static_cast<volatile int*>(tmpd);
  }

  uint64_t* d_stats;
  CUDA_CHECK(cudaMalloc(&d_stats, sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(uint64_t)));

  // current_slot resets to 0 on each self-relaunch, so the scheduler must scan
  // the ring for the next pending slot (shared_ring_mode enables that scan).
  CUDA_CHECK(cudaq_dispatch_kernel_set_shared_ring_mode(1));

  cudaq_dispatch_graph_context* ctx = nullptr;
  cudaError_t err = cudaq_create_dispatch_graph_regular(
      d_rx_flags, d_tx_flags, d_rx_data, d_tx_data, kSlotSize, kSlotSize,
      d_function_entries, 1, /*graph_io_ctx=*/nullptr, d_shutdown, d_stats,
      kNumSlots, /*num_blocks=*/1, /*threads_per_block=*/32, trig_exec, stream,
      &ctx);
  if (err != cudaSuccess) {
    cudaq_dispatch_kernel_set_shared_ring_mode(0);
    GTEST_SKIP() << "Device-side graph launch not supported: "
                 << cudaGetErrorString(err);
  }

  // No kernel is running yet, so the symbol copy cannot race a handler.
  const unsigned long long zero_calls = 0;
  CUDA_CHECK(
      cudaMemcpyToSymbol(g_trigger_handler_calls, &zero_calls, sizeof(zero_calls)));

  CUDA_CHECK(cudaq_launch_dispatch_graph(ctx, stream));

  // Poll the counter on a SEPARATE stream: the scheduler is a persistent
  // self-relaunching graph on `stream`, so a synchronous copy (null stream)
  // would deadlock waiting for it to idle.  cudaMemcpyAsync on poll_stream uses
  // the copy engine concurrently.
  cudaStream_t poll_stream;
  CUDA_CHECK(cudaStreamCreate(&poll_stream));
  int counter = 0;
  for (int elapsed = 0; elapsed < 10000; elapsed += 50) {
    CUDA_CHECK(cudaMemcpyAsync(&counter, d_counter, sizeof(int),
                               cudaMemcpyDeviceToHost, poll_stream));
    CUDA_CHECK(cudaStreamSynchronize(poll_stream));
    if (counter >= kNumMessages)
      break;
    usleep(50 * 1000);
  }

  // Stop the scheduler and wait for the self-relaunch chain to drain.
  *h_shutdown = 1;
  __sync_synchronize();
  usleep(200000);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUDA_CHECK(cudaMemcpyAsync(&counter, d_counter, sizeof(int),
                             cudaMemcpyDeviceToHost, poll_stream));
  CUDA_CHECK(cudaStreamSynchronize(poll_stream));
  CUDA_CHECK(cudaStreamDestroy(poll_stream));

  EXPECT_GT(counter, 120)
      << "Only " << counter
      << " triggered launches completed -- self-relaunch likely hit the 120 "
         "fire-and-forget cap.";
  EXPECT_EQ(counter, kNumMessages)
      << "Expected " << kNumMessages << " triggered launches, got " << counter;

  // The scheduler has exited (stream synchronized above), so the symbol can be
  // read: every thread of the 1 x 32 block called the handler for each of the
  // kNumMessages requests, yet (above) each request fired the graph once.
  unsigned long long handler_calls = 0;
  CUDA_CHECK(cudaMemcpyFromSymbol(&handler_calls, g_trigger_handler_calls,
                                  sizeof(handler_calls)));
  EXPECT_EQ(handler_calls, static_cast<unsigned long long>(kNumMessages) * 32)
      << "every thread of the scheduler block must call the handler once per "
         "request";

  // Cleanup
  CUDA_CHECK(cudaq_dispatch_kernel_set_shared_ring_mode(0));
  CUDA_CHECK(cudaq_destroy_dispatch_graph(ctx));
  CUDA_CHECK(cudaFree(d_stats));
  CUDA_CHECK(cudaFreeHost(const_cast<int*>(h_shutdown)));
  free_ring_buffer(h_rx_flags, h_rx_data);
  free_ring_buffer(h_tx_flags, h_tx_data);
  CUDA_CHECK(cudaFree(d_function_entries));
  CUDA_CHECK(cudaGraphExecDestroy(trig_exec));
  CUDA_CHECK(cudaGraphDestroy(trig_graph));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_counter));
}

// The self-relaunching scheduler is single-block by construction (block-scoped
// __shared__ state + a single device-global cursor), so
// cudaq_create_dispatch_graph_regular must reject num_blocks > 1 when a
// triggered graph is provided. The validation runs before any CUDA work, so
// this needs no device-side graph support (it runs on pre-Hopper CI too).
TEST(DeviceGraphSelfRelaunchTest, RejectsMultiBlockTriggered) {
  cudaq_dispatch_graph_context *ctx = nullptr;
  cudaGraphExec_t fake_exec = reinterpret_cast<cudaGraphExec_t>(0x1);
  cudaError_t err = cudaq_create_dispatch_graph_regular(
      /*rx_flags=*/nullptr, /*tx_flags=*/nullptr, /*rx_data=*/nullptr,
      /*tx_data=*/nullptr, /*rx_stride_sz=*/64, /*tx_stride_sz=*/64,
      /*function_table=*/nullptr, /*func_count=*/0, /*graph_io_ctx=*/nullptr,
      /*shutdown_flag=*/nullptr, /*stats=*/nullptr, /*num_slots=*/1,
      /*num_blocks=*/2, /*threads_per_block=*/32, fake_exec, /*stream=*/0,
      &ctx);
  EXPECT_EQ(err, cudaErrorInvalidValue);
  EXPECT_EQ(ctx, nullptr);
}

} // namespace

//==============================================================================
// Unified dispatch kernel (CUDAQ_KERNEL_UNIFIED)
//
// Hardware-free tests of lib/daemon/dispatcher/unified_dispatch_kernel.cu,
// which this executable compiles directly (see unittests/CMakeLists.txt).
// The kernel reaches its transport through the three `__device__` hooks of
// unified_device_transport.cuh; this file supplies them over a pinned, mapped
// ring the host drives directly.  No NIC is involved: a request is signalled
// by storing the DEVICE address of its RX frame into rx_flags[slot], the
// response goes into the TX frame with the same index, and tx_publish hands
// that frame's device address back through tx_flags[slot].
//==============================================================================

namespace {

/// Device-side view of the ring.  Every pointer is the device alias of a
/// cudaHostAllocMapped allocation the host reads and writes directly.
struct FakeDevCtx {
  volatile std::uint64_t *rx_flags; ///< [slot] = device addr of RX frame, or 0
  volatile std::uint64_t *tx_flags; ///< [slot] = device addr of TX frame, or 0
  std::uint8_t *tx_data;
  std::size_t stride;
  std::size_t num_slots;
};

/// Per-launch session the hooks keep in the kernel's session storage.
struct FakeSession {
  FakeDevCtx c;
  std::size_t cursor;
  volatile int *shutdown_flag;
};
static_assert(sizeof(FakeSession) <= CUDAQ_DEV_TRANSPORT_SESSION_BYTES,
              "FakeSession must fit the kernel's session storage");
static_assert(alignof(FakeSession) <= CUDAQ_DEV_TRANSPORT_SESSION_ALIGN,
              "FakeSession alignment exceeds the session storage's");

} // namespace

extern "C" __device__ void *
cudaq_dev_transport_attach(void *storage, void *ctx,
                           volatile int *shutdown_flag) {
  if (ctx == nullptr)
    return nullptr;
  auto *session = static_cast<FakeSession *>(storage);
  session->c = *static_cast<const FakeDevCtx *>(ctx);
  session->cursor = 0;
  session->shutdown_flag = shutdown_flag;
  return session;
}

extern "C" __device__ cudaq_rx_dev_status_t
cudaq_dev_rx_poll(void *session, void **out_request, void **out_response) {
  auto *s = static_cast<FakeSession *>(session);
  while (true) {
    __threadfence_system();
    const std::uint64_t request_addr = s->c.rx_flags[s->cursor];
    if (request_addr != 0) {
      // Acquire: the request body was written before the flag.
      __threadfence_system();
      *out_request = reinterpret_cast<void *>(request_addr);
      *out_response = s->c.tx_data + s->cursor * s->c.stride;
      return CUDAQ_RX_DEV_READY;
    }
    if (*s->shutdown_flag != 0)
      return CUDAQ_RX_DEV_SHUTDOWN;
  }
}

extern "C" __device__ void cudaq_dev_tx_publish(void *session, void *response) {
  auto *s = static_cast<FakeSession *>(session);
  const std::size_t slot =
      (static_cast<std::uint8_t *>(response) - s->c.tx_data) / s->c.stride;
  // Return the receive credit first, then release: the response body and the
  // cleared RX flag are both ordered before the TX flag the host polls, so a
  // host that reuses the slot as soon as it sees the TX flag cannot race the
  // clear.
  s->c.rx_flags[slot] = 0;
  __threadfence_system();
  s->c.tx_flags[slot] = reinterpret_cast<std::uint64_t>(response);
  s->cursor = (slot + 1) % s->c.num_slots;
}

namespace {

using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPC_MAGIC_RESPONSE;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

//==============================================================================
// Function table: rpc_increment_handler and all_threads_count_handler, the
// same two entries the ring tests use (init_rpc_function_table).
//==============================================================================

constexpr std::uint32_t kUnknownFunctionId =
    cudaq::realtime::fnv1a_hash("no_such_function");

//==============================================================================
// Host helpers
//==============================================================================

/// Poll `stream` until it is idle.  False on timeout or if the stream
/// reports a fault; never blocks unboundedly, so a wedged kernel fails the
/// test instead of hanging the run.
bool wait_stream_idle(cudaStream_t stream, int timeout_ms) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (true) {
    const cudaError_t q = cudaStreamQuery(stream);
    if (q == cudaSuccess)
      return true;
    if (q != cudaErrorNotReady) {
      ADD_FAILURE() << "stream reported " << cudaGetErrorString(q);
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline)
      return false;
    usleep(100);
  }
}

/// What the kernel reports for one request: the response header, the result
/// bytes, and the tx_flags value the publish left behind.
struct Response {
  RPCResponse header{};
  std::vector<std::uint8_t> payload;
  std::uint64_t tx_flag = 0;
};

//==============================================================================
// Fixture: mapped rings, control block, function table, fake context
//==============================================================================

class UnifiedDispatchKernelTest : public ::testing::Test {
protected:
  static constexpr std::size_t kNumSlots = 4;
  static constexpr std::size_t kSlotSize = 256;
  static constexpr int kTimeoutMs = 5000;
  static constexpr unsigned long long kThreadsPerBlock = 64;

  void SetUp() override {
    ASSERT_TRUE(allocate_ring_buffer(kNumSlots, kSlotSize, &rx_flags_host_,
                                     &rx_flags_, &rx_data_host_, &rx_data_));
    ASSERT_TRUE(allocate_ring_buffer(kNumSlots, kSlotSize, &tx_flags_host_,
                                     &tx_flags_, &tx_data_host_, &tx_data_));
    std::memset(rx_data_host_, 0, kNumSlots * kSlotSize);
    std::memset(tx_data_host_, 0, kNumSlots * kSlotSize);

    void *tmp_shutdown = nullptr;
    CUDA_CHECK(cudaHostAlloc(&tmp_shutdown, sizeof(int), cudaHostAllocMapped));
    shutdown_flag_ = static_cast<volatile int *>(tmp_shutdown);
    *shutdown_flag_ = 0;
    void *tmp_d_shutdown = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&tmp_d_shutdown, tmp_shutdown, 0));
    d_shutdown_flag_ = static_cast<volatile int *>(tmp_d_shutdown);

    CUDA_CHECK(cudaMalloc(&d_stats_, sizeof(std::uint64_t)));
    CUDA_CHECK(cudaMemset(d_stats_, 0, sizeof(std::uint64_t)));

    // Non-blocking: the legacy default stream must never wait on the
    // persistent kernel, or any synchronous CUDA call below would deadlock.
    // Created first so even the table init runs on it: a cudaDeviceSynchronize
    // here would wait on a kernel that a failed earlier test had to leak.
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));

    CUDA_CHECK(cudaMalloc(&d_entries_,
                          kRpcFuncCount * sizeof(cudaq_function_entry_t)));
    init_rpc_function_table<<<1, 1, 0, stream_>>>(d_entries_);
    CUDA_CHECK(cudaGetLastError());
    ASSERT_TRUE(wait_stream_idle(stream_, kTimeoutMs))
        << "function table init did not complete";

    // No kernel is running yet, so the symbol copy cannot race a handler.
    const unsigned long long zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(g_all_threads_count, &zero, sizeof(zero)));

    FakeDevCtx ctx{};
    ctx.rx_flags = rx_flags_;
    ctx.tx_flags = tx_flags_;
    ctx.tx_data = tx_data_;
    ctx.stride = kSlotSize;
    ctx.num_slots = kNumSlots;
    CUDA_CHECK(cudaMalloc(&d_ctx_, sizeof(FakeDevCtx)));
    CUDA_CHECK(cudaMemcpy(d_ctx_, &ctx, sizeof(ctx), cudaMemcpyHostToDevice));
  }

  void TearDown() override {
    // Unconditional, so a failed test cannot leave a kernel spinning into
    // the next one.  The dispatcher-owned kernel runs on a BLOCKING stream:
    // set the flag through the mapped host pointer first, as the bridge tool
    // does, so stop()'s own cudaMemcpy finds the kernel already leaving.
    // stop() then synchronizes that stream with no timeout, so a kernel that
    // ignored the flag would hang here; the direct-launch tests bound that
    // same shutdown path with a timeout.
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

    if (kernel_live_) {
      if (!wait_stream_idle(stream_, kTimeoutMs)) {
        // Freeing memory a live kernel polls would turn a timeout into a
        // crash; leak it and let the failure stand.
        ADD_FAILURE() << "unified kernel still running after shutdown";
        return;
      }
      kernel_live_ = false;
    }

    if (stream_)
      cudaStreamDestroy(stream_);
    if (d_ctx_)
      cudaFree(d_ctx_);
    if (d_entries_)
      cudaFree(d_entries_);
    if (d_stats_)
      cudaFree(d_stats_);
    if (shutdown_flag_)
      cudaFreeHost(const_cast<int *>(shutdown_flag_));
    free_ring_buffer(rx_flags_host_, rx_data_host_);
    free_ring_buffer(tx_flags_host_, tx_data_host_);
  }

  //--- Direct launch ---------------------------------------------------------

  /// Launch the unified kernel as one block of kThreadsPerBlock threads over
  /// `transport_ctx` (nullptr makes attach refuse).
  bool launch(void *transport_ctx) {
    cudaq_launch_unified_dispatch_device(
        transport_ctx, kSlotSize, d_entries_, kRpcFuncCount, d_shutdown_flag_,
        d_stats_, kThreadsPerBlock, stream_);
    const cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    kernel_live_ = (err == cudaSuccess);
    return kernel_live_;
  }

  /// Raise the shutdown flag and wait for the directly-launched kernel to
  /// leave.  Bounded.
  bool shutdown_and_join() {
    *shutdown_flag_ = 1;
    __sync_synchronize();
    return join();
  }

  /// Wait for the directly-launched kernel to leave without raising the
  /// flag (attach-refusal exits on its own).  Bounded.
  bool join() {
    if (!wait_stream_idle(stream_, kTimeoutMs))
      return false;
    kernel_live_ = false;
    return true;
  }

  //--- Requests and responses (host memory only) -----------------------------

  void post_request(std::size_t slot, std::uint32_t function_id,
                    const std::vector<std::uint8_t> &payload,
                    std::uint32_t request_id, std::uint64_t ptp_timestamp) {
    std::uint8_t *frame = rx_data_host_ + slot * kSlotSize;
    auto *header = reinterpret_cast<RPCHeader *>(frame);
    header->magic = RPC_MAGIC_REQUEST;
    header->function_id = function_id;
    header->arg_len = static_cast<std::uint32_t>(payload.size());
    header->request_id = request_id;
    header->ptp_timestamp = ptp_timestamp;
    if (!payload.empty())
      std::memcpy(frame + sizeof(RPCHeader), payload.data(), payload.size());
    __sync_synchronize();
    // The flag carries the DEVICE address of the frame: the kernel
    // dereferences it as the request header.
    rx_flags_host_[slot] =
        reinterpret_cast<std::uint64_t>(rx_data_ + slot * kSlotSize);
  }

  bool wait_response(std::size_t slot) {
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::milliseconds(kTimeoutMs);
    while (true) {
      __sync_synchronize();
      if (tx_flags_host_[slot] != 0)
        return true;
      if (std::chrono::steady_clock::now() >= deadline)
        return false;
      usleep(50);
    }
  }

  /// Read and consume the response in `slot` (clears its tx flag).
  Response read_response(std::size_t slot) {
    __sync_synchronize();
    Response r;
    r.tx_flag = tx_flags_host_[slot];
    const std::uint8_t *frame = tx_data_host_ + slot * kSlotSize;
    std::memcpy(&r.header, frame, sizeof(RPCResponse));
    if (r.header.result_len <= kSlotSize - sizeof(RPCResponse))
      r.payload.assign(frame + sizeof(RPCResponse),
                       frame + sizeof(RPCResponse) + r.header.result_len);
    tx_flags_host_[slot] = 0;
    return r;
  }

  bool round_trip(std::size_t slot, std::uint32_t function_id,
                  const std::vector<std::uint8_t> &payload,
                  std::uint32_t request_id, std::uint64_t ptp_timestamp,
                  Response *out) {
    post_request(slot, function_id, payload, request_id, ptp_timestamp);
    if (!wait_response(slot)) {
      ADD_FAILURE() << "timeout waiting for the response in slot " << slot;
      return false;
    }
    *out = read_response(slot);
    return true;
  }

  /// tx_publish stores the device address of the response frame.
  std::uint64_t expected_tx_flag(std::size_t slot) const {
    return reinterpret_cast<std::uint64_t>(tx_data_ + slot * kSlotSize);
  }

  //--- Device-side readbacks: only while no kernel is running ---------------

  std::uint64_t read_stats() {
    std::uint64_t stats = ~0ull;
    const cudaError_t err =
        cudaMemcpy(&stats, d_stats_, sizeof(stats), cudaMemcpyDeviceToHost);
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    return stats;
  }

  unsigned long long read_all_threads_count() {
    unsigned long long count = ~0ull;
    const cudaError_t err =
        cudaMemcpyFromSymbol(&count, g_all_threads_count, sizeof(count));
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
    return count;
  }

  //--- State -----------------------------------------------------------------

  volatile std::uint64_t *rx_flags_host_ = nullptr;
  volatile std::uint64_t *tx_flags_host_ = nullptr;
  volatile std::uint64_t *rx_flags_ = nullptr;
  volatile std::uint64_t *tx_flags_ = nullptr;
  std::uint8_t *rx_data_host_ = nullptr;
  std::uint8_t *tx_data_host_ = nullptr;
  std::uint8_t *rx_data_ = nullptr;
  std::uint8_t *tx_data_ = nullptr;

  volatile int *shutdown_flag_ = nullptr;
  volatile int *d_shutdown_flag_ = nullptr;
  std::uint64_t *d_stats_ = nullptr;
  cudaq_function_entry_t *d_entries_ = nullptr;
  FakeDevCtx *d_ctx_ = nullptr;

  cudaStream_t stream_ = nullptr;
  bool kernel_live_ = false;

  cudaq_dispatch_manager_t *manager_ = nullptr;
  cudaq_dispatcher_t *dispatcher_ = nullptr;
};

//==============================================================================
// Tests
//==============================================================================

TEST_F(UnifiedDispatchKernelTest, RoundTrip) {
  ASSERT_TRUE(launch(d_ctx_));

  Response r;
  ASSERT_TRUE(round_trip(0, RPC_INCREMENT_FUNCTION_ID, {1, 2, 3, 4}, 0x1234u,
                         0xABCDEF0123ull, &r));
  EXPECT_EQ(r.header.magic, RPC_MAGIC_RESPONSE);
  EXPECT_EQ(r.header.status, 0);
  EXPECT_EQ(r.header.result_len, 4u);
  EXPECT_EQ(r.header.request_id, 0x1234u);
  EXPECT_EQ(r.header.ptp_timestamp, 0xABCDEF0123ull);
  EXPECT_EQ(r.payload, (std::vector<std::uint8_t>{2, 3, 4, 5}));
  EXPECT_EQ(r.tx_flag, expected_tx_flag(0));

  // Every thread of the 1 x kThreadsPerBlock block calls the handler.
  ASSERT_TRUE(
      round_trip(1, ALL_THREADS_COUNT_FUNCTION_ID, {}, 2u, 200ull, &r));
  EXPECT_EQ(r.header.magic, RPC_MAGIC_RESPONSE);
  EXPECT_EQ(r.header.status, 0);
  EXPECT_EQ(r.header.result_len, 0u);
  EXPECT_EQ(r.header.request_id, 2u);
  EXPECT_EQ(r.header.ptp_timestamp, 200ull);
  EXPECT_TRUE(r.payload.empty());
  EXPECT_EQ(r.tx_flag, expected_tx_flag(1));

  ASSERT_TRUE(
      round_trip(2, RPC_INCREMENT_FUNCTION_ID, {250, 255, 7}, 3u, 300ull, &r));
  EXPECT_EQ(r.header.magic, RPC_MAGIC_RESPONSE);
  EXPECT_EQ(r.header.status, 0);
  EXPECT_EQ(r.header.result_len, 3u);
  EXPECT_EQ(r.header.request_id, 3u);
  EXPECT_EQ(r.header.ptp_timestamp, 300ull);
  EXPECT_EQ(r.payload, (std::vector<std::uint8_t>{251, 0, 8}));
  EXPECT_EQ(r.tx_flag, expected_tx_flag(2));

  ASSERT_TRUE(shutdown_and_join()) << "kernel did not exit on shutdown";
  EXPECT_EQ(read_stats(), 3u);
  EXPECT_EQ(read_all_threads_count(), kThreadsPerBlock)
      << "every thread of the block must have called the handler once";
}

TEST_F(UnifiedDispatchKernelTest, UnknownFunction) {
  ASSERT_TRUE(launch(d_ctx_));

  // Publishing is mandatory even for a frame nobody can dispatch: the
  // response goes out with status -1 and no result, and the tx flag is set.
  Response r;
  ASSERT_TRUE(
      round_trip(0, kUnknownFunctionId, {9, 9, 9}, 0x77u, 0x5555ull, &r));
  EXPECT_EQ(r.header.magic, RPC_MAGIC_RESPONSE);
  EXPECT_EQ(r.header.status, -1);
  EXPECT_EQ(r.header.result_len, 0u);
  EXPECT_EQ(r.header.request_id, 0x77u);
  EXPECT_EQ(r.header.ptp_timestamp, 0x5555ull);
  EXPECT_EQ(r.tx_flag, expected_tx_flag(0));

  ASSERT_TRUE(shutdown_and_join()) << "kernel did not exit on shutdown";
  EXPECT_EQ(read_stats(), 1u);
  EXPECT_EQ(read_all_threads_count(), 0ull);
}

TEST_F(UnifiedDispatchKernelTest, AttachRefused) {
  // attach returns nullptr for a null context; thread 0 publishes "leave" and
  // the whole block exits without the shutdown flag.
  ASSERT_TRUE(launch(nullptr));
  ASSERT_TRUE(join()) << "block did not exit by itself after attach refused";
  EXPECT_EQ(read_stats(), 0u);
  EXPECT_EQ(*shutdown_flag_, 0);
}

TEST_F(UnifiedDispatchKernelTest, DispatcherApi) {
  ASSERT_EQ(cudaq_dispatch_manager_create(&manager_), CUDAQ_OK);

  cudaq_dispatcher_config_t base{};
  base.device_id = 0;
  base.num_blocks = 1;
  base.threads_per_block = kThreadsPerBlock;
  base.num_slots = static_cast<std::uint32_t>(kNumSlots);
  base.slot_size = static_cast<std::uint32_t>(kSlotSize);
  base.vp_id = 0;
  base.kernel_type = CUDAQ_KERNEL_UNIFIED;
  base.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  base.dispatch_path = CUDAQ_DISPATCH_PATH_DEVICE;

  cudaq_function_table_t table{};
  table.entries = d_entries_;
  table.count = static_cast<std::uint32_t>(kRpcFuncCount);

  // A fully wired dispatcher for `cfg`, running the archive's unified kernel
  // (no launch override).
  auto make_dispatcher =
      [&](const cudaq_dispatcher_config_t &cfg) -> cudaq_dispatcher_t * {
    cudaq_dispatcher_t *d = nullptr;
    if (cudaq_dispatcher_create(manager_, &cfg, &d) != CUDAQ_OK)
      return nullptr;
    if (cudaq_dispatcher_set_function_table(d, &table) != CUDAQ_OK ||
        cudaq_dispatcher_set_control(d, d_shutdown_flag_, d_stats_) !=
            CUDAQ_OK ||
        cudaq_dispatcher_set_unified_launch(d, nullptr, d_ctx_) != CUDAQ_OK) {
      cudaq_dispatcher_destroy(d);
      return nullptr;
    }
    return d;
  };

  //--- Negatives: nothing below may leave a kernel running -------------------

  {
    // The block is what every DEVICE_CALL handler runs on: exactly one.
    cudaq_dispatcher_config_t cfg = base;
    cfg.num_blocks = 2;
    cudaq_dispatcher_t *d = make_dispatcher(cfg);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(cudaq_dispatcher_start(d), CUDAQ_ERR_INVALID_ARG);
    cudaq_dispatcher_destroy(d);
  }
  {
    cudaq_dispatcher_config_t cfg = base;
    cfg.threads_per_block = 0;
    cudaq_dispatcher_t *d = make_dispatcher(cfg);
    ASSERT_NE(d, nullptr);
    EXPECT_EQ(cudaq_dispatcher_start(d), CUDAQ_ERR_INVALID_ARG);
    cudaq_dispatcher_destroy(d);
  }

  //--- Positive: full start / round trip / stop flow -------------------------

  dispatcher_ = make_dispatcher(base);
  ASSERT_NE(dispatcher_, nullptr);
  ASSERT_EQ(cudaq_dispatcher_start(dispatcher_), CUDAQ_OK);

  // The dispatcher's stream is BLOCKING: from here until stop() returns, only
  // host memory may be touched (no cudaMemcpy, no get_processed).
  Response r;
  ASSERT_TRUE(round_trip(0, RPC_INCREMENT_FUNCTION_ID, {10, 20, 30, 40}, 42u,
                         4242ull, &r));
  EXPECT_EQ(r.header.magic, RPC_MAGIC_RESPONSE);
  EXPECT_EQ(r.header.status, 0);
  EXPECT_EQ(r.header.result_len, 4u);
  EXPECT_EQ(r.header.request_id, 42u);
  EXPECT_EQ(r.header.ptp_timestamp, 4242ull);
  EXPECT_EQ(r.payload, (std::vector<std::uint8_t>{11, 21, 31, 41}));
  EXPECT_EQ(r.tx_flag, expected_tx_flag(0));

  ASSERT_TRUE(
      round_trip(1, ALL_THREADS_COUNT_FUNCTION_ID, {}, 43u, 4343ull, &r));
  EXPECT_EQ(r.header.status, 0);
  EXPECT_EQ(r.header.result_len, 0u);
  EXPECT_EQ(r.header.request_id, 43u);

  // Flag first, through the mapped host pointer, then stop(): its cudaMemcpy
  // to the flag would otherwise wait on the still-running kernel.
  *shutdown_flag_ = 1;
  __sync_synchronize();
  ASSERT_EQ(cudaq_dispatcher_stop(dispatcher_), CUDAQ_OK);

  std::uint64_t processed = 0;
  ASSERT_EQ(cudaq_dispatcher_get_processed(dispatcher_, &processed), CUDAQ_OK);
  EXPECT_EQ(processed, 2u);
  EXPECT_EQ(read_all_threads_count(), kThreadsPerBlock)
      << "every thread of the block must have called the handler once";

  ASSERT_EQ(cudaq_dispatcher_destroy(dispatcher_), CUDAQ_OK);
  dispatcher_ = nullptr;
}

} // namespace
