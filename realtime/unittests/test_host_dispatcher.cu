/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <unistd.h>
#include <vector>

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

//==============================================================================
// Ring buffer helpers (same pattern as test_dispatch_kernel.cu)
//==============================================================================

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
  err = cudaHostAlloc(&host_data_ptr, num_slots * slot_size,
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

  std::memset(host_flags_ptr, 0, num_slots * sizeof(uint64_t));

  *host_flags_out = static_cast<volatile uint64_t*>(host_flags_ptr);
  *device_flags_out = static_cast<volatile uint64_t*>(device_flags_ptr);
  *host_data_out = static_cast<std::uint8_t*>(host_data_ptr);
  *device_data_out = static_cast<std::uint8_t*>(device_data_ptr);
  return true;
}

void free_ring_buffer(volatile uint64_t* host_flags, std::uint8_t* host_data) {
  if (host_flags)
    cudaFreeHost(const_cast<uint64_t*>(host_flags));
  if (host_data)
    cudaFreeHost(host_data);
}

//==============================================================================
// Minimal graph for dummy GRAPH_LAUNCH entry (so C API starts the host thread)
//==============================================================================

__global__ void noop_kernel() {}

// Creates a minimal executable graph and returns it. Caller must destroy with
// cudaGraphExecDestroy and cudaGraphDestroy.
bool create_dummy_graph(cudaGraph_t* graph_out, cudaGraphExec_t* exec_out) {
  cudaGraph_t graph = nullptr;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess)
    return false;

  cudaKernelNodeParams params = {};
  void* args[] = {};
  params.func = reinterpret_cast<void*>(noop_kernel);
  params.gridDim = dim3(1, 1, 1);
  params.blockDim = dim3(1, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = args;
  params.extra = nullptr;

  cudaGraphNode_t node = nullptr;
  if (cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params) !=
      cudaSuccess) {
    cudaGraphDestroy(graph);
    return false;
  }

  cudaGraphExec_t exec = nullptr;
  if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
    cudaGraphDestroy(graph);
    return false;
  }

  *graph_out = graph;
  *exec_out = exec;
  return true;
}

//==============================================================================
// Graph launch test: kernel that reads slot from mailbox and writes response
// in-place (same buffer as request; use single ring buffer for rx/tx).
//==============================================================================

__global__ void graph_increment_kernel(void** mailbox_slot_ptr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    void* buffer = *mailbox_slot_ptr;
    cudaq::realtime::RPCHeader* header =
        static_cast<cudaq::realtime::RPCHeader*>(buffer);
    std::uint32_t arg_len = header->arg_len;
    void* arg_buffer = static_cast<void*>(header + 1);
    std::uint8_t* data = static_cast<std::uint8_t*>(arg_buffer);
    for (std::uint32_t i = 0; i < arg_len; ++i)
      data[i] = data[i] + 1;
    cudaq::realtime::RPCResponse* response =
        static_cast<cudaq::realtime::RPCResponse*>(buffer);
    response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
    response->status = 0;
    response->result_len = arg_len;
  }
}

constexpr std::uint32_t RPC_GRAPH_INCREMENT_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("rpc_graph_increment");

/// Creates an executable graph that runs graph_increment_kernel with
/// kernel arg = d_mailbox_bank (device pointer to first mailbox slot).
/// Caller must cudaGraphExecDestroy / cudaGraphDestroy.
bool create_increment_graph(void** d_mailbox_bank, cudaGraph_t* graph_out,
                            cudaGraphExec_t* exec_out) {
  cudaGraph_t graph = nullptr;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess)
    return false;

  // kernelParams[i] must be a *pointer to* the i-th argument value.
  // The kernel takes void** so we pass &d_mailbox_bank (a void***).
  cudaKernelNodeParams params = {};
  void* kernel_args[] = {&d_mailbox_bank};
  params.func = reinterpret_cast<void*>(graph_increment_kernel);
  params.gridDim = dim3(1, 1, 1);
  params.blockDim = dim3(32, 1, 1);
  params.sharedMemBytes = 0;
  params.kernelParams = kernel_args;
  params.extra = nullptr;

  cudaGraphNode_t node = nullptr;
  if (cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params) !=
      cudaSuccess) {
    cudaGraphDestroy(graph);
    return false;
  }

  cudaGraphExec_t exec = nullptr;
  if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) != cudaSuccess) {
    cudaGraphDestroy(graph);
    return false;
  }

  *graph_out = graph;
  *exec_out = exec;
  return true;
}

//==============================================================================
// Test 1: Smoke test — host loop starts and drops slot with unknown function_id
//==============================================================================

constexpr std::uint32_t DUMMY_GRAPH_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("dummy_graph");
// Use a different function_id in the slot so the host loop does not find it.
constexpr std::uint32_t UNKNOWN_FUNCTION_ID = 0xdeadbeefu;

class HostDispatcherSmokeTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &rx_flags_host_,
                                     &rx_flags_, &rx_data_host_, &rx_data_));
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &tx_flags_host_,
                                     &tx_flags_, &tx_data_host_, &tx_data_));

    shutdown_flag_ = new (std::nothrow) int(0);
    stats_ = new (std::nothrow) uint64_t(0);
    ASSERT_NE(shutdown_flag_, nullptr);
    ASSERT_NE(stats_, nullptr);

    ASSERT_TRUE(create_dummy_graph(&dummy_graph_, &dummy_graph_exec_));

    host_table_ = new (std::nothrow) cudaq_function_entry_t[1];
    ASSERT_NE(host_table_, nullptr);
    std::memset(host_table_, 0, sizeof(cudaq_function_entry_t));
    host_table_[0].handler.graph_exec = dummy_graph_exec_;
    host_table_[0].function_id = DUMMY_GRAPH_FUNCTION_ID;
    host_table_[0].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;

    ASSERT_EQ(cudaq_dispatch_manager_create(&manager_), CUDAQ_OK);
    cudaq_dispatcher_config_t config{};
    config.device_id = 0;
    config.num_slots = static_cast<uint32_t>(num_slots_);
    config.slot_size = static_cast<uint32_t>(slot_size_);
    config.backend = CUDAQ_BACKEND_HOST_LOOP;
    ASSERT_EQ(cudaq_dispatcher_create(manager_, &config, &dispatcher_),
              CUDAQ_OK);

    cudaq_ringbuffer_t ringbuffer{};
    ringbuffer.rx_flags = rx_flags_;
    ringbuffer.tx_flags = tx_flags_;
    ringbuffer.rx_data = rx_data_;
    ringbuffer.tx_data = tx_data_;
    ringbuffer.rx_stride_sz = slot_size_;
    ringbuffer.tx_stride_sz = slot_size_;
    ringbuffer.rx_flags_host = rx_flags_host_;
    ringbuffer.tx_flags_host = tx_flags_host_;
    ringbuffer.rx_data_host = rx_data_host_;
    ringbuffer.tx_data_host = tx_data_host_;
    ASSERT_EQ(cudaq_dispatcher_set_ringbuffer(dispatcher_, &ringbuffer),
              CUDAQ_OK);

    cudaq_function_table_t table{};
    table.entries = host_table_;
    table.count = 1;
    ASSERT_EQ(cudaq_dispatcher_set_function_table(dispatcher_, &table),
              CUDAQ_OK);

    ASSERT_EQ(
        cudaq_dispatcher_set_control(dispatcher_, shutdown_flag_, stats_),
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
      delete shutdown_flag_;
    if (stats_)
      delete stats_;
    if (host_table_)
      delete[] host_table_;
    if (dummy_graph_exec_)
      cudaGraphExecDestroy(dummy_graph_exec_);
    if (dummy_graph_)
      cudaGraphDestroy(dummy_graph_);
  }

  void write_rpc_request_unknown_function(std::size_t slot) {
    std::uint8_t* slot_data =
        const_cast<std::uint8_t*>(rx_data_host_) + slot * slot_size_;
    auto* header =
        reinterpret_cast<cudaq::realtime::RPCHeader*>(slot_data);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = UNKNOWN_FUNCTION_ID;
    header->arg_len = 4;
    std::uint8_t* payload = slot_data + sizeof(cudaq::realtime::RPCHeader);
    payload[0] = 0;
    payload[1] = 1;
    payload[2] = 2;
    payload[3] = 3;
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

  int* shutdown_flag_ = nullptr;
  uint64_t* stats_ = nullptr;
  cudaq_function_entry_t* host_table_ = nullptr;
  cudaGraph_t dummy_graph_ = nullptr;
  cudaGraphExec_t dummy_graph_exec_ = nullptr;

  cudaq_dispatch_manager_t* manager_ = nullptr;
  cudaq_dispatcher_t* dispatcher_ = nullptr;
};

TEST_F(HostDispatcherSmokeTest, DropsSlotWithUnknownFunctionId) {
  write_rpc_request_unknown_function(0);

  __sync_synchronize();
  const_cast<volatile uint64_t*>(rx_flags_host_)[0] =
      reinterpret_cast<uint64_t>(rx_data_host_ + 0 * slot_size_);

  for (int i = 0; i < 50; ++i) {
    usleep(1000);
    if (tx_flags_host_[0] != 0)
      break;
  }

  EXPECT_EQ(tx_flags_host_[0], 0u)
      << "Host loop should drop slot with unknown function_id (no response)";
}

//==============================================================================
// Test 2: GRAPH_LAUNCH via host loop (full RPC round-trip) using the C API
//
// End-to-end test of: RPC in ring buffer → C API dispatcher → CUDA graph
// launch via pinned mailbox → in-place response.
//
// Flow:
//   1. Allocate pinned ring buffers and pinned mailbox (cudaHostAllocMapped).
//   2. Capture graph_increment_kernel with d_mailbox_bank baked in.
//   3. Build function table with one GRAPH_LAUNCH entry.
//   4. Wire the C API: manager → dispatcher → ringbuffer, function table,
//      control, mailbox → start.
//   5. Write an RPC request {0,1,2,3} into slot 0 and signal rx_flags.
//   6. The dispatcher picks up the slot, matches function_id → GRAPH_LAUNCH,
//      acquires the idle worker, writes the slot device pointer into the
//      pinned mailbox, and launches the graph.
//   7. The graph reads the slot pointer from the mailbox, increments each
//      payload byte, and writes an RPCResponse header in-place.
//   8. Test polls tx_flags, syncs device, then asserts the response is
//      {1,2,3,4} with correct magic/status/result_len.
//==============================================================================

TEST(HostDispatcherGraphLaunchTest, FullRpcRoundTripViaPinnedMailbox) {
  constexpr std::size_t num_slots = 2;
  constexpr std::size_t slot_size = 256;

  // --- Ring buffers ---
  // Separate flag arrays for RX and TX: the dispatcher clears rx_flags[slot]
  // right after setting tx_flags[slot], so sharing would clobber the signal.
  // Data buffers are shared (graph writes response in-place to the RX slot).
  volatile uint64_t* rx_flags_host = nullptr;
  volatile uint64_t* rx_flags_dev = nullptr;
  std::uint8_t* rx_data_host = nullptr;
  std::uint8_t* rx_data_dev = nullptr;
  volatile uint64_t* tx_flags_host = nullptr;
  volatile uint64_t* tx_flags_dev = nullptr;
  std::uint8_t* tx_data_host_unused = nullptr;
  std::uint8_t* tx_data_dev_unused = nullptr;

  ASSERT_TRUE(allocate_ring_buffer(num_slots, slot_size, &rx_flags_host,
                                   &rx_flags_dev, &rx_data_host,
                                   &rx_data_dev));
  ASSERT_TRUE(allocate_ring_buffer(num_slots, slot_size, &tx_flags_host,
                                   &tx_flags_dev, &tx_data_host_unused,
                                   &tx_data_dev_unused));

  // --- Pinned mailbox ---
  // cudaHostAllocMapped gives us host + device views of the same memory.
  // The host dispatcher writes the slot device pointer to h_mailbox_bank[0];
  // the graph reads it from d_mailbox_bank[0] (same physical location).
  void** h_mailbox_bank = nullptr;
  void** d_mailbox_bank = nullptr;
  CUDA_CHECK(cudaHostAlloc(&h_mailbox_bank, sizeof(void*),
                           cudaHostAllocMapped));
  std::memset(h_mailbox_bank, 0, sizeof(void*));
  CUDA_CHECK(
      cudaHostGetDevicePointer((void**)&d_mailbox_bank, h_mailbox_bank, 0));

  // --- Graph ---
  // Capture graph_increment_kernel with d_mailbox_bank baked in as the
  // kernel arg. At runtime the kernel reads *d_mailbox_bank to find
  // the slot, so different slots can be processed on each launch.
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graph_exec = nullptr;
  ASSERT_TRUE(
      create_increment_graph(d_mailbox_bank, &graph, &graph_exec));

  // --- Function table (one GRAPH_LAUNCH entry) ---
  cudaq_function_entry_t host_table[1];
  std::memset(host_table, 0, sizeof(host_table));
  host_table[0].function_id = RPC_GRAPH_INCREMENT_FUNCTION_ID;
  host_table[0].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
  host_table[0].handler.graph_exec = graph_exec;

  // --- C API: create manager + dispatcher ---
  cudaq_dispatch_manager_t* manager = nullptr;
  ASSERT_EQ(cudaq_dispatch_manager_create(&manager), CUDAQ_OK);

  cudaq_dispatcher_config_t disp_config{};
  disp_config.device_id = 0;
  disp_config.num_slots = static_cast<uint32_t>(num_slots);
  disp_config.slot_size = static_cast<uint32_t>(slot_size);
  disp_config.backend = CUDAQ_BACKEND_HOST_LOOP;

  cudaq_dispatcher_t* dispatcher = nullptr;
  ASSERT_EQ(cudaq_dispatcher_create(manager, &disp_config, &dispatcher),
            CUDAQ_OK);

  // --- Wire ring buffer (rx/tx flags separate, data shared for in-place) ---
  cudaq_ringbuffer_t ringbuffer{};
  ringbuffer.rx_flags = rx_flags_dev;
  ringbuffer.tx_flags = tx_flags_dev;
  ringbuffer.rx_data = rx_data_dev;
  ringbuffer.tx_data = rx_data_dev;
  ringbuffer.rx_stride_sz = slot_size;
  ringbuffer.tx_stride_sz = slot_size;
  ringbuffer.rx_flags_host = rx_flags_host;
  ringbuffer.tx_flags_host = tx_flags_host;
  ringbuffer.rx_data_host = rx_data_host;
  ringbuffer.tx_data_host = rx_data_host;
  ASSERT_EQ(cudaq_dispatcher_set_ringbuffer(dispatcher, &ringbuffer),
            CUDAQ_OK);

  cudaq_function_table_t table{};
  table.entries = host_table;
  table.count = 1;
  ASSERT_EQ(cudaq_dispatcher_set_function_table(dispatcher, &table),
            CUDAQ_OK);

  int shutdown_flag = 0;
  uint64_t stats_counter = 0;
  ASSERT_EQ(cudaq_dispatcher_set_control(dispatcher, &shutdown_flag,
                                         &stats_counter),
            CUDAQ_OK);

  // Provide the caller-allocated pinned mailbox so the dispatcher uses it
  // instead of allocating plain host memory (which the graph can't read).
  ASSERT_EQ(cudaq_dispatcher_set_mailbox(dispatcher, h_mailbox_bank),
            CUDAQ_OK);

  // --- Start ---
  ASSERT_EQ(cudaq_dispatcher_start(dispatcher), CUDAQ_OK);

  // --- Send RPC request (simulates FPGA / producer) ---
  // Write RPCHeader + payload {0,1,2,3} into slot 0, then signal rx_flags.
  std::uint8_t* slot_data = rx_data_host + 0 * slot_size;
  auto* header = reinterpret_cast<cudaq::realtime::RPCHeader*>(slot_data);
  header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
  header->function_id = RPC_GRAPH_INCREMENT_FUNCTION_ID;
  header->arg_len = 4;
  std::uint8_t* payload = slot_data + sizeof(cudaq::realtime::RPCHeader);
  payload[0] = 0;
  payload[1] = 1;
  payload[2] = 2;
  payload[3] = 3;

  __sync_synchronize();
  const_cast<volatile uint64_t*>(rx_flags_host)[0] =
      reinterpret_cast<uint64_t>(rx_data_host + 0 * slot_size);

  // --- Verify: dispatcher picked up slot and launched graph ---
  int poll_iters = 0;
  const int max_poll = 5000;
  while (tx_flags_host[0] == 0 && poll_iters < max_poll) {
    usleep(200);
    ++poll_iters;
  }
  ASSERT_NE(tx_flags_host[0], 0u) << "Timeout waiting for tx flag";
  ASSERT_NE(tx_flags_host[0] >> 48, 0xDEADu)
      << "Dispatcher reported graph launch error (check 0xDEAD...)";

  // cudaGraphLaunch is async; sync device so the in-place response is visible
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Verify: graph wrote correct response in-place ---
  auto* resp = reinterpret_cast<cudaq::realtime::RPCResponse*>(slot_data);
  ASSERT_EQ(resp->magic, cudaq::realtime::RPC_MAGIC_RESPONSE)
      << "Expected response magic (graph in-place write)";
  ASSERT_EQ(resp->status, 0);
  ASSERT_EQ(resp->result_len, 4u);
  std::uint8_t* result = slot_data + sizeof(cudaq::realtime::RPCResponse);
  EXPECT_EQ(result[0], 1);
  EXPECT_EQ(result[1], 2);
  EXPECT_EQ(result[2], 3);
  EXPECT_EQ(result[3], 4);

  // --- Teardown (C API handles thread join) ---
  shutdown_flag = 1;
  __sync_synchronize();
  cudaq_dispatcher_stop(dispatcher);
  cudaq_dispatcher_destroy(dispatcher);
  cudaq_dispatch_manager_destroy(manager);

  cudaGraphExecDestroy(graph_exec);
  cudaGraphDestroy(graph);
  cudaFreeHost(h_mailbox_bank);
  free_ring_buffer(rx_flags_host, rx_data_host);
  free_ring_buffer(tx_flags_host, tx_data_host_unused);
}

} // namespace
