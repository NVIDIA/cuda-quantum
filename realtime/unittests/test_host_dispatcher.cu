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
#include <thread>
#include <unistd.h>
#include <vector>

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

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
// Graph launch test: kernel that reads slot from mailbox and doubles payload
// in-place (for function_id routing differentiation vs increment kernel).
//==============================================================================

__global__ void graph_double_kernel(void** mailbox_slot_ptr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    void* buffer = *mailbox_slot_ptr;
    cudaq::realtime::RPCHeader* header =
        static_cast<cudaq::realtime::RPCHeader*>(buffer);
    std::uint32_t arg_len = header->arg_len;
    void* arg_buffer = static_cast<void*>(header + 1);
    std::uint8_t* data = static_cast<std::uint8_t*>(arg_buffer);
    for (std::uint32_t i = 0; i < arg_len; ++i)
      data[i] = data[i] * 2;
    cudaq::realtime::RPCResponse* response =
        static_cast<cudaq::realtime::RPCResponse*>(buffer);
    response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
    response->status = 0;
    response->result_len = arg_len;
  }
}

constexpr std::uint32_t RPC_GRAPH_DOUBLE_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("rpc_graph_double");

bool create_double_graph(void** d_mailbox_slot, cudaGraph_t* graph_out,
                         cudaGraphExec_t* exec_out) {
  cudaGraph_t graph = nullptr;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess)
    return false;

  cudaKernelNodeParams params = {};
  void* kernel_args[] = {&d_mailbox_slot};
  params.func = reinterpret_cast<void*>(graph_double_kernel);
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
// Test fixture: drives host_dispatcher_loop directly (not C API) for full
// control over idle_mask, enabling worker recycling and backpressure tests.
//==============================================================================

static constexpr std::size_t kMaxWorkers = 8;

class HostDispatcherLoopTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &rx_flags_host_,
                                     &rx_flags_dev_, &rx_data_host_,
                                     &rx_data_dev_));
    ASSERT_TRUE(allocate_ring_buffer(num_slots_, slot_size_, &tx_flags_host_,
                                     &tx_flags_dev_, &tx_data_host_,
                                     &tx_data_dev_));

    CUDA_CHECK(cudaHostAlloc(&h_mailbox_bank_,
                             kMaxWorkers * sizeof(void*),
                             cudaHostAllocMapped));
    std::memset(h_mailbox_bank_, 0, kMaxWorkers * sizeof(void*));
    CUDA_CHECK(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&d_mailbox_bank_), h_mailbox_bank_, 0));

    idle_mask_ = new cudaq::realtime::atomic_uint64_sys(0);
    live_dispatched_ = new cudaq::realtime::atomic_uint64_sys(0);
    inflight_slot_tags_ = new int[kMaxWorkers]();
    shutdown_flag_ = new cudaq::realtime::atomic_int_sys(0);
    stats_counter_ = 0;

    function_table_ = new cudaq_function_entry_t[kMaxWorkers];
    std::memset(function_table_, 0, kMaxWorkers * sizeof(cudaq_function_entry_t));

    std::memset(&ringbuffer_, 0, sizeof(ringbuffer_));
    ringbuffer_.rx_flags = rx_flags_dev_;
    ringbuffer_.tx_flags = tx_flags_dev_;
    ringbuffer_.rx_data = rx_data_dev_;
    ringbuffer_.tx_data = tx_data_dev_;
    ringbuffer_.rx_stride_sz = slot_size_;
    ringbuffer_.tx_stride_sz = slot_size_;
    ringbuffer_.rx_flags_host = rx_flags_host_;
    ringbuffer_.tx_flags_host = tx_flags_host_;
    ringbuffer_.rx_data_host = rx_data_host_;
    ringbuffer_.tx_data_host = tx_data_host_;
  }

  void TearDown() override {
    if (!loop_stopped_) {
      shutdown_flag_->store(1, cuda::std::memory_order_release);
      __sync_synchronize();
      if (loop_thread_.joinable())
        loop_thread_.join();
    }

    for (auto& w : worker_info_) {
      if (w.stream)
        cudaStreamDestroy(w.stream);
      if (w.graph_exec)
        cudaGraphExecDestroy(w.graph_exec);
      if (w.graph)
        cudaGraphDestroy(w.graph);
    }

    free_ring_buffer(rx_flags_host_, rx_data_host_);
    free_ring_buffer(tx_flags_host_, tx_data_host_);
    if (h_mailbox_bank_)
      cudaFreeHost(h_mailbox_bank_);
    delete idle_mask_;
    delete live_dispatched_;
    delete[] inflight_slot_tags_;
    delete shutdown_flag_;
    delete[] function_table_;
  }

  struct WorkerInfo {
    cudaGraphExec_t graph_exec = nullptr;
    cudaGraph_t graph = nullptr;
    cudaStream_t stream = nullptr;
  };

  void AddWorker(std::uint32_t function_id, cudaGraphExec_t exec,
                 cudaGraph_t graph) {
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    cudaq::realtime::HostDispatchWorker w;
    w.graph_exec = exec;
    w.stream = stream;
    w.function_id = function_id;
    workers_.push_back(w);
    worker_info_.push_back({exec, graph, stream});

    std::size_t idx = function_table_count_;
    function_table_[idx].handler.graph_exec = exec;
    function_table_[idx].function_id = function_id;
    function_table_[idx].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
    function_table_count_++;
  }

  void StartLoop() {
    idle_mask_->store((1ULL << workers_.size()) - 1,
                      cuda::std::memory_order_release);

    config_.rx_flags =
        reinterpret_cast<cudaq::realtime::atomic_uint64_sys*>(
            const_cast<uint64_t*>(rx_flags_host_));
    config_.tx_flags =
        reinterpret_cast<cudaq::realtime::atomic_uint64_sys*>(
            const_cast<uint64_t*>(tx_flags_host_));
    config_.rx_data_host = rx_data_host_;
    config_.rx_data_dev = rx_data_dev_;
    config_.tx_data_host = tx_data_host_;
    config_.tx_data_dev = tx_data_dev_;
    config_.tx_stride_sz = slot_size_;
    config_.h_mailbox_bank = h_mailbox_bank_;
    config_.num_slots = num_slots_;
    config_.slot_size = slot_size_;
    config_.workers = workers_;
    config_.function_table = function_table_;
    config_.function_table_count = function_table_count_;
    config_.shutdown_flag = shutdown_flag_;
    config_.stats_counter = &stats_counter_;
    config_.live_dispatched = live_dispatched_;
    config_.idle_mask = idle_mask_;
    config_.inflight_slot_tags = inflight_slot_tags_;

    loop_thread_ = std::thread(cudaq::realtime::host_dispatcher_loop, config_);
  }

  void WriteRpcRequest(std::size_t slot, std::uint32_t function_id,
                       const std::uint8_t* payload, std::size_t len) {
    ASSERT_EQ(cudaq_host_ringbuffer_write_rpc_request(
                  &ringbuffer_, static_cast<uint32_t>(slot), function_id,
                  payload, static_cast<uint32_t>(len)),
              CUDAQ_OK);
  }

  void SignalSlot(std::size_t slot) {
    cudaq_host_ringbuffer_signal_slot(&ringbuffer_, static_cast<uint32_t>(slot));
  }

  bool PollTxFlag(std::size_t slot, int timeout_ms = 2000) {
    for (int waited = 0; waited < timeout_ms * 1000; waited += 200) {
      cudaq_tx_status_t st = cudaq_host_ringbuffer_poll_tx_flag(
          &ringbuffer_, static_cast<uint32_t>(slot), nullptr);
      if (st != CUDAQ_TX_EMPTY)
        return true;
      usleep(200);
    }
    return cudaq_host_ringbuffer_poll_tx_flag(
               &ringbuffer_, static_cast<uint32_t>(slot), nullptr) !=
           CUDAQ_TX_EMPTY;
  }

  void StopLoop() {
    shutdown_flag_->store(1, cuda::std::memory_order_release);
    __sync_synchronize();
    if (loop_thread_.joinable())
      loop_thread_.join();
    loop_stopped_ = true;
  }

  void RestoreWorker(int worker_id) {
    idle_mask_->fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
  }

  void ClearSlot(std::size_t slot) {
    cudaq_host_ringbuffer_clear_slot(&ringbuffer_, static_cast<uint32_t>(slot));
    std::memset(rx_data_host_ + slot * slot_size_, 0, slot_size_);
  }

  void VerifyResponse(std::size_t slot, const std::uint8_t* expected,
                      std::size_t len) {
    int cuda_err = 0;
    cudaq_tx_status_t st = cudaq_host_ringbuffer_poll_tx_flag(
        &ringbuffer_, static_cast<uint32_t>(slot), &cuda_err);
    ASSERT_EQ(st, CUDAQ_TX_READY) << "slot " << slot
        << ": tx_flag not READY (status=" << st << " cuda_err=" << cuda_err << ")";

    std::uint8_t* slot_data = rx_data_host_ + slot * slot_size_;
    auto* resp =
        reinterpret_cast<cudaq::realtime::RPCResponse*>(slot_data);
    ASSERT_EQ(resp->magic, CUDAQ_RPC_MAGIC_RESPONSE)
        << "slot " << slot << ": expected response magic";
    ASSERT_EQ(resp->status, 0) << "slot " << slot << ": non-zero status";
    ASSERT_EQ(resp->result_len, static_cast<std::uint32_t>(len))
        << "slot " << slot << ": wrong result_len";
    std::uint8_t* result = slot_data + sizeof(cudaq::realtime::RPCResponse);
    for (std::size_t i = 0; i < len; ++i) {
      EXPECT_EQ(result[i], expected[i])
          << "slot " << slot << " byte " << i;
    }
  }

  std::size_t num_slots_ = 4;
  std::size_t slot_size_ = 256;

  volatile uint64_t* rx_flags_host_ = nullptr;
  volatile uint64_t* tx_flags_host_ = nullptr;
  volatile uint64_t* rx_flags_dev_ = nullptr;
  volatile uint64_t* tx_flags_dev_ = nullptr;
  std::uint8_t* rx_data_host_ = nullptr;
  std::uint8_t* tx_data_host_ = nullptr;
  std::uint8_t* rx_data_dev_ = nullptr;
  std::uint8_t* tx_data_dev_ = nullptr;

  void** h_mailbox_bank_ = nullptr;
  void** d_mailbox_bank_ = nullptr;

  cudaq::realtime::atomic_uint64_sys* idle_mask_ = nullptr;
  cudaq::realtime::atomic_uint64_sys* live_dispatched_ = nullptr;
  int* inflight_slot_tags_ = nullptr;
  cudaq::realtime::atomic_int_sys* shutdown_flag_ = nullptr;
  uint64_t stats_counter_ = 0;
  bool loop_stopped_ = false;

  cudaq_function_entry_t* function_table_ = nullptr;
  std::size_t function_table_count_ = 0;
  std::vector<cudaq::realtime::HostDispatchWorker> workers_;
  std::vector<WorkerInfo> worker_info_;

  cudaq_ringbuffer_t ringbuffer_{};
  cudaq::realtime::HostDispatcherConfig config_{};
  std::thread loop_thread_;
};

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

    std::memset(&ringbuffer_, 0, sizeof(ringbuffer_));
    ringbuffer_.rx_flags = rx_flags_;
    ringbuffer_.tx_flags = tx_flags_;
    ringbuffer_.rx_data = rx_data_;
    ringbuffer_.tx_data = tx_data_;
    ringbuffer_.rx_stride_sz = slot_size_;
    ringbuffer_.tx_stride_sz = slot_size_;
    ringbuffer_.rx_flags_host = rx_flags_host_;
    ringbuffer_.tx_flags_host = tx_flags_host_;
    ringbuffer_.rx_data_host = rx_data_host_;
    ringbuffer_.tx_data_host = tx_data_host_;
    ASSERT_EQ(cudaq_dispatcher_set_ringbuffer(dispatcher_, &ringbuffer_),
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
    const std::uint8_t payload[] = {0, 1, 2, 3};
    ASSERT_EQ(cudaq_host_ringbuffer_write_rpc_request(
                  &ringbuffer_, static_cast<uint32_t>(slot),
                  UNKNOWN_FUNCTION_ID, payload, 4),
              CUDAQ_OK);
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

  cudaq_ringbuffer_t ringbuffer_{};
  cudaq_dispatch_manager_t* manager_ = nullptr;
  cudaq_dispatcher_t* dispatcher_ = nullptr;
};

TEST_F(HostDispatcherSmokeTest, DropsSlotWithUnknownFunctionId) {
  write_rpc_request_unknown_function(0);
  cudaq_host_ringbuffer_signal_slot(&ringbuffer_, 0);

  for (int i = 0; i < 50; ++i) {
    usleep(1000);
    cudaq_tx_status_t st =
        cudaq_host_ringbuffer_poll_tx_flag(&ringbuffer_, 0, nullptr);
    if (st != CUDAQ_TX_EMPTY)
      break;
  }

  cudaq_tx_status_t final_st =
      cudaq_host_ringbuffer_poll_tx_flag(&ringbuffer_, 0, nullptr);
  EXPECT_EQ(final_st, CUDAQ_TX_EMPTY)
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
  const std::uint8_t payload[] = {0, 1, 2, 3};
  ASSERT_EQ(cudaq_host_ringbuffer_write_rpc_request(
                &ringbuffer, 0, RPC_GRAPH_INCREMENT_FUNCTION_ID, payload, 4),
            CUDAQ_OK);
  cudaq_host_ringbuffer_signal_slot(&ringbuffer, 0);

  // --- Verify: dispatcher picked up slot and launched graph ---
  int cuda_err = 0;
  cudaq_tx_status_t st = CUDAQ_TX_EMPTY;
  for (int i = 0; i < 5000 && st == CUDAQ_TX_EMPTY; ++i) {
    usleep(200);
    st = cudaq_host_ringbuffer_poll_tx_flag(&ringbuffer, 0, &cuda_err);
  }
  ASSERT_NE(st, CUDAQ_TX_EMPTY) << "Timeout waiting for tx flag";
  ASSERT_NE(st, CUDAQ_TX_ERROR)
      << "Dispatcher reported graph launch error (cuda_err=" << cuda_err << ")";

  // cudaGraphLaunch is async; sync device so the in-place response is visible
  CUDA_CHECK(cudaDeviceSynchronize());

  // --- Verify: graph wrote correct response in-place ---
  std::uint8_t* slot_data = rx_data_host + 0 * slot_size;
  auto* resp = reinterpret_cast<cudaq::realtime::RPCResponse*>(slot_data);
  ASSERT_EQ(resp->magic, CUDAQ_RPC_MAGIC_RESPONSE)
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

//==============================================================================
// Test 3: Multiple workers with function_id routing (internal API)
//
// Two workers: worker 0 runs graph_increment_kernel (func_id A),
// worker 1 runs graph_double_kernel (func_id B). Sends one RPC per worker
// and verifies each graph produced the expected output, confirming the
// dispatcher routed by function_id.
//==============================================================================

TEST_F(HostDispatcherLoopTest, MultiWorkerFunctionIdRouting) {
  cudaGraph_t inc_graph = nullptr;
  cudaGraphExec_t inc_exec = nullptr;
  ASSERT_TRUE(create_increment_graph(d_mailbox_bank_ + 0, &inc_graph, &inc_exec));
  AddWorker(RPC_GRAPH_INCREMENT_FUNCTION_ID, inc_exec, inc_graph);

  cudaGraph_t dbl_graph = nullptr;
  cudaGraphExec_t dbl_exec = nullptr;
  ASSERT_TRUE(create_double_graph(d_mailbox_bank_ + 1, &dbl_graph, &dbl_exec));
  AddWorker(RPC_GRAPH_DOUBLE_FUNCTION_ID, dbl_exec, dbl_graph);

  StartLoop();

  const std::uint8_t payload[] = {1, 2, 3, 4};
  WriteRpcRequest(0, RPC_GRAPH_INCREMENT_FUNCTION_ID, payload, 4);
  WriteRpcRequest(1, RPC_GRAPH_DOUBLE_FUNCTION_ID, payload, 4);
  SignalSlot(0);
  SignalSlot(1);

  ASSERT_TRUE(PollTxFlag(0)) << "Timeout on slot 0 (increment)";
  ASSERT_TRUE(PollTxFlag(1)) << "Timeout on slot 1 (double)";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::uint8_t expected_inc[] = {2, 3, 4, 5};
  const std::uint8_t expected_dbl[] = {2, 4, 6, 8};
  VerifyResponse(0, expected_inc, 4);
  VerifyResponse(1, expected_dbl, 4);
}

//==============================================================================
// Test 4: Worker recycling — idle_mask round-trip (internal API)
//
// One worker, two sequential RPCs to the same slot. The second dispatch
// can only proceed after the test restores idle_mask (simulating the
// external worker thread that returns the worker to the pool).
//==============================================================================

TEST_F(HostDispatcherLoopTest, WorkerRecycling) {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ASSERT_TRUE(create_increment_graph(d_mailbox_bank_, &graph, &exec));
  AddWorker(RPC_GRAPH_INCREMENT_FUNCTION_ID, exec, graph);

  StartLoop();

  // RPC 1 on slot 0 — after dispatch, current_slot advances to 1.
  const std::uint8_t payload1[] = {0, 1, 2, 3};
  WriteRpcRequest(0, RPC_GRAPH_INCREMENT_FUNCTION_ID, payload1, 4);
  SignalSlot(0);
  ASSERT_TRUE(PollTxFlag(0)) << "Timeout on first RPC";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::uint8_t expected1[] = {1, 2, 3, 4};
  VerifyResponse(0, expected1, 4);

  RestoreWorker(0);

  // RPC 2 on slot 1 — the dispatcher is now polling slot 1.
  // This can only dispatch if idle_mask was properly restored above.
  const std::uint8_t payload2[] = {10, 11, 12, 13};
  WriteRpcRequest(1, RPC_GRAPH_INCREMENT_FUNCTION_ID, payload2, 4);
  SignalSlot(1);
  ASSERT_TRUE(PollTxFlag(1)) << "Timeout on second RPC (worker not recycled?)";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::uint8_t expected2[] = {11, 12, 13, 14};
  VerifyResponse(1, expected2, 4);
}

//==============================================================================
// Test 5: Backpressure — dispatcher stalls when all workers are busy
//
// One worker, two slots signalled simultaneously. Slot 0 dispatches
// immediately; slot 1 stalls until the test restores idle_mask.
//==============================================================================

TEST_F(HostDispatcherLoopTest, BackpressureWhenAllBusy) {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ASSERT_TRUE(create_increment_graph(d_mailbox_bank_, &graph, &exec));
  AddWorker(RPC_GRAPH_INCREMENT_FUNCTION_ID, exec, graph);

  StartLoop();

  const std::uint8_t payload0[] = {0, 1, 2, 3};
  const std::uint8_t payload1[] = {10, 11, 12, 13};
  WriteRpcRequest(0, RPC_GRAPH_INCREMENT_FUNCTION_ID, payload0, 4);
  WriteRpcRequest(1, RPC_GRAPH_INCREMENT_FUNCTION_ID, payload1, 4);
  SignalSlot(0);
  SignalSlot(1);

  ASSERT_TRUE(PollTxFlag(0)) << "Timeout on slot 0";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  // Slot 1 should still be pending — worker is busy.
  EXPECT_EQ(tx_flags_host_[1], 0u)
      << "Slot 1 should stall while worker is busy";

  RestoreWorker(0);

  ASSERT_TRUE(PollTxFlag(1)) << "Timeout on slot 1 after restoring worker";
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  const std::uint8_t expected0[] = {1, 2, 3, 4};
  const std::uint8_t expected1[] = {11, 12, 13, 14};
  VerifyResponse(0, expected0, 4);
  VerifyResponse(1, expected1, 4);

  EXPECT_EQ(live_dispatched_->load(cuda::std::memory_order_acquire), 2u);

  StopLoop();
  EXPECT_EQ(stats_counter_, 2u);
}

//==============================================================================
// Test 6: Stats counter accuracy (internal API)
//
// Sends 5 sequential RPCs through a single worker (recycling between each)
// and verifies stats_counter == 5 at the end.
//==============================================================================

TEST_F(HostDispatcherLoopTest, StatsCounterAccuracy) {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  ASSERT_TRUE(create_increment_graph(d_mailbox_bank_, &graph, &exec));
  AddWorker(RPC_GRAPH_INCREMENT_FUNCTION_ID, exec, graph);

  StartLoop();

  // Sequential RPCs through slots 0,1,2,3,0 — the dispatcher advances
  // current_slot after each dispatch, so each RPC must target the next slot.
  // When wrapping back to slot 0 for the 5th RPC, clear its tx_flags first.
  constexpr int kNumRpcs = 5;
  for (int i = 0; i < kNumRpcs; ++i) {
    std::size_t slot = static_cast<std::size_t>(i % num_slots_);
    if (i >= static_cast<int>(num_slots_))
      ClearSlot(slot);

    std::uint8_t payload[] = {
        static_cast<std::uint8_t>(i * 10),
        static_cast<std::uint8_t>(i * 10 + 1),
        static_cast<std::uint8_t>(i * 10 + 2),
        static_cast<std::uint8_t>(i * 10 + 3)};
    WriteRpcRequest(slot, RPC_GRAPH_INCREMENT_FUNCTION_ID, payload, 4);
    SignalSlot(slot);
    ASSERT_TRUE(PollTxFlag(slot)) << "Timeout on RPC " << i << " (slot " << slot << ")";
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::uint8_t expected[] = {
        static_cast<std::uint8_t>(i * 10 + 1),
        static_cast<std::uint8_t>(i * 10 + 2),
        static_cast<std::uint8_t>(i * 10 + 3),
        static_cast<std::uint8_t>(i * 10 + 4)};
    VerifyResponse(slot, expected, 4);

    RestoreWorker(0);
  }

  EXPECT_EQ(live_dispatched_->load(cuda::std::memory_order_acquire),
            static_cast<uint64_t>(kNumRpcs));

  StopLoop();
  EXPECT_EQ(stats_counter_, static_cast<uint64_t>(kNumRpcs));
}

//==============================================================================
// Test 7: Multi-slot round-robin dispatch (internal API)
//
// 4 slots, 4 workers (all same function_id). All slots signalled at once;
// the dispatcher processes them 0 → 1 → 2 → 3 using one worker each.
//==============================================================================

TEST_F(HostDispatcherLoopTest, MultiSlotRoundRobin) {
  constexpr int kNumSlots = 4;
  cudaGraph_t graphs[kNumSlots];
  cudaGraphExec_t execs[kNumSlots];
  for (int i = 0; i < kNumSlots; ++i) {
    ASSERT_TRUE(create_increment_graph(d_mailbox_bank_ + i, &graphs[i],
                                       &execs[i]));
    AddWorker(RPC_GRAPH_INCREMENT_FUNCTION_ID, execs[i], graphs[i]);
  }

  StartLoop();

  for (int i = 0; i < kNumSlots; ++i) {
    std::uint8_t payload[] = {
        static_cast<std::uint8_t>(i * 4 + 1),
        static_cast<std::uint8_t>(i * 4 + 2),
        static_cast<std::uint8_t>(i * 4 + 3),
        static_cast<std::uint8_t>(i * 4 + 4)};
    WriteRpcRequest(static_cast<std::size_t>(i),
                    RPC_GRAPH_INCREMENT_FUNCTION_ID, payload, 4);
  }

  for (int i = 0; i < kNumSlots; ++i)
    SignalSlot(static_cast<std::size_t>(i));

  for (int i = 0; i < kNumSlots; ++i) {
    ASSERT_TRUE(PollTxFlag(static_cast<std::size_t>(i)))
        << "Timeout on slot " << i;
  }
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  for (int i = 0; i < kNumSlots; ++i) {
    std::uint8_t expected[] = {
        static_cast<std::uint8_t>(i * 4 + 2),
        static_cast<std::uint8_t>(i * 4 + 3),
        static_cast<std::uint8_t>(i * 4 + 4),
        static_cast<std::uint8_t>(i * 4 + 5)};
    VerifyResponse(static_cast<std::size_t>(i), expected, 4);
  }

  EXPECT_EQ(live_dispatched_->load(cuda::std::memory_order_acquire),
            static_cast<uint64_t>(kNumSlots));

  StopLoop();
  EXPECT_EQ(stats_counter_, static_cast<uint64_t>(kNumSlots));
}

} // namespace
