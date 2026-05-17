/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Shared-ring-mode test: brings up BOTH a HOST_LOOP CPU dispatcher AND a
// DEVICE_LOOP persistent GPU dispatcher on the SAME ring buffer, with
// shared_ring_mode = 1 on both.  The HOST_LOOP owns one function_id (a
// GRAPH_LAUNCH entry) and the DEVICE_LOOP owns a different function_id (a
// DEVICE_CALL entry).  Producer interleaves requests for the two function_ids
// across slots; the test verifies that each dispatcher services its OWN
// requests and SKIPS the peer's slots without clobbering rx_flags.

#include <gtest/gtest.h>
#include <cuda/std/atomic>
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

constexpr std::size_t kNumSlots = 8;
constexpr std::size_t kSlotSize = 256;

// function_id for the GRAPH_LAUNCH entry owned by the HOST_LOOP dispatcher.
constexpr std::uint32_t HOST_GRAPH_FN_ID =
    cudaq::realtime::fnv1a_hash("shared_ring_host_increment");

// function_id for the DEVICE_CALL entry owned by the DEVICE_LOOP dispatcher.
constexpr std::uint32_t DEVICE_CALL_FN_ID =
    cudaq::realtime::fnv1a_hash("shared_ring_device_double");

//==============================================================================
// Ring buffer / control buffer helpers
//==============================================================================

bool allocate_ring_buffer(std::size_t num_slots, std::size_t slot_size,
                          volatile uint64_t** host_flags_out,
                          volatile uint64_t** device_flags_out,
                          std::uint8_t** host_data_out,
                          std::uint8_t** device_data_out) {
  void* host_flags_ptr = nullptr;
  if (cudaHostAlloc(&host_flags_ptr, num_slots * sizeof(uint64_t),
                    cudaHostAllocMapped) != cudaSuccess)
    return false;

  void* device_flags_ptr = nullptr;
  if (cudaHostGetDevicePointer(&device_flags_ptr, host_flags_ptr, 0) !=
      cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    return false;
  }

  void* host_data_ptr = nullptr;
  if (cudaHostAlloc(&host_data_ptr, num_slots * slot_size,
                    cudaHostAllocMapped) != cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    return false;
  }

  void* device_data_ptr = nullptr;
  if (cudaHostGetDevicePointer(&device_data_ptr, host_data_ptr, 0) !=
      cudaSuccess) {
    cudaFreeHost(host_flags_ptr);
    cudaFreeHost(host_data_ptr);
    return false;
  }

  std::memset(host_flags_ptr, 0, num_slots * sizeof(uint64_t));
  std::memset(host_data_ptr, 0, num_slots * slot_size);

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
// HOST_LOOP graph kernel: reads RPC slot via mailbox, writes incremented bytes
// to the TX slot.  The dispatcher fills a GraphIOContext per-launch via its
// io_ctxs path; we use the simpler "mailbox holds raw RX slot pointer" mode
// and write the response in-place (legacy single-buffer mode is fine since
// we wire rx_data and tx_data to the SAME backing memory).
//==============================================================================

__global__ void host_graph_increment_kernel(void** mailbox_slot_ptr) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    void* buffer = *mailbox_slot_ptr;
    cudaq::realtime::RPCHeader* header =
        static_cast<cudaq::realtime::RPCHeader*>(buffer);
    std::uint32_t arg_len = header->arg_len;
    std::uint32_t request_id = header->request_id;
    std::uint8_t* data = static_cast<std::uint8_t*>(buffer) +
                         sizeof(cudaq::realtime::RPCHeader);
    for (std::uint32_t i = 0; i < arg_len; ++i)
      data[i] = data[i] + 1;
    cudaq::realtime::RPCResponse* response =
        static_cast<cudaq::realtime::RPCResponse*>(buffer);
    response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
    response->status = 0;
    response->result_len = arg_len;
    response->request_id = request_id;
  }
}

bool create_host_graph(void** d_mailbox_bank, cudaGraph_t* graph_out,
                       cudaGraphExec_t* exec_out) {
  cudaGraph_t graph = nullptr;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess)
    return false;

  cudaKernelNodeParams params = {};
  void* kernel_args[] = {&d_mailbox_bank};
  params.func = reinterpret_cast<void*>(host_graph_increment_kernel);
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
// DEVICE_LOOP device-call handler: doubles each byte.
//==============================================================================

__device__ int device_double_handler(const void* input, void* output,
                                     std::uint32_t arg_len,
                                     std::uint32_t max_result_len,
                                     std::uint32_t* result_len) {
  const std::uint8_t* in = static_cast<const std::uint8_t*>(input);
  std::uint8_t* out = static_cast<std::uint8_t*>(output);
  std::uint32_t n = arg_len;
  if (n > max_result_len)
    n = max_result_len;
  for (std::uint32_t i = 0; i < n; ++i)
    out[i] = static_cast<std::uint8_t>(in[i] * 2);
  *result_len = n;
  return 0;
}

// Populate the device function table:
//   entry 0: GRAPH_LAUNCH owned by HOST_LOOP (handler.graph_exec set below)
//   entry 1: DEVICE_CALL owned by DEVICE_LOOP (handler.device_fn_ptr =
//            device_double_handler)
//
// Both dispatchers share the SAME function table.  HOST_LOOP iterates and
// only routes GRAPH_LAUNCH entries; DEVICE_LOOP only routes DEVICE_CALL
// entries.  Under shared_ring_mode this means each peer naturally skips the
// other's slots.
__global__ void init_shared_function_table(cudaq_function_entry_t* entries,
                                           cudaGraphExec_t host_graph_exec) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[0].handler.graph_exec = host_graph_exec;
    entries[0].function_id = HOST_GRAPH_FN_ID;
    entries[0].dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
    entries[0].reserved[0] = 0;
    entries[0].reserved[1] = 0;
    entries[0].reserved[2] = 0;

    entries[1].handler.device_fn_ptr =
        reinterpret_cast<void*>(&device_double_handler);
    entries[1].function_id = DEVICE_CALL_FN_ID;
    entries[1].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    entries[1].reserved[0] = 0;
    entries[1].reserved[1] = 0;
    entries[1].reserved[2] = 0;
  }
}

//==============================================================================
// Test fixture
//==============================================================================

class SharedRingDispatcherTest : public ::testing::Test {
protected:
  void SetUp() override {
    // -- Ring buffer.  RX and TX share the same backing memory: this is
    //    valid for the HOST_LOOP's "legacy mailbox" path (graph kernel
    //    writes response in-place into the RX slot), and the DEVICE_LOOP
    //    kernel will then read its own request and overwrite it with the
    //    response in the same slot.  TX flags are a separate allocation. --
    ASSERT_TRUE(allocate_ring_buffer(kNumSlots, kSlotSize, &rx_flags_host_,
                                     &rx_flags_dev_, &rx_data_host_,
                                     &rx_data_dev_));
    void* tx_flags_host_ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(&tx_flags_host_ptr, kNumSlots * sizeof(uint64_t),
                              cudaHostAllocMapped));
    std::memset(tx_flags_host_ptr, 0, kNumSlots * sizeof(uint64_t));
    tx_flags_host_ = static_cast<volatile uint64_t*>(tx_flags_host_ptr);
    void* tx_flags_dev_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&tx_flags_dev_ptr, tx_flags_host_ptr,
                                         0));
    tx_flags_dev_ = static_cast<volatile uint64_t*>(tx_flags_dev_ptr);
    // RX and TX data buffers point to the SAME backing memory.
    tx_data_host_ = rx_data_host_;
    tx_data_dev_ = rx_data_dev_;
    tx_data_is_owned_ = false;

    // -- Shutdown flag (pinned mapped so both CPU and GPU see it) --
    void* tmp = nullptr;
    CUDA_CHECK(cudaHostAlloc(&tmp, sizeof(int), cudaHostAllocMapped));
    shutdown_flag_host_ = static_cast<volatile int*>(tmp);
    *shutdown_flag_host_ = 0;
    void* tmp_dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&tmp_dev, tmp, 0));
    shutdown_flag_dev_ = static_cast<volatile int*>(tmp_dev);

    // The HOST_LOOP wants a cuda::std::atomic<int>* opaque shutdown flag,
    // backed by the SAME pinned memory so both dispatchers stop together
    // when *shutdown_flag_host_ = 1.
    host_loop_shutdown_atomic_ =
        reinterpret_cast<cuda::std::atomic<int>*>(
            const_cast<int*>(shutdown_flag_host_));

    // -- Stats --
    CUDA_CHECK(cudaMalloc(&device_loop_stats_, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(device_loop_stats_, 0, sizeof(uint64_t)));

    // -- Function table (shared by both dispatchers).  Must be CPU-readable
    //    so the HOST_LOOP loop can do lookup_function() on it, and
    //    GPU-readable so the DEVICE_LOOP kernel can do the same.  Pinned
    //    mapped allocation gives us both views with the same address
    //    under UVA. --
    void* fn_table_host_ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(&fn_table_host_ptr,
                              2 * sizeof(cudaq_function_entry_t),
                              cudaHostAllocMapped));
    function_table_host_ =
        static_cast<cudaq_function_entry_t*>(fn_table_host_ptr);
    void* fn_table_dev_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&fn_table_dev_ptr,
                                         fn_table_host_ptr, 0));
    function_table_dev_ =
        static_cast<cudaq_function_entry_t*>(fn_table_dev_ptr);
    std::memset(function_table_host_, 0,
                 2 * sizeof(cudaq_function_entry_t));

    // -- HOST_LOOP graph (created BEFORE init_shared_function_table) --
    CUDA_CHECK(cudaHostAlloc(&h_mailbox_bank_, sizeof(void*),
                              cudaHostAllocMapped));
    h_mailbox_bank_[0] = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_mailbox_bank_void_,
                                         h_mailbox_bank_, 0));
    d_mailbox_bank_ = static_cast<void**>(d_mailbox_bank_void_);
    ASSERT_TRUE(
        create_host_graph(d_mailbox_bank_, &host_graph_, &host_graph_exec_));

    // -- Initialize function table on GPU --
    init_shared_function_table<<<1, 1>>>(function_table_dev_,
                                          host_graph_exec_);
    CUDA_CHECK(cudaDeviceSynchronize());

    // -- DEVICE_LOOP: push shared_ring_mode into the kernel's __constant__
    //    BEFORE starting the dispatcher.  This is the caller's
    //    responsibility (libcudaq-realtime.so cannot reach into the
    //    static lib's __constant__ symbol). --
    CUDA_CHECK(cudaq_dispatch_kernel_set_shared_ring_mode(1));

    // -- DEVICE_LOOP: dispatcher via the C API --
    ASSERT_EQ(cudaq_dispatch_manager_create(&device_manager_), CUDAQ_OK);

    cudaq_dispatcher_config_t device_config{};
    device_config.device_id = 0;
    device_config.num_blocks = 1;
    device_config.threads_per_block = 64;
    device_config.num_slots = static_cast<uint32_t>(kNumSlots);
    device_config.slot_size = static_cast<uint32_t>(kSlotSize);
    device_config.vp_id = 0;
    device_config.kernel_type = CUDAQ_KERNEL_REGULAR;
    device_config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    device_config.dispatch_path = CUDAQ_DISPATCH_PATH_DEVICE;
    device_config.shared_ring_mode = 1;
    ASSERT_EQ(cudaq_dispatcher_create(device_manager_, &device_config,
                                       &device_dispatcher_),
              CUDAQ_OK);

    cudaq_ringbuffer_t device_rb{};
    device_rb.rx_flags = rx_flags_dev_;
    device_rb.tx_flags = tx_flags_dev_;
    device_rb.rx_data = rx_data_dev_;
    device_rb.tx_data = tx_data_dev_;
    device_rb.rx_stride_sz = kSlotSize;
    device_rb.tx_stride_sz = kSlotSize;
    ASSERT_EQ(cudaq_dispatcher_set_ringbuffer(device_dispatcher_, &device_rb),
              CUDAQ_OK);

    cudaq_function_table_t shared_table{};
    shared_table.entries = function_table_dev_;
    shared_table.count = 2;
    ASSERT_EQ(cudaq_dispatcher_set_function_table(device_dispatcher_,
                                                    &shared_table),
              CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_set_control(device_dispatcher_,
                                            shutdown_flag_dev_,
                                            device_loop_stats_),
              CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_set_launch_fn(
                   device_dispatcher_, &cudaq_launch_dispatch_kernel_regular),
               CUDAQ_OK);

    ASSERT_EQ(cudaq_dispatcher_start(device_dispatcher_), CUDAQ_OK);

    // -- HOST_LOOP: build cudaq_host_dispatch_loop_ctx_t directly and run
    //    on a worker thread.  Using the lower-level entry point gives us
    //    explicit control over shared_ring_mode behavior under test. --
    host_workers_.push_back(cudaq_host_dispatch_worker_t{});
    cudaStream_t host_stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&host_stream));
    host_workers_[0].graph_exec = host_graph_exec_;
    host_workers_[0].stream = host_stream;
    host_workers_[0].function_id = HOST_GRAPH_FN_ID;
    host_workers_[0].pre_launch_fn = nullptr;
    host_workers_[0].pre_launch_data = nullptr;
    host_workers_[0].post_launch_fn = nullptr;
    host_workers_[0].post_launch_data = nullptr;

    host_idle_mask_ =
        new cuda::std::atomic<uint64_t>(1ULL);  // 1 worker, initially idle
    host_live_dispatched_ = new cuda::std::atomic<uint64_t>(0);
    host_inflight_slot_tags_ = new int[host_workers_.size()];
    for (size_t i = 0; i < host_workers_.size(); ++i)
      host_inflight_slot_tags_[i] = -1;

    std::memset(&host_ctx_, 0, sizeof(host_ctx_));
    host_ctx_.ringbuffer.rx_flags = rx_flags_dev_;
    host_ctx_.ringbuffer.tx_flags = tx_flags_dev_;
    host_ctx_.ringbuffer.rx_data = rx_data_dev_;
    host_ctx_.ringbuffer.tx_data = tx_data_dev_;
    host_ctx_.ringbuffer.rx_stride_sz = kSlotSize;
    host_ctx_.ringbuffer.tx_stride_sz = kSlotSize;
    host_ctx_.ringbuffer.rx_flags_host = rx_flags_host_;
    host_ctx_.ringbuffer.tx_flags_host = tx_flags_host_;
    host_ctx_.ringbuffer.rx_data_host = rx_data_host_;
    host_ctx_.ringbuffer.tx_data_host = tx_data_host_;

    host_ctx_.config.num_slots = static_cast<uint32_t>(kNumSlots);
    host_ctx_.config.slot_size = static_cast<uint32_t>(kSlotSize);
    host_ctx_.config.shared_ring_mode = 1;

    // The HOST_LOOP function table has BOTH entries; the loop will only act
    // on the GRAPH_LAUNCH entry (entry 0).  The DEVICE_CALL entry is "in our
    // table but not our mode" -- under shared_ring_mode we skip-without-clear
    // such slots.  This is the realistic configuration in cuda-qx.
    host_ctx_.function_table.entries = function_table_dev_;
    host_ctx_.function_table.count = 2;

    host_ctx_.workers = host_workers_.data();
    host_ctx_.num_workers = host_workers_.size();
    host_ctx_.h_mailbox_bank = h_mailbox_bank_;
    host_ctx_.shutdown_flag = host_loop_shutdown_atomic_;
    host_ctx_.stats_counter = &host_loop_stats_;
    host_ctx_.live_dispatched = host_live_dispatched_;
    host_ctx_.idle_mask = host_idle_mask_;
    host_ctx_.inflight_slot_tags = host_inflight_slot_tags_;
    host_ctx_.io_ctxs_host = nullptr;
    host_ctx_.io_ctxs_dev = nullptr;
    host_ctx_.skip_stream_sweep = false;

    host_loop_thread_ = std::thread([this]() {
      cudaq_host_dispatcher_loop(&host_ctx_);
    });
  }

  void TearDown() override {
    // Stop both dispatchers.
    *shutdown_flag_host_ = 1;
    __sync_synchronize();

    if (host_loop_thread_.joinable())
      host_loop_thread_.join();

    if (device_dispatcher_) {
      cudaq_dispatcher_stop(device_dispatcher_);
      cudaq_dispatcher_destroy(device_dispatcher_);
      device_dispatcher_ = nullptr;
    }
    if (device_manager_) {
      cudaq_dispatch_manager_destroy(device_manager_);
      device_manager_ = nullptr;
    }

    // Restore the dispatch kernel's __constant__ to 0 so we don't affect
    // subsequent tests in the same binary.
    (void)cudaq_dispatch_kernel_set_shared_ring_mode(0);

    if (host_graph_exec_)
      cudaGraphExecDestroy(host_graph_exec_);
    if (host_graph_)
      cudaGraphDestroy(host_graph_);
    for (auto& w : host_workers_) {
      if (w.stream)
        cudaStreamDestroy(w.stream);
    }
    host_workers_.clear();

    delete host_idle_mask_;
    delete host_live_dispatched_;
    delete[] host_inflight_slot_tags_;

    if (function_table_host_)
      cudaFreeHost(function_table_host_);
    if (device_loop_stats_)
      cudaFree(device_loop_stats_);
    if (h_mailbox_bank_)
      cudaFreeHost(h_mailbox_bank_);

    free_ring_buffer(rx_flags_host_, rx_data_host_);
    if (tx_flags_host_)
      cudaFreeHost(const_cast<uint64_t*>(tx_flags_host_));
    if (tx_data_is_owned_ && tx_data_host_)
      cudaFreeHost(tx_data_host_);

    if (shutdown_flag_host_)
      cudaFreeHost(const_cast<int*>(shutdown_flag_host_));
  }

  // Write an RPC request into a slot and signal it by storing the slot
  // address into rx_flags.  Producer-side equivalent of what the cuda-qx
  // rpc_producer will do under shared_ring_mode.
  void WriteAndSignal(std::size_t slot, std::uint32_t function_id,
                      std::uint32_t request_id,
                      const std::vector<std::uint8_t>& payload) {
    ASSERT_LT(slot, kNumSlots);
    ASSERT_LE(payload.size(),
              kSlotSize - sizeof(cudaq::realtime::RPCHeader));
    std::uint8_t* slot_host = rx_data_host_ + slot * kSlotSize;
    auto* header = reinterpret_cast<cudaq::realtime::RPCHeader*>(slot_host);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = function_id;
    header->arg_len = static_cast<std::uint32_t>(payload.size());
    header->request_id = request_id;
    header->ptp_timestamp = 0;
    std::memcpy(slot_host + sizeof(cudaq::realtime::RPCHeader),
                payload.data(), payload.size());
    __sync_synchronize();
    // Producer publishes the slot by writing the device-visible RX address
    // (the convention is that rx_flags[i] holds the address of the RX slot
    // data; both dispatchers use it as the request pointer).
    rx_flags_host_[slot] =
        reinterpret_cast<std::uint64_t>(rx_data_dev_ + slot * kSlotSize);
  }

  // Wait for the response magic to appear in the data slot.  Both
  // dispatchers write RPCResponse (magic = CUDAQ_RPC_MAGIC_RESPONSE) into
  // their tx_slot, which under our aliased rx/tx setup is the same memory
  // we wrote the request to.  This is more robust than polling tx_flags,
  // since the legacy HOST_LOOP mailbox path can leave tx_flags stuck at
  // CUDAQ_TX_FLAG_IN_FLIGHT until the worker is recycled.
  bool WaitForResponseInSlot(std::size_t slot, int timeout_ms = 5000) {
    std::uint8_t* slot_host = rx_data_host_ + slot * kSlotSize;
    auto* resp = reinterpret_cast<cudaq::realtime::RPCResponse*>(slot_host);
    for (int waited = 0; waited < timeout_ms; ++waited) {
      __sync_synchronize();
      if (resp->magic == cudaq::realtime::RPC_MAGIC_RESPONSE)
        return true;
      usleep(1000);
    }
    return false;
  }

  std::vector<std::uint8_t> ReadResponse(std::size_t slot) {
    std::vector<std::uint8_t> out;
    std::uint8_t* slot_host = rx_data_host_ + slot * kSlotSize;
    auto* resp = reinterpret_cast<cudaq::realtime::RPCResponse*>(slot_host);
    if (resp->magic == cudaq::realtime::RPC_MAGIC_RESPONSE) {
      out.resize(resp->result_len);
      std::memcpy(out.data(),
                  slot_host + sizeof(cudaq::realtime::RPCResponse),
                  resp->result_len);
    }
    return out;
  }

  // -- Ring buffer (shared by both dispatchers) --
  volatile uint64_t* rx_flags_host_ = nullptr;
  volatile uint64_t* tx_flags_host_ = nullptr;
  volatile uint64_t* rx_flags_dev_ = nullptr;
  volatile uint64_t* tx_flags_dev_ = nullptr;
  std::uint8_t* rx_data_host_ = nullptr;
  std::uint8_t* tx_data_host_ = nullptr;
  std::uint8_t* rx_data_dev_ = nullptr;
  std::uint8_t* tx_data_dev_ = nullptr;
  bool tx_data_is_owned_ = true; // false when tx aliases rx

  // -- Shared shutdown + function table --
  volatile int* shutdown_flag_host_ = nullptr;
  volatile int* shutdown_flag_dev_ = nullptr;
  cuda::std::atomic<int>* host_loop_shutdown_atomic_ = nullptr;
  cudaq_function_entry_t* function_table_host_ = nullptr;
  cudaq_function_entry_t* function_table_dev_ = nullptr;

  // -- DEVICE_LOOP dispatcher --
  cudaq_dispatch_manager_t* device_manager_ = nullptr;
  cudaq_dispatcher_t* device_dispatcher_ = nullptr;
  uint64_t* device_loop_stats_ = nullptr;

  // -- HOST_LOOP dispatcher --
  cudaGraph_t host_graph_ = nullptr;
  cudaGraphExec_t host_graph_exec_ = nullptr;
  void** h_mailbox_bank_ = nullptr;
  void* d_mailbox_bank_void_ = nullptr;
  void** d_mailbox_bank_ = nullptr;
  std::vector<cudaq_host_dispatch_worker_t> host_workers_;
  cuda::std::atomic<uint64_t>* host_idle_mask_ = nullptr;
  cuda::std::atomic<uint64_t>* host_live_dispatched_ = nullptr;
  int* host_inflight_slot_tags_ = nullptr;
  uint64_t host_loop_stats_ = 0;
  cudaq_host_dispatch_loop_ctx_t host_ctx_{};
  std::thread host_loop_thread_;
};

//==============================================================================
// Test: interleaved requests, both dispatchers running, shared_ring_mode = 1
//==============================================================================

TEST_F(SharedRingDispatcherTest, InterleavedHostAndDeviceRequests) {
  // Slot 0: HOST_LOOP (increment by 1)
  // Slot 1: DEVICE_LOOP (double)
  // Slot 2: HOST_LOOP
  // Slot 3: DEVICE_LOOP
  std::vector<std::uint8_t> p0 = {10, 20, 30, 40};
  std::vector<std::uint8_t> p1 = {3, 5, 7, 9};
  std::vector<std::uint8_t> p2 = {1, 2, 3, 4};
  std::vector<std::uint8_t> p3 = {6, 12, 24, 48};

  WriteAndSignal(0, HOST_GRAPH_FN_ID, /*request_id=*/100, p0);
  WriteAndSignal(1, DEVICE_CALL_FN_ID, /*request_id=*/101, p1);
  WriteAndSignal(2, HOST_GRAPH_FN_ID, /*request_id=*/102, p2);
  WriteAndSignal(3, DEVICE_CALL_FN_ID, /*request_id=*/103, p3);

  ASSERT_TRUE(WaitForResponseInSlot(0)) << "Slot 0 (HOST_LOOP) timed out";
  ASSERT_TRUE(WaitForResponseInSlot(1)) << "Slot 1 (DEVICE_LOOP) timed out";
  ASSERT_TRUE(WaitForResponseInSlot(2)) << "Slot 2 (HOST_LOOP) timed out";
  ASSERT_TRUE(WaitForResponseInSlot(3)) << "Slot 3 (DEVICE_LOOP) timed out";

  std::vector<std::uint8_t> r0 = ReadResponse(0);
  std::vector<std::uint8_t> r1 = ReadResponse(1);
  std::vector<std::uint8_t> r2 = ReadResponse(2);
  std::vector<std::uint8_t> r3 = ReadResponse(3);

  std::vector<std::uint8_t> e0 = {11, 21, 31, 41};   // p0 + 1
  std::vector<std::uint8_t> e1 = {6, 10, 14, 18};    // p1 * 2
  std::vector<std::uint8_t> e2 = {2, 3, 4, 5};       // p2 + 1
  std::vector<std::uint8_t> e3 = {12, 24, 48, 96};   // p3 * 2

  EXPECT_EQ(r0, e0) << "HOST_LOOP slot 0";
  EXPECT_EQ(r1, e1) << "DEVICE_LOOP slot 1";
  EXPECT_EQ(r2, e2) << "HOST_LOOP slot 2";
  EXPECT_EQ(r3, e3) << "DEVICE_LOOP slot 3";

  // Stats sanity: each dispatcher should have processed exactly two slots
  // (the two slots whose function_id matches its own table entry mode).
  // The DEVICE_LOOP stats are written when the kernel exits; we shut it
  // down explicitly below to flush.
  *shutdown_flag_host_ = 1;
  __sync_synchronize();
  if (host_loop_thread_.joinable())
    host_loop_thread_.join();
  // Stopping the DEVICE_LOOP triggers cudaStreamSynchronize on the
  // dispatcher's stream, which flushes the kernel's final atomicAdd into
  // device_loop_stats_.
  cudaq_dispatcher_stop(device_dispatcher_);

  uint64_t dev_count = 0;
  CUDA_CHECK(cudaMemcpy(&dev_count, device_loop_stats_, sizeof(uint64_t),
                         cudaMemcpyDeviceToHost));
  EXPECT_EQ(dev_count, 2u)
      << "DEVICE_LOOP should have processed exactly 2 slots, got "
      << dev_count;
  EXPECT_EQ(host_loop_stats_, 2u)
      << "HOST_LOOP should have processed exactly 2 slots, got "
      << host_loop_stats_;

  // Mark device dispatcher as already stopped so TearDown doesn't double
  // stop / destroy.
  cudaq_dispatcher_destroy(device_dispatcher_);
  device_dispatcher_ = nullptr;
}

} // namespace
