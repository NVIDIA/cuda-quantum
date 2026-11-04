/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <thread>
#include <unistd.h>
#include <vector>

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatch_loop.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

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

// Plain host memory for the host-only fixtures: the host loops and the ring
// helpers use only the _host views of a cudaq_ringbuffer_t, so the ring needs
// to be CUDA-mapped only when a GPU consumer touches it
// (SharedRingDispatcherTest below).  Zero-initialized.
bool allocate_host_ring_buffer(std::size_t num_slots, std::size_t slot_size,
                               volatile uint64_t** flags_out,
                               std::uint8_t** data_out) {
  auto* flags = new (std::nothrow) uint64_t[num_slots]();
  auto* data = new (std::nothrow) std::uint8_t[num_slots * slot_size]();
  if (!flags || !data) {
    delete[] flags;
    delete[] data;
    return false;
  }
  *flags_out = flags;
  *data_out = data;
  return true;
}

void free_host_ring_buffer(volatile uint64_t* flags, std::uint8_t* data) {
  delete[] const_cast<uint64_t*>(flags);
  delete[] data;
}

//==============================================================================
// HOST_CALL handler (two-pointer ABI): increments the rx payload into tx.
//==============================================================================

void host_increment(const void* rx_slot, void* tx_slot, std::size_t slot_size) {
  const auto* h = static_cast<const RPCHeader*>(rx_slot);
  if (h->magic != cudaq::realtime::RPC_MAGIC_REQUEST)
    return;
  const std::uint32_t arg_len = h->arg_len;
  const std::uint32_t request_id = h->request_id;
  const std::uint64_t ptp = h->ptp_timestamp;
  if (slot_size < sizeof(RPCResponse) + arg_len)
    return;
  const auto* in =
      static_cast<const std::uint8_t*>(rx_slot) + sizeof(RPCHeader);
  auto* resp = static_cast<RPCResponse*>(tx_slot);
  resp->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
  resp->status = 0;
  resp->result_len = arg_len;
  resp->request_id = request_id;
  resp->ptp_timestamp = ptp;
  auto* out = static_cast<std::uint8_t*>(tx_slot) + sizeof(RPCResponse);
  for (std::uint32_t i = 0; i < arg_len; ++i)
    out[i] = in[i] + 1;
}

constexpr std::uint32_t HOST_INCREMENT_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("host_increment");

//==============================================================================
// Test fixture: drives the ring loop (cudaq_host_ring_dispatch_loop) directly
// over a HOST_CALL function table, for full control over slot sequencing and
// the stats counter.  Separate RX and TX rings: the handler reads the request
// from the RX slot and writes the response into the TX slot.
//==============================================================================

static constexpr std::size_t kMaxFunctionEntries = 8;

class HostDispatcherLoopTest : public ::testing::Test {
protected:
  void SetUp() override {
    std::memset(&config_, 0, sizeof(config_));
    ASSERT_TRUE(allocate_host_ring_buffer(num_slots_, slot_size_,
                                          &rx_flags_host_, &rx_data_host_));
    ASSERT_TRUE(allocate_host_ring_buffer(num_slots_, slot_size_,
                                          &tx_flags_host_, &tx_data_host_));

    stats_counter_ = 0;

    function_table_ = new cudaq_function_entry_t[kMaxFunctionEntries];
    std::memset(function_table_, 0,
                kMaxFunctionEntries * sizeof(cudaq_function_entry_t));

    // Host views only; the device views stay null (no GPU consumer).
    std::memset(&ringbuffer_, 0, sizeof(ringbuffer_));
    ringbuffer_.rx_stride_sz = slot_size_;
    ringbuffer_.tx_stride_sz = slot_size_;
    ringbuffer_.rx_flags_host = rx_flags_host_;
    ringbuffer_.tx_flags_host = tx_flags_host_;
    ringbuffer_.rx_data_host = rx_data_host_;
    ringbuffer_.tx_data_host = tx_data_host_;

    config_.num_slots = static_cast<uint32_t>(num_slots_);
    config_.slot_size = static_cast<uint32_t>(slot_size_);
  }

  void TearDown() override {
    if (!loop_stopped_) {
      shutdown_flag_ = 1;
      __sync_synchronize();
      if (loop_thread_.joinable())
        loop_thread_.join();
    }

    free_host_ring_buffer(rx_flags_host_, rx_data_host_);
    free_host_ring_buffer(tx_flags_host_, tx_data_host_);
    delete[] function_table_;
  }

  // Register a HOST_CALL entry.
  void AddHostCall(std::uint32_t function_id, cudaq_host_rpc_fn_t fn) {
    std::size_t idx = function_table_count_;
    function_table_[idx].handler.host_fn = fn;
    function_table_[idx].function_id = function_id;
    function_table_[idx].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
    function_table_count_++;
  }

  void StartLoop() {
    table_.entries = function_table_;
    table_.count = static_cast<uint32_t>(function_table_count_);

    loop_thread_ = std::thread([this]() {
      cudaq_host_ring_dispatch_loop(&ringbuffer_, &table_, &config_,
                                    &shutdown_flag_, &stats_counter_);
    });
  }

  void WriteRpcRequest(std::size_t slot, std::uint32_t function_id,
                       const std::uint8_t* payload, std::size_t len) {
    ASSERT_EQ(cudaq_host_ringbuffer_write_rpc_request(
                  &ringbuffer_, static_cast<uint32_t>(slot), function_id,
                  payload, static_cast<uint32_t>(len), 0, 0),
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
    shutdown_flag_ = 1;
    __sync_synchronize();
    if (loop_thread_.joinable())
      loop_thread_.join();
    loop_stopped_ = true;
  }

  void ClearSlot(std::size_t slot) {
    cudaq_host_ringbuffer_clear_slot(&ringbuffer_, static_cast<uint32_t>(slot));
    std::memset(tx_data_host_ + slot * slot_size_, 0, slot_size_);
  }

  void VerifyResponse(std::size_t slot, const std::uint8_t* expected,
                      std::size_t len) {
    cudaq_tx_status_t st = cudaq_host_ringbuffer_poll_tx_flag(
        &ringbuffer_, static_cast<uint32_t>(slot), nullptr);
    ASSERT_EQ(st, CUDAQ_TX_READY)
        << "slot " << slot << ": tx_flag unexpected (status=" << st << ")";

    std::uint8_t* slot_data = tx_data_host_ + slot * slot_size_;
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
  std::uint8_t* rx_data_host_ = nullptr;
  std::uint8_t* tx_data_host_ = nullptr;

  volatile int shutdown_flag_ = 0;
  uint64_t stats_counter_ = 0;
  bool loop_stopped_ = false;

  cudaq_function_entry_t* function_table_ = nullptr;
  std::size_t function_table_count_ = 0;

  cudaq_ringbuffer_t ringbuffer_{};
  cudaq_function_table_t table_{};
  cudaq_dispatcher_config_t config_;
  std::thread loop_thread_;
};

//==============================================================================
// Smoke test: the C API starts the host loop over a HOST_CALL table, and the
// loop drops a slot whose function_id is not in the table (no response).
//==============================================================================

// Use a different function_id in the slot so the host loop does not find it.
constexpr std::uint32_t UNKNOWN_FUNCTION_ID = 0xdeadbeefu;

class HostDispatcherSmokeTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(allocate_host_ring_buffer(num_slots_, slot_size_,
                                          &rx_flags_host_, &rx_data_host_));
    ASSERT_TRUE(allocate_host_ring_buffer(num_slots_, slot_size_,
                                          &tx_flags_host_, &tx_data_host_));

    shutdown_flag_ = new (std::nothrow) int(0);
    stats_ = new (std::nothrow) uint64_t(0);
    ASSERT_NE(shutdown_flag_, nullptr);
    ASSERT_NE(stats_, nullptr);

    host_table_ = new (std::nothrow) cudaq_function_entry_t[1];
    ASSERT_NE(host_table_, nullptr);
    std::memset(host_table_, 0, sizeof(cudaq_function_entry_t));
    host_table_[0].handler.host_fn = &host_increment;
    host_table_[0].function_id = HOST_INCREMENT_FUNCTION_ID;
    host_table_[0].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

    ASSERT_EQ(cudaq_dispatch_manager_create(&manager_), CUDAQ_OK);
    cudaq_dispatcher_config_t config{};
    config.num_slots = static_cast<uint32_t>(num_slots_);
    config.slot_size = static_cast<uint32_t>(slot_size_);
    config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
    ASSERT_EQ(cudaq_dispatcher_create(manager_, &config, &dispatcher_),
              CUDAQ_OK);

    // Host views only; the device views stay null (no GPU consumer).
    std::memset(&ringbuffer_, 0, sizeof(ringbuffer_));
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
    free_host_ring_buffer(rx_flags_host_, rx_data_host_);
    free_host_ring_buffer(tx_flags_host_, tx_data_host_);
    if (shutdown_flag_)
      delete shutdown_flag_;
    if (stats_)
      delete stats_;
    if (host_table_)
      delete[] host_table_;
  }

  void write_rpc_request_unknown_function(std::size_t slot) {
    const std::uint8_t payload[] = {0, 1, 2, 3};
    ASSERT_EQ(cudaq_host_ringbuffer_write_rpc_request(
                  &ringbuffer_, static_cast<uint32_t>(slot),
                  UNKNOWN_FUNCTION_ID, payload, 4, 0, 0),
              CUDAQ_OK);
  }

  static constexpr std::size_t num_slots_ = 2;
  std::size_t slot_size_ = 256;

  volatile uint64_t* rx_flags_host_ = nullptr;
  volatile uint64_t* tx_flags_host_ = nullptr;
  std::uint8_t* rx_data_host_ = nullptr;
  std::uint8_t* tx_data_host_ = nullptr;

  int* shutdown_flag_ = nullptr;
  uint64_t* stats_ = nullptr;
  cudaq_function_entry_t* host_table_ = nullptr;

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
// start() refuses a host-path table carrying an entry the host loop cannot
// serve (DEVICE_CALL or GRAPH_LAUNCH) unless shared_ring_mode says a peer
// dispatcher on the same ring serves it.
//==============================================================================

class HostDispatcherTableTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(allocate_host_ring_buffer(num_slots_, slot_size_,
                                          &rx_flags_host_, &rx_data_host_));
    ASSERT_TRUE(allocate_host_ring_buffer(num_slots_, slot_size_,
                                          &tx_flags_host_, &tx_data_host_));
    std::memset(&ringbuffer_, 0, sizeof(ringbuffer_));
    ringbuffer_.rx_stride_sz = slot_size_;
    ringbuffer_.tx_stride_sz = slot_size_;
    ringbuffer_.rx_flags_host = rx_flags_host_;
    ringbuffer_.tx_flags_host = tx_flags_host_;
    ringbuffer_.rx_data_host = rx_data_host_;
    ringbuffer_.tx_data_host = tx_data_host_;

    std::memset(entries_, 0, sizeof(entries_));
    entries_[0].handler.host_fn = &host_increment;
    entries_[0].function_id = HOST_INCREMENT_FUNCTION_ID;
    entries_[0].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
    entries_[1].function_id = HOST_INCREMENT_FUNCTION_ID + 1;

    ASSERT_EQ(cudaq_dispatch_manager_create(&manager_), CUDAQ_OK);
  }

  void TearDown() override {
    shutdown_flag_ = 1;
    __sync_synchronize();
    if (dispatcher_) {
      cudaq_dispatcher_stop(dispatcher_);
      cudaq_dispatcher_destroy(dispatcher_);
    }
    if (manager_)
      cudaq_dispatch_manager_destroy(manager_);
    free_host_ring_buffer(rx_flags_host_, rx_data_host_);
    free_host_ring_buffer(tx_flags_host_, tx_data_host_);
  }

  // Create a host-path dispatcher over entries_ (a HOST_CALL entry plus one
  // of `peer_mode`) and return what start() says about it.
  cudaq_status_t StartWith(cudaq_dispatch_mode_t peer_mode,
                           uint32_t shared_ring_mode) {
    entries_[1].dispatch_mode = peer_mode;
    cudaq_dispatcher_config_t config{};
    config.num_slots = static_cast<uint32_t>(num_slots_);
    config.slot_size = static_cast<uint32_t>(slot_size_);
    config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
    config.shared_ring_mode = shared_ring_mode;
    EXPECT_EQ(cudaq_dispatcher_create(manager_, &config, &dispatcher_),
              CUDAQ_OK);
    cudaq_function_table_t table{};
    table.entries = entries_;
    table.count = 2;
    EXPECT_EQ(cudaq_dispatcher_set_ringbuffer(dispatcher_, &ringbuffer_),
              CUDAQ_OK);
    EXPECT_EQ(cudaq_dispatcher_set_function_table(dispatcher_, &table),
              CUDAQ_OK);
    EXPECT_EQ(cudaq_dispatcher_set_control(dispatcher_, &shutdown_flag_,
                                           &stats_),
              CUDAQ_OK);
    return cudaq_dispatcher_start(dispatcher_);
  }

  static constexpr std::size_t num_slots_ = 2;
  std::size_t slot_size_ = 256;
  volatile uint64_t* rx_flags_host_ = nullptr;
  volatile uint64_t* tx_flags_host_ = nullptr;
  std::uint8_t* rx_data_host_ = nullptr;
  std::uint8_t* tx_data_host_ = nullptr;
  volatile int shutdown_flag_ = 0;
  uint64_t stats_ = 0;
  cudaq_function_entry_t entries_[2];
  cudaq_ringbuffer_t ringbuffer_{};
  cudaq_dispatch_manager_t* manager_ = nullptr;
  cudaq_dispatcher_t* dispatcher_ = nullptr;
};

TEST_F(HostDispatcherTableTest, RefusesGraphLaunchEntry) {
  EXPECT_EQ(StartWith(CUDAQ_DISPATCH_GRAPH_LAUNCH, 0), CUDAQ_ERR_UNSUPPORTED);
}

TEST_F(HostDispatcherTableTest, RefusesDeviceCallEntry) {
  EXPECT_EQ(StartWith(CUDAQ_DISPATCH_DEVICE_CALL, 0), CUDAQ_ERR_UNSUPPORTED);
}

TEST_F(HostDispatcherTableTest, SharedRingLeavesPeerEntriesToThePeer) {
  EXPECT_EQ(StartWith(CUDAQ_DISPATCH_GRAPH_LAUNCH, 1), CUDAQ_OK);
}

//==============================================================================
// Stats counter accuracy (internal API)
//
// Sends 5 sequential RPCs through one HOST_CALL entry and verifies
// stats_counter == 5 at the end.
//==============================================================================

TEST_F(HostDispatcherLoopTest, StatsCounterAccuracy) {
  AddHostCall(HOST_INCREMENT_FUNCTION_ID, &host_increment);

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
    WriteRpcRequest(slot, HOST_INCREMENT_FUNCTION_ID, payload, 4);
    SignalSlot(slot);
    ASSERT_TRUE(PollTxFlag(slot)) << "Timeout on RPC " << i << " (slot " << slot << ")";

    std::uint8_t expected[] = {
        static_cast<std::uint8_t>(i * 10 + 1),
        static_cast<std::uint8_t>(i * 10 + 2),
        static_cast<std::uint8_t>(i * 10 + 3),
        static_cast<std::uint8_t>(i * 10 + 4)};
    VerifyResponse(slot, expected, 4);
  }

  StopLoop();
  EXPECT_EQ(stats_counter_, static_cast<uint64_t>(kNumRpcs));
}

//==============================================================================
// Multi-slot round-robin dispatch (internal API)
//
// 4 slots, one HOST_CALL entry.  All slots signalled at once; the dispatcher
// processes them 0 → 1 → 2 → 3.
//==============================================================================

TEST_F(HostDispatcherLoopTest, MultiSlotRoundRobin) {
  constexpr int kNumSlots = 4;
  AddHostCall(HOST_INCREMENT_FUNCTION_ID, &host_increment);

  StartLoop();

  for (int i = 0; i < kNumSlots; ++i) {
    std::uint8_t payload[] = {
        static_cast<std::uint8_t>(i * 4 + 1),
        static_cast<std::uint8_t>(i * 4 + 2),
        static_cast<std::uint8_t>(i * 4 + 3),
        static_cast<std::uint8_t>(i * 4 + 4)};
    WriteRpcRequest(static_cast<std::size_t>(i), HOST_INCREMENT_FUNCTION_ID,
                    payload, 4);
  }

  for (int i = 0; i < kNumSlots; ++i)
    SignalSlot(static_cast<std::size_t>(i));

  for (int i = 0; i < kNumSlots; ++i) {
    ASSERT_TRUE(PollTxFlag(static_cast<std::size_t>(i)))
        << "Timeout on slot " << i;
  }

  for (int i = 0; i < kNumSlots; ++i) {
    std::uint8_t expected[] = {
        static_cast<std::uint8_t>(i * 4 + 2),
        static_cast<std::uint8_t>(i * 4 + 3),
        static_cast<std::uint8_t>(i * 4 + 4),
        static_cast<std::uint8_t>(i * 4 + 5)};
    VerifyResponse(static_cast<std::size_t>(i), expected, 4);
  }

  StopLoop();
  EXPECT_EQ(stats_counter_, static_cast<uint64_t>(kNumSlots));
}

//==============================================================================
// Shared-ring-mode test: HOST_LOOP + DEVICE_LOOP on one RX ring
//
// The function table holds one HOST_CALL entry and one DEVICE_CALL entry.
// The device kernel SKIPs the HOST_CALL slots and the host loop leaves the
// DEVICE_CALL slots for its peer; each request is answered exactly once.
//==============================================================================

constexpr std::size_t kSharedRingNumSlots = 8;
constexpr std::size_t kSharedRingSlotSize = 256;

constexpr std::uint32_t HOST_CALL_FN_ID =
    cudaq::realtime::fnv1a_hash("shared_ring_host_increment");

constexpr std::uint32_t DEVICE_CALL_FN_ID =
    cudaq::realtime::fnv1a_hash("shared_ring_device_double");

// DEVICE_CALL handler (same two-pointer form as host_increment above, plus a
// returned status): doubles the rx payload into tx and sets result_len.
__device__ int device_double_handler(const void* rx_slot, void* tx_slot,
                                     std::size_t slot_size) {
  // The dispatcher calls this on every thread of its block: each thread
  // doubles its own bytes.  That is safe on the in-place ring of
  // SharedRingDispatcherTest (tx_slot aliases rx_slot) because no byte is
  // touched by more than one thread.
  const auto* h = static_cast<const RPCHeader*>(rx_slot);
  const auto* in =
      static_cast<const std::uint8_t*>(rx_slot) + sizeof(RPCHeader);
  auto* resp = static_cast<RPCResponse*>(tx_slot);
  auto* out = static_cast<std::uint8_t*>(tx_slot) + sizeof(RPCResponse);
  const std::uint32_t max_result_len = slot_size - sizeof(RPCResponse);
  std::uint32_t n = h->arg_len;
  if (n > max_result_len)
    n = max_result_len;
  for (std::uint32_t i = threadIdx.x; i < n; i += blockDim.x)
    out[i] = static_cast<std::uint8_t>(in[i] * 2);
  // result_len aliases arg_len on the in-place ring: every thread must have
  // read the header before thread 0 overwrites it.
  __syncthreads();
  if (threadIdx.x == 0)
    resp->result_len = n;
  return 0;
}

// The DEVICE_CALL entry needs the device function's address, which only device
// code can take; the HOST_CALL entry is written from the host.
__global__ void init_shared_function_table(cudaq_function_entry_t* entries) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[1].handler.device_fn_ptr =
        reinterpret_cast<void*>(&device_double_handler);
    entries[1].function_id = DEVICE_CALL_FN_ID;
    entries[1].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    entries[1].reserved[0] = 0;
    entries[1].reserved[1] = 0;
    entries[1].reserved[2] = 0;
  }
}

class SharedRingDispatcherTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_TRUE(allocate_ring_buffer(kSharedRingNumSlots, kSharedRingSlotSize,
                                     &rx_flags_host_, &rx_flags_dev_,
                                     &rx_data_host_, &rx_data_dev_));
    void* tx_flags_host_ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(&tx_flags_host_ptr,
                             kSharedRingNumSlots * sizeof(uint64_t),
                             cudaHostAllocMapped));
    std::memset(tx_flags_host_ptr, 0,
                kSharedRingNumSlots * sizeof(uint64_t));
    tx_flags_host_ = static_cast<volatile uint64_t*>(tx_flags_host_ptr);
    void* tx_flags_dev_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&tx_flags_dev_ptr, tx_flags_host_ptr,
                                        0));
    tx_flags_dev_ = static_cast<volatile uint64_t*>(tx_flags_dev_ptr);
    // RX and TX data buffers intentionally point to the same backing memory.
    tx_data_host_ = rx_data_host_;
    tx_data_dev_ = rx_data_dev_;
    tx_data_is_owned_ = false;

    void* tmp = nullptr;
    CUDA_CHECK(cudaHostAlloc(&tmp, sizeof(int), cudaHostAllocMapped));
    shutdown_flag_host_ = static_cast<volatile int*>(tmp);
    *shutdown_flag_host_ = 0;
    void* tmp_dev = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&tmp_dev, tmp, 0));
    shutdown_flag_dev_ = static_cast<volatile int*>(tmp_dev);

    CUDA_CHECK(cudaMalloc(&device_loop_stats_, sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(device_loop_stats_, 0, sizeof(uint64_t)));

    void* fn_table_host_ptr = nullptr;
    CUDA_CHECK(cudaHostAlloc(&fn_table_host_ptr,
                             2 * sizeof(cudaq_function_entry_t),
                             cudaHostAllocMapped));
    function_table_host_ =
        static_cast<cudaq_function_entry_t*>(fn_table_host_ptr);
    void* fn_table_dev_ptr = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&fn_table_dev_ptr, fn_table_host_ptr,
                                        0));
    function_table_dev_ =
        static_cast<cudaq_function_entry_t*>(fn_table_dev_ptr);
    std::memset(function_table_host_, 0,
                2 * sizeof(cudaq_function_entry_t));

    function_table_host_[0].handler.host_fn = &host_increment;
    function_table_host_[0].function_id = HOST_CALL_FN_ID;
    function_table_host_[0].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

    init_shared_function_table<<<1, 1>>>(function_table_dev_);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaq_dispatch_kernel_set_shared_ring_mode(1));

    ASSERT_EQ(cudaq_dispatch_manager_create(&device_manager_), CUDAQ_OK);

    cudaq_dispatcher_config_t device_config{};
    device_config.device_id = 0;
    device_config.num_blocks = 1;
    device_config.threads_per_block = 64;
    device_config.num_slots = static_cast<uint32_t>(kSharedRingNumSlots);
    device_config.slot_size = static_cast<uint32_t>(kSharedRingSlotSize);
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
    device_rb.rx_stride_sz = kSharedRingSlotSize;
    device_rb.tx_stride_sz = kSharedRingSlotSize;
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

    std::memset(&host_ringbuffer_, 0, sizeof(host_ringbuffer_));
    host_ringbuffer_.rx_flags = rx_flags_dev_;
    host_ringbuffer_.tx_flags = tx_flags_dev_;
    host_ringbuffer_.rx_data = rx_data_dev_;
    host_ringbuffer_.tx_data = tx_data_dev_;
    host_ringbuffer_.rx_stride_sz = kSharedRingSlotSize;
    host_ringbuffer_.tx_stride_sz = kSharedRingSlotSize;
    host_ringbuffer_.rx_flags_host = rx_flags_host_;
    host_ringbuffer_.tx_flags_host = tx_flags_host_;
    host_ringbuffer_.rx_data_host = rx_data_host_;
    host_ringbuffer_.tx_data_host = tx_data_host_;

    host_table_.entries = function_table_host_;
    host_table_.count = 2;

    std::memset(&host_config_, 0, sizeof(host_config_));
    host_config_.num_slots = static_cast<uint32_t>(kSharedRingNumSlots);
    host_config_.slot_size = static_cast<uint32_t>(kSharedRingSlotSize);
    host_config_.shared_ring_mode = 1;

    host_loop_thread_ = std::thread([this]() {
      cudaq_host_ring_dispatch_loop(&host_ringbuffer_, &host_table_,
                                    &host_config_, shutdown_flag_host_,
                                    &host_loop_stats_);
    });
  }

  void TearDown() override {
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

    (void)cudaq_dispatch_kernel_set_shared_ring_mode(0);

    if (function_table_host_)
      cudaFreeHost(function_table_host_);
    if (device_loop_stats_)
      cudaFree(device_loop_stats_);

    free_ring_buffer(rx_flags_host_, rx_data_host_);
    if (tx_flags_host_)
      cudaFreeHost(const_cast<uint64_t*>(tx_flags_host_));
    if (tx_data_is_owned_ && tx_data_host_)
      cudaFreeHost(tx_data_host_);

    if (shutdown_flag_host_)
      cudaFreeHost(const_cast<int*>(shutdown_flag_host_));
  }

  void WriteAndSignal(std::size_t slot, std::uint32_t function_id,
                      std::uint32_t request_id,
                      const std::vector<std::uint8_t>& payload) {
    ASSERT_LT(slot, kSharedRingNumSlots);
    ASSERT_LE(payload.size(),
              kSharedRingSlotSize - sizeof(cudaq::realtime::RPCHeader));
    std::uint8_t* slot_host = rx_data_host_ + slot * kSharedRingSlotSize;
    auto* header = reinterpret_cast<cudaq::realtime::RPCHeader*>(slot_host);
    header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
    header->function_id = function_id;
    header->arg_len = static_cast<std::uint32_t>(payload.size());
    header->request_id = request_id;
    header->ptp_timestamp = 0;
    std::memcpy(slot_host + sizeof(cudaq::realtime::RPCHeader),
                payload.data(), payload.size());
    __sync_synchronize();
    rx_flags_host_[slot] = reinterpret_cast<std::uint64_t>(
        rx_data_dev_ + slot * kSharedRingSlotSize);
  }

  bool WaitForResponseInSlot(std::size_t slot, int timeout_ms = 5000) {
    std::uint8_t* slot_host = rx_data_host_ + slot * kSharedRingSlotSize;
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
    std::uint8_t* slot_host = rx_data_host_ + slot * kSharedRingSlotSize;
    auto* resp = reinterpret_cast<cudaq::realtime::RPCResponse*>(slot_host);
    std::vector<std::uint8_t> out(resp->result_len);
    std::memcpy(out.data(),
                slot_host + sizeof(cudaq::realtime::RPCResponse),
                resp->result_len);
    return out;
  }

  volatile uint64_t* rx_flags_host_ = nullptr;
  volatile uint64_t* tx_flags_host_ = nullptr;
  volatile uint64_t* rx_flags_dev_ = nullptr;
  volatile uint64_t* tx_flags_dev_ = nullptr;
  std::uint8_t* rx_data_host_ = nullptr;
  std::uint8_t* tx_data_host_ = nullptr;
  std::uint8_t* rx_data_dev_ = nullptr;
  std::uint8_t* tx_data_dev_ = nullptr;
  bool tx_data_is_owned_ = true;

  volatile int* shutdown_flag_host_ = nullptr;
  volatile int* shutdown_flag_dev_ = nullptr;
  cudaq_function_entry_t* function_table_host_ = nullptr;
  cudaq_function_entry_t* function_table_dev_ = nullptr;

  cudaq_dispatch_manager_t* device_manager_ = nullptr;
  cudaq_dispatcher_t* device_dispatcher_ = nullptr;
  uint64_t* device_loop_stats_ = nullptr;

  cudaq_ringbuffer_t host_ringbuffer_{};
  cudaq_function_table_t host_table_{};
  cudaq_dispatcher_config_t host_config_{};
  uint64_t host_loop_stats_ = 0;
  std::thread host_loop_thread_;
};

TEST_F(SharedRingDispatcherTest, InterleavedHostAndDeviceRequests) {
  std::vector<std::uint8_t> p0 = {10, 20, 30, 40};
  std::vector<std::uint8_t> p1 = {3, 5, 7, 9};
  std::vector<std::uint8_t> p2 = {1, 2, 3, 4};
  std::vector<std::uint8_t> p3 = {6, 12, 24, 48};

  WriteAndSignal(0, HOST_CALL_FN_ID, /*request_id=*/100, p0);
  WriteAndSignal(1, DEVICE_CALL_FN_ID, /*request_id=*/101, p1);
  WriteAndSignal(2, HOST_CALL_FN_ID, /*request_id=*/102, p2);
  WriteAndSignal(3, DEVICE_CALL_FN_ID, /*request_id=*/103, p3);

  ASSERT_TRUE(WaitForResponseInSlot(0)) << "Slot 0 (HOST_LOOP) timed out";
  ASSERT_TRUE(WaitForResponseInSlot(1)) << "Slot 1 (DEVICE_LOOP) timed out";
  ASSERT_TRUE(WaitForResponseInSlot(2)) << "Slot 2 (HOST_LOOP) timed out";
  ASSERT_TRUE(WaitForResponseInSlot(3)) << "Slot 3 (DEVICE_LOOP) timed out";

  EXPECT_EQ(ReadResponse(0), (std::vector<std::uint8_t>{11, 21, 31, 41}));
  EXPECT_EQ(ReadResponse(1), (std::vector<std::uint8_t>{6, 10, 14, 18}));
  EXPECT_EQ(ReadResponse(2), (std::vector<std::uint8_t>{2, 3, 4, 5}));
  EXPECT_EQ(ReadResponse(3), (std::vector<std::uint8_t>{12, 24, 48, 96}));

  *shutdown_flag_host_ = 1;
  __sync_synchronize();
  if (host_loop_thread_.joinable())
    host_loop_thread_.join();
  cudaq_dispatcher_stop(device_dispatcher_);

  uint64_t dev_count = 0;
  CUDA_CHECK(cudaMemcpy(&dev_count, device_loop_stats_, sizeof(uint64_t),
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(dev_count, 2u);
  EXPECT_EQ(host_loop_stats_, 2u);

  cudaq_dispatcher_destroy(device_dispatcher_);
  device_dispatcher_ = nullptr;
}

//==============================================================================
// Single-thread unified loop (cudaq_host_unified_loop) over a fake ring
// data-plane.  The fake stages test-written RX slots on rx_poll and records
// published TX slots on tx_publish.
//==============================================================================

class FakeRing {
public:
  FakeRing(std::size_t num_slots, std::size_t slot_size, bool in_place)
      : num_slots_(num_slots), slot_size_(slot_size) {
    ok_ = allocate_host_ring_buffer(num_slots, slot_size, &rx_flags_host_,
                                    &rx_data_host_) &&
          allocate_host_ring_buffer(num_slots, slot_size, &tx_flags_host_,
                                    &tx_data_host_);
    // std::atomic<int> is neither copyable nor movable, so size the vector via
    // default-insertion (not assign) and zero each element explicitly.
    published_ = std::vector<std::atomic<int>>(num_slots);
    for (auto& p : published_)
      p.store(0, std::memory_order_relaxed);
    // Host views only; the device views stay null (no GPU consumer).
    rb_.rx_stride_sz = slot_size_;
    rb_.tx_stride_sz = slot_size_;
    rb_.rx_flags_host = rx_flags_host_;
    rb_.tx_flags_host = tx_flags_host_;
    rb_.rx_data_host = rx_data_host_;
    rb_.tx_data_host = in_place ? rx_data_host_ : tx_data_host_;
  }
  ~FakeRing() {
    free_host_ring_buffer(rx_flags_host_, rx_data_host_);
    free_host_ring_buffer(tx_flags_host_, tx_data_host_);
  }

  bool ok() const { return ok_; }
  const cudaq_ringbuffer_t& ring() const { return rb_; }
  std::uint8_t* tx_host(std::size_t slot) {
    return reinterpret_cast<std::uint8_t*>(rb_.tx_data_host) +
           slot * slot_size_;
  }

  // Producer side (test): enqueue an already-written RX slot for delivery.
  // Single producer.  The slot must be stored before tail_ advances: rx_poll
  // reads pending_[head] as soon as it sees head != tail.
  void enqueue(std::uint32_t slot) {
    const std::uint32_t t = tail_.load(std::memory_order_relaxed);
    pending_[t % kCap] = slot;
    tail_.store(t + 1, std::memory_order_release);
  }
  bool published(std::uint32_t slot) {
    return published_[slot].load(std::memory_order_acquire) != 0;
  }

  cudaq_cpu_dataplane_t dataplane() {
    cudaq_cpu_dataplane_t dp{};
    dp.ctx = this;
    dp.ring = rb_;
    dp.rx_poll = &FakeRing::rx_poll;
    dp.tx_publish = &FakeRing::tx_publish;
    return dp;
  }

private:
  static cudaq_rx_status_t rx_poll(void* ctx, std::uint32_t* out_slot) {
    auto* f = static_cast<FakeRing*>(ctx);
    std::uint32_t h = f->head_.load(std::memory_order_acquire);
    if (h == f->tail_.load(std::memory_order_acquire))
      return CUDAQ_RX_EMPTY;
    *out_slot = f->pending_[h % kCap];
    f->head_.store(h + 1, std::memory_order_release);
    return CUDAQ_RX_READY;
  }
  static cudaq_status_t tx_publish(void* ctx, std::uint32_t slot) {
    auto* f = static_cast<FakeRing*>(ctx);
    f->published_[slot].store(1, std::memory_order_release);
    return CUDAQ_OK;
  }

  static constexpr std::size_t kCap = 64;
  std::size_t num_slots_;
  std::size_t slot_size_;
  bool ok_ = false;
  cudaq_ringbuffer_t rb_{};
  volatile uint64_t* rx_flags_host_ = nullptr;
  volatile uint64_t* tx_flags_host_ = nullptr;
  std::uint8_t* rx_data_host_ = nullptr;
  std::uint8_t* tx_data_host_ = nullptr;
  std::atomic<std::uint32_t> head_{0};
  std::atomic<std::uint32_t> tail_{0};
  std::uint32_t pending_[kCap]{};
  std::vector<std::atomic<int>> published_;
};

TEST(UnifiedLoopTest, HostCallOverRingDataplane) {
  constexpr std::size_t num_slots = 4;
  constexpr std::size_t slot_size = 256;
  FakeRing fake(num_slots, slot_size, /*in_place=*/false);
  ASSERT_TRUE(fake.ok());

  cudaq_function_entry_t entry{};
  entry.handler.host_fn = &host_increment;
  entry.function_id = HOST_INCREMENT_FUNCTION_ID;
  entry.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

  cudaq_cpu_dataplane_t dp = fake.dataplane();
  volatile int shutdown_flag = 0;
  uint64_t stats = 0;
  cudaq_function_table_t table{&entry, 1};
  std::thread loop([&] {
    cudaq_host_unified_loop(&dp, &table, &shutdown_flag, &stats);
  });

  const std::uint8_t payload[] = {10, 20, 30, 40};
  ASSERT_EQ(cudaq_host_ringbuffer_write_rpc_request(
                &fake.ring(), 1, HOST_INCREMENT_FUNCTION_ID, payload, 4, 7,
                0),
            CUDAQ_OK);
  fake.enqueue(1);

  for (int i = 0; i < 5000 && !fake.published(1); ++i)
    usleep(200);
  ASSERT_TRUE(fake.published(1)) << "HOST_CALL response not published";

  auto* resp = reinterpret_cast<RPCResponse*>(fake.tx_host(1));
  EXPECT_EQ(resp->magic, CUDAQ_RPC_MAGIC_RESPONSE);
  EXPECT_EQ(resp->result_len, 4u);
  EXPECT_EQ(resp->request_id, 7u);
  std::uint8_t* result = fake.tx_host(1) + sizeof(RPCResponse);
  EXPECT_EQ(result[0], 11);
  EXPECT_EQ(result[1], 21);
  EXPECT_EQ(result[2], 31);
  EXPECT_EQ(result[3], 41);

  shutdown_flag = 1;
  __sync_synchronize();
  loop.join();
  EXPECT_EQ(stats, 1u);
}

// The unified loop validates RPC_MAGIC_REQUEST before resolving/dispatching a
// slot (parity with the ring loop).  A slot with bad framing must be dropped:
// no dispatch, no tx_publish, not counted.
TEST(UnifiedLoopTest, DropsBadMagicOverRingDataplane) {
  constexpr std::size_t num_slots = 4;
  constexpr std::size_t slot_size = 256;
  FakeRing fake(num_slots, slot_size, /*in_place=*/false);
  ASSERT_TRUE(fake.ok());

  cudaq_function_entry_t entry{};
  entry.handler.host_fn = &host_increment;
  entry.function_id = HOST_INCREMENT_FUNCTION_ID;
  entry.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

  cudaq_cpu_dataplane_t dp = fake.dataplane();
  volatile int shutdown_flag = 0;
  uint64_t stats = 0;
  cudaq_function_table_t table{&entry, 1};
  std::thread loop([&] {
    cudaq_host_unified_loop(&dp, &table, &shutdown_flag, &stats);
  });

  const std::uint8_t payload[] = {10, 20, 30, 40};

  // Slot 1: a well-formed request whose magic we then corrupt.
  ASSERT_EQ(cudaq_host_ringbuffer_write_rpc_request(
                &fake.ring(), 1, HOST_INCREMENT_FUNCTION_ID, payload, 4, 7,
                0),
            CUDAQ_OK);
  auto* bad = reinterpret_cast<RPCHeader*>(
      reinterpret_cast<std::uint8_t*>(fake.ring().rx_data_host) +
      1 * slot_size);
  bad->magic = 0; // bust the RPC_MAGIC_REQUEST framing
  fake.enqueue(1);

  // Slot 2: a valid request used as a fence.  rx_poll is FIFO and the loop is
  // single-threaded, so once slot 2 is published slot 1 has already been
  // processed (and, correctly, dropped).
  ASSERT_EQ(cudaq_host_ringbuffer_write_rpc_request(
                &fake.ring(), 2, HOST_INCREMENT_FUNCTION_ID, payload, 4, 8,
                0),
            CUDAQ_OK);
  fake.enqueue(2);

  for (int i = 0; i < 5000 && !fake.published(2); ++i)
    usleep(200);
  ASSERT_TRUE(fake.published(2)) << "fence request not published";
  EXPECT_FALSE(fake.published(1)) << "bad-magic slot must be dropped";

  shutdown_flag = 1;
  __sync_synchronize();
  loop.join();
  EXPECT_EQ(stats, 1u); // only the fence request was dispatched
}

} // namespace
