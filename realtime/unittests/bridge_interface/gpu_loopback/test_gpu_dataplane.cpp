/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_gpu_dataplane.cpp
/// @brief A DEVICE unified dispatcher over the gpu_loopback provider: the
/// dispatcher's embedded kernel, the provider's hooks and this binary's
/// handler, linked at run time.  The test plays the wire through the
/// provider's host views of its rings.  Skips without a GPU.

#include "driver_launch.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/device_code/device_code_reader.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <string>
#include <thread>
#include <vector>

#ifndef GPU_LOOPBACK_BRIDGE_PATH
#error "GPU_LOOPBACK_BRIDGE_PATH must name the built gpu_loopback provider"
#endif

// test_gpu_dataplane_device.cu
extern "C" void *gpu_dataplane_test_own_module_handler();
// test_gpu_dataplane_second_handler.cu
extern "C" cudaq_status_t
gpu_dataplane_test_lookup_second_handler(cudaq_dispatcher_t *dispatcher,
                                         void **out_fn);
// no_device_code.cpp, a library built without -rdc.
extern "C" cudaq_status_t
gpu_dataplane_test_no_device_code(cudaq_device_image_t *image);

namespace {

using cudaq::realtime::RPCResponse;

constexpr std::uint32_t kFunctionId =
    cudaq::realtime::fnv1a_hash("gpu_dataplane_test");
constexpr const char *kHandler = "gpu_dataplane_test_handler";
// unittests/dispatchers/single_thread_dispatcher.cu
constexpr const char *kDispatcher = "single_thread_dispatcher";

cudaq_dispatcher_config_t make_config(cudaq_dispatch_path_t path,
                                      cudaq_kernel_type_t kernel_type,
                                      std::uint32_t num_blocks) {
  cudaq_dispatcher_config_t config{};
  config.dispatch_path = path;
  config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  config.kernel_type = kernel_type;
  config.num_blocks = num_blocks;
  config.threads_per_block = 32;
  return config;
}

// The application's launcher.  Its type carries no user data, so the kernel
// it starts, looked up in the dispatcher's linked module, and the grid come
// from here.  One thread per block: the kernel works on the grid's first
// thread, where the block-partitioned handler then does all its work.
cudaKernel_t own_dispatcher_kernel = nullptr;
unsigned own_dispatcher_blocks = 1;
bool own_dispatcher_cooperative = false;

// Through the driver API, which takes the kernel on CUDA 12 and 13 alike.
cudaq_status_t launch_own_dispatcher(void *transport_ctx, size_t tx_stride_sz,
                                     cudaq_function_entry_t *function_table,
                                     size_t func_count,
                                     volatile int *shutdown_flag,
                                     uint64_t *stats, uint32_t,
                                     cudaStream_t stream) {
  void *args[] = {&transport_ctx,  &tx_stride_sz, &shutdown_flag,
                  &function_table, &func_count,   &stats};
  CUlaunchAttribute cooperative{};
  cooperative.id = CU_LAUNCH_ATTRIBUTE_COOPERATIVE;
  cooperative.value.cooperative = 1;
  CUlaunchConfig config{};
  config.gridDimX = own_dispatcher_blocks;
  config.gridDimY = config.gridDimZ = 1;
  config.blockDimX = config.blockDimY = config.blockDimZ = 1;
  config.hStream = stream;
  if (own_dispatcher_cooperative) {
    config.attrs = &cooperative;
    config.numAttrs = 1;
  }
  return cudaq::realtime::testing::launch_kernel_ex(
      &config, own_dispatcher_kernel, args);
}

// Recorded on the launcher's stream after the kernel by launch_then_fail.
cudaEvent_t own_dispatcher_done = nullptr;

// Enqueues the application's kernel, then fails anyway.
cudaq_status_t launch_then_fail(void *transport_ctx, size_t tx_stride_sz,
                                cudaq_function_entry_t *function_table,
                                size_t func_count, volatile int *shutdown_flag,
                                uint64_t *stats, uint32_t threads_per_block,
                                cudaStream_t stream) {
  if (launch_own_dispatcher(transport_ctx, tx_stride_sz, function_table,
                            func_count, shutdown_flag, stats, threads_per_block,
                            stream) != CUDAQ_OK ||
      cudaEventRecord(own_dispatcher_done, stream) != cudaSuccess)
    return CUDAQ_ERR_CUDA;
  return CUDAQ_ERR_INTERNAL;
}

class GpuDataplaneTest : public ::testing::Test {
protected:
  void SetUp() override {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0)
      GTEST_SKIP() << "no CUDA device";
#if CUDART_VERSION < 12080
    // The run-time link uses this toolkit's nvJitLink, which only targets
    // sm_100 and newer from CUDA 12.8.
    int major = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0) ==
            cudaSuccess &&
        major >= 10)
      GTEST_SKIP() << "nvJitLink " << CUDART_VERSION / 1000 << "."
                   << CUDART_VERSION % 1000 / 10 << " cannot target sm_"
                   << major << "x";
#endif
    ASSERT_EQ(cudaq_dispatch_manager_create(&manager), CUDAQ_OK);
    ASSERT_EQ(cudaMalloc(&d_entries, sizeof(cudaq_function_entry_t)),
              cudaSuccess);
  }

  void TearDown() override {
    for (auto *dispatcher : dispatchers) {
      cudaq_dispatcher_stop(dispatcher);
      cudaq_dispatcher_destroy(dispatcher);
    }
    for (auto bridge : bridges)
      EXPECT_EQ(cudaq_bridge_destroy(bridge), CUDAQ_OK);
    if (manager)
      cudaq_dispatch_manager_destroy(manager);
    cudaFree(d_entries);
    // The set_control override's words, when a test allocated them: freed
    // after the dispatcher that polled them has been stopped.
    if (override_stats)
      cudaFree(override_stats);
    if (override_shutdown)
      cudaFree(override_shutdown);
  }

  cudaq_realtime_bridge_handle_t bridge(std::vector<std::string> args = {}) {
    std::vector<char *> argv;
    for (auto &arg : args)
      argv.push_back(arg.data());
    cudaq_realtime_bridge_handle_t b = nullptr;
    EXPECT_EQ(cudaq_bridge_create_from_library(&b, GPU_LOOPBACK_BRIDGE_PATH,
                                               static_cast<int>(argv.size()),
                                               argv.data()),
              CUDAQ_OK);
    if (b)
      bridges.push_back(b);
    return b;
  }

  cudaq_dispatcher_t *
  dispatcher(cudaq_dispatch_path_t path = CUDAQ_DISPATCH_PATH_DEVICE,
             cudaq_kernel_type_t kernel_type = CUDAQ_KERNEL_UNIFIED,
             std::uint32_t num_blocks = 1) {
    const auto config = make_config(path, kernel_type, num_blocks);
    cudaq_dispatcher_t *d = nullptr;
    EXPECT_EQ(cudaq_dispatcher_create(manager, &config, &d), CUDAQ_OK);
    if (d)
      dispatchers.push_back(d);
    return d;
  }

  // One DEVICE_CALL entry for `fn`.
  void install(cudaq_dispatcher_t *d, void *fn) {
    cudaq_function_entry_t entry;
    std::memset(&entry, 0, sizeof(entry));
    entry.handler.device_fn_ptr = fn;
    entry.function_id = kFunctionId;
    entry.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    ASSERT_EQ(
        cudaMemcpy(d_entries, &entry, sizeof(entry), cudaMemcpyHostToDevice),
        cudaSuccess);
    cudaq_function_table_t table{d_entries, 1};
    ASSERT_EQ(cudaq_dispatcher_set_function_table(d, &table), CUDAQ_OK);
  }

  // The deprecated override: device words of this fixture's in place of the
  // dispatcher's own.
  void override_control(cudaq_dispatcher_t *d) {
    ASSERT_EQ(cudaMalloc(&override_shutdown, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMemset(override_shutdown, 0, sizeof(int)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&override_stats, sizeof(std::uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(override_stats, 0, sizeof(std::uint64_t)),
              cudaSuccess);
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    ASSERT_EQ(
        cudaq_dispatcher_set_control(d, override_shutdown, override_stats),
        CUDAQ_OK);
#pragma GCC diagnostic pop
  }

  // Plays the wire for started dispatcher `d` over bridge `b`, then stops it
  // and checks that every request was counted.
  void round_trip(cudaq_realtime_bridge_handle_t b, cudaq_dispatcher_t *d) {
    cudaq_ringbuffer_t ring{};
    ASSERT_EQ(cudaq_bridge_get_transport_context(b, RING_BUFFER, &ring),
              CUDAQ_OK);
    std::uint32_t num_slots = 0, slot_size = 0;
    ASSERT_EQ(cudaq_bridge_get_ring_geometry(b, &num_slots, &slot_size),
              CUDAQ_OK);

    // Two and a half times around the ring.
    const std::uint32_t requests = num_slots * 5 / 2;
    for (std::uint32_t i = 0; i < requests; ++i) {
      const std::uint32_t slot = i % num_slots;
      const std::uint8_t payload[4] = {
          static_cast<std::uint8_t>(i), static_cast<std::uint8_t>(i + 1),
          static_cast<std::uint8_t>(i + 2), static_cast<std::uint8_t>(i + 3)};
      ASSERT_EQ(
          cudaq_host_ringbuffer_write_rpc_request(
              &ring, slot, kFunctionId, payload, sizeof(payload), 100 + i, 0),
          CUDAQ_OK);
      __atomic_store_n(const_cast<std::uint64_t *>(&ring.rx_flags_host[slot]),
                       1, __ATOMIC_RELEASE);

      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::seconds(5);
      while (__atomic_load_n(
                 const_cast<std::uint64_t *>(&ring.tx_flags_host[slot]),
                 __ATOMIC_ACQUIRE) == 0) {
        ASSERT_LT(std::chrono::steady_clock::now(), deadline)
            << "no response for request " << i;
        std::this_thread::yield();
      }
      const auto *response = reinterpret_cast<const RPCResponse *>(
          ring.tx_data_host + slot * ring.tx_stride_sz);
      EXPECT_EQ(response->magic, cudaq::realtime::RPC_MAGIC_RESPONSE);
      EXPECT_EQ(response->status, 0);
      EXPECT_EQ(response->request_id, 100 + i);
      ASSERT_EQ(response->result_len, sizeof(payload));
      const auto *result = reinterpret_cast<const std::uint8_t *>(response + 1);
      for (std::size_t k = 0; k < sizeof(payload); ++k)
        EXPECT_EQ(result[k], static_cast<std::uint8_t>(payload[k] + 1));
      __atomic_store_n(const_cast<std::uint64_t *>(&ring.tx_flags_host[slot]),
                       0, __ATOMIC_RELEASE);
    }

    ASSERT_EQ(cudaq_dispatcher_stop(d), CUDAQ_OK);
    std::uint64_t processed = 0;
    ASSERT_EQ(cudaq_dispatcher_get_processed(d, &processed), CUDAQ_OK);
    EXPECT_EQ(processed, requests);
  }

  // Wires `d` to `b` with this binary's handler, and has the application's
  // launcher run this binary's own dispatch kernel.
  void use_own_dispatcher(cudaq_realtime_bridge_handle_t b,
                          cudaq_dispatcher_t *d, unsigned blocks,
                          bool cooperative) {
    ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
    void *fn = nullptr;
    ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn), CUDAQ_OK);
    install(d, fn);
    ASSERT_EQ(cudaq_dispatcher_get_device_kernel(d, kDispatcher,
                                                 &own_dispatcher_kernel),
              CUDAQ_OK);
    own_dispatcher_blocks = blocks;
    own_dispatcher_cooperative = cooperative;
    ASSERT_EQ(
        cudaq_dispatcher_set_unified_launch(d, launch_own_dispatcher, nullptr),
        CUDAQ_OK);
  }

  cudaq_dispatch_manager_t *manager = nullptr;
  cudaq_function_entry_t *d_entries = nullptr;
  int *override_shutdown = nullptr;
  std::uint64_t *override_stats = nullptr;
  std::vector<cudaq_dispatcher_t *> dispatchers;
  std::vector<cudaq_realtime_bridge_handle_t> bridges;
};

TEST_F(GpuDataplaneTest, RoundTripWrapsTheRing) {
  auto b = bridge();
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);

  void *fn = nullptr;
  ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn), CUDAQ_OK);
  install(d, fn);
  ASSERT_EQ(cudaq_dispatcher_start(d), CUDAQ_OK);
  round_trip(b, d);
}

// Nothing has run before start(), so the count is zero rather than an error.
TEST_F(GpuDataplaneTest, GetProcessedIsZeroBeforeStart) {
  auto *d = dispatcher();
  std::uint64_t processed = 42;
  ASSERT_EQ(cudaq_dispatcher_get_processed(d, &processed), CUDAQ_OK);
  EXPECT_EQ(processed, 0u);
}

// stop() ends the kernel on its own (round_trip raises no flag); a second
// stop() is a no-op, and the count survives it.
TEST_F(GpuDataplaneTest, StopIsIdempotentAndKeepsTheCount) {
  auto b = bridge();
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  void *fn = nullptr;
  ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn), CUDAQ_OK);
  install(d, fn);
  ASSERT_EQ(cudaq_dispatcher_start(d), CUDAQ_OK);
  round_trip(b, d);
  std::uint64_t before = 0;
  ASSERT_EQ(cudaq_dispatcher_get_processed(d, &before), CUDAQ_OK);
  ASSERT_EQ(cudaq_dispatcher_stop(d), CUDAQ_OK);
  std::uint64_t after = 0;
  ASSERT_EQ(cudaq_dispatcher_get_processed(d, &after), CUDAQ_OK);
  EXPECT_GT(after, 0u);
  EXPECT_EQ(after, before);
}

// Each start() zeroes the dispatcher's own words: started again after stop(),
// it reports only the second run's requests.
TEST_F(GpuDataplaneTest, RestartZeroesTheCount) {
  auto b = bridge();
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  void *fn = nullptr;
  ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn), CUDAQ_OK);
  install(d, fn);
  ASSERT_EQ(cudaq_dispatcher_start(d), CUDAQ_OK);
  round_trip(b, d);
  std::uint64_t first = 0;
  ASSERT_EQ(cudaq_dispatcher_get_processed(d, &first), CUDAQ_OK);
  ASSERT_EQ(cudaq_dispatcher_start(d), CUDAQ_OK);
  round_trip(b, d);
  std::uint64_t second = 0;
  ASSERT_EQ(cudaq_dispatcher_get_processed(d, &second), CUDAQ_OK);
  EXPECT_EQ(second, first);
}

// The deprecated override still routes the kernel's flag and count to the
// caller's device words: stop() raises that flag, and the count lands there.
TEST_F(GpuDataplaneTest, SetControlStillOverridesTheWords) {
  auto b = bridge();
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  void *fn = nullptr;
  ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn), CUDAQ_OK);
  install(d, fn);
  override_control(d);
  ASSERT_EQ(cudaq_dispatcher_start(d), CUDAQ_OK);
  round_trip(b, d);
  std::uint64_t processed = 0, counted = 0;
  int flag = 0;
  ASSERT_EQ(cudaq_dispatcher_get_processed(d, &processed), CUDAQ_OK);
  ASSERT_EQ(cudaMemcpy(&counted, override_stats, sizeof(counted),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&flag, override_shutdown, sizeof(flag),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  EXPECT_GT(counted, 0u);
  EXPECT_EQ(counted, processed);
  EXPECT_EQ(flag, 1);
}

// The application's own kernel, from this binary's device code, started by
// its own launcher: one thread receives, dispatches and publishes.
TEST_F(GpuDataplaneTest, AnApplicationDispatcherRoundTripsAcrossTheRing) {
  auto b = bridge();
  auto *d = dispatcher();
  use_own_dispatcher(b, d, 1, false);
  ASSERT_EQ(cudaq_dispatcher_start(d), CUDAQ_OK);
  round_trip(b, d);
}

// The launcher owns the grid: the dispatcher's own kernel needs one block,
// a launcher may start a cooperative grid of two.
TEST_F(GpuDataplaneTest, AnApplicationLauncherPicksItsOwnGrid) {
  auto b = bridge();
  auto *d = dispatcher(CUDAQ_DISPATCH_PATH_DEVICE, CUDAQ_KERNEL_UNIFIED, 2);
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  void *fn = nullptr;
  ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn), CUDAQ_OK);
  install(d, fn);
  EXPECT_EQ(cudaq_dispatcher_start(d), CUDAQ_ERR_INVALID_ARG);

  ASSERT_EQ(cudaq_dispatcher_get_device_kernel(d, kDispatcher,
                                               &own_dispatcher_kernel),
            CUDAQ_OK);
  own_dispatcher_blocks = 2;
  own_dispatcher_cooperative = true;
  ASSERT_EQ(
      cudaq_dispatcher_set_unified_launch(d, launch_own_dispatcher, nullptr),
      CUDAQ_OK);
  ASSERT_EQ(cudaq_dispatcher_start(d), CUDAQ_OK);
  round_trip(b, d);
}

// A launch the driver refuses fails start, though the launcher returns
// nothing.
TEST_F(GpuDataplaneTest, StartFailsWhenTheApplicationLaunchFails) {
  auto b = bridge();
  auto *d = dispatcher();
  // Far more cooperative blocks than can be resident at once.
  use_own_dispatcher(b, d, 1u << 20, true);
  EXPECT_EQ(cudaq_dispatcher_start(d), CUDAQ_ERR_CUDA);
}

// A launcher that fails after enqueueing its kernel: start stops the kernel
// before returning, so destroy() never unloads the module under it.
TEST_F(GpuDataplaneTest, StartStopsAKernelItsLauncherEnqueuedBeforeFailing) {
  auto b = bridge();
  auto *d = dispatcher();
  use_own_dispatcher(b, d, 1, false);
  ASSERT_EQ(
      cudaEventCreateWithFlags(&own_dispatcher_done, cudaEventDisableTiming),
      cudaSuccess);
  ASSERT_EQ(cudaq_dispatcher_set_unified_launch(d, launch_then_fail, nullptr),
            CUDAQ_OK);
  EXPECT_EQ(cudaq_dispatcher_start(d), CUDAQ_ERR_INTERNAL);
  // The kernel, and the marker behind it, finished before start returned.
  EXPECT_EQ(cudaEventQuery(own_dispatcher_done), cudaSuccess);
  cudaEventDestroy(own_dispatcher_done);
  own_dispatcher_done = nullptr;
}

// The application's launcher does not lift the guard on handler addresses.
TEST_F(GpuDataplaneTest, AnApplicationLauncherStillNeedsLinkedHandlers) {
  auto b = bridge();
  auto *d = dispatcher();
  use_own_dispatcher(b, d, 1, false);
  install(d, gpu_dataplane_test_own_module_handler());
  EXPECT_EQ(cudaq_dispatcher_start(d), CUDAQ_ERR_INVALID_ARG);
}

TEST_F(GpuDataplaneTest, GetDeviceKernelRefusesUnknownNamesAndTheBuiltInOne) {
  auto b = bridge();
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  cudaKernel_t kernel = nullptr;
  EXPECT_EQ(cudaq_dispatcher_get_device_kernel(d, "no_such_kernel", &kernel),
            CUDAQ_ERR_INVALID_ARG);
  EXPECT_EQ(cudaq_dispatcher_get_device_kernel(
                d, "cudaq_unified_dispatch_kernel", &kernel),
            CUDAQ_ERR_INVALID_ARG);
  EXPECT_EQ(cudaq_dispatcher_get_device_kernel(d, kDispatcher, &kernel),
            CUDAQ_OK);
  EXPECT_NE(kernel, nullptr);
}

// Over a GPU data plane, the bridge's context is the one the launcher gets,
// and the launcher is set before start.
TEST_F(GpuDataplaneTest, SetUnifiedLaunchKeepsTheBridgesContext) {
  auto b = bridge();
  auto *d = dispatcher();
  use_own_dispatcher(b, d, 1, false);
  int other_context = 0;
  EXPECT_EQ(cudaq_dispatcher_set_unified_launch(d, launch_own_dispatcher,
                                                &other_context),
            CUDAQ_ERR_INVALID_ARG);

  ASSERT_EQ(cudaq_dispatcher_start(d), CUDAQ_OK);
  EXPECT_EQ(
      cudaq_dispatcher_set_unified_launch(d, launch_own_dispatcher, nullptr),
      CUDAQ_ERR_INVALID_ARG);
}

// An address taken in this binary's own module points at code the linked
// module does not hold.
TEST_F(GpuDataplaneTest, StartRefusesAHandlerFromAnotherModule) {
  auto b = bridge();
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  void *own = gpu_dataplane_test_own_module_handler();
  ASSERT_NE(own, nullptr);
  void *linked = nullptr;
  ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &linked),
            CUDAQ_OK);
  ASSERT_NE(own, linked);
  install(d, own);
  EXPECT_EQ(cudaq_dispatcher_start(d), CUDAQ_ERR_INVALID_ARG);
}

TEST_F(GpuDataplaneTest, EnforcesSetBridgeThenLookupThenStart) {
  auto b = bridge();
  auto *d = dispatcher();
  // Nothing to link with before the bridge has handed over its hooks.
  void *fn = nullptr;
  EXPECT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn),
            CUDAQ_ERR_INVALID_ARG);

  // Without a lookup, start() refuses a handler address it did not hand out,
  // before it links anything.
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  install(d, reinterpret_cast<void *>(0x1));
  EXPECT_EQ(cudaq_dispatcher_start(d), CUDAQ_ERR_INVALID_ARG);

  // So the lookup still links this binary's code, and start() then succeeds.
  ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn), CUDAQ_OK);
  install(d, fn);
  EXPECT_EQ(cudaq_dispatcher_start(d), CUDAQ_OK);
}

// The first lookup links; later ones reuse the module, from the same binary
// only.
TEST_F(GpuDataplaneTest, RefusesHandlersFromASecondBinary) {
  auto b = bridge();
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  void *first = nullptr;
  void *again = nullptr;
  ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &first), CUDAQ_OK);
  ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &again), CUDAQ_OK);
  EXPECT_EQ(first, again);

  const unsigned char other[64] = {1, 2, 3};
  const cudaq_device_image_t other_binary = {other, sizeof(other)};
  void *fn = nullptr;
  EXPECT_EQ(cudaq_dispatcher_get_device_handler_in(d, &other_binary, 1,
                                                   kHandler, &fn),
            CUDAQ_ERR_INVALID_ARG);
}

// This binary's two handlers sit in two fatbins of one section, and the
// second is looked up from its own translation unit.
TEST_F(GpuDataplaneTest, LinksEveryFatbinOfTheBinary) {
  auto b = bridge();
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  void *first = nullptr;
  void *second = nullptr;
  ASSERT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &first), CUDAQ_OK);
  ASSERT_EQ(gpu_dataplane_test_lookup_second_handler(d, &second), CUDAQ_OK);
  EXPECT_NE(first, nullptr);
  EXPECT_NE(second, nullptr);
  EXPECT_NE(first, second);
}

TEST_F(GpuDataplaneTest, RelinksAfterAFailedLink) {
  auto b = bridge();
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  const unsigned char junk[64] = {1, 2, 3};
  const cudaq_device_image_t bad = {junk, sizeof(junk)};
  void *fn = nullptr;
  EXPECT_NE(cudaq_dispatcher_get_device_handler_in(d, &bad, 1, kHandler, &fn),
            CUDAQ_OK);
  EXPECT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn), CUDAQ_OK);
}

TEST_F(GpuDataplaneTest, RefusesUnknownNames) {
  auto b = bridge();
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  void *ptr = nullptr;
  EXPECT_EQ(cudaq_dispatcher_get_device_handler(d, "no_such_handler", &ptr),
            CUDAQ_ERR_INVALID_ARG);
  // A refused name does not spoil the next lookup.
  EXPECT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &ptr), CUDAQ_OK);
}

TEST_F(GpuDataplaneTest, OnlyADeviceUnifiedDispatcherLinks) {
  void *fn = nullptr;
  EXPECT_EQ(cudaq_dispatcher_get_device_handler(
                dispatcher(CUDAQ_DISPATCH_PATH_HOST, CUDAQ_KERNEL_UNIFIED),
                kHandler, &fn),
            CUDAQ_ERR_INVALID_ARG);
  EXPECT_EQ(cudaq_dispatcher_get_device_handler(
                dispatcher(CUDAQ_DISPATCH_PATH_DEVICE, CUDAQ_KERNEL_REGULAR),
                kHandler, &fn),
            CUDAQ_ERR_INVALID_ARG);
}

TEST_F(GpuDataplaneTest, ReaderSeesNoCodeInABinaryWithoutIt) {
  cudaq_device_image_t image{};
  EXPECT_EQ(gpu_dataplane_test_no_device_code(&image), CUDAQ_ERR_UNSUPPORTED);
}

// The provider and this test each hand over their own code, never the
// other's.
TEST_F(GpuDataplaneTest, ReaderGivesEachBinaryItsOwnCode) {
  auto b = bridge();
  cudaq_gpu_dataplane_t dataplane{};
  ASSERT_EQ(cudaq_bridge_get_gpu_dataplane(b, &dataplane), CUDAQ_OK);
  cudaq_device_image_t image{};
  ASSERT_EQ(cudaq_device_code_of_this_binary(&image), CUDAQ_OK);
  ASSERT_GT(dataplane.device_code_count, 0u);
  const auto *q = static_cast<const char *>(image.data);
  for (size_t i = 0; i < dataplane.device_code_count; ++i) {
    const auto *p = static_cast<const char *>(dataplane.device_code[i].data);
    EXPECT_TRUE(p + dataplane.device_code[i].size <= q || q + image.size <= p);
  }
}

TEST_F(GpuDataplaneTest, AProviderImageThatCannotLinkFailsTheLink) {
  auto b = bridge({"--explicit-garbage"});
  auto *d = dispatcher();
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  void *fn = nullptr;
  EXPECT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn),
            CUDAQ_ERR_INTERNAL);
}

// Only CUDAQ_ERR_UNSUPPORTED means "no launch override"; a failed query must
// not fall through to the GPU data plane.
TEST_F(GpuDataplaneTest, SetBridgeRefusesAFailedOverrideQuery) {
  auto b = bridge({"--unified-context-fails"});
  auto *d = dispatcher();
  EXPECT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_ERR_INTERNAL);
}

TEST_F(GpuDataplaneTest, SetBridgeRefusesAProviderWithoutDeviceCode) {
  auto b = bridge({"--no-device-code"});
  auto *d = dispatcher();
  EXPECT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_ERR_INVALID_ARG);
}

} // namespace
