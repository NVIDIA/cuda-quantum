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

cudaq_dispatcher_config_t make_config(cudaq_dispatch_path_t path,
                                      cudaq_kernel_type_t kernel_type) {
  cudaq_dispatcher_config_t config{};
  config.dispatch_path = path;
  config.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  config.kernel_type = kernel_type;
  return config;
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
    // The kernel polls the flag, so the host must reach it while the kernel
    // runs: mapped pinned memory, written through its host view.
    void *h_flag = nullptr;
    ASSERT_EQ(cudaHostAlloc(&h_flag, sizeof(int), cudaHostAllocMapped),
              cudaSuccess);
    h_shutdown = static_cast<volatile int *>(h_flag);
    *h_shutdown = 0;
    void *d_flag = nullptr;
    ASSERT_EQ(cudaHostGetDevicePointer(&d_flag, h_flag, 0), cudaSuccess);
    d_shutdown = static_cast<volatile int *>(d_flag);
    ASSERT_EQ(cudaMalloc(&d_stats, sizeof(std::uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_stats, 0, sizeof(std::uint64_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_entries, sizeof(cudaq_function_entry_t)),
              cudaSuccess);
  }

  void TearDown() override {
    // A resident kernel only sees the flag through its host view; stop()'s
    // own write would queue behind the kernel.
    if (h_shutdown)
      *h_shutdown = 1;
    for (auto *dispatcher : dispatchers) {
      cudaq_dispatcher_stop(dispatcher);
      cudaq_dispatcher_destroy(dispatcher);
    }
    for (auto bridge : bridges)
      EXPECT_EQ(cudaq_bridge_destroy(bridge), CUDAQ_OK);
    if (manager)
      cudaq_dispatch_manager_destroy(manager);
    cudaFree(d_entries);
    cudaFree(d_stats);
    if (h_shutdown)
      cudaFreeHost(const_cast<int *>(h_shutdown));
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
             cudaq_kernel_type_t kernel_type = CUDAQ_KERNEL_UNIFIED) {
    const auto config = make_config(path, kernel_type);
    cudaq_dispatcher_t *d = nullptr;
    EXPECT_EQ(cudaq_dispatcher_create(manager, &config, &d), CUDAQ_OK);
    if (d)
      dispatchers.push_back(d);
    return d;
  }

  // One DEVICE_CALL entry for `fn`, plus the control words.
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
    ASSERT_EQ(cudaq_dispatcher_set_control(d, d_shutdown, d_stats), CUDAQ_OK);
  }

  cudaq_dispatch_manager_t *manager = nullptr;
  volatile int *h_shutdown = nullptr;
  volatile int *d_shutdown = nullptr;
  std::uint64_t *d_stats = nullptr;
  cudaq_function_entry_t *d_entries = nullptr;
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
    ASSERT_EQ(cudaq_host_ringbuffer_write_rpc_request(&ring, slot, kFunctionId,
                                                      payload, sizeof(payload),
                                                      100 + i, 0),
              CUDAQ_OK);
    __atomic_store_n(const_cast<std::uint64_t *>(&ring.rx_flags_host[slot]), 1,
                     __ATOMIC_RELEASE);

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (
        __atomic_load_n(const_cast<std::uint64_t *>(&ring.tx_flags_host[slot]),
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
    __atomic_store_n(const_cast<std::uint64_t *>(&ring.tx_flags_host[slot]), 0,
                     __ATOMIC_RELEASE);
  }

  *h_shutdown = 1;
  ASSERT_EQ(cudaq_dispatcher_stop(d), CUDAQ_OK);
  std::uint64_t processed = 0;
  ASSERT_EQ(cudaq_dispatcher_get_processed(d, &processed), CUDAQ_OK);
  EXPECT_EQ(processed, requests);
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

  // Without a lookup, start() links the kernel and the hooks alone, then
  // refuses a handler address it did not hand out.
  ASSERT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_OK);
  install(d, reinterpret_cast<void *>(0x1));
  EXPECT_EQ(cudaq_dispatcher_start(d), CUDAQ_ERR_INVALID_ARG);

  // That module holds none of this binary's code, so a lookup now is refused.
  EXPECT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &fn),
            CUDAQ_ERR_INVALID_ARG);
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
  // A failed lookup leaves no error behind for the next CUDA call.
  EXPECT_EQ(cudaq_dispatcher_get_device_handler(d, kHandler, &ptr), CUDAQ_OK);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
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

TEST_F(GpuDataplaneTest, SetBridgeRefusesAProviderWithoutDeviceCode) {
  auto b = bridge({"--no-device-code"});
  auto *d = dispatcher();
  EXPECT_EQ(cudaq_dispatcher_set_bridge(d, b), CUDAQ_ERR_INVALID_ARG);
}

} // namespace
