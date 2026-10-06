/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Pure-HOST_CALL device_call host dispatch must work without a visible CUDA
// device: the handler runs inline on the host dispatcher thread and nothing
// reads the ring buffer from a device. This binary hides every device from
// itself (CUDA_VISIBLE_DEVICES="" in main(), before the CUDA runtime
// initializes) so it exercises the no-device path on any machine, and it is
// registered without the `gpu_required` ctest label so it also runs on
// GPU-less CI runners.

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/device_call_service.h"
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>

namespace cudaq_internal::device_call {
void initializeDeviceCallRuntime(int argc, char **argv);
void finalizeDeviceCallRuntime();
} // namespace cudaq_internal::device_call

extern "C" std::int32_t __cudaq_device_call_acquire_realtime_frame(
    std::uint32_t deviceId, std::uint32_t functionId,
    std::uint64_t requestBytes, std::uint64_t responseCapacity,
    void **frameHandle, void **requestPayload, void **responsePayload);
extern "C" std::int32_t
__cudaq_device_call_dispatch_realtime_frame(void *frameHandle,
                                            std::uint64_t *responseBytes);
extern "C" void
__cudaq_device_call_safely_release_realtime_frame(void *frameHandle);

namespace {

using cudaq::realtime::DeviceCallDispatchMode;
using cudaq::realtime::DeviceCallDispatchTable;
using cudaq::realtime::DeviceCallService;
using cudaq::realtime::DeviceCallServicePluginInfo;
using cudaq::realtime::DeviceCallServiceSession;

constexpr std::uint32_t AddThemFunctionId =
    cudaq::realtime::fnv1a_hash("addThem");

void addThemHandler(const void *rxSlot, void *txSlot, std::size_t slotSize) {
  if (!rxSlot || !txSlot || slotSize < sizeof(cudaq::realtime::RPCResponse))
    return;

  const auto *const request =
      static_cast<const cudaq::realtime::RPCHeader *>(rxSlot);
  auto *const response = static_cast<cudaq::realtime::RPCResponse *>(txSlot);
  auto *const result =
      reinterpret_cast<std::int32_t *>(static_cast<std::uint8_t *>(txSlot) +
                                       sizeof(cudaq::realtime::RPCResponse));

  std::int32_t status = 0;
  std::uint32_t resultLen = sizeof(std::int32_t);
  if (request->magic != cudaq::realtime::RPC_MAGIC_REQUEST ||
      request->arg_len != 2 * sizeof(std::int32_t) ||
      slotSize < sizeof(cudaq::realtime::RPCResponse) + sizeof(std::int32_t)) {
    status = 103;
    resultLen = 0;
  } else {
    const auto *const args =
        reinterpret_cast<const std::int32_t *>(request + 1);
    *result = args[0] + args[1];
  }

  response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
  response->request_id = request->request_id;
  response->ptp_timestamp = request->ptp_timestamp;
  response->status = status;
  response->result_len = resultLen;
}

// A session exporting a single HOST_CALL entry. Deliberately makes no CUDA
// runtime calls: HOST_CALL tables need no graph, mailbox, or device memory.
class HostOnlySession : public DeviceCallServiceSession {
public:
  HostOnlySession() {
    entry = {};
    entry.handler.host_fn = addThemHandler;
    entry.function_id = AddThemFunctionId;
    entry.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
    entry.schema.num_args = 2;
    entry.schema.num_results = 1;
    entry.schema.args[0].type_id = CUDAQ_TYPE_INT32;
    entry.schema.args[0].size_bytes = sizeof(std::int32_t);
    entry.schema.args[0].num_elements = 1;
    entry.schema.args[1] = entry.schema.args[0];
    entry.schema.results[0] = entry.schema.args[0];

    table.mode = DeviceCallDispatchMode::Host;
    table.entries = &entry;
    table.count = 1;
    table.deviceId = 0;
  }

  const DeviceCallDispatchTable &dispatchTable() const noexcept override {
    return table;
  }

private:
  cudaq_function_entry_t entry{};
  DeviceCallDispatchTable table;
};

class HostOnlyService : public DeviceCallService {
public:
  std::unique_ptr<DeviceCallServiceSession>
  createDispatchSession(DeviceCallDispatchMode mode) override {
    if (mode != DeviceCallDispatchMode::Host)
      return nullptr;
    return std::make_unique<HostOnlySession>();
  }
};

DeviceCallService *getHostOnlyService() {
  static HostOnlyService service;
  return &service;
}

class HostDispatchNoGpuTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Guard against the environment override in main() being defeated (e.g. a
    // CUDA context created before it took effect). The whole point of this
    // test is exercising dispatch with no usable device.
    int deviceCount = 0;
    const cudaError_t err = cudaGetDeviceCount(&deviceCount);
    ASSERT_TRUE(err != cudaSuccess || deviceCount == 0)
        << "expected no visible CUDA device, found " << deviceCount;

    char program[] = "test_host_dispatch_no_gpu";
    char option[] = "--cudaq-device-call=host-dispatch";
    char *argv[] = {program, option};
    ASSERT_NO_THROW(
        cudaq_internal::device_call::initializeDeviceCallRuntime(2, argv));
  }

  void TearDown() override {
    ASSERT_NO_THROW(cudaq_internal::device_call::finalizeDeviceCallRuntime());
  }
};

TEST_F(HostDispatchNoGpuTest, DispatchesHostCallWithoutCudaDevice) {
  void *frame = nullptr;
  void *requestPayload = nullptr;
  void *responsePayload = nullptr;
  ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                   0, AddThemFunctionId, 2 * sizeof(std::int32_t),
                   sizeof(std::int32_t), &frame, &requestPayload,
                   &responsePayload));
  ASSERT_NE(nullptr, frame);
  ASSERT_NE(nullptr, requestPayload);
  ASSERT_NE(nullptr, responsePayload);

  auto *const args = static_cast<std::int32_t *>(requestPayload);
  args[0] = 19;
  args[1] = 23;

  std::uint64_t responseLen = 0;
  EXPECT_EQ(0,
            __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
  EXPECT_EQ(sizeof(std::int32_t), responseLen);
  EXPECT_EQ(42, *static_cast<std::int32_t *>(responsePayload));

  __cudaq_device_call_safely_release_realtime_frame(frame);
}

TEST_F(HostDispatchNoGpuTest, ReusesRingSlotsAcrossDispatches) {
  // More dispatches than ring slots, so slot recycling also runs on the
  // plain-host-memory fallback storage.
  for (int i = 0; i < 8; ++i) {
    std::array<std::int32_t, 2> request{i, 100 + i};

    void *frame = nullptr;
    void *requestPayload = nullptr;
    void *responsePayload = nullptr;
    ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                     0, AddThemFunctionId, 2 * sizeof(std::int32_t),
                     sizeof(std::int32_t), &frame, &requestPayload,
                     &responsePayload));
    std::memcpy(requestPayload, request.data(), sizeof(request));

    std::uint64_t responseLen = 0;
    EXPECT_EQ(0,
              __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
    EXPECT_EQ(sizeof(std::int32_t), responseLen);
    EXPECT_EQ(100 + 2 * i, *static_cast<std::int32_t *>(responsePayload));

    __cudaq_device_call_safely_release_realtime_frame(frame);
  }
}

} // namespace

// Discovered via dlsym(RTLD_DEFAULT, ...); requires ENABLE_EXPORTS on the
// test executable.
extern "C" cudaq::realtime::DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo() {
  return {"host-only-test", &getHostOnlyService};
}

int main(int argc, char **argv) {
  // Hide every CUDA device before the CUDA runtime can initialize, so the
  // no-device path is exercised even on machines that have GPUs. On a
  // GPU-less runner this is a no-op.
  setenv("CUDA_VISIBLE_DEVICES", "", /*overwrite=*/1);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
