/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq_internal/device_call/DeviceCallService.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cudaq_internal::device_call {
void initializeDeviceCallRuntime(int argc, char **argv);
void finalizeDeviceCallRuntime();

namespace test {
bool createGraphAddThem(void **dMailbox, cudaGraph_t *graphOut,
                        cudaGraphExec_t *execOut);
void fillHostGraphAddEntry(cudaq_function_entry_t &entry,
                           cudaGraphExec_t graphExec);
int populateAddThemTable(cudaq_function_entry_t *entries, bool useOffset,
                         cudaStream_t stream);
} // namespace test
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

using namespace cudaq_internal::device_call;

constexpr std::uint32_t AddThemFunctionId =
    cudaq::realtime::fnv1a_hash("addThem");
constexpr std::uint32_t GraphAddThemFunctionId =
    cudaq::realtime::fnv1a_hash("graphAddThem");
constexpr std::uint32_t HostAddThemFunctionId =
    cudaq::realtime::fnv1a_hash("hostAddThem");
constexpr std::int32_t DeviceCallSuccessStatus =
    toAbiStatus(DeviceCallStatus::Success);
constexpr std::int32_t DeviceCallInvalidArgumentStatus =
    toAbiStatus(DeviceCallStatus::InvalidArgument);
constexpr std::int32_t DeviceCallNotInitializedStatus =
    toAbiStatus(DeviceCallStatus::NotInitialized);
constexpr std::int32_t DeviceCallResponseTooLargeStatus =
    toAbiStatus(DeviceCallStatus::ResponseTooLarge);

// A payload buffer must be non-null whenever its declared length is nonzero;
// a zero-length buffer is allowed to be null.
constexpr bool isValidBuffer(const void *buffer, std::uint64_t length) {
  return length == 0 || buffer != nullptr;
}

void hostAddThemHandler(const void *rxSlot, void *txSlot,
                        std::size_t slotSize) {
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

void fillHostCallAddEntry(cudaq_function_entry_t &entry) {
  entry = {};
  entry.handler.host_fn = hostAddThemHandler;
  entry.function_id = HostAddThemFunctionId;
  entry.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  entry.schema.num_args = 2;
  entry.schema.num_results = 1;
  entry.schema.args[0].type_id = CUDAQ_TYPE_INT32;
  entry.schema.args[0].size_bytes = sizeof(std::int32_t);
  entry.schema.args[0].num_elements = 1;
  entry.schema.args[1].type_id = CUDAQ_TYPE_INT32;
  entry.schema.args[1].size_bytes = sizeof(std::int32_t);
  entry.schema.args[1].num_elements = 1;
  entry.schema.results[0].type_id = CUDAQ_TYPE_INT32;
  entry.schema.results[0].size_bytes = sizeof(std::int32_t);
  entry.schema.results[0].num_elements = 1;
}

std::int32_t dispatchUsingFrameLease(std::uint32_t deviceId,
                                     std::uint32_t functionId,
                                     const void *request,
                                     std::uint64_t requestLen, void *response,
                                     std::uint64_t responseCapacity,
                                     std::uint64_t *responseLen) {
  if (!isValidBuffer(request, requestLen) || !responseLen ||
      !isValidBuffer(response, responseCapacity))
    return DeviceCallInvalidArgumentStatus;

  void *frame = nullptr;
  void *requestPayload = nullptr;
  void *responsePayload = nullptr;
  std::int32_t status = __cudaq_device_call_acquire_realtime_frame(
      deviceId, functionId, requestLen, responseCapacity, &frame,
      &requestPayload, &responsePayload);
  if (status != DeviceCallSuccessStatus)
    return status;
  if (!isValidBuffer(requestPayload, requestLen) ||
      !isValidBuffer(responsePayload, responseCapacity)) {
    __cudaq_device_call_safely_release_realtime_frame(frame);
    return DeviceCallInvalidArgumentStatus;
  }

  if (requestLen > 0)
    std::memcpy(requestPayload, request, requestLen);

  status = __cudaq_device_call_dispatch_realtime_frame(frame, responseLen);
  if (status == DeviceCallSuccessStatus && *responseLen > responseCapacity)
    status = DeviceCallResponseTooLargeStatus;
  if (status == DeviceCallSuccessStatus && *responseLen > 0)
    std::memcpy(response, responsePayload, *responseLen);

  __cudaq_device_call_safely_release_realtime_frame(frame);
  return status;
}

enum class TestGpuTable { AddThem, AddThemOffset };
enum class TestHostTable { GraphAddThem, HostAddThem, MixedAddThem };

TestGpuTable selectedGpuTable = TestGpuTable::AddThem;
TestHostTable selectedHostTable = TestHostTable::GraphAddThem;

class TestRealtimeService : public DeviceCallService {
public:
  int create(const void *, std::size_t) override { return 0; }

  int destroy() noexcept override {
    teardownHostDispatch();
    return 0;
  }

  std::uint32_t getFunctionCount() const override { return 1; }

  int populateTable(cudaq_function_entry_t *entries, std::uint32_t capacity,
                    cudaStream_t stream) override {
    if (!entries || capacity < 1)
      return 1;
    return test::populateAddThemTable(
        entries, selectedGpuTable == TestGpuTable::AddThemOffset, stream);
  }

  cudaq_dispatch_launch_fn_t getDeviceDispatchLaunch() const override {
    return cudaq_launch_dispatch_kernel_regular;
  }

  int getHostDispatchTable(DeviceCallHostDispatchTable &table) override {
    if (setupHostDispatch() != 0)
      return 1;
    table.entries = hostEntries.data();
    table.count = hostEntryCount;
    table.deviceId = 0;
    table.mailbox = h_mailbox;
    return 0;
  }

  int stop() noexcept override {
    teardownHostDispatch();
    return 0;
  }

private:
  int setupHostDispatch() {
    if (selectedHostTable == TestHostTable::HostAddThem) {
      teardownHostDispatch();
      fillHostCallAddEntry(hostEntries[0]);
      hostEntryCount = 1;
      return 0;
    }

    if (h_mailbox && graphExec)
      return 0;

    if (cudaHostAlloc(&h_mailbox, sizeof(void *), cudaHostAllocMapped) !=
        cudaSuccess)
      return 1;
    std::memset(h_mailbox, 0, sizeof(void *));
    if (cudaHostGetDevicePointer(reinterpret_cast<void **>(&d_mailbox),
                                 h_mailbox, 0) != cudaSuccess) {
      teardownHostDispatch();
      return 1;
    }
    if (!test::createGraphAddThem(d_mailbox, &graph, &graphExec)) {
      teardownHostDispatch();
      return 1;
    }

    test::fillHostGraphAddEntry(hostEntries[0], graphExec);
    hostEntryCount = 1;
    if (selectedHostTable == TestHostTable::MixedAddThem) {
      fillHostCallAddEntry(hostEntries[1]);
      hostEntryCount = 2;
    }
    return 0;
  }

  void teardownHostDispatch() noexcept {
    if (graphExec)
      cudaGraphExecDestroy(graphExec);
    if (graph)
      cudaGraphDestroy(graph);
    if (h_mailbox)
      cudaFreeHost(h_mailbox);

    graphExec = nullptr;
    graph = nullptr;
    h_mailbox = nullptr;
    d_mailbox = nullptr;
    hostEntries = {};
    hostEntryCount = 0;
  }

  void **h_mailbox = nullptr;
  void **d_mailbox = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  std::array<cudaq_function_entry_t, 2> hostEntries{};
  std::uint32_t hostEntryCount = 0;
};

DeviceCallService *getTestRealtimeService() {
  static TestRealtimeService service;
  return &service;
}

} // namespace

extern "C" cudaq_internal::device_call::DeviceCallServicePluginInfo
cudaqGetDeviceCallServicePluginInfo() {
  return {"test-device-call", &getTestRealtimeService};
}

namespace {

void initializeGpuRuntime(TestGpuTable table = TestGpuTable::AddThem) {
  selectedGpuTable = table;
  char program[] = "test_device_call_dispatch";
  char *argv[] = {program};
  cudaq_internal::device_call::initializeDeviceCallRuntime(1, argv);
}

void initializeHostRuntime(TestHostTable table) {
  selectedHostTable = table;
  char program[] = "test_device_call_dispatch";
  char option[] = "--cudaq-device-call=host-dispatch";
  char *argv[] = {program, option};
  cudaq_internal::device_call::initializeDeviceCallRuntime(2, argv);
}

void initializeHostRuntime() {
  initializeHostRuntime(TestHostTable::GraphAddThem);
}

void finalizeRuntime() {
  cudaq_internal::device_call::finalizeDeviceCallRuntime();
}

class DeviceCallDispatchTest : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_NO_THROW(initializeGpuRuntime()); }

  void TearDown() override { ASSERT_NO_THROW(finalizeRuntime()); }
};

TEST_F(DeviceCallDispatchTest, DispatchesI32AddHandler) {
  std::array<std::int32_t, 2> request{};
  auto *const args = request.data();
  args[0] = 19;
  args[1] = 23;

  std::int32_t response = 0;
  std::uint64_t responseLen = 0;
  ASSERT_EQ(0,
            dispatchUsingFrameLease(0, AddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &response, sizeof(response), &responseLen));
  EXPECT_EQ(sizeof(response), responseLen);
  EXPECT_EQ(42, response);
}

TEST_F(DeviceCallDispatchTest, DispatchesI32AddHandlerThroughFrameLease) {
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
  ASSERT_EQ(0,
            __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
  EXPECT_EQ(sizeof(std::int32_t), responseLen);
  EXPECT_EQ(42, *static_cast<std::int32_t *>(responsePayload));

  __cudaq_device_call_safely_release_realtime_frame(frame);
}

TEST_F(DeviceCallDispatchTest, DispatchesVoidFireAndForgetThroughFrameLease) {
  void *frame = nullptr;
  void *requestPayload = nullptr;
  void *responsePayload = nullptr;
  ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                   0, AddThemFunctionId, 2 * sizeof(std::int32_t), 0, &frame,
                   &requestPayload, &responsePayload));
  ASSERT_NE(nullptr, frame);
  ASSERT_NE(nullptr, requestPayload);
  EXPECT_EQ(nullptr, responsePayload);

  auto *args = static_cast<std::int32_t *>(requestPayload);
  args[0] = 19;
  args[1] = 23;

  std::uint64_t responseLen = 123;
  ASSERT_EQ(0,
            __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
  EXPECT_EQ(0u, responseLen);

  __cudaq_device_call_safely_release_realtime_frame(frame);

  for (int i = 0; i < 2; ++i) {
    frame = nullptr;
    requestPayload = nullptr;
    responsePayload = nullptr;
    ASSERT_EQ(0, __cudaq_device_call_acquire_realtime_frame(
                     0, AddThemFunctionId, 2 * sizeof(std::int32_t),
                     sizeof(std::int32_t), &frame, &requestPayload,
                     &responsePayload));
    args = static_cast<std::int32_t *>(requestPayload);
    args[0] = 19;
    args[1] = 23;

    responseLen = 0;
    ASSERT_EQ(0,
              __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
    EXPECT_EQ(sizeof(std::int32_t), responseLen);
    EXPECT_EQ(42, *static_cast<std::int32_t *>(responsePayload));

    __cudaq_device_call_safely_release_realtime_frame(frame);
  }
}

TEST_F(DeviceCallDispatchTest, ReinitializesThroughDiscoveredPlugin) {
  ASSERT_NO_THROW(finalizeRuntime());
  ASSERT_NO_THROW(initializeGpuRuntime(TestGpuTable::AddThemOffset));

  std::array<std::int32_t, 2> request{19, 23};
  std::int32_t response = 0;
  std::uint64_t responseLen = 0;
  ASSERT_EQ(0,
            dispatchUsingFrameLease(0, AddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &response, sizeof(response), &responseLen));
  EXPECT_EQ(sizeof(response), responseLen);
  EXPECT_EQ(142, response);
}

class HostGraphDispatchFrameTest : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_NO_THROW(initializeHostRuntime()); }

  void TearDown() override {
    if (frame)
      __cudaq_device_call_safely_release_realtime_frame(frame);
    ASSERT_NO_THROW(finalizeRuntime());
  }

  void *frame = nullptr;
};

TEST_F(HostGraphDispatchFrameTest, DispatchesGraphLaunchThroughFrameLease) {
  void *requestPayload = nullptr;
  void *responsePayload = nullptr;
  ASSERT_EQ(DeviceCallSuccessStatus,
            __cudaq_device_call_acquire_realtime_frame(
                0, GraphAddThemFunctionId, 2 * sizeof(std::int32_t),
                sizeof(std::int32_t), &frame, &requestPayload,
                &responsePayload));
  ASSERT_NE(nullptr, frame);
  ASSERT_NE(nullptr, requestPayload);
  ASSERT_NE(nullptr, responsePayload);

  auto *const args = static_cast<std::int32_t *>(requestPayload);
  args[0] = 19;
  args[1] = 23;

  std::uint64_t responseLen = 0;
  ASSERT_EQ(DeviceCallSuccessStatus,
            __cudaq_device_call_dispatch_realtime_frame(frame, &responseLen));
  EXPECT_EQ(sizeof(std::int32_t), responseLen);
  EXPECT_EQ(42, *static_cast<std::int32_t *>(responsePayload));

  __cudaq_device_call_safely_release_realtime_frame(frame);
  frame = nullptr;
}

class HostCallDispatchFrameTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_NO_THROW(initializeHostRuntime(TestHostTable::HostAddThem));
  }

  void TearDown() override { ASSERT_NO_THROW(finalizeRuntime()); }
};

TEST_F(HostCallDispatchFrameTest, DispatchesHostCallThroughFrameLease) {
  std::array<std::int32_t, 2> request{19, 23};
  std::int32_t response = 0;
  std::uint64_t responseLen = 0;

  ASSERT_EQ(DeviceCallSuccessStatus,
            dispatchUsingFrameLease(0, HostAddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &response, sizeof(response), &responseLen));
  EXPECT_EQ(sizeof(response), responseLen);
  EXPECT_EQ(42, response);
}

class HostMixedDispatchFrameTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_NO_THROW(initializeHostRuntime(TestHostTable::MixedAddThem));
  }

  void TearDown() override { ASSERT_NO_THROW(finalizeRuntime()); }
};

TEST_F(HostMixedDispatchFrameTest, DispatchesHostCallAndGraphLaunchEntries) {
  std::array<std::int32_t, 2> request{19, 23};

  std::int32_t hostResponse = 0;
  std::uint64_t hostResponseLen = 0;
  ASSERT_EQ(DeviceCallSuccessStatus,
            dispatchUsingFrameLease(0, HostAddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &hostResponse, sizeof(hostResponse),
                                    &hostResponseLen));
  EXPECT_EQ(sizeof(hostResponse), hostResponseLen);
  EXPECT_EQ(42, hostResponse);

  std::int32_t graphResponse = 0;
  std::uint64_t graphResponseLen = 0;
  ASSERT_EQ(DeviceCallSuccessStatus,
            dispatchUsingFrameLease(0, GraphAddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &graphResponse, sizeof(graphResponse),
                                    &graphResponseLen));
  EXPECT_EQ(sizeof(graphResponse), graphResponseLen);
  EXPECT_EQ(42, graphResponse);
}

class DeviceCallServicePluginTest : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_NO_THROW(initializeGpuRuntime()); }

  void TearDown() override { ASSERT_NO_THROW(finalizeRuntime()); }
};

TEST_F(DeviceCallServicePluginTest, DispatchesThroughDiscoveredPlugin) {
  std::array<std::int32_t, 2> request{19, 23};
  std::int32_t response = 0;
  std::uint64_t responseLen = 0;

  ASSERT_EQ(0,
            dispatchUsingFrameLease(0, AddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &response, sizeof(response), &responseLen));
  EXPECT_EQ(sizeof(response), responseLen);
  EXPECT_EQ(42, response);
}

TEST_F(DeviceCallServicePluginTest, FinalizeClearsPluginSession) {
  ASSERT_NO_THROW(finalizeRuntime());

  std::array<std::int32_t, 2> request{19, 23};
  std::int32_t response = 0;
  std::uint64_t responseLen = 0;
  EXPECT_EQ(DeviceCallNotInitializedStatus,
            dispatchUsingFrameLease(0, AddThemFunctionId, request.data(),
                                    request.size() * sizeof(request[0]),
                                    &response, sizeof(response), &responseLen));
}

} // namespace
