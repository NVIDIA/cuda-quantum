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
#include <cstdint>
#include <cstring>

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

using namespace cudaq_internal::device_call;

constexpr std::uint32_t AddThemFunctionId =
    cudaq::realtime::fnv1a_hash("addThem");
constexpr std::uint32_t GraphAddThemFunctionId =
    cudaq::realtime::fnv1a_hash("graphAddThem");
constexpr std::int32_t DeviceCallSuccessStatus =
    toAbiStatus(DeviceCallStatus::Success);
constexpr std::int32_t DeviceCallInvalidArgumentStatus =
    toAbiStatus(DeviceCallStatus::InvalidArgument);
constexpr std::int32_t DeviceCallNotInitializedStatus =
    toAbiStatus(DeviceCallStatus::NotInitialized);
constexpr std::int32_t DeviceCallResponseTooLargeStatus =
    toAbiStatus(DeviceCallStatus::ResponseTooLarge);

#define ASSERT_CUDA_SUCCESS(expr)                                              \
  do {                                                                         \
    const cudaError_t err = (expr);                                            \
    ASSERT_EQ(cudaSuccess, err)                                                \
        << #expr << " failed at " << __FILE__ << ":" << __LINE__ << ": "       \
        << cudaGetErrorString(err);                                            \
  } while (false)

__device__ int addThemHandler(const void *input, void *output,
                              std::uint32_t argLen, std::uint32_t maxResultLen,
                              std::uint32_t *resultLen) {
  if (argLen != 2 * sizeof(std::int32_t))
    return 101;
  if (maxResultLen < sizeof(std::int32_t))
    return 102;

  const auto *const args = static_cast<const std::int32_t *>(input);
  *static_cast<std::int32_t *>(output) = args[0] + args[1];
  *resultLen = sizeof(std::int32_t);
  return 0;
}

__device__ int addThemOffsetHandler(const void *input, void *output,
                                    std::uint32_t argLen,
                                    std::uint32_t maxResultLen,
                                    std::uint32_t *resultLen) {
  if (argLen != 2 * sizeof(std::int32_t))
    return 101;
  if (maxResultLen < sizeof(std::int32_t))
    return 102;

  const auto *const args = static_cast<const std::int32_t *>(input);
  *static_cast<std::int32_t *>(output) = args[0] + args[1] + 100;
  *resultLen = sizeof(std::int32_t);
  return 0;
}

__global__ void graphAddThemHandler(void **mailboxSlotPtr) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  auto *const ioContext =
      reinterpret_cast<cudaq::realtime::GraphIOContext *>(*mailboxSlotPtr);
  if (!ioContext || !ioContext->rx_slot || !ioContext->tx_slot)
    return;

  const auto *const request =
      static_cast<const cudaq::realtime::RPCHeader *>(ioContext->rx_slot);
  auto *const response =
      reinterpret_cast<cudaq::realtime::RPCResponse *>(ioContext->tx_slot);
  auto *const result = reinterpret_cast<std::int32_t *>(
      ioContext->tx_slot + sizeof(cudaq::realtime::RPCResponse));

  std::int32_t status = 0;
  std::uint32_t resultLen = sizeof(std::int32_t);
  if (request->magic != cudaq::realtime::RPC_MAGIC_REQUEST ||
      request->arg_len != 2 * sizeof(std::int32_t) ||
      ioContext->tx_stride_sz <
          sizeof(cudaq::realtime::RPCResponse) + sizeof(std::int32_t)) {
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

  __threadfence_system();
  if (ioContext->tx_flag)
    *ioContext->tx_flag = ioContext->tx_flag_value;
}

bool createGraphAddThem(void **d_mailbox, cudaGraph_t *graphOut,
                        cudaGraphExec_t *execOut) {
  cudaGraph_t graph = nullptr;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess)
    return false;

  void *kernelArgs[] = {&d_mailbox};
  const cudaKernelNodeParams params = [&] {
    cudaKernelNodeParams result{};
    result.func = reinterpret_cast<void *>(graphAddThemHandler);
    result.gridDim = dim3(1, 1, 1);
    result.blockDim = dim3(32, 1, 1);
    result.kernelParams = kernelArgs;
    return result;
  }();

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

  *graphOut = graph;
  *execOut = exec;
  return true;
}

std::int32_t dispatchUsingFrameLease(std::uint32_t deviceId,
                                     std::uint32_t functionId,
                                     const void *request,
                                     std::uint64_t requestLen, void *response,
                                     std::uint64_t responseCapacity,
                                     std::uint64_t *responseLen) {
  if ((requestLen > 0 && !request) || !responseLen)
    return DeviceCallInvalidArgumentStatus;
  if (responseCapacity > 0 && !response)
    return DeviceCallInvalidArgumentStatus;

  void *frame = nullptr;
  void *requestPayload = nullptr;
  void *responsePayload = nullptr;
  std::int32_t status = __cudaq_device_call_acquire_realtime_frame(
      deviceId, functionId, requestLen, responseCapacity, &frame,
      &requestPayload, &responsePayload);
  if (status != DeviceCallSuccessStatus)
    return status;
  if ((requestLen > 0 && !requestPayload) ||
      (responseCapacity > 0 && !responsePayload)) {
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

__global__ void initDeviceCallTable(cudaq_function_entry_t *entries) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  entries[0].handler.device_fn_ptr = reinterpret_cast<void *>(&addThemHandler);
  entries[0].function_id = AddThemFunctionId;
  entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  entries[0].reserved[0] = 0;
  entries[0].reserved[1] = 0;
  entries[0].reserved[2] = 0;
  entries[0].schema.num_args = 2;
  entries[0].schema.num_results = 1;
  entries[0].schema.reserved = 0;
  entries[0].schema.args[0].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.args[0].size_bytes = sizeof(std::int32_t);
  entries[0].schema.args[0].num_elements = 1;
  entries[0].schema.args[1].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.args[1].size_bytes = sizeof(std::int32_t);
  entries[0].schema.args[1].num_elements = 1;
  entries[0].schema.results[0].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.results[0].size_bytes = sizeof(std::int32_t);
  entries[0].schema.results[0].num_elements = 1;
}

__global__ void initDeviceCallTableWithOffset(cudaq_function_entry_t *entries) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

  entries[0].handler.device_fn_ptr =
      reinterpret_cast<void *>(&addThemOffsetHandler);
  entries[0].function_id = AddThemFunctionId;
  entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  entries[0].reserved[0] = 0;
  entries[0].reserved[1] = 0;
  entries[0].reserved[2] = 0;
  entries[0].schema.num_args = 2;
  entries[0].schema.num_results = 1;
  entries[0].schema.reserved = 0;
  entries[0].schema.args[0].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.args[0].size_bytes = sizeof(std::int32_t);
  entries[0].schema.args[0].num_elements = 1;
  entries[0].schema.args[1].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.args[1].size_bytes = sizeof(std::int32_t);
  entries[0].schema.args[1].num_elements = 1;
  entries[0].schema.results[0].type_id = CUDAQ_TYPE_INT32;
  entries[0].schema.results[0].size_bytes = sizeof(std::int32_t);
  entries[0].schema.results[0].num_elements = 1;
}

void fillHostGraphAddEntry(cudaq_function_entry_t &entry,
                           cudaGraphExec_t graphExec) {
  entry = {};
  entry.handler.graph_exec = graphExec;
  entry.function_id = GraphAddThemFunctionId;
  entry.dispatch_mode = CUDAQ_DISPATCH_GRAPH_LAUNCH;
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

enum class TestGpuTable { AddThem, AddThemOffset };

TestGpuTable selectedGpuTable = TestGpuTable::AddThem;

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
    if (selectedGpuTable == TestGpuTable::AddThemOffset)
      initDeviceCallTableWithOffset<<<1, 1, 0, stream>>>(entries);
    else
      initDeviceCallTable<<<1, 1, 0, stream>>>(entries);
    return cudaGetLastError() == cudaSuccess ? 0 : 1;
  }

  cudaq_dispatch_launch_fn_t getDeviceDispatchLaunch() const override {
    return cudaq_launch_dispatch_kernel_regular;
  }

  int getHostDispatchTable(DeviceCallHostDispatchTable &table) override {
    if (setupHostDispatch() != 0)
      return 1;
    table.entries = hostEntries.data();
    table.count = static_cast<std::uint32_t>(hostEntries.size());
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
    if (h_mailbox && graphExec)
      return 0;

    if (cudaHostAlloc(&h_mailbox, sizeof(void *), cudaHostAllocMapped) !=
        cudaSuccess)
      return 1;
    std::memset(h_mailbox, 0, sizeof(void *));
    if (cudaHostGetDevicePointer(reinterpret_cast<void **>(&d_mailbox), h_mailbox,
                                 0) != cudaSuccess) {
      teardownHostDispatch();
      return 1;
    }
    if (!createGraphAddThem(d_mailbox, &graph, &graphExec)) {
      teardownHostDispatch();
      return 1;
    }

    fillHostGraphAddEntry(hostEntries[0], graphExec);
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
  }

  void **h_mailbox = nullptr;
  void **d_mailbox = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  std::array<cudaq_function_entry_t, 1> hostEntries{};
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

void initializeHostRuntime() {
  char program[] = "test_device_call_dispatch";
  char option[] = "--cudaq-device-call=host-dispatch";
  char *argv[] = {program, option};
  cudaq_internal::device_call::initializeDeviceCallRuntime(2, argv);
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
