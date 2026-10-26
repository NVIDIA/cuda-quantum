/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/DeviceCallError.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace cudaq_internal::device_call::test {

constexpr std::uint32_t AddThemFunctionId =
    cudaq::realtime::fnv1a_hash("addThem");
constexpr std::uint32_t GraphAddThemFunctionId =
    cudaq::realtime::fnv1a_hash("graphAddThem");

// DEVICE_CALL handlers use the same two-pointer form as the host-call
// handler (see hostAddThemHandler in DeviceCallDispatchTester.cpp), plus a
// returned status: the request sits in rxSlot, the result goes after the
// RPCResponse in txSlot, and the handler sets result_len.
__device__ int addThemHandler(const void *rxSlot, void *txSlot,
                              std::size_t slotSize) {
  const auto *const request =
      static_cast<const cudaq::realtime::RPCHeader *>(rxSlot);
  auto *const response = static_cast<cudaq::realtime::RPCResponse *>(txSlot);
  if (request->arg_len != 2 * sizeof(std::int32_t))
    return 101;
  if (slotSize < sizeof(cudaq::realtime::RPCResponse) + sizeof(std::int32_t))
    return 102;

  const auto *const args = reinterpret_cast<const std::int32_t *>(request + 1);
  *reinterpret_cast<std::int32_t *>(response + 1) = args[0] + args[1];
  response->result_len = sizeof(std::int32_t);
  return 0;
}

__device__ int addThemOffsetHandler(const void *rxSlot, void *txSlot,
                                    std::size_t slotSize) {
  const auto *const request =
      static_cast<const cudaq::realtime::RPCHeader *>(rxSlot);
  auto *const response = static_cast<cudaq::realtime::RPCResponse *>(txSlot);
  if (request->arg_len != 2 * sizeof(std::int32_t))
    return 101;
  if (slotSize < sizeof(cudaq::realtime::RPCResponse) + sizeof(std::int32_t))
    return 102;

  const auto *const args = reinterpret_cast<const std::int32_t *>(request + 1);
  *reinterpret_cast<std::int32_t *>(response + 1) = args[0] + args[1] + 100;
  response->result_len = sizeof(std::int32_t);
  return 0;
}

// Graph node kernel behind the graph-launching HOST_CALL entry. `ioContext` is
// the device alias of the pinned context that graphAddThemHostHandler fills
// before each launch. The kernel writes the response but never touches the TX
// flag: the host dispatcher publishes the slot after the handler returns.
__global__ void
graphAddThemHandler(cudaq::realtime::GraphIOContext *ioContext) {
  if (threadIdx.x != 0 || blockIdx.x != 0)
    return;

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
}

namespace {

// Resources behind the graph-launching HOST_CALL entry: one graph exec, one
// non-blocking stream and one pinned, mapped GraphIOContext, created by
// setupGraphAddThem and released by teardownGraphAddThem.
cudaGraphExec_t g_graphExec = nullptr;
cudaStream_t g_stream = nullptr;
cudaq::realtime::GraphIOContext *g_hIoContext = nullptr;
cudaq::realtime::GraphIOContext *g_dIoContext = nullptr;
int g_deviceId = 0;

bool createGraphAddThem(cudaGraphExec_t *execOut) {
  cudaGraph_t graph = nullptr;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess)
    return false;

  void *kernelArgs[] = {&g_dIoContext};
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

  const cudaError_t err =
      cudaGraphInstantiate(execOut, graph, nullptr, nullptr, 0);
  cudaGraphDestroy(graph);
  return err == cudaSuccess;
}

cudaError_t launchGraphAddThem(const void *rxSlot, void *txSlot,
                               std::size_t slotSize) {
  // The dispatcher thread has no device selected.
  cudaError_t err = cudaSetDevice(g_deviceId);
  if (err != cudaSuccess)
    return err;

  // The ring slots are cudaHostAllocMapped memory; the kernel needs their
  // device views.
  cudaPointerAttributes rxAttributes{};
  err = cudaPointerGetAttributes(&rxAttributes, rxSlot);
  if (err != cudaSuccess)
    return err;
  cudaPointerAttributes txAttributes{};
  err = cudaPointerGetAttributes(&txAttributes, txSlot);
  if (err != cudaSuccess)
    return err;
  if (!rxAttributes.devicePointer || !txAttributes.devicePointer)
    return cudaErrorInvalidValue;

  g_hIoContext->rx_slot = rxAttributes.devicePointer;
  g_hIoContext->tx_slot =
      static_cast<std::uint8_t *>(txAttributes.devicePointer);
  g_hIoContext->tx_flag = nullptr;
  g_hIoContext->tx_flag_value = 0;
  g_hIoContext->tx_stride_sz = slotSize;

  err = cudaGraphLaunch(g_graphExec, g_stream);
  if (err != cudaSuccess)
    return err;
  return cudaStreamSynchronize(g_stream);
}

void writeCudaErrorResponse(const void *rxSlot, void *txSlot,
                            std::size_t slotSize) {
  if (!rxSlot || !txSlot || slotSize < sizeof(cudaq::realtime::RPCResponse))
    return;
  const auto *const request =
      static_cast<const cudaq::realtime::RPCHeader *>(rxSlot);
  auto *const response = static_cast<cudaq::realtime::RPCResponse *>(txSlot);
  response->magic = cudaq::realtime::RPC_MAGIC_RESPONSE;
  response->request_id = request->request_id;
  response->ptp_timestamp = request->ptp_timestamp;
  response->status = toAbiStatus(DeviceCallStatus::CudaError);
  response->result_len = 0;
}

} // namespace

void teardownGraphAddThem() {
  if (g_graphExec)
    cudaGraphExecDestroy(g_graphExec);
  if (g_stream)
    cudaStreamDestroy(g_stream);
  if (g_hIoContext)
    cudaFreeHost(g_hIoContext);
  g_graphExec = nullptr;
  g_stream = nullptr;
  g_hIoContext = nullptr;
  g_dIoContext = nullptr;
}

bool setupGraphAddThem(int deviceId) {
  if (g_graphExec)
    return true;
  g_deviceId = deviceId;
  if (cudaSetDevice(deviceId) != cudaSuccess)
    return false;
  if (cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking) !=
          cudaSuccess ||
      cudaHostAlloc(reinterpret_cast<void **>(&g_hIoContext),
                    sizeof(cudaq::realtime::GraphIOContext),
                    cudaHostAllocMapped) != cudaSuccess) {
    teardownGraphAddThem();
    return false;
  }
  std::memset(g_hIoContext, 0, sizeof(cudaq::realtime::GraphIOContext));
  if (cudaHostGetDevicePointer(reinterpret_cast<void **>(&g_dIoContext),
                               g_hIoContext, 0) != cudaSuccess ||
      !createGraphAddThem(&g_graphExec)) {
    teardownGraphAddThem();
    return false;
  }
  return true;
}

// HOST_CALL handler that launches the graph above, with the same shape as the
// compiler-driver shim's trampoline (cudaq/test/Driver/Inputs/
// DeviceCallLibrary.h): select the device, translate the slot pointers to
// device views, launch, synchronize, and report a CUDA failure in-band as an
// RPCResponse with status CudaError and no result bytes.
void graphAddThemHostHandler(const void *rxSlot, void *txSlot,
                             std::size_t slotSize) {
  const cudaError_t err = launchGraphAddThem(rxSlot, txSlot, slotSize);
  if (err != cudaSuccess)
    writeCudaErrorResponse(rxSlot, txSlot, slotSize);
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

void fillHostGraphAddEntry(cudaq_function_entry_t &entry) {
  entry = {};
  entry.handler.host_fn = graphAddThemHostHandler;
  entry.function_id = GraphAddThemFunctionId;
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

int populateAddThemTable(cudaq_function_entry_t *entries, bool useOffset,
                         cudaStream_t stream) {
  if (useOffset)
    initDeviceCallTableWithOffset<<<1, 1, 0, stream>>>(entries);
  else
    initDeviceCallTable<<<1, 1, 0, stream>>>(entries);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

} // namespace cudaq_internal::device_call::test
