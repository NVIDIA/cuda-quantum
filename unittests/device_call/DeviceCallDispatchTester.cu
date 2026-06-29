/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace cudaq_internal::device_call::test {

constexpr std::uint32_t AddThemFunctionId =
    cudaq::realtime::fnv1a_hash("addThem");
constexpr std::uint32_t GraphAddThemFunctionId =
    cudaq::realtime::fnv1a_hash("graphAddThem");

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

bool createGraphAddThem(void **dMailbox, cudaGraph_t *graphOut,
                        cudaGraphExec_t *execOut) {
  cudaGraph_t graph = nullptr;
  if (cudaGraphCreate(&graph, 0) != cudaSuccess)
    return false;

  void *kernelArgs[] = {&dMailbox};
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

int populateAddThemTable(cudaq_function_entry_t *entries, bool useOffset,
                         cudaStream_t stream) {
  if (useOffset)
    initDeviceCallTableWithOffset<<<1, 1, 0, stream>>>(entries);
  else
    initDeviceCallTable<<<1, 1, 0, stream>>>(entries);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

} // namespace cudaq_internal::device_call::test
