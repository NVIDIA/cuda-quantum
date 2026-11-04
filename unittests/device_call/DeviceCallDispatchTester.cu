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

#include <cstddef>
#include <cstdint>

namespace cudaq_internal::device_call::test {

constexpr std::uint32_t AddThemFunctionId =
    cudaq::realtime::fnv1a_hash("addThem");

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

int populateAddThemTable(cudaq_function_entry_t *entries, bool useOffset,
                         cudaStream_t stream) {
  if (useOffset)
    initDeviceCallTableWithOffset<<<1, 1, 0, stream>>>(entries);
  else
    initDeviceCallTable<<<1, 1, 0, stream>>>(entries);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

} // namespace cudaq_internal::device_call::test
