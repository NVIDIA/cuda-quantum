/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file init_rpc_increment_function_table.cu
/// @brief Device-side increment RPC handler and function table initialisation.
///
/// This file is compiled by nvcc so that the __device__ function pointer
/// can be taken.  The host-callable setup_rpc_increment_function_table()
/// wrapper is extern "C" so that the bridge .cpp (compiled by g++) can
/// call it without needing CUDA kernel launch syntax.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_modes.h"
#include "rpc_increment_handler.cuh"

#include <cstdint>
#include <cstring> // std::memset
#include <cuda_runtime.h>

namespace {

constexpr std::uint32_t RPC_INCREMENT_FUNCTION_ID =
    cudaq::realtime::fnv1a_hash("rpc_increment");

/// @brief Kernel to populate a cudaq_function_entry_t with the increment
///        handler.
__global__ void init_function_table_kernel(cudaq_function_entry_t *entries) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    entries[0].handler.device_fn_ptr =
        reinterpret_cast<void *>(&rpc_increment_handler);
    entries[0].function_id = RPC_INCREMENT_FUNCTION_ID;
    entries[0].dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    entries[0].reserved[0] = 0;
    entries[0].reserved[1] = 0;
    entries[0].reserved[2] = 0;

    // Schema: 1 array argument (uint8), 1 array result (uint8)
    entries[0].schema.num_args = 1;
    entries[0].schema.num_results = 1;
    entries[0].schema.reserved = 0;
    entries[0].schema.args[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
    entries[0].schema.args[0].reserved[0] = 0;
    entries[0].schema.args[0].reserved[1] = 0;
    entries[0].schema.args[0].reserved[2] = 0;
    entries[0].schema.args[0].size_bytes = 0;
    entries[0].schema.args[0].num_elements = 0;
    entries[0].schema.results[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
    entries[0].schema.results[0].reserved[0] = 0;
    entries[0].schema.results[0].reserved[1] = 0;
    entries[0].schema.results[0].reserved[2] = 0;
    entries[0].schema.results[0].size_bytes = 0;
    entries[0].schema.results[0].num_elements = 0;
  }
}

} // anonymous namespace

//==============================================================================
// Host-Callable Wrapper
//==============================================================================

extern "C" void
setup_rpc_increment_function_table(cudaq_function_entry_t *d_entries) {
  init_function_table_kernel<<<1, 1>>>(d_entries);
  cudaDeviceSynchronize();
}

/// Same entry as setup_rpc_increment_function_table, with the handler address
/// the run-time linked `dispatcher` hands out
/// (cudaq_dispatcher_get_device_handler); filled on the host, since no kernel
/// of this binary runs in that dispatcher's module.
extern "C" cudaq_status_t
setup_rpc_increment_function_table_linked(cudaq_function_entry_t *d_entries,
                                          cudaq_dispatcher_t *dispatcher) {
  void *fn = nullptr;
  const cudaq_status_t status = cudaq_dispatcher_get_device_handler(
      dispatcher, "rpc_increment_handler", &fn);
  if (status != CUDAQ_OK)
    return status;
  cudaq_function_entry_t entry;
  std::memset(&entry, 0, sizeof(entry));
  entry.handler.device_fn_ptr = fn;
  entry.function_id = RPC_INCREMENT_FUNCTION_ID;
  entry.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
  // Schema: 1 array argument (uint8), 1 array result (uint8)
  entry.schema.num_args = 1;
  entry.schema.num_results = 1;
  entry.schema.args[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
  entry.schema.results[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
  cudaMemcpy(d_entries, &entry, sizeof(entry), cudaMemcpyHostToDevice);
  return CUDAQ_OK;
}
