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

#include <cstdint>
#include <cuda_runtime.h>

namespace {

//==============================================================================
// Increment RPC Handler
//==============================================================================

/// @brief Simple RPC handler that increments each byte of the payload by 1.
///
/// Matches the DeviceRPCFunction signature (same two-pointer form as the
/// host-call handler): reads the RPCHeader + payload from rx_slot, writes the
/// incremented payload after the RPCResponse in tx_slot (no overlap) and sets
/// result_len.  The dispatch kernel fills the other response header fields.
__device__ int rpc_increment_handler(const void *rx_slot, void *tx_slot,
                                     std::size_t slot_size) {
  const auto *header = static_cast<const cudaq::realtime::RPCHeader *>(rx_slot);
  const std::uint8_t *in_data =
      reinterpret_cast<const std::uint8_t *>(header + 1);
  auto *response = static_cast<cudaq::realtime::RPCResponse *>(tx_slot);
  std::uint8_t *out_data = reinterpret_cast<std::uint8_t *>(response + 1);
  const std::uint32_t arg_len = header->arg_len;
  const std::uint32_t max_result_len =
      slot_size - sizeof(cudaq::realtime::RPCResponse);
  std::uint32_t len = (arg_len < max_result_len) ? arg_len : max_result_len;
  // Every thread of the dispatch block runs this; each increments its own
  // bytes.
  for (std::uint32_t i = threadIdx.x; i < len; i += blockDim.x)
    out_data[i] = static_cast<std::uint8_t>(in_data[i] + 1);
  if (threadIdx.x == 0)
    response->result_len = len;
  return 0;
}

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
