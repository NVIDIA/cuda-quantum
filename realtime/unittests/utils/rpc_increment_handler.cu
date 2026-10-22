/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file rpc_increment_handler.cu
/// @brief The device increment RPC handler, defined once: the 3-kernel init
/// kernel (init_rpc_increment_function_table.cu) takes its address in this
/// binary's own module, and the export lets a run-time linked unified
/// dispatcher find it with cudaq_dispatcher_get_device_handler.
///
/// An INTERFACE source of rpc_increment_ft, so each executable compiles it:
/// nothing on the host references it, so an archive member holding it would
/// be left out of the binary's device code.

#include "rpc_increment_handler.cuh"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/device_code/device_code.h"
#include <cstddef>
#include <cstdint>

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
CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(rpc_increment_handler)
