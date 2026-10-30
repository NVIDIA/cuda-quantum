/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file unified_dispatch_device.cuh
/// @brief Device helpers shared by unified dispatch kernels: the dispatcher's
/// own (`unified_dispatch_kernel.cu`) and any an application launches through
/// `cudaq_dispatcher_set_unified_launch`.
///
/// Internal linkage throughout: every binary that includes this gets its own
/// copy, so the run-time link never sees two definitions of one symbol.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include <cstddef>
#include <cstdint>

namespace cudaq::realtime {

__device__ static inline const cudaq_function_entry_t *
unified_lookup_entry(std::uint32_t function_id, cudaq_function_entry_t *entries,
                     std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (entries[i].function_id == function_id)
      return &entries[i];
  }
  return nullptr;
}

/// Fill the response header.  Written for every frame, bad magic included: the
/// publish is mandatory, and an unwritten response frame would put whatever
/// the slot last held on the wire.  The handler owns result_len; it is forced
/// to 0 for unhandled frames and failed handlers.
__device__ static inline void
write_response_header(RPCResponse *response, int status,
                      std::uint32_t request_id, std::uint64_t ptp_timestamp) {
  response->magic = RPC_MAGIC_RESPONSE;
  response->status = status;
  if (status != 0)
    response->result_len = 0;
  response->request_id = request_id;
  response->ptp_timestamp = ptp_timestamp;
}

} // namespace cudaq::realtime
