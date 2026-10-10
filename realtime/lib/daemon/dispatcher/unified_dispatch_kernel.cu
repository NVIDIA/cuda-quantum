/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file unified_dispatch_kernel.cu
/// @brief Unified dispatch: RX + RPC dispatch + TX in one kernel, with no
/// inter-kernel flag handoff.
///
/// This file carries no transport dependency of any kind -- it is compiled
/// without DOCA, verbs or HSB include paths, and that exclusion is enforced by
/// its CMake target rather than by convention.  Everything transport-specific
/// arrives through the three `__device__` hooks declared in
/// unified_device_transport.cuh, which a transport implements in its own TU
/// and device-links into the same shared library (see the LINKAGE note there).
///
/// Compiled into libcudaq-realtime, together with whichever TU defines the
/// hooks: bridge/gpu_roce/gpu_roce_unified_transport.cu when that transport
/// is configured, otherwise dispatcher/unified_transport_none.cu.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/unified_device_transport.cuh"

#include <cstdint>
#include <cuda_runtime.h>

using namespace cudaq::realtime;

//==============================================================================
// Device helpers
//==============================================================================

__device__ static inline const cudaq_function_entry_t *
unified_lookup_entry(std::uint32_t function_id, cudaq_function_entry_t *entries,
                     std::size_t count) {
  for (std::size_t i = 0; i < count; ++i) {
    if (entries[i].function_id == function_id)
      return &entries[i];
  }
  return nullptr;
}

//==============================================================================
// Unified dispatch kernel -- single thread, single block.
//
// Each turn claims a request and its response frame from the transport, runs
// the handler from one into the other, and hands the response back.  The
// wait, the wire transfer and all receive bookkeeping live behind the hooks,
// so what remains here is framing and dispatch.
//==============================================================================

__global__ void
cudaq_unified_dispatch_kernel(void *transport_ctx, std::size_t tx_stride_sz,
                              volatile int *shutdown_flag,
                              cudaq_function_entry_t *function_table,
                              std::size_t func_count, std::uint64_t *stats) {
  void *session = cudaq_dev_transport_attach(transport_ctx, shutdown_flag);
  if (session == nullptr)
    return;

  std::uint64_t packet_count = 0;

  while (true) {
    void *request = nullptr;
    void *response_frame = nullptr;
    if (cudaq_dev_rx_poll(session, &request, &response_frame) !=
        CUDAQ_RX_DEV_READY)
      break;

    auto *header = static_cast<RPCHeader *>(request);
    auto *response = static_cast<RPCResponse *>(response_frame);

    int status = -1;

    if (header->magic == RPC_MAGIC_REQUEST) {
      const cudaq_function_entry_t *entry =
          unified_lookup_entry(header->function_id, function_table, func_count);

      if (entry != nullptr &&
          entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL) {
        auto func =
            reinterpret_cast<DeviceRPCFunction>(entry->handler.device_fn_ptr);
        // Same two-pointer call as the host path (host_dispatcher.cu): the
        // handler reads the request from `request`, writes its result after
        // the RPCResponse in `response_frame` and sets response->result_len.
        status = func(request, response_frame, tx_stride_sz);
      }
    }

    // Written for every frame, bad magic included: the publish below is
    // mandatory, and an unwritten response frame would put whatever the slot
    // last held on the wire.
    response->magic = RPC_MAGIC_RESPONSE;
    response->status = status;
    // The handler owns result_len; force it to 0 for unhandled frames and
    // failed handlers.
    if (status != 0)
      response->result_len = 0;
    response->request_id = header->request_id;
    response->ptp_timestamp = header->ptp_timestamp;

    // Published unconditionally, including for a frame whose magic did not
    // match: tx_publish is also what returns the slot's receive credit, so
    // dropping one here would starve the transport a slot at a time.
    cudaq_dev_tx_publish(session, response_frame);

    packet_count++;
  }

  atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
}

//==============================================================================
// Host launch wrapper -- matches cudaq_unified_launch_fn_t signature.
//==============================================================================

extern "C" void cudaq_launch_unified_dispatch_device(
    void *transport_ctx, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, cudaStream_t stream) {
  cudaq_unified_dispatch_kernel<<<1, 1, 0, stream>>>(
      transport_ctx, tx_stride_sz, shutdown_flag, function_table, func_count,
      stats);
}
