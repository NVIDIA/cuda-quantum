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
/// Compiled into libcudaq-realtime-unified-dispatch-core.a.  The DOCA
/// implementation of the hooks lives in
/// bridge/gpu_roce/gpu_roce_unified_transport.cu.

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
// Each turn claims one slot from the transport, runs its handler in place, and
// hands the slot back.  The wait, the wire transfer and all receive
// bookkeeping live behind the hooks, so what remains here is framing and
// dispatch.
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
    void *frame = nullptr;
    if (cudaq_dev_rx_poll(session, &frame) != CUDAQ_RX_DEV_READY)
      break;

    auto *slot = static_cast<std::uint8_t *>(frame);
    auto *header = reinterpret_cast<RPCHeader *>(slot);

    if (header->magic == RPC_MAGIC_REQUEST) {
      std::uint32_t function_id = header->function_id;
      std::uint32_t arg_len = header->arg_len;
      std::uint32_t request_id = header->request_id;
      std::uint64_t ptp_timestamp = header->ptp_timestamp;

      const cudaq_function_entry_t *entry =
          unified_lookup_entry(function_id, function_table, func_count);

      int status = -1;
      std::uint32_t result_len = 0;

      if (entry != nullptr &&
          entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL) {
        auto func =
            reinterpret_cast<DeviceRPCFunction>(entry->handler.device_fn_ptr);
        void *arg_buffer = static_cast<void *>(header + 1);
        auto *output_buffer = slot + sizeof(RPCResponse);
        auto max_result_len =
            static_cast<std::uint32_t>(tx_stride_sz - sizeof(RPCResponse));

        status = func(arg_buffer, output_buffer, arg_len, max_result_len,
                      &result_len);
      }

      auto *response = reinterpret_cast<RPCResponse *>(slot);
      response->magic = RPC_MAGIC_RESPONSE;
      response->status = status;
      response->result_len = result_len;
      response->request_id = request_id;
      response->ptp_timestamp = ptp_timestamp;
    }

    // Published unconditionally, including for a frame whose magic did not
    // match: tx_publish is also what returns the slot's receive credit, so
    // dropping one here would starve the transport a slot at a time.
    cudaq_dev_tx_publish(session, frame);

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
