/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file single_thread_dispatcher.cu
/// @brief An application's own unified dispatch kernel: one thread receives,
/// calls the handler and publishes, with no block barriers.
///
/// Not an ordinary source of the binaries that use it: it calls the transport
/// hooks, which only the bridge defines, so their own device link could not
/// resolve it.  cudaq_realtime_add_dispatch_kernel
/// (realtime/cmake/CudaqRealtimeDispatchKernel.cmake) compiles it on its own
/// to LTO-IR and embeds that in the binary's __nv_relfatbin section, where the
/// dispatcher's run-time link finds it.

#include "cudaq/realtime/daemon/dispatcher/unified_device_transport.cuh"
#include "cudaq/realtime/daemon/dispatcher/unified_dispatch_device.cuh"
#include <cstddef>
#include <cstdint>

#if defined(CUDAQ_REALTIME_TEST_EXPECT_CUDA_DEFINITION) &&                     \
    !defined(CUDAQ_REALTIME_TEST_CUDA_DEFINITION)
#error "the custom dispatch kernel did not receive CUDA compile definitions"
#endif

using namespace cudaq::realtime;

/// The parameters of the dispatcher's own kernel, which the launchers pass
/// on.  Only the grid's first thread works, so any grid and launch form
/// serves; with one thread per block, a block-partitioned handler does all of
/// its work on that thread.
extern "C" __global__ void
single_thread_dispatcher(void *transport_ctx, std::size_t tx_stride_sz,
                         volatile int *shutdown_flag,
                         cudaq_function_entry_t *function_table,
                         std::size_t func_count, std::uint64_t *stats) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  alignas(CUDAQ_DEV_TRANSPORT_SESSION_ALIGN) unsigned char
      session_storage[CUDAQ_DEV_TRANSPORT_SESSION_BYTES];
  void *session =
      cudaq_dev_enter_kernel(session_storage, transport_ctx, shutdown_flag);
  if (session == nullptr)
    return;

  std::uint64_t packet_count = 0;
  void *request = nullptr;
  void *response = nullptr;
  while (cudaq_dev_rx_poll(session, &request, &response) ==
         CUDAQ_RX_DEV_READY) {
    const auto *header = static_cast<const RPCHeader *>(request);
    const std::uint32_t request_id = header->request_id;
    const std::uint64_t ptp_timestamp = header->ptp_timestamp;
    int status = -1;
    if (header->magic == RPC_MAGIC_REQUEST) {
      const cudaq_function_entry_t *entry =
          unified_lookup_entry(header->function_id, function_table, func_count);
      if (entry != nullptr &&
          entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL)
        status = reinterpret_cast<DeviceRPCFunction>(
            entry->handler.device_fn_ptr)(request, response, tx_stride_sz);
    }
    write_response_header(static_cast<RPCResponse *>(response), status,
                          request_id, ptp_timestamp);
    cudaq_dev_tx_publish(session, response);
    packet_count++;
  }
  atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
}
