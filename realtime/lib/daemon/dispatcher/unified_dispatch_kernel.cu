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
/// without DOCA, verbs or HSB include paths, which the CMake build scopes to
/// the transport's own TU.  Everything transport-specific arrives through the
/// three `__device__` hooks declared in unified_device_transport.cuh, which a
/// transport implements in its own TU (see the LINKAGE note there).
///
/// Compiled on its own to relocatable LTO-IR and embedded in
/// libcudaq-realtime.so (unified_dispatch_kernel_image.cpp).  The dispatcher
/// links it at run time with the bridge's hooks and the application's
/// DEVICE_CALL handlers (cudaq_dispatcher_get_device_handler), so the kernel, the
/// hooks and the handlers the function table points at share one CUDA module.
/// The kernel is `extern "C"` so that it can be found in that module by name.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/unified_device_transport.cuh"
#include "cudaq/realtime/daemon/dispatcher/unified_dispatch_device.cuh"

#include <cstdint>
#include <cuda_runtime.h>

using namespace cudaq::realtime;

//==============================================================================
// Unified dispatch kernel -- one block, every thread runs the handler.
//
// Thread 0 owns the transport session: it claims each request and its
// response frame from the hooks, hands a DEVICE_CALL request to the block
// through a __shared__ descriptor, and after every thread has run the handler
// writes the response header and publishes.  Frames nothing can be dispatched
// to (bad magic, unknown function, wrong mode) are answered by thread 0 alone
// without waking the block.  The wait, the wire transfer and all receive
// bookkeeping live behind the hooks, so what remains here is framing and
// dispatch.
//==============================================================================

namespace {

/// Work descriptor thread 0 hands to the block for one DEVICE_CALL request.
/// Written before the first barrier of a turn, read by every thread between
/// the two barriers, written again only after the second.  `func == nullptr`
/// means "leave the loop".
struct UnifiedWork {
  DeviceRPCFunction func;
  void *request;
  void *response;
  std::uint32_t request_id;
  std::uint64_t ptp_timestamp;
};

} // namespace

extern "C" __global__ void
cudaq_unified_dispatch_kernel(void *transport_ctx, std::size_t tx_stride_sz,
                              volatile int *shutdown_flag,
                              cudaq_function_entry_t *function_table,
                              std::size_t func_count, std::uint64_t *stats) {
  if (blockIdx.x != 0) // one block, as for the ring kernels
    return;
  __shared__ UnifiedWork s_work;
  // The session belongs to thread 0.  Every thread declares the storage (it
  // is kernel-local); only thread 0 attaches and calls the hooks.
  alignas(CUDAQ_DEV_TRANSPORT_SESSION_ALIGN) unsigned char
      session_storage[CUDAQ_DEV_TRANSPORT_SESSION_BYTES];
  void *session = nullptr;
  std::uint64_t packet_count = 0;
  if (threadIdx.x == 0)
    session =
        cudaq_dev_enter_kernel(session_storage, transport_ctx, shutdown_flag);
  // A refused attach (nullptr) makes the first turn publish "leave" below, so
  // the whole block exits without a separate broadcast.

  while (true) {
    if (threadIdx.x == 0) {
      DeviceRPCFunction func = nullptr;
      while (session != nullptr) {
        void *request = nullptr;
        void *response_frame = nullptr;
        if (cudaq_dev_rx_poll(session, &request, &response_frame) !=
            CUDAQ_RX_DEV_READY)
          break; // shutdown: func stays nullptr

        auto *header = static_cast<RPCHeader *>(request);
        // Read before anything is written: after the handler the compiler
        // has to assume the request may have changed.
        const std::uint32_t request_id = header->request_id;
        const std::uint64_t ptp_timestamp = header->ptp_timestamp;

        if (header->magic == RPC_MAGIC_REQUEST) {
          const cudaq_function_entry_t *entry = unified_lookup_entry(
              header->function_id, function_table, func_count);
          if (entry != nullptr &&
              entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL)
            func = reinterpret_cast<DeviceRPCFunction>(
                entry->handler.device_fn_ptr);
        }
        if (func != nullptr) {
          s_work.request = request;
          s_work.response = response_frame;
          s_work.request_id = request_id;
          s_work.ptp_timestamp = ptp_timestamp;
          break;
        }

        // Unhandled frame: answered here by thread 0 alone.  Published
        // unconditionally: tx_publish is also what returns the slot's receive
        // credit, so dropping one here would starve the transport a slot at a
        // time.
        write_response_header(static_cast<RPCResponse *>(response_frame), -1,
                              request_id, ptp_timestamp);
        cudaq_dev_tx_publish(session, response_frame);
        packet_count++;
      }
      s_work.func = func;
    }

    __syncthreads(); // (1) the descriptor is published to the block
    const UnifiedWork work = s_work;
    if (work.func == nullptr)
      break; // uniform: every thread read the same descriptor

    // Same two-pointer call as the host path (host_dispatcher.cpp): the handler
    // reads the request from `request`, writes its result after the
    // RPCResponse in `response` and sets response->result_len.  Every thread
    // of the block makes the call; thread 0's return value is the one
    // recorded.
    const int status = work.func(work.request, work.response, tx_stride_sz);

    __syncthreads(); // (2) every thread's handler writes are done

    if (threadIdx.x == 0) {
      write_response_header(static_cast<RPCResponse *>(work.response), status,
                            work.request_id, work.ptp_timestamp);
      cudaq_dev_tx_publish(session, work.response);
      packet_count++;
    }
  }

  if (threadIdx.x == 0)
    atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
}
