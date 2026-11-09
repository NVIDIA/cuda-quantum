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
// Thread 0 owns the transport session: each turn it claims one request and
// its response frame from the hooks and hands the turn to the block through a
// __shared__ descriptor; every thread runs the handler, if there is one, and
// thread 0 then writes the response header and publishes.  Frames nothing can
// be dispatched to (bad magic, unknown function, wrong mode) take the same
// turn without a handler.  One straight-line turn with a single publish keeps
// the frame pointers rx_poll returns, and what it told the compiler about
// them, all the way to tx_publish.  The wait, the wire transfer and all
// receive bookkeeping live behind the hooks, so what remains here is framing
// and dispatch.
//==============================================================================

namespace {

/// Work descriptor thread 0 hands to the block for one turn.  Written before
/// the first barrier of a turn, read by every thread between the two
/// barriers, written again only after the second.  `func == nullptr` means no
/// handler for this frame.
struct UnifiedWork {
  bool leave;
  DeviceRPCFunction func;
  void *request;
  void *response;
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
  // The hooks get the storage itself, never the pointer enter_kernel returns:
  // only a session pointer the compiler can see is this array lets it hold
  // the session in registers once the hooks are inlined.
  void *const session = session_storage;
  bool attached = false;
  std::uint64_t packet_count = 0;
  if (threadIdx.x == 0)
    attached = cudaq_dev_enter_kernel(session_storage, transport_ctx,
                                      shutdown_flag) != nullptr;
  // A refused attach (nullptr) makes the first turn publish "leave" below, so
  // the whole block exits without a separate broadcast.

  while (true) {
    // Thread 0's view of this turn, kept for the publish below rather than
    // reloaded from the shared descriptor, which would lose what rx_poll told
    // the compiler about the frame pointers (that they are global memory).
    void *response_frame;
    std::uint32_t request_id;
    std::uint64_t ptp_timestamp;
    if (threadIdx.x == 0) {
      // Set by rx_poll only, never initialized here: a null on the shutdown
      // path would reach the publish below as one more possible value, and
      // the frame pointer would again be generic.
      void *request;
      DeviceRPCFunction func = nullptr;
      const bool leave =
          !attached || cudaq_dev_rx_poll(session, &request, &response_frame) !=
                           CUDAQ_RX_DEV_READY;
      if (!leave) {
        s_work.request = request;
        s_work.response = response_frame;
        auto *header = static_cast<RPCHeader *>(request);
        // Read before anything is written: after the handler the compiler
        // has to assume the request may have changed.
        request_id = header->request_id;
        ptp_timestamp = header->ptp_timestamp;
        if (header->magic == RPC_MAGIC_REQUEST) {
          const cudaq_function_entry_t *entry = unified_lookup_entry(
              header->function_id, function_table, func_count);
          if (entry != nullptr &&
              entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL)
            func = reinterpret_cast<DeviceRPCFunction>(
                entry->handler.device_fn_ptr);
        }
      }
      s_work.leave = leave;
      s_work.func = func;
    }

    __syncthreads(); // (1) the descriptor is published to the block
    const UnifiedWork work = s_work;
    if (work.leave)
      break; // uniform: every thread read the same descriptor

    // Same two-pointer call as the host path (host_dispatcher.cpp): the handler
    // reads the request from `request`, writes its result after the
    // RPCResponse in `response` and sets response->result_len.  Every thread
    // of the block makes the call; thread 0's return value is the one
    // recorded.  A frame nothing handles is answered with -1.
    int status = -1;
    if (work.func != nullptr)
      status = work.func(work.request, work.response, tx_stride_sz);

    __syncthreads(); // (2) every thread's handler writes are done

    if (threadIdx.x == 0) {
      // Frames are global memory (see cudaq_dev_rx_poll).  What rx_poll told
      // the compiler does not reach here, past the barriers, and a generic
      // frame pointer makes the transport's send order every frame load after
      // its own shared-memory stores.
      __builtin_assume(__isGlobal(response_frame));
      // Published unconditionally, including for a frame nothing handled:
      // tx_publish is also what returns the slot's receive credit, so
      // dropping one here would starve the transport a slot at a time.
      write_response_header(static_cast<RPCResponse *>(response_frame), status,
                            request_id, ptp_timestamp);
      cudaq_dev_tx_publish(session, response_frame);
      packet_count++;
    }
  }

  if (threadIdx.x == 0)
    atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
}
