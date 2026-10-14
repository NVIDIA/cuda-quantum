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
/// Two variants share one kernel template, selected by the KernelType
/// parameter from kernel_types.h:
///   - RegularKernel (CUDAQ_KERNEL_UNIFIED): <<<1,1>>>, one thread does
///     everything.
///   - CooperativeKernel (CUDAQ_KERNEL_UNIFIED_COOPERATIVE): launched with
///     cudaLaunchCooperativeKernel over num_blocks x threads_per_block.
///     Thread 0 of block 0 alone drives the transport; every thread calls the
///     handler between two grid barriers, mirroring the ring kernel's
///     cooperative path in dispatch_kernel.cu.
///
/// Compiled into libcudaq-realtime-dispatch.a, together with whichever TU
/// defines the hooks: bridge/gpu_roce/gpu_roce_unified_transport.cu when that
/// transport is configured, otherwise dispatcher/unified_transport_none.cu.
/// The consumer device-links the archive with its DEVICE_CALL handlers, so the
/// kernel, the hooks and the handlers the function table points at share one
/// CUDA module.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/kernel_types.h"
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
// Cooperative variant: work descriptor the transport thread (block 0, thread
// 0) hands to the rest of the grid.  Device-global so every block can read it
// after the grid barrier that follows the write -- one live cooperative
// unified kernel per module, as with the ring kernel's d_* statics in
// dispatch_kernel.cu.  Reused every turn without a third barrier: all readers
// of turn i are done before barrier (2) of turn i, and the next write happens
// after it.
//==============================================================================

namespace {

struct UnifiedCoopWork {
  DeviceRPCFunction func; ///< nullptr: a frame the kernel cannot dispatch
  void *request;
  void *response;
  std::uint32_t request_id;
  std::uint64_t ptp_timestamp;
  int shutdown; ///< non-zero: the whole grid leaves the loop
};

__device__ UnifiedCoopWork g_unified_coop_work;

} // namespace

//==============================================================================
// Unified dispatch kernel.
//
// Each turn claims a request and its response frame from the transport, runs
// the handler from one into the other, and hands the response back.  The
// wait, the wire transfer and all receive bookkeeping live behind the hooks,
// so what remains here is framing and dispatch.
//==============================================================================

template <typename KernelType>
__global__ void
cudaq_unified_dispatch_kernel(void *transport_ctx, std::size_t tx_stride_sz,
                              volatile int *shutdown_flag,
                              cudaq_function_entry_t *function_table,
                              std::size_t func_count, std::uint64_t *stats) {
  if constexpr (KernelType::is_cooperative) {
    //==========================================================================
    // Cooperative path.  tid 0 is the only thread that touches the transport:
    // it attaches, claims each request and publishes each response.  In
    // between it hands the request to the grid through g_unified_coop_work
    // and two grid barriers:
    //   (1) after the descriptor is written -- every thread reads it and
    //       calls the handler (which may grid.sync() itself);
    //   (2) after the handler -- tid 0 writes the response header and
    //       publishes.
    // Shutdown and an attach refusal travel through the same descriptor, so
    // the grid leaves uniformly and no thread waits at a barrier the others
    // have left.  While tid 0 blocks in rx_poll the rest of the grid is
    // parked at barrier (1); see the THREADING note in
    // unified_device_transport.cuh for what that asks of the hooks.
    //==========================================================================
    const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

    alignas(CUDAQ_DEV_TRANSPORT_SESSION_ALIGN) unsigned char
        session_storage[CUDAQ_DEV_TRANSPORT_SESSION_BYTES];
    void *session = nullptr;
    if (tid == 0) {
      session = cudaq_dev_transport_attach(session_storage, transport_ctx,
                                           shutdown_flag);
      // Also clears whatever a previous launch left behind.
      g_unified_coop_work.shutdown = (session == nullptr) ? 1 : 0;
    }
    KernelType::sync();
    if (g_unified_coop_work.shutdown)
      return;

    std::uint64_t packet_count = 0;

    while (true) {
      if (tid == 0) {
        void *request = nullptr;
        void *response_frame = nullptr;
        if (cudaq_dev_rx_poll(session, &request, &response_frame) !=
            CUDAQ_RX_DEV_READY) {
          g_unified_coop_work.shutdown = 1;
        } else {
          auto *header = static_cast<RPCHeader *>(request);
          DeviceRPCFunction func = nullptr;
          if (header->magic == RPC_MAGIC_REQUEST) {
            const cudaq_function_entry_t *entry = unified_lookup_entry(
                header->function_id, function_table, func_count);
            if (entry != nullptr &&
                entry->dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL)
              func = reinterpret_cast<DeviceRPCFunction>(
                  entry->handler.device_fn_ptr);
          }
          g_unified_coop_work.func = func;
          g_unified_coop_work.request = request;
          g_unified_coop_work.response = response_frame;
          // Saved before the handler runs, as on the regular path below.
          g_unified_coop_work.request_id = header->request_id;
          g_unified_coop_work.ptp_timestamp = header->ptp_timestamp;
        }
      }

      // (1) Broadcast.  Every thread takes its own copy.
      KernelType::sync();
      const UnifiedCoopWork work = g_unified_coop_work;
      if (work.shutdown)
        break;

      // ALL threads call the handler -- the same two-pointer call as the
      // regular path.  A frame the kernel cannot dispatch gets status -1 and
      // is still answered below.
      int status = -1;
      if (work.func != nullptr)
        status = work.func(work.request, work.response, tx_stride_sz);

      // (2) The handler has finished everywhere; tid 0 answers.
      KernelType::sync();
      if (tid == 0) {
        auto *response = static_cast<RPCResponse *>(work.response);
        response->magic = RPC_MAGIC_RESPONSE;
        response->status = status;
        // The handler owns result_len; force it to 0 for unhandled frames and
        // failed handlers.
        if (status != 0)
          response->result_len = 0;
        response->request_id = work.request_id;
        response->ptp_timestamp = work.ptp_timestamp;
        // Mandatory for every claimed frame, bad magic included: tx_publish
        // is also what returns the slot's receive credit.
        cudaq_dev_tx_publish(session, work.response);
        packet_count++;
      }
    }

    if (tid == 0)
      atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
  } else {
    //==========================================================================
    // Regular path: single thread, single block.
    //==========================================================================
    alignas(CUDAQ_DEV_TRANSPORT_SESSION_ALIGN) unsigned char
        session_storage[CUDAQ_DEV_TRANSPORT_SESSION_BYTES];
    void *session = cudaq_dev_transport_attach(session_storage, transport_ctx,
                                               shutdown_flag);
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

      // Read before the handler runs: after its call the compiler has to
      // assume the request may have changed and reload it behind the response
      // stores.
      const std::uint32_t request_id = header->request_id;
      const std::uint64_t ptp_timestamp = header->ptp_timestamp;

      int status = -1;

      if (header->magic == RPC_MAGIC_REQUEST) {
        const cudaq_function_entry_t *entry = unified_lookup_entry(
            header->function_id, function_table, func_count);

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
      // mandatory, and an unwritten response frame would put whatever the
      // slot last held on the wire.
      response->magic = RPC_MAGIC_RESPONSE;
      response->status = status;
      // The handler owns result_len; force it to 0 for unhandled frames and
      // failed handlers.
      if (status != 0)
        response->result_len = 0;
      response->request_id = request_id;
      response->ptp_timestamp = ptp_timestamp;

      // Published unconditionally, including for a frame whose magic did not
      // match: tx_publish is also what returns the slot's receive credit, so
      // dropping one here would starve the transport a slot at a time.
      cudaq_dev_tx_publish(session, response_frame);

      packet_count++;
    }

    atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
  }
}

//==============================================================================
// Host launch wrappers
//==============================================================================

// Regular variant; matches cudaq_unified_launch_fn_t.
extern "C" void cudaq_launch_unified_dispatch_device(
    void *transport_ctx, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, cudaStream_t stream) {
  cudaq_unified_dispatch_kernel<RegularKernel>
      <<<1, 1, 0, stream>>>(transport_ctx, tx_stride_sz, shutdown_flag,
                            function_table, func_count, stats);
}

// Cooperative variant (CUDAQ_KERNEL_UNIFIED_COOPERATIVE).  Returns the launch
// result -- including cudaErrorCooperativeLaunchTooLarge for a grid that
// cannot be co-resident -- for cudaq_dispatcher_start to report: it runs in
// the consumer's CUDA runtime, whose cudaGetLastError() state the dispatcher
// library's own runtime cannot see.
extern "C" cudaError_t cudaq_launch_unified_dispatch_device_cooperative(
    void *transport_ctx, size_t tx_stride_sz,
    cudaq_function_entry_t *function_table, size_t func_count,
    volatile int *shutdown_flag, uint64_t *stats, uint32_t num_blocks,
    uint32_t threads_per_block, cudaStream_t stream) {
  void *kernel_args[] = {
      &transport_ctx,  &tx_stride_sz, const_cast<int **>(&shutdown_flag),
      &function_table, &func_count,   &stats};
  const cudaError_t err = cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(
          cudaq_unified_dispatch_kernel<CooperativeKernel>),
      dim3(num_blocks), dim3(threads_per_block), kernel_args, 0, stream);
  // The return value is the reporting channel: consume the pending last-error
  // here, in the runtime that recorded it, so it is not attributed to the
  // consumer's next CUDA call.
  if (err != cudaSuccess)
    (void)cudaGetLastError();
  return err;
}

extern "C" cudaError_t
cudaq_unified_dispatch_cooperative_query_occupancy(int *out_blocks,
                                                   uint32_t threads_per_block) {
  int num_blocks = 0;
  cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &num_blocks, cudaq_unified_dispatch_kernel<CooperativeKernel>,
      threads_per_block, 0);
  if (err != cudaSuccess)
    num_blocks = 0; // documented: 0 on error
  if (out_blocks)
    *out_blocks = num_blocks;
  return err;
}
