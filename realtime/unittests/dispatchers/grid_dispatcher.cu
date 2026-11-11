/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file grid_dispatcher.cu
/// @brief An application's cooperative unified dispatcher: one resident block
/// per SM, with every grid thread running each DEVICE_CALL handler.
///
/// Exactly one fixed leader owns the transport hooks.  It hands each turn to
/// its own block through shared memory and to the other blocks through a
/// device-scope release store.  Thread 0 of every other block adds a release
/// increment to an arrival counter after its block finishes; the leader waits
/// for the counter, then writes and publishes the response.  There is no
/// grid-wide barrier on the request path.
///
/// cudaq_realtime_add_dispatch_kernel compiles this file separately to LTO-IR
/// and embeds it in gpu_roce_app.  The dispatcher links it at run time with the
/// gpu_roce hooks and the application's exported handler.

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/unified_device_transport.cuh"
#include "cudaq/realtime/daemon/dispatcher/unified_dispatch_device.cuh"
#include <cooperative_groups.h>
#include <cstddef>
#include <cstdint>
#include <cuda/atomic>

namespace cg = cooperative_groups;
using namespace cudaq::realtime;

namespace {

struct GridWork {
  bool leave;
  DeviceRPCFunction func;
  void *request;
  void *response;
};

/// A function table entry as every block keeps it in shared memory, so that a
/// block resolves a request's handler without reading the table.  `func` is
/// null for an entry that is not DEVICE_CALL.
struct HandlerSlot {
  std::uint32_t function_id;
  DeviceRPCFunction func;
};

/// A turn as four words, each holding the turn's number in its high half and
/// half of the request or response pointer in its low half.  A block that sees
/// every word carry the turn it waits for has the whole turn after a single
/// round trip, with no flag read before the data.
struct alignas(32) GridTurn {
  unsigned long long words[4];
};

__device__ GridTurn g_turn;
__device__ unsigned long long g_arrived;

using device_word =
    cuda::atomic_ref<unsigned long long, cuda::thread_scope_device>;

__device__ void publish_turn(void *request, void *response,
                             std::uint32_t turn) {
  const std::uint64_t values[2] = {reinterpret_cast<std::uintptr_t>(request),
                                   reinterpret_cast<std::uintptr_t>(response)};
  const std::uint64_t tag = static_cast<std::uint64_t>(turn) << 32;
  cuda::atomic_thread_fence(cuda::memory_order_release,
                            cuda::thread_scope_device);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const std::uint32_t half =
        static_cast<std::uint32_t>(values[i / 2] >> (i % 2 ? 32 : 0));
    device_word(g_turn.words[i]).store(tag | half, cuda::memory_order_relaxed);
  }
}

__device__ GridWork wait_turn(std::uint32_t turn) {
  unsigned long long words[4];
  bool ready;
  do {
#pragma unroll
    for (int i = 0; i < 3; ++i)
      words[i] = device_word(g_turn.words[i]).load(cuda::memory_order_relaxed);
    words[3] = device_word(g_turn.words[3]).load(cuda::memory_order_acquire);
    ready = true;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      ready &= static_cast<std::uint32_t>(words[i] >> 32) == turn;
  } while (!ready);
  GridWork work;
  work.request =
      reinterpret_cast<void *>((words[0] & 0xffffffffull) | (words[1] << 32));
  work.response =
      reinterpret_cast<void *>((words[2] & 0xffffffffull) | (words[3] << 32));
  work.func = nullptr;
  work.leave = work.request == nullptr;
  return work;
}

__device__ DeviceRPCFunction select_handler(
    const RPCHeader *header, const HandlerSlot *handlers, bool cached,
    cudaq_function_entry_t *function_table, std::size_t func_count) {
  const std::uint32_t magic = header->magic;
  const std::uint32_t function_id = header->function_id;
  if (magic != RPC_MAGIC_REQUEST)
    return nullptr;
  if (cached) {
    for (std::size_t i = 0; i < func_count; ++i)
      if (handlers[i].function_id == function_id)
        return handlers[i].func;
    return nullptr;
  }
  const cudaq_function_entry_t *entry =
      unified_lookup_entry(function_id, function_table, func_count);
  if (entry == nullptr || entry->dispatch_mode != CUDAQ_DISPATCH_DEVICE_CALL)
    return nullptr;
  return reinterpret_cast<DeviceRPCFunction>(entry->handler.device_fn_ptr);
}

__device__ std::uint32_t dynamic_shared_bytes() {
  std::uint32_t bytes;
  asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(bytes));
  return bytes;
}

__device__ unsigned long long load_acquire(unsigned long long *address) {
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> ref(*address);
  return ref.load(cuda::memory_order_acquire);
}

__device__ void arrive() {
  cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> ref(
      g_arrived);
  ref.fetch_add(1, cuda::memory_order_release);
}

__device__ std::uint32_t current_smid() {
  std::uint32_t smid;
  asm("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
}

} // namespace

extern "C" __global__ void
grid_dispatcher(void *transport_ctx, std::size_t tx_stride_sz,
                volatile int *shutdown_flag,
                cudaq_function_entry_t *function_table, std::size_t func_count,
                std::uint64_t *stats, std::uint32_t *sm_ids) {
  const bool leader = blockIdx.x == 0 && threadIdx.x == 0;
  cg::grid_group grid = cg::this_grid();
  __shared__ GridWork s_work;
  extern __shared__ HandlerSlot s_handlers[];

  alignas(CUDAQ_DEV_TRANSPORT_SESSION_ALIGN) unsigned char
      session_storage[CUDAQ_DEV_TRANSPORT_SESSION_BYTES];
  void *const session = session_storage;
  bool attached = false;
  std::uint64_t packet_count = 0;

  if (threadIdx.x == 0)
    sm_ids[blockIdx.x] = current_smid();
  const bool cached =
      func_count <= dynamic_shared_bytes() / sizeof(HandlerSlot);
  if (cached)
    for (std::size_t i = threadIdx.x; i < func_count; i += blockDim.x) {
      const cudaq_function_entry_t &entry = function_table[i];
      s_handlers[i] = {
          entry.function_id,
          entry.dispatch_mode == CUDAQ_DISPATCH_DEVICE_CALL
              ? reinterpret_cast<DeviceRPCFunction>(entry.handler.device_fn_ptr)
              : nullptr};
    }
  if (leader) {
    g_turn = GridTurn{};
    g_arrived = 0;
    attached = cudaq_dev_enter_kernel(session_storage, transport_ctx,
                                      shutdown_flag) != nullptr;
  }
  grid.sync();

  const unsigned long long others = gridDim.x - 1;
  unsigned long long last = 0;
  unsigned long long target = 0;
  while (true) {
    void *response_frame = nullptr;
    std::uint32_t request_id = 0;
    std::uint64_t ptp_timestamp = 0;
    if (leader) {
      void *request = nullptr;
      DeviceRPCFunction func = nullptr;
      const bool leave =
          !attached || cudaq_dev_rx_poll(session, &request, &response_frame) !=
                           CUDAQ_RX_DEV_READY;
      if (leave)
        request = nullptr;
      last++;
      if (others != 0)
        publish_turn(request, response_frame, static_cast<std::uint32_t>(last));
      if (!leave) {
        auto *header = static_cast<RPCHeader *>(request);
        request_id = header->request_id;
        ptp_timestamp = header->ptp_timestamp;
        func = select_handler(header, s_handlers, cached, function_table,
                              func_count);
      }
      s_work = GridWork{leave, func, request, response_frame};
    } else if (threadIdx.x == 0) {
      last++;
      GridWork work = wait_turn(static_cast<std::uint32_t>(last));
      if (!work.leave)
        work.func =
            select_handler(static_cast<const RPCHeader *>(work.request),
                           s_handlers, cached, function_table, func_count);
      s_work = work;
    }
    __syncthreads();

    const GridWork work = s_work;
    if (work.leave)
      break;

    __builtin_assume(__isGlobal(work.request));
    __builtin_assume(__isGlobal(work.response));
    int status = -1;
    if (work.func != nullptr)
      status = work.func(work.request, work.response, tx_stride_sz);

    __syncthreads();
    if (leader) {
      target += others;
      while (load_acquire(&g_arrived) < target)
        ;
      __builtin_assume(__isGlobal(response_frame));
      write_response_header(static_cast<RPCResponse *>(response_frame), status,
                            request_id, ptp_timestamp);
      cudaq_dev_tx_publish(session, response_frame);
      packet_count++;
    } else if (threadIdx.x == 0) {
      arrive();
    }
  }

  if (leader)
    atomicAdd(reinterpret_cast<unsigned long long *>(stats), packet_count);
}
