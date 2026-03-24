/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include <cuda/std/atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>

using atomic_uint64_sys = cuda::std::atomic<uint64_t>;
using atomic_int_sys = cuda::std::atomic<int>;

struct cudaq_host_dispatcher_handle {
  std::thread thread;
  cudaq_host_dispatch_worker_t *workers = nullptr;
  size_t num_workers = 0;
  atomic_uint64_sys *idle_mask = nullptr;
  int *inflight_slot_tags = nullptr;
  void **h_mailbox_bank = nullptr;
  bool owns_mailbox = false;
  void *io_ctxs_pinned = nullptr;
};

static void free_handle(cudaq_host_dispatcher_handle *handle) {
  if (!handle)
    return;
  delete[] handle->workers;
  delete handle->idle_mask;
  delete[] handle->inflight_slot_tags;
  if (handle->owns_mailbox)
    delete[] handle->h_mailbox_bank;
  if (handle->io_ctxs_pinned)
    cudaFreeHost(handle->io_ctxs_pinned);
  delete handle;
}

static size_t count_graph_launch_workers(const cudaq_function_table_t *table) {
  size_t n = 0;
  for (uint32_t i = 0; i < table->count; ++i) {
    if (table->entries[i].dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH)
      ++n;
  }
  return n;
}

extern "C" cudaq_host_dispatcher_handle_t *cudaq_host_dispatcher_start_thread(
    const cudaq_ringbuffer_t *ringbuffer, const cudaq_function_table_t *table,
    const cudaq_dispatcher_config_t *config, volatile int *shutdown_flag,
    uint64_t *stats, void **external_mailbox) {
  if (!ringbuffer || !table || !config || !shutdown_flag || !stats)
    return nullptr;
  if (!ringbuffer->rx_flags_host || !ringbuffer->tx_flags_host ||
      !ringbuffer->rx_data_host || !ringbuffer->tx_data_host)
    return nullptr;
  if (!table->entries || table->count == 0)
    return nullptr;
  if (config->num_slots == 0 || config->slot_size == 0)
    return nullptr;

  const size_t num_workers = count_graph_launch_workers(table);
  if (num_workers == 0)
    return nullptr;

  auto *handle = new (std::nothrow) cudaq_host_dispatcher_handle();
  if (!handle)
    return nullptr;

  handle->workers = new (std::nothrow) cudaq_host_dispatch_worker_t[num_workers];
  handle->idle_mask = new (std::nothrow) atomic_uint64_sys(0);
  handle->inflight_slot_tags = new (std::nothrow) int[num_workers];
  if (external_mailbox) {
    handle->h_mailbox_bank = external_mailbox;
    handle->owns_mailbox = false;
  } else {
    handle->h_mailbox_bank = new (std::nothrow) void *[num_workers];
    handle->owns_mailbox = true;
  }
  if (!handle->workers || !handle->idle_mask || !handle->inflight_slot_tags ||
      !handle->h_mailbox_bank) {
    free_handle(handle);
    return nullptr;
  }

  std::memset(handle->inflight_slot_tags, 0, num_workers * sizeof(int));
  std::memset(handle->workers, 0,
              num_workers * sizeof(cudaq_host_dispatch_worker_t));

  size_t worker_idx = 0;
  for (uint32_t i = 0; i < table->count; ++i) {
    if (table->entries[i].dispatch_mode != CUDAQ_DISPATCH_GRAPH_LAUNCH)
      continue;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      for (size_t j = 0; j < worker_idx; ++j)
        cudaStreamDestroy(handle->workers[j].stream);
      free_handle(handle);
      return nullptr;
    }
    handle->workers[worker_idx].graph_exec =
        table->entries[i].handler.graph_exec;
    handle->workers[worker_idx].stream = stream;
    handle->workers[worker_idx].function_id = table->entries[i].function_id;
    handle->workers[worker_idx].pre_launch_fn = nullptr;
    handle->workers[worker_idx].pre_launch_data = nullptr;
    handle->workers[worker_idx].post_launch_fn = nullptr;
    handle->workers[worker_idx].post_launch_data = nullptr;
    worker_idx++;
  }
  handle->num_workers = num_workers;

  handle->idle_mask->store((1ULL << num_workers) - 1,
                           cuda::std::memory_order_release);

  // Allocate per-worker GraphIOContext array only when the caller wired
  // separate RX and TX data buffers.  When rx_data == tx_data (in-place),
  // the legacy path writes a raw slot pointer into the mailbox instead.
  void *io_ctxs_host_ptr = nullptr;
  void *io_ctxs_dev_ptr = nullptr;
  if (ringbuffer->rx_data != ringbuffer->tx_data) {
    size_t io_ctxs_bytes =
        num_workers * sizeof(cudaq::realtime::GraphIOContext);
    if (cudaHostAlloc(&io_ctxs_host_ptr, io_ctxs_bytes,
                      cudaHostAllocMapped) != cudaSuccess) {
      for (size_t j = 0; j < worker_idx; ++j)
        cudaStreamDestroy(handle->workers[j].stream);
      free_handle(handle);
      return nullptr;
    }
    std::memset(io_ctxs_host_ptr, 0, io_ctxs_bytes);
    if (cudaHostGetDevicePointer(&io_ctxs_dev_ptr, io_ctxs_host_ptr, 0) !=
        cudaSuccess) {
      cudaFreeHost(io_ctxs_host_ptr);
      for (size_t j = 0; j < worker_idx; ++j)
        cudaStreamDestroy(handle->workers[j].stream);
      free_handle(handle);
      return nullptr;
    }
  }
  handle->io_ctxs_pinned = io_ctxs_host_ptr;

  cudaq_host_dispatch_loop_ctx_t host_config;
  std::memset(&host_config, 0, sizeof(host_config));
  host_config.rx_flags = (void *)(uintptr_t)ringbuffer->rx_flags_host;
  host_config.tx_flags = (void *)(uintptr_t)ringbuffer->tx_flags_host;
  host_config.rx_data_host = ringbuffer->rx_data_host;
  host_config.rx_data_dev = ringbuffer->rx_data;
  host_config.tx_data_host = ringbuffer->tx_data_host;
  host_config.tx_data_dev = ringbuffer->tx_data;
  host_config.tx_stride_sz = ringbuffer->tx_stride_sz;
  host_config.h_mailbox_bank = handle->h_mailbox_bank;
  host_config.num_slots = config->num_slots;
  host_config.slot_size = config->slot_size;
  host_config.workers = handle->workers;
  host_config.num_workers = num_workers;
  host_config.function_table = table->entries;
  host_config.function_table_count = table->count;
  // The C API takes volatile int* for ABI stability; internally the dispatch
  // loop accesses it via cuda::std::atomic<int>* for acquire semantics.
  // This is safe: cuda::std::atomic<int> is lock-free and layout-compatible
  // with int on all CUDA-supported platforms.
  host_config.shutdown_flag = (void *)(uintptr_t)shutdown_flag;
  host_config.stats_counter = stats;
  host_config.live_dispatched = nullptr;
  host_config.idle_mask = handle->idle_mask;
  host_config.inflight_slot_tags = handle->inflight_slot_tags;
  host_config.tx_flags_dev = ringbuffer->tx_flags;
  host_config.io_ctxs_host = io_ctxs_host_ptr;
  host_config.io_ctxs_dev = io_ctxs_dev_ptr;

  handle->thread = std::thread(
      [cfg = host_config]() { cudaq_host_dispatcher_loop(&cfg); });
  return handle;
}

extern "C" cudaq_status_t
cudaq_host_dispatcher_release_worker(cudaq_host_dispatcher_handle_t *handle,
                                     int worker_id) {
  if (!handle || !handle->idle_mask)
    return CUDAQ_ERR_INVALID_ARG;
  if (worker_id < 0 || static_cast<size_t>(worker_id) >= handle->num_workers)
    return CUDAQ_ERR_INVALID_ARG;
  handle->idle_mask->fetch_or(1ULL << worker_id,
                              cuda::std::memory_order_release);
  return CUDAQ_OK;
}

extern "C" void
cudaq_host_dispatcher_stop(cudaq_host_dispatcher_handle_t *handle) {
  if (!handle)
    return;
  if (handle->thread.joinable())
    handle->thread.join();
  for (size_t i = 0; i < handle->num_workers; ++i)
    cudaStreamDestroy(handle->workers[i].stream);
  free_handle(handle);
}
