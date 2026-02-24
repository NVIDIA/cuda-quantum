/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

struct cudaq_host_dispatcher_handle {
  std::thread thread;
  std::vector<cudaq::realtime::HostDispatchWorker> workers;
  cudaq::realtime::atomic_uint64_sys* idle_mask = nullptr;
  int* inflight_slot_tags = nullptr;
  void** h_mailbox_bank = nullptr;
  size_t num_workers = 0;
};

static size_t count_graph_launch_workers(const cudaq_function_table_t* table) {
  size_t n = 0;
  for (uint32_t i = 0; i < table->count; ++i) {
    if (table->entries[i].dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH)
      ++n;
  }
  return n;
}

extern "C" cudaq_host_dispatcher_handle_t* cudaq_host_dispatcher_start_thread(
    const cudaq_ringbuffer_t* ringbuffer,
    const cudaq_function_table_t* table,
    const cudaq_dispatcher_config_t* config,
    volatile int* shutdown_flag,
    uint64_t* stats) {
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

  auto* handle = new (std::nothrow) cudaq_host_dispatcher_handle();
  if (!handle)
    return nullptr;

  handle->idle_mask = new (std::nothrow) cudaq::realtime::atomic_uint64_sys(0);
  handle->inflight_slot_tags = new (std::nothrow) int[num_workers];
  handle->h_mailbox_bank = new (std::nothrow) void*[num_workers];
  if (!handle->idle_mask || !handle->inflight_slot_tags || !handle->h_mailbox_bank) {
    delete handle->idle_mask;
    delete[] handle->inflight_slot_tags;
    delete[] handle->h_mailbox_bank;
    delete handle;
    return nullptr;
  }

  std::memset(handle->inflight_slot_tags, 0, num_workers * sizeof(int));

  handle->workers.reserve(num_workers);
  for (uint32_t i = 0; i < table->count; ++i) {
    if (table->entries[i].dispatch_mode != CUDAQ_DISPATCH_GRAPH_LAUNCH)
      continue;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      for (auto& w : handle->workers)
        cudaStreamDestroy(w.stream);
      delete handle->idle_mask;
      delete[] handle->inflight_slot_tags;
      delete[] handle->h_mailbox_bank;
      delete handle;
      return nullptr;
    }
    cudaq::realtime::HostDispatchWorker w;
    w.graph_exec = table->entries[i].handler.graph_exec;
    w.stream = stream;
    w.function_id = table->entries[i].function_id;
    handle->workers.push_back(w);
  }
  handle->num_workers = num_workers;

  handle->idle_mask->store((1ULL << num_workers) - 1,
                           cuda::std::memory_order_release);

  cudaq::realtime::HostDispatcherConfig host_config;
  host_config.rx_flags =
      (cudaq::realtime::atomic_uint64_sys*)(uintptr_t)ringbuffer->rx_flags_host;
  host_config.tx_flags =
      (cudaq::realtime::atomic_uint64_sys*)(uintptr_t)ringbuffer->tx_flags_host;
  host_config.rx_data_host = ringbuffer->rx_data_host;
  host_config.rx_data_dev = ringbuffer->rx_data;
  host_config.tx_data_host = ringbuffer->tx_data_host;
  host_config.tx_data_dev = ringbuffer->tx_data;
  host_config.tx_stride_sz = ringbuffer->tx_stride_sz;
  host_config.h_mailbox_bank = handle->h_mailbox_bank;
  host_config.num_slots = config->num_slots;
  host_config.slot_size = config->slot_size;
  host_config.workers = handle->workers;
  host_config.function_table = table->entries;
  host_config.function_table_count = table->count;
  host_config.shutdown_flag =
      (cudaq::realtime::atomic_int_sys*)(uintptr_t)shutdown_flag;
  host_config.stats_counter = stats;
  host_config.live_dispatched = nullptr;
  host_config.idle_mask = handle->idle_mask;
  host_config.inflight_slot_tags = handle->inflight_slot_tags;

  handle->thread = std::thread(cudaq::realtime::host_dispatcher_loop, host_config);
  return handle;
}

extern "C" void cudaq_host_dispatcher_stop(cudaq_host_dispatcher_handle_t* handle) {
  if (!handle)
    return;
  if (handle->thread.joinable())
    handle->thread.join();
  for (auto& w : handle->workers)
    cudaStreamDestroy(w.stream);
  delete handle->idle_mask;
  delete[] handle->inflight_slot_tags;
  delete[] handle->h_mailbox_bank;
  delete handle;
}
