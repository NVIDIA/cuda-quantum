/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/graph_launch_engine.h"

#include <cstdint>
#include <new>
#include <thread>

// Lifecycle wrapper for the 3-thread ring shape: own a standalone GRAPH_LAUNCH
// engine (NULL for a HOST_CALL-only table) and run cudaq_host_ring_dispatch_loop
// on a dedicated thread.  The engine owns the worker streams / idle_mask /
// mailbox / GraphIOContext array, so the handle is just the thread + engine.
struct cudaq_host_dispatcher_handle {
  std::thread thread;
  cudaq_graph_launch_engine_t *engine = nullptr;
};

static bool has_host_dispatch_work(const cudaq_function_table_t *table) {
  for (uint32_t i = 0; i < table->count; ++i) {
    const cudaq_function_entry_t &entry = table->entries[i];
    if (entry.dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH)
      return true;
    if (entry.dispatch_mode == CUDAQ_DISPATCH_HOST_CALL &&
        entry.handler.host_fn)
      return true;
  }
  return false;
}

static void free_handle(cudaq_host_dispatcher_handle *handle) {
  if (!handle)
    return;
  if (handle->engine)
    cudaq_graph_launch_engine_destroy(handle->engine);
  delete handle;
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
  if (!has_host_dispatch_work(table))
    return nullptr;

  auto *handle = new (std::nothrow) cudaq_host_dispatcher_handle();
  if (!handle)
    return nullptr;

  // A HOST_CALL-only table yields a NULL engine (nothing graph-related is
  // instantiated); the loop then runs the inline HOST_CALL path only.  A
  // non-OK status is a hard create failure.
  cudaq_status_t st = CUDAQ_OK;
  handle->engine = cudaq_graph_launch_engine_create(ringbuffer, table, config,
                                                    external_mailbox, &st);
  if (st != CUDAQ_OK) {
    free_handle(handle);
    return nullptr;
  }

  cudaq_ringbuffer_t rb = *ringbuffer;
  cudaq_function_table_t tbl = *table;
  cudaq_dispatcher_config_t cfg = *config;
  cudaq_graph_launch_engine_t *engine = handle->engine;
  handle->thread = std::thread([rb, tbl, cfg, engine, shutdown_flag, stats]() {
    cudaq_host_ring_dispatch_loop(&rb, &tbl, &cfg, engine, shutdown_flag, stats);
  });
  return handle;
}

extern "C" cudaq_status_t
cudaq_host_dispatcher_release_worker(cudaq_host_dispatcher_handle_t *handle,
                                     int worker_id) {
  if (!handle || !handle->engine)
    return CUDAQ_ERR_INVALID_ARG;
  return cudaq_graph_launch_engine_release_worker(handle->engine, worker_id);
}

extern "C" void
cudaq_host_dispatcher_stop(cudaq_host_dispatcher_handle_t *handle) {
  if (!handle)
    return;
  if (handle->thread.joinable())
    handle->thread.join();
  free_handle(handle);
}
