/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/graph_launch_engine.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <new>
#include <thread>

struct cudaq_dispatch_manager_t {
  int reserved = 0;
};

struct cudaq_dispatcher_t {
  cudaq_dispatcher_config_t config{};
  cudaq_ringbuffer_t ringbuffer{};
  cudaq_function_table_t table{};
  cudaq_dispatch_launch_fn_t launch_fn = nullptr;
  cudaq_unified_launch_fn_t unified_launch_fn = nullptr;
  void *transport_ctx = nullptr;
  volatile int *shutdown_flag = nullptr;
  uint64_t *stats = nullptr;
  cudaStream_t stream = nullptr;
  bool running = false;
  void **h_mailbox_bank = nullptr;
  // The unified loop's data-plane (HOST + UNIFIED path).  The public setter is
  // void* to keep the API transport-agnostic; stored typed here since this TU
  // includes bridge_interface.h.
  cudaq_cpu_dataplane_t *cpu_dataplane = nullptr;
  // Both HOST dispatch paths (ring and unified) run their loop on this thread
  // and drive GRAPH_LAUNCH work through `engine`, both owned here: created in
  // cudaq_dispatcher_start, torn down in stop/destroy.  `engine` is NULL for a
  // HOST_CALL-only table.
  std::thread host_thread;
  cudaq_graph_launch_engine_t *engine = nullptr;
};

// True for HOST + UNIFIED: the dispatcher runs cudaq_host_unified_loop on its
// own thread over the ring data-plane (HOST_CALL inline + GRAPH_LAUNCH via the
// engine).
static bool is_host_unified_dispatcher(const cudaq_dispatcher_t *dispatcher) {
  return dispatcher &&
         dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST &&
         dispatcher->config.kernel_type == CUDAQ_KERNEL_UNIFIED;
}

// True when the table has any GRAPH_LAUNCH entry.  Such entries need a GPU (the
// engine creates CUDA streams / graphs); a HOST_CALL-only table runs GPU-less
// on either HOST transport.
static bool table_has_graph_launch(const cudaq_function_table_t *table) {
  if (!table || !table->entries)
    return false;
  for (uint32_t i = 0; i < table->count; ++i)
    if (table->entries[i].dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH)
      return true;
  return false;
}

static bool is_valid_kernel_type(cudaq_kernel_type_t kernel_type) {
  switch (kernel_type) {
  case CUDAQ_KERNEL_REGULAR:
  case CUDAQ_KERNEL_COOPERATIVE:
  case CUDAQ_KERNEL_UNIFIED:
    return true;
  default:
    return false;
  }
}

static bool is_valid_dispatch_mode(cudaq_dispatch_mode_t dispatch_mode) {
  switch (dispatch_mode) {
  case CUDAQ_DISPATCH_DEVICE_CALL:
  case CUDAQ_DISPATCH_GRAPH_LAUNCH:
  case CUDAQ_DISPATCH_HOST_CALL:
    return true;
  default:
    return false;
  }
}

static cudaq_status_t validate_dispatcher(cudaq_dispatcher_t *dispatcher) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (!is_valid_kernel_type(dispatcher->config.kernel_type))
    return CUDAQ_ERR_INVALID_ARG;
  if (!dispatcher->shutdown_flag || !dispatcher->stats)
    return CUDAQ_ERR_INVALID_ARG;
  if (!dispatcher->table.entries || dispatcher->table.count == 0)
    return CUDAQ_ERR_INVALID_ARG;

  if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
    if (dispatcher->config.kernel_type == CUDAQ_KERNEL_UNIFIED) {
      if (!dispatcher->cpu_dataplane)
        return CUDAQ_ERR_INVALID_ARG;
      return CUDAQ_OK;
    }
    if (!dispatcher->ringbuffer.rx_flags_host ||
        !dispatcher->ringbuffer.tx_flags_host ||
        !dispatcher->ringbuffer.rx_data_host ||
        !dispatcher->ringbuffer.tx_data_host)
      return CUDAQ_ERR_INVALID_ARG;
    return CUDAQ_OK;
  }

  if (dispatcher->config.kernel_type == CUDAQ_KERNEL_UNIFIED) {
    if (!dispatcher->unified_launch_fn || !dispatcher->transport_ctx)
      return CUDAQ_ERR_INVALID_ARG;
  } else {
    if (!dispatcher->launch_fn)
      return CUDAQ_ERR_INVALID_ARG;
    if (!dispatcher->ringbuffer.rx_flags || !dispatcher->ringbuffer.tx_flags)
      return CUDAQ_ERR_INVALID_ARG;
    if (dispatcher->config.num_blocks == 0 ||
        dispatcher->config.threads_per_block == 0 ||
        dispatcher->config.num_slots == 0 || dispatcher->config.slot_size == 0)
      return CUDAQ_ERR_INVALID_ARG;
    if (!is_valid_dispatch_mode(dispatcher->config.dispatch_mode))
      return CUDAQ_ERR_INVALID_ARG;
  }
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatch_manager_create(cudaq_dispatch_manager_t **out_mgr) {
  if (!out_mgr)
    return CUDAQ_ERR_INVALID_ARG;
  auto *mgr = new (std::nothrow) cudaq_dispatch_manager_t();
  if (!mgr)
    return CUDAQ_ERR_INTERNAL;
  *out_mgr = mgr;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatch_manager_destroy(cudaq_dispatch_manager_t *mgr) {
  if (mgr)
    delete mgr;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_create(cudaq_dispatch_manager_t *,
                                       const cudaq_dispatcher_config_t *config,
                                       cudaq_dispatcher_t **out_dispatcher) {
  if (!config || !out_dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  auto *dispatcher = new (std::nothrow) cudaq_dispatcher_t();
  if (!dispatcher)
    return CUDAQ_ERR_INTERNAL;
  dispatcher->config = *config;
  *out_dispatcher = dispatcher;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_destroy(cudaq_dispatcher_t *dispatcher) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->running) {
    if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
      // `const_cast` drops the flag's `volatile` qualifier (`reinterpret_cast`
      // can't cast away cv-qualifiers) so it can be written as a plain atomic.
      if (dispatcher->shutdown_flag)
        reinterpret_cast<std::atomic<int> *>(
            const_cast<int *>(dispatcher->shutdown_flag))
            ->store(1, std::memory_order_relaxed);
      if (dispatcher->host_thread.joinable())
        dispatcher->host_thread.join();
      cudaq_graph_launch_engine_destroy(dispatcher->engine);
      dispatcher->engine = nullptr;
    }
    dispatcher->running = false;
  }
  delete dispatcher;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_ringbuffer(cudaq_dispatcher_t *dispatcher,
                                const cudaq_ringbuffer_t *ringbuffer) {
  if (!dispatcher || !ringbuffer)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->ringbuffer = *ringbuffer;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_function_table(cudaq_dispatcher_t *dispatcher,
                                    const cudaq_function_table_t *table) {
  if (!dispatcher || !table)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->table = *table;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_set_control(cudaq_dispatcher_t *dispatcher,
                                            volatile int *shutdown_flag,
                                            uint64_t *stats) {
  if (!dispatcher || !shutdown_flag || !stats)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->shutdown_flag = shutdown_flag;
  dispatcher->stats = stats;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_launch_fn(cudaq_dispatcher_t *dispatcher,
                               cudaq_dispatch_launch_fn_t launch_fn) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST &&
      launch_fn != nullptr)
    return CUDAQ_ERR_INVALID_ARG;
  if (dispatcher->config.dispatch_path != CUDAQ_DISPATCH_PATH_HOST &&
      !launch_fn)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->launch_fn = launch_fn;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_set_mailbox(cudaq_dispatcher_t *dispatcher,
                                            void **h_mailbox_bank) {
  if (!dispatcher || !h_mailbox_bank)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->h_mailbox_bank = h_mailbox_bank;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_unified_launch(cudaq_dispatcher_t *dispatcher,
                                    cudaq_unified_launch_fn_t unified_launch_fn,
                                    void *transport_ctx) {
  if (!dispatcher || !unified_launch_fn || !transport_ctx)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->unified_launch_fn = unified_launch_fn;
  dispatcher->transport_ctx = transport_ctx;
  return CUDAQ_OK;
}

cudaq_status_t
cudaq_dispatcher_set_cpu_dataplane(cudaq_dispatcher_t *dispatcher,
                                   void *cpu_dataplane) {
  if (!dispatcher || !cpu_dataplane)
    return CUDAQ_ERR_INVALID_ARG;
  dispatcher->cpu_dataplane =
      static_cast<cudaq_cpu_dataplane_t *>(cpu_dataplane);
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_start(cudaq_dispatcher_t *dispatcher) {
  cudaq_status_t status = validate_dispatcher(dispatcher);
  if (status != CUDAQ_OK)
    return status;
  if (dispatcher->running)
    return CUDAQ_OK;

  if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
    const bool unified = is_host_unified_dispatcher(dispatcher);

    // Check if the table has any GRAPH_LAUNCH entries. If so, we need to make
    // sure the device is set and create a graph launch engine.
    if (table_has_graph_launch(&dispatcher->table)) {
      int device_id = dispatcher->config.device_id;
      if (device_id < 0)
        device_id = 0;
      if (cudaSetDevice(device_id) != cudaSuccess)
        return CUDAQ_ERR_CUDA;

      const cudaq_ringbuffer_t *engine_ring =
          unified ? &dispatcher->cpu_dataplane->ring : &dispatcher->ringbuffer;

      // Unified must keep the markers: its publish_ready reads the per-slot
      // CUDAQ_TX_FLAG_IN_FLIGHT sentinel to tell a still-running graph from a
      // completed one.  The ring path may skip them (config), since it recycles
      // workers via cudaStreamQuery and its transport thread owns TX.
      const int skip_tx_markers =
          unified ? 0 : dispatcher->config.skip_tx_markers;

      cudaq_status_t engine_st = CUDAQ_OK;
      dispatcher->engine = cudaq_graph_launch_engine_create(
          engine_ring, &dispatcher->table, skip_tx_markers,
          dispatcher->h_mailbox_bank, &engine_st);
      if (engine_st != CUDAQ_OK)
        return CUDAQ_ERR_INTERNAL;
    }

    cudaq_graph_launch_engine_t *engine = dispatcher->engine;
    volatile int *shutdown_flag = dispatcher->shutdown_flag;
    uint64_t *stats = dispatcher->stats;
    cudaq_function_table_t tbl = dispatcher->table;
    if (unified) {
      cudaq_cpu_dataplane_t *cpu_dataplane = dispatcher->cpu_dataplane;
      dispatcher->host_thread =
          std::thread([engine, cpu_dataplane, tbl, shutdown_flag, stats] {
            cudaq_host_unified_loop(cpu_dataplane, &tbl, engine, shutdown_flag,
                                    stats);
          });
    } else {
      cudaq_ringbuffer_t rb = dispatcher->ringbuffer;
      cudaq_dispatcher_config_t cfg = dispatcher->config;
      dispatcher->host_thread =
          std::thread([rb, tbl, cfg, engine, shutdown_flag, stats] {
            cudaq_host_ring_dispatch_loop(&rb, &tbl, &cfg, engine,
                                          shutdown_flag, stats);
          });
    }
    dispatcher->running = true;
    return CUDAQ_OK;
  }

  int device_id = dispatcher->config.device_id;
  if (device_id < 0)
    device_id = 0;
  // The HOST dispatch path may run without a usable CUDA device: HOST_CALL
  // entries never touch the device, and GRAPH_LAUNCH workers fail at their
  // own CUDA calls (cudaStreamCreate) when the device is truly needed. Only
  // the no-device error class is tolerated, though: if devices are
  // enumerable, a cudaSetDevice failure means `device_id` names a bad device
  // (e.g. cudaErrorInvalidDevice), and swallowing it would leave host-side
  // GRAPH_LAUNCH workers silently running on the default device instead of
  // the configured one.
  if (cudaSetDevice(device_id) != cudaSuccess) {
    if (dispatcher->config.dispatch_path != CUDAQ_DISPATCH_PATH_HOST)
      return CUDAQ_ERR_CUDA;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0)
      return CUDAQ_ERR_CUDA;
    (void)cudaGetLastError();
  }

  if (cudaStreamCreate(&dispatcher->stream) != cudaSuccess)
    return CUDAQ_ERR_CUDA;

  // NOTE on config.shared_ring_mode for DEVICE_LOOP:
  //
  // The device dispatch kernel reads shared_ring_mode from a __constant__
  // symbol that lives in libcudaq-realtime-dispatch.a (the static lib).
  // libcudaq-realtime.so does NOT link the static lib (architecturally
  // separate: consumers link the static lib themselves), so we cannot
  // call cudaq_dispatch_kernel_set_shared_ring_mode() from here.
  //
  // Callers that want shared_ring_mode for DEVICE_LOOP must invoke
  // cudaq_dispatch_kernel_set_shared_ring_mode(1) themselves BEFORE
  // cudaq_dispatcher_start().  The HOST_LOOP path reads
  // config.shared_ring_mode directly from this struct (it has no
  // __constant__ indirection) -- nothing needed here.

  if (dispatcher->config.kernel_type == CUDAQ_KERNEL_UNIFIED) {
    dispatcher->unified_launch_fn(
        dispatcher->transport_ctx, dispatcher->table.entries,
        dispatcher->table.count, dispatcher->shutdown_flag, dispatcher->stats,
        dispatcher->stream);
  } else {
    dispatcher->launch_fn(
        dispatcher->ringbuffer.rx_flags, dispatcher->ringbuffer.tx_flags,
        dispatcher->ringbuffer.rx_data, dispatcher->ringbuffer.tx_data,
        dispatcher->ringbuffer.rx_stride_sz,
        dispatcher->ringbuffer.tx_stride_sz, dispatcher->table.entries,
        dispatcher->table.count, dispatcher->shutdown_flag, dispatcher->stats,
        dispatcher->config.num_slots, dispatcher->config.num_blocks,
        dispatcher->config.threads_per_block, dispatcher->stream);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error in dispatcher launch: %s (%d)\n",
            cudaGetErrorString(err), err);
    cudaStreamDestroy(dispatcher->stream);
    dispatcher->stream = nullptr;
    return CUDAQ_ERR_CUDA;
  }

  dispatcher->running = true;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_stop(cudaq_dispatcher_t *dispatcher) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  if (!dispatcher->running)
    return CUDAQ_OK;

  if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
    // Both HOST paths (ring + unified) run on host_thread and own `engine`.
    // `const_cast` drops the flag's `volatile` qualifier (`reinterpret_cast`
    // can't cast away cv-qualifiers) so it can be written as a plain atomic.
    if (dispatcher->shutdown_flag)
      reinterpret_cast<std::atomic<int> *>(
          const_cast<int *>(dispatcher->shutdown_flag))
          ->store(1, std::memory_order_relaxed);
    if (dispatcher->host_thread.joinable())
      dispatcher->host_thread.join();
    cudaq_graph_launch_engine_destroy(dispatcher->engine);
    dispatcher->engine = nullptr;
    dispatcher->running = false;
    return CUDAQ_OK;
  }

  int shutdown = 1;
  if (cudaMemcpy(const_cast<int *>(dispatcher->shutdown_flag), &shutdown,
                 sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)
    return CUDAQ_ERR_CUDA;
  cudaStreamSynchronize(dispatcher->stream);
  cudaStreamDestroy(dispatcher->stream);
  dispatcher->stream = nullptr;
  dispatcher->running = false;
  return CUDAQ_OK;
}

cudaq_status_t cudaq_dispatcher_get_processed(cudaq_dispatcher_t *dispatcher,
                                              uint64_t *out_packets) {
  if (!dispatcher || !out_packets || !dispatcher->stats)
    return CUDAQ_ERR_INVALID_ARG;

  if (dispatcher->config.dispatch_path == CUDAQ_DISPATCH_PATH_HOST) {
    *out_packets = *dispatcher->stats;
    return CUDAQ_OK;
  }

  if (cudaMemcpy(out_packets, dispatcher->stats, sizeof(uint64_t),
                 cudaMemcpyDeviceToHost) != cudaSuccess)
    return CUDAQ_ERR_CUDA;

  return CUDAQ_OK;
}

//==============================================================================
// Ring buffer slot helpers
//==============================================================================

static inline uint64_t atomic_load_u64(volatile uint64_t *ptr) {
  auto *ap =
      reinterpret_cast<std::atomic<uint64_t> *>(const_cast<uint64_t *>(ptr));
  return ap->load(std::memory_order_acquire);
}

static inline void atomic_store_u64(volatile uint64_t *ptr, uint64_t val) {
  auto *ap =
      reinterpret_cast<std::atomic<uint64_t> *>(const_cast<uint64_t *>(ptr));
  ap->store(val, std::memory_order_release);
}

cudaq_status_t cudaq_host_ringbuffer_write_rpc_request(
    const cudaq_ringbuffer_t *rb, uint32_t slot_idx, uint32_t function_id,
    const void *payload, uint32_t payload_len, uint32_t request_id,
    uint64_t ptp_timestamp) {
  if (!rb || !rb->rx_data_host)
    return CUDAQ_ERR_INVALID_ARG;
  if (CUDAQ_RPC_HEADER_SIZE + payload_len > rb->rx_stride_sz)
    return CUDAQ_ERR_INVALID_ARG;

  uint8_t *slot = rb->rx_data_host + slot_idx * rb->rx_stride_sz;
  uint32_t *hdr32 = reinterpret_cast<uint32_t *>(slot);
  hdr32[0] = CUDAQ_RPC_MAGIC_REQUEST;
  hdr32[1] = function_id;
  hdr32[2] = payload_len;
  hdr32[3] = request_id;
  uint64_t *hdr64 = reinterpret_cast<uint64_t *>(slot + 16);
  *hdr64 = ptp_timestamp;

  if (payload && payload_len > 0)
    std::memcpy(slot + CUDAQ_RPC_HEADER_SIZE, payload, payload_len);

  return CUDAQ_OK;
}

void cudaq_host_ringbuffer_signal_slot(const cudaq_ringbuffer_t *rb,
                                       uint32_t slot_idx) {
  uint64_t addr = reinterpret_cast<uint64_t>(rb->rx_data_host +
                                             slot_idx * rb->rx_stride_sz);
  atomic_store_u64(&rb->rx_flags_host[slot_idx], addr);
}

cudaq_tx_status_t
cudaq_host_ringbuffer_poll_tx_flag(const cudaq_ringbuffer_t *rb,
                                   uint32_t slot_idx, int *out_cuda_error) {
  uint64_t v = atomic_load_u64(&rb->tx_flags_host[slot_idx]);
  if (v == 0)
    return CUDAQ_TX_EMPTY;
  if (v == CUDAQ_TX_FLAG_IN_FLIGHT)
    return CUDAQ_TX_IN_FLIGHT;
  if ((v >> 48) == CUDAQ_TX_FLAG_ERROR_TAG) {
    if (out_cuda_error)
      *out_cuda_error = static_cast<int>(v & 0xFFFF);
    return CUDAQ_TX_ERROR;
  }
  return CUDAQ_TX_READY;
}

int cudaq_host_ringbuffer_slot_available(const cudaq_ringbuffer_t *rb,
                                         uint32_t slot_idx) {
  return atomic_load_u64(&rb->rx_flags_host[slot_idx]) == 0 &&
         atomic_load_u64(&rb->tx_flags_host[slot_idx]) == 0;
}

void cudaq_host_ringbuffer_clear_slot(const cudaq_ringbuffer_t *rb,
                                      uint32_t slot_idx) {
  atomic_store_u64(&rb->tx_flags_host[slot_idx], 0);
}

cudaq_status_t cudaq_host_release_worker(cudaq_dispatcher_t *dispatcher,
                                         int worker_id) {
  if (!dispatcher)
    return CUDAQ_ERR_INVALID_ARG;
  // External worker release is a 3-thread ring-path feature; the unified path
  // recycles workers internally via publish_ready.
  if (dispatcher->config.dispatch_path != CUDAQ_DISPATCH_PATH_HOST ||
      is_host_unified_dispatcher(dispatcher) || !dispatcher->engine)
    return CUDAQ_ERR_INVALID_ARG;
  return cudaq_graph_launch_engine_release_worker(dispatcher->engine,
                                                  worker_id);
}
