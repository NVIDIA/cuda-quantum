/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

/// @file graph_launch_engine.h
/// @brief CUDA-graph offload engine for the GRAPH_LAUNCH dispatch mode.
///
/// Owns the per-RPC graph worker pool: one stream and one pre-instantiated
/// `cudaGraphExec` per GRAPH_LAUNCH function-table entry, an idle bitmask, and
/// a pinned mailbox.  A host dispatch loop drives it per slot
/// (`acquire` -> `launch` -> `sweep`) over a device-visible ring
/// (`cudaq_ringbuffer_t`).  `cudaq_graph_launch_engine_create` returns NULL for
/// a table with zero GRAPH_LAUNCH entries (a HOST_CALL-only deployment needs no
/// engine).

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Maximum GRAPH_LAUNCH workers (one per function-table entry).  The idle pool
/// is tracked in a single `uint64_t` bitmask, so worker ids are limited to
/// [0, CUDAQ_GRAPH_LAUNCH_MAX_WORKERS).
#define CUDAQ_GRAPH_LAUNCH_MAX_WORKERS 64

/// One graph worker: a stream + the pre-instantiated graph for a GRAPH_LAUNCH
/// function-table entry, plus optional pre/post launch hooks.
typedef struct {
  cudaGraphExec_t graph_exec;
  cudaStream_t stream;
  uint32_t function_id;
  void (*pre_launch_fn)(void *user_data, void *slot_dev, cudaStream_t stream);
  void *pre_launch_data;
  void (*post_launch_fn)(void *user_data, void *slot_dev, cudaStream_t stream);
  void *post_launch_data;
  /// Optional sub-routing key for `function_id` collisions across workers.
  /// When several workers share the same `function_id` but back different
  /// captured graphs, the monitor uses (function_id, routing_key) to
  /// disambiguate.  The runtime routing key comes from the request
  /// payload's first 8 bytes (arg0); a worker matches only if both
  /// function_id and routing_key match.  Set to 0 when sub-routing isn't
  /// needed (the historical function_id-only match).
  uint64_t routing_key;
} cudaq_host_dispatch_worker_t;

/// The GRAPH_LAUNCH engine state.  Heap-owned via
/// `cudaq_graph_launch_engine_create`, which allocates streams / idle_mask /
/// mailbox / GraphIOContext array.
typedef struct {
  /// Device-visible ring the engine launches graphs over.  `launch` reads the
  /// device rx/tx data + tx flags and the tx stride from here.
  cudaq_ringbuffer_t ringbuffer;

  cudaq_host_dispatch_worker_t *workers;
  size_t num_workers;
  /// Precomputed at create: bit i set for valid worker ids [0, num_workers).
  uint64_t worker_mask;

  /// Opaque `cuda::std::atomic<uint64_t>*`; bit i == 1 means worker i is free.
  void *idle_mask;
  /// worker_id -> origin slot index it is currently servicing.
  int *inflight_slot_tags;
  /// Per-worker mailbox: `launch` writes the device GraphIOContext (or raw
  /// device slot pointer) here for the graph kernel to read.
  void **h_mailbox_bank;

  /// Per-worker `GraphIOContext` array (host + device views of one pinned
  /// mapped allocation) when RX and TX buffers are separate; NULL for the
  /// in-place (rx_data == tx_data) mode.
  void *io_ctxs_host;
  void *io_ctxs_dev;

  /// Mirror of `cudaq_dispatcher_config_t::skip_tx_markers`.
  int skip_tx_markers;
  /// 1 when this engine owns `h_mailbox_bank` (allocated it) and must free it.
  int owns_mailbox;
} cudaq_graph_launch_engine_t;

//===----------------------------------------------------------------------===//
// Lifecycle (heap-owned engine)
//===----------------------------------------------------------------------===//

/// Build a heap-owned engine for the GRAPH_LAUNCH entries in `table`.
///
/// Returns NULL when `table` has zero GRAPH_LAUNCH entries (a HOST_CALL only
/// deployment instantiates no engine) -- in that case `*out_status` is
/// CUDAQ_OK.  Returns NULL with `*out_status == CUDAQ_ERR_INVALID_ARG` when the
/// table has more than `CUDAQ_GRAPH_LAUNCH_MAX_WORKERS` GRAPH_LAUNCH entries.
/// Returns NULL with other non-OK `*out_status` on allocation / CUDA failure.
/// `out_status` may be NULL.
///
/// If `external_mailbox` is non-NULL it is used (caller-owned, must be pinned
/// mapped and sized >= num_graph_launch_entries * sizeof(void*)); otherwise the
/// engine allocates and owns its own.
///
/// `skip_tx_markers`: when non-zero, the engine does NOT write the
/// CUDAQ_TX_FLAG_IN_FLIGHT sentinel before a graph launch (set it when an
/// external consumer polls the same tx_flags, e.g. the Hololink TX kernel). The
/// single-thread unified loop needs the markers (its publish_ready
/// distinguishes in-flight from done), so it must pass 0.
cudaq_graph_launch_engine_t *cudaq_graph_launch_engine_create(
    const cudaq_ringbuffer_t *ringbuffer, const cudaq_function_table_t *table,
    int skip_tx_markers, void **external_mailbox, cudaq_status_t *out_status);

/// Destroy worker streams and free engine-owned resources.  Call after the
/// driving loop has stopped (and after `..._drain`).
void cudaq_graph_launch_engine_destroy(cudaq_graph_launch_engine_t *engine);

//===----------------------------------------------------------------------===//
// Per-dispatch mechanics
//===----------------------------------------------------------------------===//

/// Pick a free worker matched to `function_id` (and `routing_key`, the arg0
/// sub-filter for the multi-instance GRAPH_LAUNCH pattern).  Returns the worker
/// id, or -1 when none is free (backpressure).
int cudaq_graph_launch_engine_acquire(const cudaq_graph_launch_engine_t *engine,
                                      uint32_t function_id,
                                      uint64_t routing_key);

/// Fire the graph for `worker_id` on origin `current_slot`.  Fills the
/// GraphIOContext / mailbox from the engine's ring + slot, marks the worker
/// busy, optionally writes the IN_FLIGHT TX marker, and launches.  `slot_host`
/// is the host view of the RX slot (used to derive the device pointer).
void cudaq_graph_launch_engine_launch(const cudaq_graph_launch_engine_t *engine,
                                      int worker_id, void *slot_host,
                                      size_t current_slot);

/// Poll `cudaStreamQuery` on busy workers and return finished ones to the idle
/// pool (worker recycling).  No-op shape for callers that recycle externally.
void cudaq_graph_launch_engine_sweep(const cudaq_graph_launch_engine_t *engine);

/// Synchronize every worker stream (call once when the driving loop exits).
void cudaq_graph_launch_engine_drain(const cudaq_graph_launch_engine_t *engine);

/// Publish + recycle every worker whose graph has signaled completion via its
/// TX doorbell: `publish(ctx, slot)` runs for each completed slot (skipped for
/// launch-error completions) before its worker is recycled, so a finished
/// worker is never freed before its response is sent.  A single non-blocking
/// pass; slot-addressed, so responses may be published out of the order their
/// requests were launched.  TX-publishing counterpart to
/// cudaq_graph_launch_engine_sweep (which only recycles); the single-thread
/// unified loop uses this, while the ring loop uses sweep because its transport
/// threads own TX.
void cudaq_graph_launch_engine_publish_ready(
    const cudaq_graph_launch_engine_t *engine,
    cudaq_status_t (*publish)(void *ctx, uint32_t slot), void *ctx);

/// Return a worker to the idle pool (consumer-side recycle counterpart).
cudaq_status_t
cudaq_graph_launch_engine_release_worker(cudaq_graph_launch_engine_t *engine,
                                         int worker_id);

//===----------------------------------------------------------------------===//
// 3-thread ring dispatch loop
//===----------------------------------------------------------------------===//

/// Poll `ringbuffer.rx_flags_host`, run HOST_CALL entries inline, and offload
/// GRAPH_LAUNCH entries to `engine` (which may be NULL for a HOST_CALL only
/// table).  Blocks until `*shutdown_flag != 0`; meant to run on its own thread.
/// The bridge's RX/TX adapter threads move bytes wire<->ring around it.
void cudaq_host_ring_dispatch_loop(const cudaq_ringbuffer_t *ringbuffer,
                                   const cudaq_function_table_t *table,
                                   const cudaq_dispatcher_config_t *config,
                                   cudaq_graph_launch_engine_t *engine,
                                   volatile int *shutdown_flag,
                                   uint64_t *stats);

//===----------------------------------------------------------------------===//
// Single-thread unified dispatch loop
//===----------------------------------------------------------------------===//

/// Drive `cpu_dataplane` from one thread: poll rx_poll for a ready slot, run
/// its dispatch mode inline (HOST_CALL run inline, GRAPH_LAUNCH via `engine`),
/// then publish it with tx_publish.  `engine` is created and destroyed by the
/// caller (NULL for a HOST_CALL-only table).  Blocks until `*shutdown_flag !=
/// 0`.
void cudaq_host_unified_loop(cudaq_cpu_dataplane_t *cpu_dataplane,
                             const cudaq_function_table_t *table,
                             cudaq_graph_launch_engine_t *engine,
                             volatile int *shutdown_flag, uint64_t *stats);

#ifdef __cplusplus
}
#endif
