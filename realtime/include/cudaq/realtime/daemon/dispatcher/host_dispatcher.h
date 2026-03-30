/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifndef CUDAQ_REALTIME_CPU_RELAX
#if defined(__x86_64__)
#include <immintrin.h>
#define CUDAQ_REALTIME_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define CUDAQ_REALTIME_CPU_RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define CUDAQ_REALTIME_CPU_RELAX()                                             \
  do {                                                                         \
  } while (0)
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  cudaGraphExec_t graph_exec;
  cudaStream_t stream;
  uint32_t function_id;
  void (*pre_launch_fn)(void *user_data, void *slot_dev, cudaStream_t stream);
  void *pre_launch_data;
  void (*post_launch_fn)(void *user_data, void *slot_dev, cudaStream_t stream);
  void *post_launch_data;
} cudaq_host_dispatch_worker_t;

typedef struct {
  // Composed public API structs
  cudaq_ringbuffer_t ringbuffer;
  cudaq_dispatcher_config_t config;
  cudaq_function_table_t function_table;

  // Host dispatch runtime state
  cudaq_host_dispatch_worker_t *workers;
  size_t num_workers;
  void **h_mailbox_bank;
  void *shutdown_flag; ///< opaque cuda::std::atomic<int>*
  uint64_t *stats_counter;
  void *live_dispatched; ///< opaque cuda::std::atomic<uint64_t>*
  void *idle_mask;       ///< opaque cuda::std::atomic<uint64_t>*, 1=free 0=busy
  int *inflight_slot_tags; ///< worker_id -> origin FPGA slot for tx_flags
                           ///< routing

  /// Per-worker GraphIOContext array for separate RX/TX buffer support.
  /// When non-NULL, launch_graph_worker fills a GraphIOContext per dispatch
  /// and writes its device address into h_mailbox_bank[worker_id].
  /// When NULL, legacy mode: raw RX slot pointer written to mailbox.
  void *io_ctxs_host; ///< host view of GraphIOContext[num_workers]
  void *io_ctxs_dev;  ///< device view of same pinned mapped memory
} cudaq_host_dispatch_loop_ctx_t;

/// Run the host-side dispatcher loop. Blocks until `*config->shutdown_flag`
/// becomes non-zero. Call from a dedicated thread.
/// Uses dynamic worker pool: allocates via idle_mask, tags with
/// inflight_slot_tags.
void cudaq_host_dispatcher_loop(const cudaq_host_dispatch_loop_ctx_t *config);

#ifdef __cplusplus
}
#endif
