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
  void *rx_flags; ///< opaque cuda::std::atomic<uint64_t>*
  void *tx_flags; ///< opaque cuda::std::atomic<uint64_t>*
  uint8_t *rx_data_host;
  uint8_t *rx_data_dev;
  uint8_t *tx_data_host;
  uint8_t *tx_data_dev;
  size_t tx_stride_sz;
  void **h_mailbox_bank;
  size_t num_slots;
  size_t slot_size;
  cudaq_host_dispatch_worker_t *workers;
  size_t num_workers;
  /// Host-visible function table for lookup by function_id (GRAPH_LAUNCH only;
  /// others dropped).
  cudaq_function_entry_t *function_table;
  size_t function_table_count;
  void *shutdown_flag; ///< opaque cuda::std::atomic<int>*
  uint64_t *stats_counter;
  void *live_dispatched; ///< opaque cuda::std::atomic<uint64_t>*
  void *idle_mask;       ///< opaque cuda::std::atomic<uint64_t>*, 1=free 0=busy
  int *inflight_slot_tags; ///< worker_id -> origin FPGA slot for tx_flags
                           ///< routing

  /// Device view of tx_flags (needed for GraphIOContext.tx_flag).
  /// NULL when tx_flags is already a device-accessible pointer.
  volatile uint64_t *tx_flags_dev;

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
