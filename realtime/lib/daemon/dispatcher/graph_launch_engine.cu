/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

/// @file graph_launch_engine.cu
/// @brief CUDA-graph offload engine for the GRAPH_LAUNCH dispatch mode.

#include "cudaq/realtime/daemon/dispatcher/graph_launch_engine.h"

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cpu_relax.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cstdint>
#include <cstring>
#include <cuda/std/atomic>
#include <new>

using atomic_uint64_sys = cuda::std::atomic<uint64_t>;

namespace {

inline atomic_uint64_sys *as_atomic_u64(void *p) {
  return static_cast<atomic_uint64_sys *>(p);
}
inline atomic_uint64_sys *as_atomic_u64(volatile uint64_t *p) {
  return reinterpret_cast<atomic_uint64_sys *>(const_cast<uint64_t *>(p));
}

size_t count_graph_launch_workers(const cudaq_function_table_t *table) {
  size_t n = 0;
  for (uint32_t i = 0; i < table->count; ++i) {
    if (table->entries[i].dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH)
      ++n;
  }
  return n;
}

uint64_t make_worker_mask(size_t num_workers) {
  return num_workers >= CUDAQ_GRAPH_LAUNCH_MAX_WORKERS
             ? ~0ULL
             : ((1ULL << num_workers) - 1);
}

} // namespace

extern "C" {

int cudaq_graph_launch_engine_acquire(const cudaq_graph_launch_engine_t *engine,
                                      uint32_t function_id,
                                      uint64_t routing_key) {
  uint64_t mask =
      as_atomic_u64(engine->idle_mask)->load(cuda::std::memory_order_acquire);
  while (mask != 0) {
    int worker_id = __builtin_ffsll(static_cast<long long>(mask)) - 1;
    const cudaq_host_dispatch_worker_t &w =
        engine->workers[static_cast<size_t>(worker_id)];
    // routing_key == 0 is a wildcard worker (no sub-routing); otherwise the
    // worker's key must equal the request's routing_key (arg0).
    if (w.function_id == function_id &&
        (w.routing_key == 0 || w.routing_key == routing_key))
      return worker_id;
    mask &= ~(1ULL << worker_id);
  }
  return -1;
}

void cudaq_graph_launch_engine_sweep(
    const cudaq_graph_launch_engine_t *engine) {
  uint64_t busy =
      ~as_atomic_u64(engine->idle_mask)->load(cuda::std::memory_order_acquire);
  busy &= engine->worker_mask;
  while (busy != 0) {
    int w = __builtin_ffsll(static_cast<long long>(busy)) - 1;
    if (cudaStreamQuery(engine->workers[w].stream) == cudaSuccess) {
      as_atomic_u64(engine->idle_mask)
          ->fetch_or(1ULL << w, cuda::std::memory_order_release);
    }
    busy &= ~(1ULL << w);
  }
}

void cudaq_graph_launch_engine_launch(const cudaq_graph_launch_engine_t *engine,
                                      int worker_id, void *slot_host,
                                      size_t current_slot) {
  using cudaq::realtime::GraphIOContext;

  as_atomic_u64(engine->idle_mask)
      ->fetch_and(~(1ULL << worker_id), cuda::std::memory_order_release);
  engine->inflight_slot_tags[worker_id] = static_cast<int>(current_slot);

  ptrdiff_t offset =
      static_cast<uint8_t *>(slot_host) - engine->ringbuffer.rx_data_host;
  void *data_dev = static_cast<void *>(engine->ringbuffer.rx_data + offset);

  if (engine->io_ctxs_host != nullptr) {
    auto *h_ctxs = static_cast<GraphIOContext *>(engine->io_ctxs_host);
    auto *d_ctxs = static_cast<uint8_t *>(engine->io_ctxs_dev);
    GraphIOContext *h_ctx = &h_ctxs[worker_id];

    h_ctx->rx_slot = data_dev;
    h_ctx->tx_slot = engine->ringbuffer.tx_data +
                     current_slot * engine->ringbuffer.tx_stride_sz;
    h_ctx->tx_flag = &engine->ringbuffer.tx_flags[current_slot];
    h_ctx->tx_flag_value = reinterpret_cast<uint64_t>(h_ctx->tx_slot);
    h_ctx->tx_stride_sz = engine->ringbuffer.tx_stride_sz;

    void *d_ctx = d_ctxs + worker_id * sizeof(GraphIOContext);
    engine->h_mailbox_bank[worker_id] = d_ctx;

    if (!engine->skip_tx_markers) {
      as_atomic_u64(engine->ringbuffer.tx_flags_host)[current_slot].store(
          CUDAQ_TX_FLAG_IN_FLIGHT, cuda::std::memory_order_release);
    }
    __sync_synchronize();
  } else {
    engine->h_mailbox_bank[worker_id] = data_dev;
  }
  __sync_synchronize();

  const size_t w = static_cast<size_t>(worker_id);
  if (engine->workers[w].pre_launch_fn)
    engine->workers[w].pre_launch_fn(engine->workers[w].pre_launch_data,
                                     data_dev, engine->workers[w].stream);
  cudaError_t err = cudaGraphLaunch(engine->workers[w].graph_exec,
                                    engine->workers[w].stream);

  if (err != cudaSuccess) {
    uint64_t error_val = CUDAQ_TX_FLAG_ERROR_TAG << 48 | (uint64_t)err;
    as_atomic_u64(engine->ringbuffer.tx_flags_host)[current_slot].store(
        error_val, cuda::std::memory_order_release);
    as_atomic_u64(engine->idle_mask)
        ->fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
  } else {
    if (engine->workers[w].post_launch_fn)
      engine->workers[w].post_launch_fn(engine->workers[w].post_launch_data,
                                        data_dev, engine->workers[w].stream);
    if (engine->io_ctxs_host == nullptr && !engine->skip_tx_markers) {
      as_atomic_u64(engine->ringbuffer.tx_flags_host)[current_slot].store(
          CUDAQ_TX_FLAG_IN_FLIGHT, cuda::std::memory_order_release);
    }
  }
}

void cudaq_graph_launch_engine_drain(
    const cudaq_graph_launch_engine_t *engine) {
  for (size_t i = 0; i < engine->num_workers; ++i)
    cudaStreamSynchronize(engine->workers[i].stream);
}

cudaq_status_t cudaq_graph_launch_engine_release_worker(
    cudaq_graph_launch_engine_t *engine, int worker_id) {
  if (!engine || !engine->idle_mask)
    return CUDAQ_ERR_INVALID_ARG;
  if (worker_id < 0 || static_cast<size_t>(worker_id) >= engine->num_workers)
    return CUDAQ_ERR_INVALID_ARG;
  as_atomic_u64(engine->idle_mask)
      ->fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
  return CUDAQ_OK;
}

cudaq_graph_launch_engine_t *cudaq_graph_launch_engine_create(
    const cudaq_ringbuffer_t *ringbuffer, const cudaq_function_table_t *table,
    int skip_tx_markers, void **external_mailbox, cudaq_status_t *out_status) {
  auto set_status = [&](cudaq_status_t s) {
    if (out_status)
      *out_status = s;
  };
  if (!ringbuffer || !table || !table->entries) {
    set_status(CUDAQ_ERR_INVALID_ARG);
    return nullptr;
  }

  // HOST_CALL-only table: no GRAPH_LAUNCH workers, so no engine.  Not an error.
  const size_t num_workers = count_graph_launch_workers(table);
  if (num_workers == 0) {
    set_status(CUDAQ_OK);
    return nullptr;
  }
  if (num_workers > CUDAQ_GRAPH_LAUNCH_MAX_WORKERS) {
    set_status(CUDAQ_ERR_INVALID_ARG);
    return nullptr;
  }

  const uint64_t worker_mask = make_worker_mask(num_workers);

  auto *engine = new (std::nothrow) cudaq_graph_launch_engine_t();
  if (!engine) {
    set_status(CUDAQ_ERR_INTERNAL);
    return nullptr;
  }
  std::memset(engine, 0, sizeof(*engine));
  engine->ringbuffer = *ringbuffer;
  engine->num_workers = num_workers;
  engine->worker_mask = worker_mask;
  engine->skip_tx_markers = skip_tx_markers;

  engine->workers =
      new (std::nothrow) cudaq_host_dispatch_worker_t[num_workers];
  auto *idle = new (std::nothrow) atomic_uint64_sys(0);
  engine->idle_mask = idle;
  engine->inflight_slot_tags = new (std::nothrow) int[num_workers];
  if (external_mailbox) {
    engine->h_mailbox_bank = external_mailbox;
    engine->owns_mailbox = 0;
  } else {
    engine->h_mailbox_bank = new (std::nothrow) void *[num_workers];
    engine->owns_mailbox = 1;
  }
  if (!engine->workers || !engine->idle_mask || !engine->inflight_slot_tags ||
      !engine->h_mailbox_bank) {
    cudaq_graph_launch_engine_destroy(engine);
    set_status(CUDAQ_ERR_INTERNAL);
    return nullptr;
  }
  std::memset(engine->inflight_slot_tags, 0, num_workers * sizeof(int));
  std::memset(engine->workers, 0,
              num_workers * sizeof(cudaq_host_dispatch_worker_t));

  size_t worker_idx = 0;
  for (uint32_t i = 0; i < table->count; ++i) {
    if (table->entries[i].dispatch_mode != CUDAQ_DISPATCH_GRAPH_LAUNCH)
      continue;
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      cudaq_graph_launch_engine_destroy(engine);
      set_status(CUDAQ_ERR_CUDA);
      return nullptr;
    }
    engine->workers[worker_idx].graph_exec = table->entries[i].handler.graph_exec;
    engine->workers[worker_idx].stream = stream;
    engine->workers[worker_idx].function_id = table->entries[i].function_id;
    engine->workers[worker_idx].routing_key = table->entries[i].routing_key;
    worker_idx++;
  }

  idle->store(worker_mask, cuda::std::memory_order_release);

  // Per-worker GraphIOContext array only when RX and TX data buffers differ.
  // For in-place rings (rx_data == tx_data) the legacy path writes a raw device
  // slot pointer into the mailbox instead.
  if (ringbuffer->rx_data != ringbuffer->tx_data) {
    void *io_ctxs_host = nullptr;
    void *io_ctxs_dev = nullptr;
    size_t bytes = num_workers * sizeof(cudaq::realtime::GraphIOContext);
    if (cudaHostAlloc(&io_ctxs_host, bytes, cudaHostAllocMapped) !=
        cudaSuccess) {
      cudaq_graph_launch_engine_destroy(engine);
      set_status(CUDAQ_ERR_CUDA);
      return nullptr;
    }
    std::memset(io_ctxs_host, 0, bytes);
    if (cudaHostGetDevicePointer(&io_ctxs_dev, io_ctxs_host, 0) !=
        cudaSuccess) {
      cudaFreeHost(io_ctxs_host);
      cudaq_graph_launch_engine_destroy(engine);
      set_status(CUDAQ_ERR_CUDA);
      return nullptr;
    }
    engine->io_ctxs_host = io_ctxs_host;
    engine->io_ctxs_dev = io_ctxs_dev;
  }

  set_status(CUDAQ_OK);
  return engine;
}

void cudaq_graph_launch_engine_destroy(cudaq_graph_launch_engine_t *engine) {
  if (!engine)
    return;
  if (engine->workers) {
    for (size_t i = 0; i < engine->num_workers; ++i) {
      if (engine->workers[i].stream)
        cudaStreamDestroy(engine->workers[i].stream);
    }
    delete[] engine->workers;
  }
  delete as_atomic_u64(engine->idle_mask);
  delete[] engine->inflight_slot_tags;
  if (engine->owns_mailbox)
    delete[] engine->h_mailbox_bank;
  if (engine->io_ctxs_host)
    cudaFreeHost(engine->io_ctxs_host);
  delete engine;
}

void cudaq_graph_launch_engine_publish_ready(
    const cudaq_graph_launch_engine_t *engine,
    cudaq_status_t (*publish)(void *ctx, uint32_t slot), void *ctx) {
  const cudaq_ringbuffer_t &ring = engine->ringbuffer;
  uint64_t busy =
      ~as_atomic_u64(engine->idle_mask)->load(cuda::std::memory_order_acquire);
  busy &= (1ULL << engine->num_workers) - 1;
  while (busy != 0) {
    int w = __builtin_ffsll(static_cast<long long>(busy)) - 1;
    busy &= ~(1ULL << w);
    const int slot = engine->inflight_slot_tags[w];
    const uint64_t v = as_atomic_u64(ring.tx_flags_host)[slot].load(
        cuda::std::memory_order_acquire);
    if (v == 0 || v == CUDAQ_TX_FLAG_IN_FLIGHT)
      continue; // still running; leave the worker busy
    // TODO: on the error we skip the publish and fall through to clear the flag
    // and recycle the worker, so the requester sees the failure as a timeout.
    // We need a strategy to handle errors instead of dropping them.
    if ((v >> 48) != CUDAQ_TX_FLAG_ERROR_TAG)
      publish(ctx, static_cast<uint32_t>(slot));
    as_atomic_u64(ring.tx_flags_host)[slot].store(
        0, cuda::std::memory_order_release);
    as_atomic_u64(engine->idle_mask)
        ->fetch_or(1ULL << w, cuda::std::memory_order_release);
  }
}

} // extern "C"
