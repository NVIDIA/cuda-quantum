/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include <cstring>
#include <cuda/std/atomic>

using atomic_uint64_sys = cuda::std::atomic<uint64_t>;
using atomic_int_sys = cuda::std::atomic<int>;

static inline atomic_uint64_sys *as_atomic_u64(void *p) {
  return static_cast<atomic_uint64_sys *>(p);
}
static inline atomic_uint64_sys *as_atomic_u64(volatile uint64_t *p) {
  return reinterpret_cast<atomic_uint64_sys *>(const_cast<uint64_t *>(p));
}
static inline atomic_int_sys *as_atomic_int(void *p) {
  return static_cast<atomic_int_sys *>(p);
}

namespace {

using namespace cudaq::realtime;

static const cudaq_function_entry_t *
lookup_function(cudaq_function_entry_t *table, size_t count,
                uint32_t function_id) {
  for (size_t i = 0; i < count; ++i) {
    if (table[i].function_id == function_id)
      return &table[i];
  }
  return nullptr;
}

// Acquire an idle GRAPH_LAUNCH worker that matches both `function_id` and
// `routing_key`.  The routing_key parameter sub-routes within a shared
// `function_id` -- see [host_api.bs Routing-Key Sub-filter for GRAPH_LAUNCH
// Workers].  Workloads that don't use sub-routing pass routing_key == 0 and
// register every worker with routing_key == 0, in which case this loop
// degenerates to the historical `function_id`-only match.
static int
find_idle_graph_worker_for_function(const cudaq_host_dispatch_loop_ctx_t *ctx,
                                    uint32_t function_id,
                                    uint64_t routing_key) {
  uint64_t mask = as_atomic_u64(ctx->idle_mask)->load(
      cuda::std::memory_order_acquire);
  while (mask != 0) {
    int worker_id = __builtin_ffsll(static_cast<long long>(mask)) - 1;
    const cudaq_host_dispatch_worker_t &w =
        ctx->workers[static_cast<size_t>(worker_id)];
    if (w.function_id == function_id && w.routing_key == routing_key)
      return worker_id;
    mask &= ~(1ULL << worker_id);
  }
  return -1;
}

struct ParsedSlot {
  uint32_t function_id = 0;
  uint64_t routing_key = 0; // arg0 of the payload (or 0 if arg_len < 8)
  const cudaq_function_entry_t *entry = nullptr;
  bool drop = false; // bad header -- clear rx_flags and advance
  bool skip = false; // function not in our table -- advance WITHOUT clearing
                     // (only set when shared_ring_mode is non-zero)
};

static ParsedSlot
parse_slot_with_function_table(void *slot_host,
                               const cudaq_host_dispatch_loop_ctx_t *ctx) {
  ParsedSlot out;
  const RPCHeader *header = static_cast<const RPCHeader *>(slot_host);
  if (header->magic != RPC_MAGIC_REQUEST) {
    out.drop = true;
    return out;
  }
  out.function_id = header->function_id;
  // Routing-key sub-filter: read arg0 (first 8 bytes of payload) when the
  // payload is large enough.  Workloads that don't use sub-routing leave
  // the worker's routing_key == 0, and any arg0 (or absent arg0) still
  // matches via the routing_key == 0 worker.  See
  // proposals/cudaq_realtime_host_api.bs#host-path-graph-routing-key.
  if (header->arg_len >= sizeof(uint64_t)) {
    const uint8_t *slot_bytes = static_cast<const uint8_t *>(slot_host);
    out.routing_key = *reinterpret_cast<const uint64_t *>(slot_bytes +
                                                          sizeof(RPCHeader));
  }
  out.entry = lookup_function(ctx->function_table.entries,
                              ctx->function_table.count, out.function_id);
  if (!out.entry) {
    if (ctx->config.shared_ring_mode)
      out.skip = true;
    else
      out.drop = true;
  }
  return out;
}

static void finish_slot_and_advance(const cudaq_host_dispatch_loop_ctx_t *ctx,
                                    size_t &current_slot, size_t num_slots,
                                    uint64_t &packets_dispatched) {
  as_atomic_u64(ctx->ringbuffer.rx_flags_host)[current_slot].store(
      0, cuda::std::memory_order_release);
  packets_dispatched++;
  if (ctx->live_dispatched)
    as_atomic_u64(ctx->live_dispatched)
        ->fetch_add(1, cuda::std::memory_order_relaxed);
  current_slot = (current_slot + 1) % num_slots;
}

// CUDAQ_DISPATCH_HOST_CALL handler: synchronous pure-C++ dispatch.  No GPU
// graph worker is acquired (HOST_CALL bypasses the worker pool entirely).
// Two-pointer ABI (`cudaq_host_rpc_fn_t(const void *rx, void *tx, size_t)`):
// the dispatcher hands the handler the RX slot (inbound request) and the TX
// slot (outbound response) directly, so the handler reads the request from RX
// and writes its response straight into the TX ring where the consumer (e.g.
// CpuRoceTransceiver TX thread or Hololink TX kernel) expects it -- no
// intermediate RX->TX copy.
static void handle_host_call(const cudaq_host_dispatch_loop_ctx_t *ctx,
                             const cudaq_function_entry_t *entry,
                             void *slot_host, size_t current_slot) {
  if (!entry || !entry->handler.host_fn)
    return;
  const size_t rx_stride = ctx->ringbuffer.rx_stride_sz;
  const size_t tx_stride = ctx->ringbuffer.tx_stride_sz;
  // Usable slot size handed to the handler.  RX and TX strides are equal (the
  // bridge configures both from the same slot_size); the smaller is passed
  // defensively against a future asymmetric configuration.
  const size_t slot_size = rx_stride < tx_stride ? rx_stride : tx_stride;
  uint8_t *tx_slot = ctx->ringbuffer.tx_data_host + current_slot * tx_stride;
  entry->handler.host_fn(slot_host, tx_slot, slot_size);
  // Publish: writing the slot's address to tx_flag signals "fresh data" to
  // the consumer.  No in-flight sentinel needed because this entire path
  // is synchronous from the consumer's POV (the store happens after the
  // host_fn has already filled tx_slot).
  as_atomic_u64(ctx->ringbuffer.tx_flags_host)[current_slot].store(
      reinterpret_cast<uint64_t>(tx_slot),
      cuda::std::memory_order_release);
}

static int acquire_graph_worker(const cudaq_host_dispatch_loop_ctx_t *ctx,
                                bool use_function_table,
                                const cudaq_function_entry_t *entry,
                                uint32_t function_id,
                                uint64_t routing_key) {
  if (use_function_table && entry &&
      entry->dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH)
    return find_idle_graph_worker_for_function(ctx, function_id, routing_key);
  uint64_t mask =
      as_atomic_u64(ctx->idle_mask)->load(cuda::std::memory_order_acquire);
  if (mask == 0)
    return -1;
  return __builtin_ffsll(static_cast<long long>(mask)) - 1;
}

static void
sweep_completed_workers(const cudaq_host_dispatch_loop_ctx_t *ctx) {
  // HOST_CALL dispatch uses no graph worker pool, so idle_mask/workers are
  // null and num_workers is 0.  Guard against that here (not just via the
  // caller's skip_stream_sweep flag) so a future caller can't crash on a null
  // idle_mask if they forget to set it.
  if (!ctx->idle_mask || !ctx->workers || ctx->num_workers == 0)
    return;
  uint64_t busy =
      ~as_atomic_u64(ctx->idle_mask)->load(cuda::std::memory_order_acquire);
  busy &= (1ULL << ctx->num_workers) - 1;
  while (busy != 0) {
    int w = __builtin_ffsll(static_cast<long long>(busy)) - 1;
    if (cudaStreamQuery(ctx->workers[w].stream) == cudaSuccess) {
      as_atomic_u64(ctx->idle_mask)
          ->fetch_or(1ULL << w, cuda::std::memory_order_release);
    }
    busy &= ~(1ULL << w);
  }
}

static void launch_graph_worker(const cudaq_host_dispatch_loop_ctx_t *ctx,
                                int worker_id, void *slot_host,
                                size_t current_slot) {
  as_atomic_u64(ctx->idle_mask)
      ->fetch_and(~(1ULL << worker_id), cuda::std::memory_order_release);
  ctx->inflight_slot_tags[worker_id] = static_cast<int>(current_slot);

  ptrdiff_t offset =
      static_cast<uint8_t *>(slot_host) - ctx->ringbuffer.rx_data_host;
  void *data_dev = static_cast<void *>(ctx->ringbuffer.rx_data + offset);

  if (ctx->io_ctxs_host != nullptr) {
    auto *h_ctxs = static_cast<GraphIOContext *>(ctx->io_ctxs_host);
    auto *d_ctxs = static_cast<uint8_t *>(ctx->io_ctxs_dev);
    GraphIOContext *h_ctx = &h_ctxs[worker_id];

    h_ctx->rx_slot = data_dev;
    h_ctx->tx_slot = ctx->ringbuffer.tx_data +
                     current_slot * ctx->ringbuffer.tx_stride_sz;
    h_ctx->tx_flag = &ctx->ringbuffer.tx_flags[current_slot];
    h_ctx->tx_flag_value =
        reinterpret_cast<uint64_t>(h_ctx->tx_slot);
    h_ctx->tx_stride_sz = ctx->ringbuffer.tx_stride_sz;

    void *d_ctx = d_ctxs + worker_id * sizeof(GraphIOContext);
    ctx->h_mailbox_bank[worker_id] = d_ctx;

    if (!ctx->config.skip_tx_markers) {
      as_atomic_u64(ctx->ringbuffer.tx_flags_host)[current_slot].store(
          CUDAQ_TX_FLAG_IN_FLIGHT, cuda::std::memory_order_release);
    }
    __sync_synchronize();
  } else {
    ctx->h_mailbox_bank[worker_id] = data_dev;
  }
  __sync_synchronize();

  const size_t w = static_cast<size_t>(worker_id);
  if (ctx->workers[w].pre_launch_fn)
    ctx->workers[w].pre_launch_fn(ctx->workers[w].pre_launch_data,
                                     data_dev, ctx->workers[w].stream);
  cudaError_t err = cudaGraphLaunch(ctx->workers[w].graph_exec,
                                    ctx->workers[w].stream);

  if (err != cudaSuccess) {
    uint64_t error_val = CUDAQ_TX_FLAG_ERROR_TAG << 48 | (uint64_t)err;
    as_atomic_u64(ctx->ringbuffer.tx_flags_host)[current_slot].store(
        error_val, cuda::std::memory_order_release);
    as_atomic_u64(ctx->idle_mask)
        ->fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
  } else {
    if (ctx->workers[w].post_launch_fn)
      ctx->workers[w].post_launch_fn(ctx->workers[w].post_launch_data,
                                        data_dev, ctx->workers[w].stream);
    if (ctx->io_ctxs_host == nullptr && !ctx->config.skip_tx_markers) {
      as_atomic_u64(ctx->ringbuffer.tx_flags_host)[current_slot].store(
          CUDAQ_TX_FLAG_IN_FLIGHT, cuda::std::memory_order_release);
    }
  }
}

} // anonymous namespace

extern "C" void
cudaq_host_dispatcher_loop(const cudaq_host_dispatch_loop_ctx_t *ctx) {
  size_t current_slot = 0;
  const size_t num_slots = ctx->config.num_slots;
  uint64_t packets_dispatched = 0;
  const bool use_function_table =
      (ctx->function_table.entries != nullptr && ctx->function_table.count > 0);

  while (as_atomic_int(ctx->shutdown_flag)
             ->load(cuda::std::memory_order_acquire) == 0) {
    uint64_t rx_value =
        as_atomic_u64(ctx->ringbuffer.rx_flags_host)[current_slot].load(
            cuda::std::memory_order_acquire);

    if (rx_value == 0) {
      if (!ctx->skip_stream_sweep)
        sweep_completed_workers(ctx);
      // Under shared_ring_mode, rx_value == 0 at our local cursor does NOT
      // mean "no work anywhere on the ring" -- the peer dispatcher may
      // have cleared this slot after handling it.  Scan the rest of the
      // ring looking for ANY non-zero rx_flag; if we find one, jump our
      // cursor there.  If we wrap all the way back without finding any,
      // fall through to the normal CPU_RELAX wait.
      if (ctx->config.shared_ring_mode) {
        size_t probe = (current_slot + 1) % num_slots;
        size_t scanned = 0;
        while (scanned < num_slots - 1) {
          uint64_t v = as_atomic_u64(ctx->ringbuffer.rx_flags_host)[probe]
                           .load(cuda::std::memory_order_acquire);
          if (v != 0) {
            current_slot = probe;
            break;
          }
          probe = (probe + 1) % num_slots;
          ++scanned;
        }
        if (scanned >= num_slots - 1) {
          // Truly idle: no slot has work for anyone right now.
          CUDAQ_REALTIME_CPU_RELAX();
          continue;
        }
        // Re-load rx_value at the new cursor position and fall through.
        rx_value =
            as_atomic_u64(ctx->ringbuffer.rx_flags_host)[current_slot].load(
                cuda::std::memory_order_acquire);
        if (rx_value == 0) {
          CUDAQ_REALTIME_CPU_RELAX();
          continue;
        }
      } else {
        CUDAQ_REALTIME_CPU_RELAX();
        continue;
      }
    }

    void *slot_host = reinterpret_cast<void *>(rx_value);
    uint32_t function_id = 0;
    uint64_t routing_key = 0;
    const cudaq_function_entry_t *entry = nullptr;

    // TODO: Remove non-function-table path; RPC framing is always required.
    if (use_function_table) {
      ParsedSlot parsed = parse_slot_with_function_table(slot_host, ctx);
      if (parsed.drop) {
        as_atomic_u64(ctx->ringbuffer.rx_flags_host)[current_slot].store(
            0, cuda::std::memory_order_release);
        current_slot = (current_slot + 1) % num_slots;
        continue;
      }
      if (parsed.skip) {
        // shared_ring_mode: leave rx_flags set so a peer dispatcher can pick
        // this slot up; just advance our local cursor.
        current_slot = (current_slot + 1) % num_slots;
        continue;
      }
      function_id = parsed.function_id;
      routing_key = parsed.routing_key;
      entry = parsed.entry;
    }

    // Mode dispatch.  HOST_CALL is handled synchronously inline (no graph
    // worker pool).  DEVICE_CALL slots are dropped on the host loop (the
    // header comment in cudaq_realtime.h documents this — device calls
    // belong on the GPU dispatch path).  GRAPH_LAUNCH falls through to the
    // worker-pool path below.
    if (entry && entry->dispatch_mode == CUDAQ_DISPATCH_HOST_CALL) {
      handle_host_call(ctx, entry, slot_host, current_slot);
      finish_slot_and_advance(ctx, current_slot, num_slots,
                              packets_dispatched);
      continue;
    }
    if (entry && entry->dispatch_mode != CUDAQ_DISPATCH_GRAPH_LAUNCH) {
      if (ctx->config.shared_ring_mode) {
        // Entry is in our table but is not a GRAPH_LAUNCH (e.g. a DEVICE_CALL
        // entry registered for a peer dispatcher).  Under shared_ring_mode
        // the peer will service it -- skip without clearing rx_flags.
        current_slot = (current_slot + 1) % num_slots;
        continue;
      }
      as_atomic_u64(ctx->ringbuffer.rx_flags_host)[current_slot].store(
          0, cuda::std::memory_order_release);
      current_slot = (current_slot + 1) % num_slots;
      continue;
    }

    if (!ctx->skip_stream_sweep)
      sweep_completed_workers(ctx);
    int worker_id = acquire_graph_worker(ctx, use_function_table, entry,
                                         function_id, routing_key);
    if (worker_id < 0) {
      CUDAQ_REALTIME_CPU_RELAX();
      continue;
    }

    launch_graph_worker(ctx, worker_id, slot_host, current_slot);
    finish_slot_and_advance(ctx, current_slot, num_slots,
                            packets_dispatched);
  }

  for (size_t i = 0; i < ctx->num_workers; ++i) {
    cudaStreamSynchronize(ctx->workers[i].stream);
  }

  if (ctx->stats_counter) {
    *ctx->stats_counter = packets_dispatched;
  }
}
