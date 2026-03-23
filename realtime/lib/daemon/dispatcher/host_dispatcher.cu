/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include <cuda/std/atomic>

using atomic_uint64_sys = cuda::std::atomic<uint64_t>;
using atomic_int_sys = cuda::std::atomic<int>;

static inline atomic_uint64_sys *as_atomic_u64(void *p) {
  return static_cast<atomic_uint64_sys *>(p);
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

static int
find_idle_graph_worker_for_function(const cudaq_host_dispatcher_config_t *config,
                                    uint32_t function_id) {
  uint64_t mask = as_atomic_u64(config->idle_mask)->load(
      cuda::std::memory_order_acquire);
  while (mask != 0) {
    int worker_id = __builtin_ffsll(static_cast<long long>(mask)) - 1;
    if (config->workers[static_cast<size_t>(worker_id)].function_id ==
        function_id)
      return worker_id;
    mask &= ~(1ULL << worker_id);
  }
  return -1;
}

struct ParsedSlot {
  uint32_t function_id = 0;
  const cudaq_function_entry_t *entry = nullptr;
  bool drop = false;
};

static ParsedSlot
parse_slot_with_function_table(void *slot_host,
                               const cudaq_host_dispatcher_config_t *config) {
  ParsedSlot out;
  const RPCHeader *header = static_cast<const RPCHeader *>(slot_host);
  if (header->magic != RPC_MAGIC_REQUEST) {
    out.drop = true;
    return out;
  }
  out.function_id = header->function_id;
  out.entry = lookup_function(config->function_table,
                              config->function_table_count, out.function_id);
  if (!out.entry)
    out.drop = true;
  return out;
}

static void finish_slot_and_advance(const cudaq_host_dispatcher_config_t *config,
                                    size_t &current_slot, size_t num_slots,
                                    uint64_t &packets_dispatched) {
  as_atomic_u64(config->rx_flags)[current_slot].store(
      0, cuda::std::memory_order_release);
  packets_dispatched++;
  if (config->live_dispatched)
    as_atomic_u64(config->live_dispatched)
        ->fetch_add(1, cuda::std::memory_order_relaxed);
  current_slot = (current_slot + 1) % num_slots;
}

static int acquire_graph_worker(const cudaq_host_dispatcher_config_t *config,
                                bool use_function_table,
                                const cudaq_function_entry_t *entry,
                                uint32_t function_id) {
  if (use_function_table && entry &&
      entry->dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH)
    return find_idle_graph_worker_for_function(config, function_id);
  uint64_t mask =
      as_atomic_u64(config->idle_mask)->load(cuda::std::memory_order_acquire);
  if (mask == 0)
    return -1;
  return __builtin_ffsll(static_cast<long long>(mask)) - 1;
}

static void launch_graph_worker(const cudaq_host_dispatcher_config_t *config,
                                int worker_id, void *slot_host,
                                size_t current_slot) {
  as_atomic_u64(config->idle_mask)
      ->fetch_and(~(1ULL << worker_id), cuda::std::memory_order_release);
  config->inflight_slot_tags[worker_id] = static_cast<int>(current_slot);

  ptrdiff_t offset =
      static_cast<uint8_t *>(slot_host) - config->rx_data_host;
  void *data_dev = static_cast<void *>(config->rx_data_dev + offset);

  if (config->io_ctxs_host != nullptr) {
    // GraphIOContext mode: fill per-worker context with separate RX/TX info.
    auto *h_ctxs = static_cast<GraphIOContext *>(config->io_ctxs_host);
    auto *d_ctxs = static_cast<uint8_t *>(config->io_ctxs_dev);
    GraphIOContext *h_ctx = &h_ctxs[worker_id];

    h_ctx->rx_slot = data_dev;
    h_ctx->tx_slot = config->tx_data_dev + current_slot * config->tx_stride_sz;
    h_ctx->tx_flag = &config->tx_flags_dev[current_slot];
    h_ctx->tx_flag_value =
        reinterpret_cast<uint64_t>(h_ctx->tx_slot);
    h_ctx->tx_stride_sz = config->tx_stride_sz;

    void *d_ctx = d_ctxs + worker_id * sizeof(GraphIOContext);
    config->h_mailbox_bank[worker_id] = d_ctx;

    // In GraphIOContext mode the graph kernel writes tx_flag_value (READY)
    // to tx_flags from the GPU.  Set the in-flight marker BEFORE launch so
    // the kernel's READY write is never clobbered by a late host write.
    as_atomic_u64(config->tx_flags)[current_slot].store(
        CUDAQ_TX_FLAG_IN_FLIGHT, cuda::std::memory_order_release);
    __sync_synchronize();
  } else {
    config->h_mailbox_bank[worker_id] = data_dev;
  }
  __sync_synchronize();

  const size_t w = static_cast<size_t>(worker_id);
  if (config->workers[w].pre_launch_fn)
    config->workers[w].pre_launch_fn(config->workers[w].pre_launch_data,
                                     data_dev, config->workers[w].stream);
  cudaError_t err = cudaGraphLaunch(config->workers[w].graph_exec,
                                    config->workers[w].stream);

  if (err != cudaSuccess) {
    uint64_t error_val = CUDAQ_TX_FLAG_ERROR_TAG << 48 | (uint64_t)err;
    as_atomic_u64(config->tx_flags)[current_slot].store(
        error_val, cuda::std::memory_order_release);
    as_atomic_u64(config->idle_mask)
        ->fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
  } else {
    if (config->workers[w].post_launch_fn)
      config->workers[w].post_launch_fn(config->workers[w].post_launch_data,
                                        data_dev, config->workers[w].stream);
    if (config->io_ctxs_host == nullptr) {
      as_atomic_u64(config->tx_flags)[current_slot].store(
          CUDAQ_TX_FLAG_IN_FLIGHT, cuda::std::memory_order_release);
    }
  }
}

} // anonymous namespace

extern "C" void
cudaq_host_dispatcher_loop(const cudaq_host_dispatcher_config_t *config) {
  size_t current_slot = 0;
  const size_t num_slots = config->num_slots;
  uint64_t packets_dispatched = 0;
  const bool use_function_table =
      (config->function_table != nullptr && config->function_table_count > 0);

  while (as_atomic_int(config->shutdown_flag)
             ->load(cuda::std::memory_order_acquire) == 0) {
    uint64_t rx_value = as_atomic_u64(config->rx_flags)[current_slot].load(
        cuda::std::memory_order_acquire);

    if (rx_value == 0) {
      CUDAQ_REALTIME_CPU_RELAX();
      continue;
    }

    void *slot_host = reinterpret_cast<void *>(rx_value);
    uint32_t function_id = 0;
    const cudaq_function_entry_t *entry = nullptr;

    // TODO: Remove non-function-table path; RPC framing is always required.
    if (use_function_table) {
      ParsedSlot parsed = parse_slot_with_function_table(slot_host, config);
      if (parsed.drop) {
        as_atomic_u64(config->rx_flags)[current_slot].store(
            0, cuda::std::memory_order_release);
        current_slot = (current_slot + 1) % num_slots;
        continue;
      }
      function_id = parsed.function_id;
      entry = parsed.entry;
    }

    if (entry && entry->dispatch_mode != CUDAQ_DISPATCH_GRAPH_LAUNCH) {
      as_atomic_u64(config->rx_flags)[current_slot].store(
          0, cuda::std::memory_order_release);
      current_slot = (current_slot + 1) % num_slots;
      continue;
    }

    int worker_id =
        acquire_graph_worker(config, use_function_table, entry, function_id);
    if (worker_id < 0) {
      CUDAQ_REALTIME_CPU_RELAX();
      continue;
    }

    launch_graph_worker(config, worker_id, slot_host, current_slot);
    finish_slot_and_advance(config, current_slot, num_slots,
                            packets_dispatched);
  }

  for (size_t i = 0; i < config->num_workers; ++i) {
    cudaStreamSynchronize(config->workers[i].stream);
  }

  if (config->stats_counter) {
    *config->stats_counter = packets_dispatched;
  }
}
