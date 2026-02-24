/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

namespace cudaq::realtime {

//-----------------------------------------------------------------------------
// Helpers: function table lookup
//-----------------------------------------------------------------------------

static const cudaq_function_entry_t* lookup_function(cudaq_function_entry_t* table,
                                                     size_t count,
                                                     uint32_t function_id) {
  for (size_t i = 0; i < count; ++i) {
    if (table[i].function_id == function_id)
      return &table[i];
  }
  return nullptr;
}

static int find_idle_graph_worker_for_function(const HostDispatcherConfig& config,
                                               uint32_t function_id) {
  uint64_t mask = config.idle_mask->load(cuda::std::memory_order_acquire);
  while (mask != 0) {
    int worker_id = __builtin_ffsll(static_cast<long long>(mask)) - 1;
    if (config.workers[static_cast<size_t>(worker_id)].function_id == function_id)
      return worker_id;
    mask &= ~(1ULL << worker_id);
  }
  return -1;
}

/// Result of parsing the slot when a function table is in use.
struct ParsedSlot {
  uint32_t function_id = 0;
  const cudaq_function_entry_t* entry = nullptr;
  bool drop = false;  // true => invalid magic or unknown function_id; clear slot and advance
};

static ParsedSlot parse_slot_with_function_table(void* slot_host,
                                                 const HostDispatcherConfig& config) {
  ParsedSlot out;
  const RPCHeader* header = static_cast<const RPCHeader*>(slot_host);
  if (header->magic != RPC_MAGIC_REQUEST) {
    out.drop = true;
    return out;
  }
  out.function_id = header->function_id;
  out.entry = lookup_function(config.function_table, config.function_table_count,
                             out.function_id);
  if (!out.entry)
    out.drop = true;
  return out;
}

/// Clear rx_flag for this slot, increment stats, advance slot index.
static void finish_slot_and_advance(const HostDispatcherConfig& config,
                                    size_t& current_slot,
                                    size_t num_slots,
                                    uint64_t& packets_dispatched) {
  config.rx_flags[current_slot].store(0, cuda::std::memory_order_release);
  packets_dispatched++;
  if (config.live_dispatched)
    config.live_dispatched->fetch_add(1, cuda::std::memory_order_relaxed);
  current_slot = (current_slot + 1) % num_slots;
}

/// Acquire a graph worker (by function_id if table in use, else any idle worker).
static int acquire_graph_worker(const HostDispatcherConfig& config,
                                bool use_function_table,
                                const cudaq_function_entry_t* entry,
                                uint32_t function_id) {
  if (use_function_table && entry && entry->dispatch_mode == CUDAQ_DISPATCH_GRAPH_LAUNCH)
    return find_idle_graph_worker_for_function(config, function_id);
  uint64_t mask = config.idle_mask->load(cuda::std::memory_order_acquire);
  if (mask == 0)
    return -1;
  return __builtin_ffsll(static_cast<long long>(mask)) - 1;
}

/// Launch the graph for the given worker; set tx_flags on success or error.
static void launch_graph_worker(const HostDispatcherConfig& config,
                                int worker_id,
                                void* slot_host,
                                size_t current_slot) {
  config.idle_mask->fetch_and(~(1ULL << worker_id), cuda::std::memory_order_release);
  config.inflight_slot_tags[worker_id] = static_cast<int>(current_slot);

  ptrdiff_t offset = static_cast<uint8_t*>(slot_host) - config.rx_data_host;
  void* data_dev = static_cast<void*>(config.rx_data_dev + offset);
  config.h_mailbox_bank[worker_id] = data_dev;
  __sync_synchronize();

  const size_t w = static_cast<size_t>(worker_id);
  cudaError_t err = cudaGraphLaunch(config.workers[w].graph_exec, config.workers[w].stream);

  if (err != cudaSuccess) {
    uint64_t error_val = (uint64_t)0xDEAD << 48 | (uint64_t)err;
    config.tx_flags[current_slot].store(error_val, cuda::std::memory_order_release);
    config.idle_mask->fetch_or(1ULL << worker_id, cuda::std::memory_order_release);
  } else {
    uint64_t tx_slot_addr =
        (config.tx_data_host != nullptr && config.tx_data_dev != nullptr)
            ? reinterpret_cast<uint64_t>(config.tx_data_host +
                                         current_slot * config.tx_stride_sz)
            : 0xEEEEEEEEEEEEEEEEULL;
    config.tx_flags[current_slot].store(tx_slot_addr, cuda::std::memory_order_release);
  }
}

//-----------------------------------------------------------------------------
// Main loop
//-----------------------------------------------------------------------------

void host_dispatcher_loop(const HostDispatcherConfig& config) {
  size_t current_slot = 0;
  const size_t num_slots = config.num_slots;
  uint64_t packets_dispatched = 0;
  const bool use_function_table =
      (config.function_table != nullptr && config.function_table_count > 0);

  while (config.shutdown_flag->load(cuda::std::memory_order_acquire) == 0) {
    uint64_t rx_value = config.rx_flags[current_slot].load(cuda::std::memory_order_acquire);

    if (rx_value == 0) {
      QEC_CPU_RELAX();
      continue;
    }

    void* slot_host = reinterpret_cast<void*>(rx_value);
    uint32_t function_id = 0;
    const cudaq_function_entry_t* entry = nullptr;

    if (use_function_table) {
      ParsedSlot parsed = parse_slot_with_function_table(slot_host, config);
      if (parsed.drop) {
        config.rx_flags[current_slot].store(0, cuda::std::memory_order_release);
        current_slot = (current_slot + 1) % num_slots;
        continue;
      }
      function_id = parsed.function_id;
      entry = parsed.entry;
    }

    // Only GRAPH_LAUNCH is dispatched; HOST_CALL and DEVICE_CALL are dropped.
    if (entry && entry->dispatch_mode != CUDAQ_DISPATCH_GRAPH_LAUNCH) {
      config.rx_flags[current_slot].store(0, cuda::std::memory_order_release);
      current_slot = (current_slot + 1) % num_slots;
      continue;
    }

    int worker_id = acquire_graph_worker(config, use_function_table, entry, function_id);
    if (worker_id < 0) {
      QEC_CPU_RELAX();
      continue;
    }

    launch_graph_worker(config, worker_id, slot_host, current_slot);
    finish_slot_and_advance(config, current_slot, num_slots, packets_dispatched);
  }

  for (const auto& w : config.workers) {
    cudaStreamSynchronize(w.stream);
  }

  if (config.stats_counter) {
    *config.stats_counter = packets_dispatched;
  }
}

} // namespace cudaq::realtime
