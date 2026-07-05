/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cpu_relax.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/graph_launch_engine.h"

#include <cstring>
#include <cuda/std/atomic>

using atomic_int_sys = cuda::std::atomic<int>;
using atomic_uint64_sys = cuda::std::atomic<uint64_t>;

static inline atomic_int_sys *as_atomic_int(volatile int *p) {
  // `const_cast` drops the flag's `volatile` qualifier (`reinterpret_cast`
  // can't cast away cv-qualifiers) so it can be written as a plain atomic.
  return reinterpret_cast<atomic_int_sys *>(const_cast<int *>(p));
}
static inline atomic_uint64_sys *as_atomic_u64(void *p) {
  return static_cast<atomic_uint64_sys *>(p);
}
static inline atomic_uint64_sys *as_atomic_u64(volatile uint64_t *p) {
  return reinterpret_cast<atomic_uint64_sys *>(const_cast<uint64_t *>(p));
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
                               const cudaq_function_table_t *table,
                               uint32_t shared_ring_mode) {
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
  // matches via the routing_key == 0 worker.
  if (header->arg_len >= sizeof(uint64_t)) {
    const uint8_t *slot_bytes = static_cast<const uint8_t *>(slot_host);
    out.routing_key = *reinterpret_cast<const uint64_t *>(slot_bytes +
                                                          sizeof(RPCHeader));
  }
  out.entry =
      lookup_function(table->entries, table->count, out.function_id);
  if (!out.entry) {
    if (shared_ring_mode)
      out.skip = true;
    else
      out.drop = true;
  }
  return out;
}

// live_dispatched is an optional `cuda::std::atomic<uint64_t>*` (live in-flight
// counter); NULL skips it.
static void finish_slot_and_advance(const cudaq_ringbuffer_t *rb,
                                    void *live_dispatched, size_t &current_slot,
                                    size_t num_slots,
                                    uint64_t &packets_dispatched) {
  as_atomic_u64(rb->rx_flags_host)[current_slot].store(
      0, cuda::std::memory_order_release);
  packets_dispatched++;
  if (live_dispatched)
    as_atomic_u64(live_dispatched)
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
static void handle_host_call(const cudaq_ringbuffer_t *rb,
                             const cudaq_function_entry_t *entry,
                             void *slot_host, size_t current_slot) {
  if (!entry || !entry->handler.host_fn)
    return;
  const size_t rx_stride = rb->rx_stride_sz;
  const size_t tx_stride = rb->tx_stride_sz;
  // Usable slot size handed to the handler.  RX and TX strides are equal (the
  // bridge configures both from the same slot_size); the smaller is passed
  // defensively against a future asymmetric configuration.
  const size_t slot_size = rx_stride < tx_stride ? rx_stride : tx_stride;
  uint8_t *tx_slot = rb->tx_data_host + current_slot * tx_stride;
  entry->handler.host_fn(slot_host, tx_slot, slot_size);
  // Publish: writing the slot's address to tx_flag signals "fresh data" to
  // the consumer.  No in-flight sentinel needed because this entire path
  // is synchronous from the consumer's POV (the store happens after the
  // host_fn has already filled tx_slot).
  as_atomic_u64(rb->tx_flags_host)[current_slot].store(
      reinterpret_cast<uint64_t>(tx_slot), cuda::std::memory_order_release);
}

} // anonymous namespace

// 3-thread ring dispatch loop.  The transport's RX/TX adapter threads move
// bytes wire<->ring (and set/drain the flags); this loop polls rx_flags_host,
// runs HOST_CALL inline, and offloads GRAPH_LAUNCH to `engine` (NULL for a
// HOST_CALL-only table, in which case no graph machinery runs).  Blocks until
// `*shutdown_flag != 0`.
extern "C" void
cudaq_host_ring_dispatch_loop(const cudaq_ringbuffer_t *ringbuffer,
                              const cudaq_function_table_t *table,
                              const cudaq_dispatcher_config_t *config,
                              cudaq_graph_launch_engine_t *engine,
                              volatile int *shutdown_flag, uint64_t *stats) {
  if (!ringbuffer || !table || !config || !shutdown_flag)
    return;
  size_t current_slot = 0;
  const size_t num_slots = config->num_slots;
  uint64_t packets_dispatched = 0;
  const bool use_function_table =
      (table->entries != nullptr && table->count > 0);

  while (as_atomic_int(shutdown_flag)->load(cuda::std::memory_order_relaxed) ==
         0) {
    uint64_t rx_value =
        as_atomic_u64(ringbuffer->rx_flags_host)[current_slot].load(
            cuda::std::memory_order_acquire);

    if (rx_value == 0) {
      if (engine)
        cudaq_graph_launch_engine_sweep(engine);
      // Under shared_ring_mode, rx_value == 0 at our local cursor does NOT
      // mean "no work anywhere on the ring" -- the peer dispatcher may
      // have cleared this slot after handling it.  Scan the rest of the
      // ring looking for ANY non-zero rx_flag; if we find one, jump our
      // cursor there.  If we wrap all the way back without finding any,
      // fall through to the normal CPU_RELAX wait.
      if (config->shared_ring_mode) {
        size_t probe = (current_slot + 1) % num_slots;
        size_t scanned = 0;
        while (scanned < num_slots - 1) {
          uint64_t v = as_atomic_u64(ringbuffer->rx_flags_host)[probe]
                           .load(cuda::std::memory_order_acquire);
          if (v != 0) {
            current_slot = probe;
            break;
          }
          probe = (probe + 1) % num_slots;
          ++scanned;
        }
        if (scanned >= num_slots - 1) {
          CUDAQ_REALTIME_CPU_RELAX();
          continue;
        }
        rx_value =
            as_atomic_u64(ringbuffer->rx_flags_host)[current_slot].load(
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
    if (use_function_table) {
      ParsedSlot parsed = parse_slot_with_function_table(
          slot_host, table, config->shared_ring_mode);
      if (parsed.drop) {
        as_atomic_u64(ringbuffer->rx_flags_host)[current_slot].store(
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

    if (entry && entry->dispatch_mode == CUDAQ_DISPATCH_HOST_CALL) {
      handle_host_call(ringbuffer, entry, slot_host, current_slot);
      finish_slot_and_advance(ringbuffer, nullptr, current_slot, num_slots,
                              packets_dispatched);
      continue;
    }
    if (entry && entry->dispatch_mode != CUDAQ_DISPATCH_GRAPH_LAUNCH) {
      if (config->shared_ring_mode) {
        // Entry is in our table but is not a GRAPH_LAUNCH (e.g. a DEVICE_CALL
        // entry registered for a peer dispatcher).  Under shared_ring_mode
        // the peer will service it -- skip without clearing rx_flags.
        current_slot = (current_slot + 1) % num_slots;
        continue;
      }
      as_atomic_u64(ringbuffer->rx_flags_host)[current_slot].store(
          0, cuda::std::memory_order_release);
      current_slot = (current_slot + 1) % num_slots;
      continue;
    }
    // GRAPH_LAUNCH with no engine wired: drop the slot.
    if (!engine) {
      as_atomic_u64(ringbuffer->rx_flags_host)[current_slot].store(
          0, cuda::std::memory_order_release);
      current_slot = (current_slot + 1) % num_slots;
      continue;
    }

    cudaq_graph_launch_engine_sweep(engine);
    int worker_id = cudaq_graph_launch_engine_acquire(
        engine, use_function_table, entry, function_id, routing_key);
    if (worker_id < 0) {
      CUDAQ_REALTIME_CPU_RELAX();
      continue;
    }

    cudaq_graph_launch_engine_launch(engine, worker_id, slot_host,
                                     current_slot);
    finish_slot_and_advance(ringbuffer, nullptr, current_slot, num_slots,
                            packets_dispatched);
  }

  if (engine)
    cudaq_graph_launch_engine_drain(engine);

  if (stats)
    *stats += packets_dispatched;
}
