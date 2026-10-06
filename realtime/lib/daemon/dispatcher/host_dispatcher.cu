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

#include <cassert>
#include <cstring>
#include <cuda/std/atomic>
#include <type_traits>
#include <utility>

using atomic_int_sys = cuda::std::atomic<int>;
using atomic_uint64_sys = cuda::std::atomic<uint64_t>;

static inline atomic_int_sys *as_atomic_int(void *p) {
  return static_cast<atomic_int_sys *>(p);
}
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

//===----------------------------------------------------------------------===//
// Shared primitives
//===----------------------------------------------------------------------===//

const cudaq_function_entry_t *lookup_function(cudaq_function_entry_t *table,
                                              size_t count,
                                              uint32_t function_id) {
  for (size_t i = 0; i < count; ++i) {
    if (table[i].function_id == function_id)
      return &table[i];
  }
  return nullptr;
}

// Run a resolved HOST_CALL entry via the two-pointer ABI
// (`cudaq_host_rpc_fn_t(const void *rx, void *tx, size_t)`) and return the
// framed RPCResponse length (sizeof(RPCResponse) + result_len), or 0 to drop
// (entry is null / not a HOST_CALL / null handler).  Publish-agnostic: the
// caller signals the filled tx slot however its transport requires.
size_t invoke_host_call(const cudaq_function_entry_t *entry,
                        const void *rx_slot, void *tx_slot, size_t slot_size) {
  if (!entry || entry->dispatch_mode != CUDAQ_DISPATCH_HOST_CALL ||
      !entry->handler.host_fn)
    return 0;
  entry->handler.host_fn(rx_slot, tx_slot, slot_size);
  const auto *resp = static_cast<const RPCResponse *>(tx_slot);
  return sizeof(RPCResponse) + resp->result_len;
}

//===----------------------------------------------------------------------===//
// Transport policy
//===----------------------------------------------------------------------===//
//
// dispatch_core() runs the shared per-slot skeleton and delegates the parts that
// differ between the two deployment shapes -- how a request is received, how a
// response is sent, and how completed GRAPH_LAUNCH work is finalized -- to a
// Transport policy.  A policy provides:
//
//   const cudaq_ringbuffer_t *ring() const;
//       The ring buffer the loop works on (rx/tx data pointers + slot strides).
//
//   bool poll_request(RxItem &out);
//       Non-blocking.  If an inbound request is ready, fill `out` (its slot
//       index + a host pointer to the request bytes) and return true; otherwise
//       return false, and the core relaxes and retries next turn.
//
//   void send_response(const RxItem &it, void *tx_host);
//       Publish the response the dispatcher has already written into `tx_host`
//       (the request's TX slot) back to the requester.
//
//   void retire_request(const RxItem &it, SlotOutcome outcome);
//       Finish with a request slot once the core is done with it, releasing it
//       according to `outcome` (reclaim it, or leave it for a peer dispatcher).
//
//   void service_completions(cudaq_graph_launch_engine_t *engine);
//       Finalize GRAPH_LAUNCH graphs that have completed: recycle their workers
//       and, where the dispatcher owns TX, publish their responses.  Called once
//       per turn and again while waiting for a free worker under backpressure.
//
// The policy is a template parameter (not a vtable), so these calls inline on
// the hot path.  RingTransport and UnifiedTransport implement it below; the
// static_asserts after them pin both to this exact signature set.

// A request the loop is about to dispatch: its ring slot index and a host
// pointer to the inbound request bytes.
struct RxItem {
  uint32_t slot = 0;
  void *rx_host = nullptr;
};

// The core's verdict on a slot, handed to retire_request so the transport can
// release it correctly:
//   Handled - dispatched by this dispatcher; consume the slot.
//   Foreign - not for this dispatcher (unknown function_id, or a dispatch mode
//             it does not service); under shared_ring_mode a peer may take it.
//   Drop    - undispatchable (bad framing, or GRAPH_LAUNCH with no engine);
//             consume it.
enum class SlotOutcome { Handled, Foreign, Drop };

// Compile-time contract for the Transport policy.  C++20 concepts would be the
// natural tool, but this .cu compiles as CUDA C++17, so the shared API is pinned
// with a static_assert (see after the policies): any signature drift between the
// two transports is caught here -- checking both the required call and its
// return type -- rather than as a cryptic dispatch_core<> instantiation error.
template <class T>
constexpr bool satisfies_transport_policy() {
  return std::is_same<decltype(std::declval<T &>().ringbuffer()),
                      const cudaq_ringbuffer_t *>::value &&
         std::is_same<decltype(std::declval<T &>().poll_request(
                          std::declval<RxItem &>())),
                      bool>::value &&
         std::is_same<decltype(std::declval<T &>().send_response(
                          std::declval<const RxItem &>(),
                          std::declval<void *>())),
                      void>::value &&
         std::is_same<decltype(std::declval<T &>().retire_request(
                          std::declval<const RxItem &>(),
                          std::declval<SlotOutcome>())),
                      void>::value &&
         std::is_same<decltype(std::declval<T &>().service_completions(
                          std::declval<cudaq_graph_launch_engine_t *>())),
                      void>::value;
}

// Transport policy for the 3-thread ring shape: dedicated RX/TX adapter threads
// move bytes wire<->ring and own TX, so dispatch_core only polls/clears rx_flags
// and stores the response address into tx_flags (address-as-flag).  Completed
// GRAPH_LAUNCH graphs are recycled via sweep -- the adapter thread publishes
// them.
struct RingTransport {
  const cudaq_ringbuffer_t *rb;
  const cudaq_dispatcher_config_t *config;
  size_t num_slots;
  size_t current_slot = 0;

  RingTransport(const cudaq_ringbuffer_t *rb_,
                const cudaq_dispatcher_config_t *config_)
      : rb(rb_), config(config_), num_slots(config_->num_slots) {}

  const cudaq_ringbuffer_t *ringbuffer() const { return rb; }

  // Poll rx_flags_host at the local cursor.  Under shared_ring_mode, when the
  // cursor's slot is empty, scan the ring for any pending slot (a peer may have
  // consumed ours) and jump the cursor there.  Returns false when idle.
  bool poll_request(RxItem &out) {
    uint64_t rx_value = as_atomic_u64(rb->rx_flags_host)[current_slot].load(
        cuda::std::memory_order_acquire);
    if (rx_value == 0) {
      if (!config->shared_ring_mode)
        return false;
      size_t probe = (current_slot + 1) % num_slots;
      size_t scanned = 0;
      while (scanned < num_slots - 1) {
        uint64_t v = as_atomic_u64(rb->rx_flags_host)[probe].load(
            cuda::std::memory_order_acquire);
        if (v != 0) {
          current_slot = probe;
          break;
        }
        probe = (probe + 1) % num_slots;
        ++scanned;
      }
      if (scanned >= num_slots - 1)
        return false;
      rx_value = as_atomic_u64(rb->rx_flags_host)[current_slot].load(
          cuda::std::memory_order_acquire);
      if (rx_value == 0)
        return false;
    }
    out.slot = static_cast<uint32_t>(current_slot);
    out.rx_host = reinterpret_cast<void *>(rx_value);
    return true;
  }

  // Store the tx-slot address into tx_flags_host; this "address-as-flag" write
  // signals fresh response data to the consumer transport thread.
  void send_response(const RxItem &it, void *tx_host) {
    as_atomic_u64(rb->tx_flags_host)[it.slot].store(
        reinterpret_cast<uint64_t>(tx_host), cuda::std::memory_order_release);
  }

  // Release the slot and step the cursor.  Foreign under shared_ring_mode leaves
  // rx_flags set so a peer dispatcher can pick the slot up (advance cursor
  // only); every other outcome claims the slot by clearing its rx_flag.
  void retire_request(const RxItem &it, SlotOutcome outcome) {
    if (!(outcome == SlotOutcome::Foreign && config->shared_ring_mode))
      as_atomic_u64(rb->rx_flags_host)[it.slot].store(
          0, cuda::std::memory_order_release);
    current_slot = (static_cast<size_t>(it.slot) + 1) % num_slots;
  }

  // Recycle finished GRAPH_LAUNCH workers; the adapter thread publishes their
  // responses off tx_flags, so there is nothing to publish here.
  void service_completions(cudaq_graph_launch_engine_t *engine) {
    cudaq_graph_launch_engine_sweep(engine);
  }
};

// Transport policy for the single-thread unified shape: there are no adapter
// threads, so dispatch_core drives the transport itself through the rx_poll /
// tx_publish hooks.  The hooks own wire<->ring, so retire_request has nothing
// to reclaim.
struct UnifiedTransport {
  cudaq_cpu_dataplane_t *dp;

  explicit UnifiedTransport(cudaq_cpu_dataplane_t *dp_) : dp(dp_) {}

  const cudaq_ringbuffer_t *ringbuffer() const { return &dp->ring; }

  // Claim the next ready request via the transport's rx_poll hook.
  bool poll_request(RxItem &out) {
    uint32_t slot = 0;
    if (dp->rx_poll(dp->ctx, &slot) != CUDAQ_RX_READY)
      return false;
    out.slot = slot;
    out.rx_host = dp->ring.rx_data_host +
                  static_cast<size_t>(slot) * dp->ring.rx_stride_sz;
    return true;
  }

  // Publish via the transport's tx_publish hook; the response is already in the
  // TX slot, so the hook only needs the slot index (tx_host is unused here).
  void send_response(const RxItem &it, void * /*tx_host*/) {
    // TODO: tx_publish returns a cudaq_status_t, but the hot dispatch path has
    // no recovery action for a failed publish today.
    (void)dp->tx_publish(dp->ctx, it.slot);
  }

  // Nothing to reclaim: rx_poll already claimed the slot and the hook transport
  // is single-consumer (no peer to leave it for).
  void retire_request(const RxItem &, SlotOutcome) {}

  // Recycle finished GRAPH_LAUNCH workers AND publish their responses: under
  // this shape the dispatcher owns TX, so graph completions (which may land out
  // of order) are published here via tx_publish.
  void service_completions(cudaq_graph_launch_engine_t *engine) {
    cudaq_graph_launch_engine_publish_ready(engine, dp->tx_publish, dp->ctx);
  }
};

// Both policies must expose the identical API dispatch_core drives.  These
// assertions fail the build the moment either one drifts.
static_assert(satisfies_transport_policy<RingTransport>(),
              "RingTransport must match the dispatch_core Transport policy");
static_assert(satisfies_transport_policy<UnifiedTransport>(),
              "UnifiedTransport must match the dispatch_core Transport policy");

//===----------------------------------------------------------------------===//
// Shared dispatch skeleton
//===----------------------------------------------------------------------===//

// Drive `tp`'s ring until `*shutdown_flag != 0`.  Each turn: service completed
// GRAPH_LAUNCH work, poll the next request, validate framing + resolve the
// entry, then run its dispatch mode (HOST_CALL inline via invoke_host_call +
// tp.send_response; GRAPH_LAUNCH via the engine).  `engine` may be NULL for a
// HOST_CALL-only table.  Does NOT own `engine`; it only drains it on exit (the
// caller keeps it, or services completions + destroys, as appropriate).
template <class Transport>
void dispatch_core(Transport &tp, const cudaq_function_table_t *table,
                   cudaq_graph_launch_engine_t *engine,
                   volatile int *shutdown_flag, uint64_t *stats) {
  const cudaq_ringbuffer_t *rb = tp.ringbuffer();
  // The dispatcher operates on a registered function table; there is nothing to
  // dispatch without one.  (The engine, when present, is still drained so its
  // worker streams are synchronized on exit.)
  if (!table || !table->entries || table->count == 0) {
    if (engine)
      cudaq_graph_launch_engine_drain(engine);
    return;
  }
  // RX and TX strides are equal by configuration; the smaller is passed to
  // handlers defensively against a future asymmetric layout.
  const size_t slot_size = rb->rx_stride_sz < rb->tx_stride_sz
                               ? rb->rx_stride_sz
                               : rb->tx_stride_sz;
  uint64_t count = 0;

  while (as_atomic_int(shutdown_flag)->load(cuda::std::memory_order_relaxed) ==
         0) {
    if (engine)
      tp.service_completions(engine);

    RxItem it;
    if (!tp.poll_request(it)) {
      CUDAQ_REALTIME_CPU_RELAX();
      continue;
    }

    const RPCHeader *hdr = static_cast<const RPCHeader *>(it.rx_host);
    if (hdr->magic != RPC_MAGIC_REQUEST) {
      tp.retire_request(it, SlotOutcome::Drop);
      continue;
    }

    const cudaq_function_entry_t *entry =
        lookup_function(table->entries, table->count, hdr->function_id);
    if (!entry) {
      tp.retire_request(it, SlotOutcome::Foreign);
      continue;
    }

    // Sub-routing key: arg0 (first 8 payload bytes) when present, else 0.  Only
    // consulted for GRAPH_LAUNCH worker matching; a 0 key is a wildcard.
    uint64_t routing_key = 0;
    if (hdr->arg_len >= sizeof(uint64_t))
      routing_key = *reinterpret_cast<const uint64_t *>(
          static_cast<const uint8_t *>(it.rx_host) + sizeof(RPCHeader));

    if (entry->dispatch_mode == CUDAQ_DISPATCH_HOST_CALL) {
      void *tx_host =
          rb->tx_data_host + static_cast<size_t>(it.slot) * rb->tx_stride_sz;
      if (invoke_host_call(entry, it.rx_host, tx_host, slot_size)) {
        tp.send_response(it, tx_host);
        ++count;
      }
      tp.retire_request(it, SlotOutcome::Handled);
      continue;
    }

    if (entry->dispatch_mode != CUDAQ_DISPATCH_GRAPH_LAUNCH) {
      tp.retire_request(it, SlotOutcome::Foreign); // e.g. a DEVICE_CALL entry
      continue;
    }
    if (!engine) {
      tp.retire_request(it, SlotOutcome::Drop); // GRAPH_LAUNCH but no engine wired
      continue;
    }

    // GRAPH_LAUNCH: acquire a worker matched to the function; spin (reaping)
    // under backpressure until one frees.  The request stays staged in its slot
    // while we wait.
    int worker_id =
        cudaq_graph_launch_engine_acquire(engine, hdr->function_id, routing_key);
    while (worker_id < 0) {
      if (as_atomic_int(shutdown_flag)->load(cuda::std::memory_order_relaxed) !=
          0)
        break;
      tp.service_completions(engine);
      CUDAQ_REALTIME_CPU_RELAX();
      worker_id = cudaq_graph_launch_engine_acquire(engine, hdr->function_id,
                                                    routing_key);
    }
    if (worker_id < 0)
      break; // shutdown raced

    cudaq_graph_launch_engine_launch(engine, worker_id, it.rx_host, it.slot);
    ++count;
    tp.retire_request(it, SlotOutcome::Handled);
  }

  if (engine)
    cudaq_graph_launch_engine_drain(engine);

  if (stats)
    *stats += count;
}

} // anonymous namespace

extern "C" void
cudaq_host_ring_dispatch_loop(const cudaq_ringbuffer_t *ringbuffer,
                              const cudaq_function_table_t *table,
                              const cudaq_dispatcher_config_t *config,
                              cudaq_graph_launch_engine_t *engine,
                              volatile int *shutdown_flag, uint64_t *stats) {
  assert(ringbuffer && "ringbuffer must be non-null");
  assert(table && "table must be non-null");
  assert(config && "config must be non-null");
  assert(shutdown_flag && "shutdown_flag must be non-null");

  RingTransport tp(ringbuffer, config);
  dispatch_core(tp, table, engine, shutdown_flag, stats);
}

extern "C" void
cudaq_host_unified_loop(cudaq_cpu_dataplane_t *cpu_dataplane,
                        const cudaq_function_table_t *table,
                        cudaq_graph_launch_engine_t *engine,
                        volatile int *shutdown_flag, uint64_t *stats) {
  assert(cpu_dataplane && "cpu_dataplane must be non-null");
  assert(cpu_dataplane->rx_poll && "data-plane must provide rx_poll");
  assert(cpu_dataplane->tx_publish && "data-plane must provide tx_publish");
  assert(table && "table must be non-null");
  assert(shutdown_flag && "shutdown_flag must be non-null");

  UnifiedTransport tp(cpu_dataplane);
  dispatch_core(tp, table, engine, shutdown_flag, stats);

  // dispatch_core drained in-flight graphs; publish any completions that landed
  // during the drain.
  if (engine)
    tp.service_completions(engine);
}
