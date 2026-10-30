/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "cudaq/realtime/daemon/dispatcher/host_dispatch_loop.h"

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cpu_relax.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <type_traits>
#include <utility>

// The ring flags and the shutdown flag are plain `volatile` words shared with
// other threads (and, on a CUDA-mapped ring, with the GPU).  Each helper views
// one such word through a std::atomic_ref for a single load or store.
// `const_cast` drops the word's `volatile` qualifier, which std::atomic_ref
// does not accept.
static inline std::atomic_ref<int> as_atomic_int(volatile int *p) {
  return std::atomic_ref<int>(*const_cast<int *>(p));
}
static inline std::atomic_ref<uint64_t> as_atomic_u64(volatile uint64_t *p) {
  return std::atomic_ref<uint64_t>(*const_cast<uint64_t *>(p));
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
// dispatch_core() runs the shared per-slot skeleton and delegates the parts
// that differ between the two deployment shapes -- how a request is received,
// how a response is sent, and how a finished slot is released -- to a Transport
// policy.  A policy provides:
//
//   const cudaq_ringbuffer_t *ringbuffer() const;
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
//   Drop    - undispatchable (bad framing); consume it.
enum class SlotOutcome { Handled, Foreign, Drop };

// Compile-time contract for the Transport policy, pinned with a static_assert
// (see after the policies): any signature drift between the two transports is
// caught here -- checking both the required call and its return type -- rather
// than as a cryptic dispatch_core<> instantiation error.
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
                      void>::value;
}

// Transport policy for the 3-thread ring shape: dedicated RX/TX adapter threads
// move bytes wire<->ring and own TX, so dispatch_core only polls/clears
// rx_flags and stores the response address into tx_flags (address-as-flag).
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
    uint64_t rx_value = as_atomic_u64(&rb->rx_flags_host[current_slot])
                            .load(std::memory_order_acquire);
    if (rx_value == 0) {
      if (!config->shared_ring_mode)
        return false;
      size_t probe = (current_slot + 1) % num_slots;
      size_t scanned = 0;
      while (scanned < num_slots - 1) {
        uint64_t v = as_atomic_u64(&rb->rx_flags_host[probe])
                         .load(std::memory_order_acquire);
        if (v != 0) {
          current_slot = probe;
          break;
        }
        probe = (probe + 1) % num_slots;
        ++scanned;
      }
      if (scanned >= num_slots - 1)
        return false;
      rx_value = as_atomic_u64(&rb->rx_flags_host[current_slot])
                     .load(std::memory_order_acquire);
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
    as_atomic_u64(&rb->tx_flags_host[it.slot])
        .store(reinterpret_cast<uint64_t>(tx_host), std::memory_order_release);
  }

  // Release the slot and step the cursor.  Foreign under shared_ring_mode
  // leaves rx_flags set so a peer dispatcher can pick the slot up (advance
  // cursor only); every other outcome claims the slot by clearing its rx_flag.
  void retire_request(const RxItem &it, SlotOutcome outcome) {
    if (!(outcome == SlotOutcome::Foreign && config->shared_ring_mode))
      as_atomic_u64(&rb->rx_flags_host[it.slot])
          .store(0, std::memory_order_release);
    current_slot = (static_cast<size_t>(it.slot) + 1) % num_slots;
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

// Drive `tp`'s ring until `*shutdown_flag != 0`.  Each turn: poll the next
// request, validate framing + resolve the entry, run its HOST_CALL handler
// inline via invoke_host_call and publish the response with tp.send_response.
// An entry of any other dispatch mode belongs to the device path and is
// retired as Foreign.
template <class Transport>
void dispatch_core(Transport &tp, const cudaq_function_table_t *table,
                   volatile int *shutdown_flag, uint64_t *stats) {
  const cudaq_ringbuffer_t *rb = tp.ringbuffer();
  // The dispatcher operates on a registered function table; there is nothing to
  // dispatch without one.
  if (!table || !table->entries || table->count == 0)
    return;
  // RX and TX strides are equal by configuration; the smaller is passed to
  // handlers defensively against a future asymmetric layout.
  const size_t slot_size =
      rb->rx_stride_sz < rb->tx_stride_sz ? rb->rx_stride_sz : rb->tx_stride_sz;
  uint64_t count = 0;

  while (as_atomic_int(shutdown_flag).load(std::memory_order_relaxed) == 0) {
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
    // DEVICE_CALL and GRAPH_LAUNCH entries belong to the device path: under
    // shared_ring_mode the slot is left for that peer, otherwise it is cleared.
    if (!entry || entry->dispatch_mode != CUDAQ_DISPATCH_HOST_CALL) {
      tp.retire_request(it, SlotOutcome::Foreign);
      continue;
    }

    void *tx_host =
        rb->tx_data_host + static_cast<size_t>(it.slot) * rb->tx_stride_sz;
    if (invoke_host_call(entry, it.rx_host, tx_host, slot_size)) {
      tp.send_response(it, tx_host);
      ++count;
    }
    tp.retire_request(it, SlotOutcome::Handled);
  }

  if (stats)
    *stats += count;
}

} // anonymous namespace

extern "C" void
cudaq_host_ring_dispatch_loop(const cudaq_ringbuffer_t *ringbuffer,
                              const cudaq_function_table_t *table,
                              const cudaq_dispatcher_config_t *config,
                              volatile int *shutdown_flag, uint64_t *stats) {
  assert(ringbuffer && "ringbuffer must be non-null");
  assert(table && "table must be non-null");
  assert(config && "config must be non-null");
  assert(shutdown_flag && "shutdown_flag must be non-null");

  RingTransport tp(ringbuffer, config);
  dispatch_core(tp, table, shutdown_flag, stats);
}

extern "C" void cudaq_host_unified_loop(cudaq_cpu_dataplane_t *cpu_dataplane,
                                        const cudaq_function_table_t *table,
                                        volatile int *shutdown_flag,
                                        uint64_t *stats) {
  assert(cpu_dataplane && "cpu_dataplane must be non-null");
  assert(cpu_dataplane->rx_poll && "data-plane must provide rx_poll");
  assert(cpu_dataplane->tx_publish && "data-plane must provide tx_publish");
  assert(table && "table must be non-null");
  assert(shutdown_flag && "shutdown_flag must be non-null");

  UnifiedTransport tp(cpu_dataplane);
  dispatch_core(tp, table, shutdown_flag, stats);
}
