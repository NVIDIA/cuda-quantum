/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file unified_device_transport.cuh
/// @brief Device-side data-plane a transport provides to the unified GPU
/// dispatch kernel.
///
/// The unified kernel (`unified_dispatch_core.cu`) does RDMA RX, RPC dispatch
/// and RDMA TX in one persistent single-thread kernel.  Everything
/// transport-specific reaches it through the three `__device__` hooks below
/// plus an opaque `void *ctx`, so the loop itself compiles with no transport
/// headers -- no DOCA, no verbs, no mlx5.
///
/// `rx_poll` and `tx_publish` deliberately mirror `cudaq_cpu_rx_poll_fn_t` and
/// `cudaq_cpu_tx_publish_fn_t` in `bridge/bridge_interface.h`, which serve the
/// equivalent host-side loop.  The one behavioural difference is documented on
/// `cudaq_dev_rx_poll`.
///
/// LINKAGE.  These are `extern __device__` functions resolved by `nvcc -dlink`,
/// NOT function pointers: a device function pointer is only valid inside the
/// module that defines it, so the implementations must be device-linked into
/// the same CUDA module as the kernel.  In practice that means the transport's
/// implementation TU and the core TU end up in one shared library.  Runtime
/// (`dlopen`) selection of a device data plane is not expressible in CUDA.

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/// Result of a device-side RX poll.
///
/// Deliberately not `cudaq_rx_status_t`, which the host data plane uses: that
/// enum carries CUDAQ_RX_EMPTY for "nothing yet, ask again", which a blocking
/// poll can never mean.  Keeping the types apart makes the value the device
/// contract forbids impossible to return, rather than something the dispatch
/// loop has to detect and reject.
typedef enum {
  CUDAQ_RX_DEV_READY = 0,    ///< A request is ready in `*out_frame`
  CUDAQ_RX_DEV_SHUTDOWN = 1, ///< Shutdown signalled; leave the loop
} cudaq_rx_dev_status_t;

/// Open the transport's device-side session.  Called once per block at kernel
/// entry, before any other hook.
///
/// The implementation materialises whatever per-block state it needs (derived
/// handles, cursors, shared-memory descriptors) from `ctx` and returns the
/// handle passed back to the other two hooks.
///
///   - `ctx`           : device-accessible transport context, forwarded
///                       verbatim from the launch wrapper.  Opaque here.
///   - `shutdown_flag` : device-visible flag `cudaq_dispatcher_stop` writes.
///                       The transport retains it, because `rx_poll` blocks
///                       and is therefore the only place it can be observed.
///
/// Returns the session handle, or `nullptr` if the transport cannot run (the
/// kernel then exits immediately).
extern __device__ void *cudaq_dev_transport_attach(void *ctx,
                                                   volatile int *shutdown_flag);

/// Claim the next inbound request.
///
/// Behavior: block until a request is ready, then point `*out_frame` at its
/// RPC header and return CUDAQ_RX_DEV_READY.  Return CUDAQ_RX_DEV_SHUTDOWN
/// instead, leaving `*out_frame` unchanged, once the shutdown flag handed to
/// `attach` is set.  Frames the transport itself cannot dispatch (bad
/// descriptors, out-of-range slots) are recycled internally and never
/// surface.
///
/// The frame pointer, not a slot index, is the handle: it is what the loop
/// must dereference, and handing it over directly keeps the transport's slot
/// addressing -- base, stride, count -- private inside `ctx`.  Pass it back
/// verbatim to `tx_publish`.
///
/// Contract: MUST observe the shutdown flag while it waits, since the dispatch
/// loop cannot check it meanwhile.  Blocking here is intentional and is the
/// one divergence from `cudaq_cpu_rx_poll_fn_t`, which must not block: the
/// host loop has to return every turn to service graph-launch completions,
/// whereas this path runs device calls only and has no other work to
/// interleave.  Returning on every empty poll would also put a cross-TU call
/// in the spin, so the wait stays inside the implementation.
extern __device__ cudaq_rx_dev_status_t cudaq_dev_rx_poll(void *session,
                                                          void **out_frame);

/// Transmit the response the dispatch loop has written into `frame`, and
/// return that slot's receive credit to the transport.
///
/// Behavior: whatever it takes to put the frame on the wire -- ordering
/// fences, descriptor preparation, ringing the doorbell -- plus re-arming the
/// slot to receive again.  Coupling the two is what keeps the transport's
/// receive bookkeeping out of the dispatch loop; it is also why a frame may
/// not be dropped silently.
///
/// Contract: `frame` MUST be a pointer `rx_poll` handed out, and MUST be
/// passed here exactly once -- INCLUDING frames the dispatcher could not
/// dispatch (unknown function, bad framing).  Skipping one leaks that slot's
/// receive credit, and the transport stalls once every slot has leaked.
extern __device__ void cudaq_dev_tx_publish(void *session, void *frame);

#ifdef __cplusplus
}
#endif
