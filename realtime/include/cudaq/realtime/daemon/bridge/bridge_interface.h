/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file bridge_interface.h
/// @brief Interface Bindings for transport layer providers (e.g. Hololink).
///
/// Different transport providers can be loaded at runtime via `dlopen`,
/// allowing for dynamic selection and initialization of the desired transport
/// layer.  Callers name the provider library directly via
/// `cudaq_bridge_create_from_library`; loaded libraries are cached per
/// process keyed by that name, so multiple distinct providers can coexist in
/// one process.  The enum-based `cudaq_bridge_create` remains as a
/// convenience wrapper (built-in Hololink name, or the library named by the
/// CUDAQ_REALTIME_BRIDGE_LIB environment variable).

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#ifdef __cplusplus
extern "C" {
#endif

///@brief Opaque data structure storing the details of the transport layer
/// connection
typedef void *cudaq_realtime_bridge_handle_t;

typedef enum {
  CUDAQ_PROVIDER_HOLOLINK =
      0, /// Hololink GPU-RoCE transceiver (built-in provider)
  CUDAQ_PROVIDER_EXTERNAL = 1, /// Externally managed transport

} cudaq_realtime_transport_provider_t;

typedef enum {
  RING_BUFFER = 0, // Ring buffer context (for Hololink provider)
  UNIFIED = 1,     /// Unified transport context  for unified dispatch
} cudaq_realtime_transport_context_t;

/// Result of a non-blocking RX poll on the ringbuffer dataplane.
typedef enum {
  CUDAQ_RX_READY = 0, ///< A request is ready
  CUDAQ_RX_EMPTY = 1, ///< No request ready
} cudaq_rx_status_t;

/// Non-blocking inbound-poll hook.
///
/// This hook is necessary for providers supporting the unified dispatcher loop.
///
/// Behavior: check whether the transport has a fully received RPC request
/// ready for the dispatcher to process:
///   - If one is ready, claim it (advance the transport's RX cursor), write its
///     ringbuffer slot index to `*out_slot`, and return CUDAQ_RX_READY. The
///     request's RPCHeader + payload MUST already be fully written and visible
///     to the CPU before this returns (i.e. publish with acquire/release
///     ordering, not just a bare store).
///
///   - Otherwise leave `*out_slot` unchanged and return CUDAQ_RX_EMPTY.
///
/// Interface:
///   - `ctx`      : the provider's own state pointer, handed back unchanged
///                  from the `ctx` field the provider set in
///                  `cudaq_cpu_dataplane_t`.
///
///   - `out_slot` : output parameter, set to a slot index in `[0, num_slots)`
///                  only when returning CUDAQ_RX_READY.
///
/// Contract: MUST NOT block (return CUDAQ_RX_EMPTY instead of waiting) and
/// should be cheap, the dispatch loop calls it in a tight spin.
typedef cudaq_rx_status_t (*cudaq_cpu_rx_poll_fn_t)(void *ctx,
                                                    uint32_t *out_slot);

/// Outbound-publish hook a unified transport provider must implement.
///
/// This hook is necessary for providers supporting the unified dispatcher loop.
///
/// Behavior: transmit the response the dispatch loop has just written into the
/// TX slot at `slot`. The implementation is responsible for whatever it takes
/// to put that slot on the wire: an ordering/visibility fence, any
/// host<->device or device<->wire copy, and ringing the transport's TX
/// doorbell.
///
/// Interface:
///   - `ctx`    : the provider's own state pointer, handed back unchanged from
///                the `ctx` field the provider set in `cudaq_cpu_dataplane_t`.
///   - `slot`   : the ring slot index to publish.
///   - returns  : CUDAQ_OK on success, or a cudaq_status_t error code.
///
/// Contract: invoked only from the dispatcher's single unified thread.  It is
/// slot-addressed rather than FIFO, so responses MAY be published out of the
/// order their requests arrived.
typedef cudaq_status_t (*cudaq_cpu_tx_publish_fn_t)(void *ctx, uint32_t slot);

/// The CPU/host-driven unified data-plane: a device-visible ringbuffer plus the
/// two host-callable ops that drive it, which a transport provider exposes for
/// the CPU dispatch loop.
///
/// The library's single-thread unified dispatch loop
/// (`cudaq_host_unified_loop`, used for CUDAQ_DISPATCH_PATH_HOST +
/// CUDAQ_KERNEL_UNIFIED) owns one of these and, on its own thread, repeatedly
/// calls `rx_poll` to pull the next request out of `ring`, dispatches it
/// (HOST_CALL inline, GRAPH_LAUNCH via the graph engine), and calls
/// `tx_publish` to send each response back out `ring`.  A provider that
/// supports this shape returns a fully populated instance from its
/// `get_cpu_dataplane` callback; providers that do not leave that callback
/// NULL.
typedef struct {
  void *ctx; ///< Transport-resident ring state.  Passed verbatim as the first
             ///< argument to `rx_poll` and `tx_publish`; opaque to the library.
  cudaq_ringbuffer_t ring; ///< Device-visible RX/TX data + tx flags together
                           ///< with their host-mapped views.  The unified loop
                           ///< reads `rx_data_host` / `tx_data_host` (and
                           ///< `tx_flags_host` to detect GRAPH_LAUNCH graph
                           ///< completion), so the host-view pointers and slot
                           ///< strides MUST be populated.
  cudaq_cpu_rx_poll_fn_t rx_poll; ///< Non-blocking inbound poll; required.
  cudaq_cpu_tx_publish_fn_t tx_publish; ///< Outbound publish; required.
} cudaq_cpu_dataplane_t;

/// @brief Create and initialize a transport bridge from an explicit provider
/// library.  `library` is any string `dlopen` accepts: a bare soname resolved
/// via the usual load paths (e.g. "libcudaq-realtime-bridge-udp.so") or an
/// absolute/relative path.  Provider libraries are cached per process keyed
/// by this string, so any number of DISTINCT provider libraries may coexist
/// in one process, each serving any number of bridge instances.  This is the
/// preferred entry point; every provider -- including the ones shipped with
/// this library -- is just a library name here.
cudaq_status_t cudaq_bridge_create_from_library(
    cudaq_realtime_bridge_handle_t *out_bridge_handle, const char *library,
    int argc, char **argv);

/// @brief Create and initialize a transport bridge for the specified provider
/// enum.  A convenience wrapper over `cudaq_bridge_create_from_library`:
/// CUDAQ_PROVIDER_HOLOLINK resolves to the bundled Hololink library name, and
/// CUDAQ_PROVIDER_EXTERNAL resolves to the library named by the
/// CUDAQ_REALTIME_BRIDGE_LIB environment variable.  New callers should prefer
/// `cudaq_bridge_create_from_library` and pass the library name directly.
cudaq_status_t
cudaq_bridge_create(cudaq_realtime_bridge_handle_t *out_bridge_handle,
                    cudaq_realtime_transport_provider_t provider, int argc,
                    char **argv);

/// @brief Destroy the transport bridge and release all associated resources.
cudaq_status_t cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t bridge);

/// @brief Retrieve the transport context for the given bridge.
/// This could be a ring buffer or unified context.
cudaq_status_t cudaq_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t bridge,
    cudaq_realtime_transport_context_t context_type, void *out_context);

/// @brief Connect the transport bridge.
cudaq_status_t cudaq_bridge_connect(cudaq_realtime_bridge_handle_t bridge);

/// @brief Launch the transport bridge's main processing loop (e.g. start
/// Hololink kernels).
cudaq_status_t cudaq_bridge_launch(cudaq_realtime_bridge_handle_t bridge);

/// @brief Disconnect the transport bridge (e.g. stop Hololink kernels and
/// disconnect).
cudaq_status_t cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t bridge);

/// @brief Retrieve the CPU data-plane for the single-thread unified host
/// dispatch loop.  Returns CUDAQ_ERR_UNSUPPORTED when the provider predates
/// interface version 2 or does not implement the unified shape.
cudaq_status_t
cudaq_bridge_get_cpu_dataplane(cudaq_realtime_bridge_handle_t bridge,
                               cudaq_cpu_dataplane_t *out_dataplane);

/// @brief Write a one-line, space-separated `key=value` description of the
/// provider's live endpoint (e.g. "transport=udp port=45678" or
/// "transport=cpu_roce port=9000 roce_ip=10.0.0.2 qp=0x1a rkey=1234") into
/// `buf`.  Valid as soon as create() returns, so a server can publish its
/// rendezvous endpoint BEFORE connect() blocks waiting for the peer.  Returns
/// CUDAQ_ERR_UNSUPPORTED when the provider predates interface version 2 or
/// has nothing to report.
cudaq_status_t
cudaq_bridge_get_endpoint_info(cudaq_realtime_bridge_handle_t bridge, char *buf,
                               size_t buf_len);

/// @brief Retrieve the provider's ring geometry so dispatcher configuration
/// can be derived from the transport instead of duplicated by the caller.
/// Returns CUDAQ_ERR_UNSUPPORTED when the provider predates interface
/// version 2.
cudaq_status_t
cudaq_bridge_get_ring_geometry(cudaq_realtime_bridge_handle_t bridge,
                               uint32_t *out_num_slots,
                               uint32_t *out_slot_size);

/// Version 2 adds the capability queries after `disconnect`:
/// `get_cpu_dataplane`, `get_endpoint_info`, and `get_ring_geometry`.  The
/// loader accepts providers reporting any version in [1, CURRENT]; fields
/// beyond `disconnect` are only read from providers reporting version >= 2
/// (a v1 provider's struct may simply end at `disconnect`).  A v2 provider
/// sets entries it does not support to NULL; the corresponding API calls
/// return CUDAQ_ERR_UNSUPPORTED.
#define CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION 2

/// @brief Interface struct for transport layer providers.  Each provider must
/// implement this interface and provide a `getter` function
/// (`cudaq_realtime_get_bridge_interface`) that returns a pointer to a
/// statically allocated instance of this struct with the function pointers set
/// to the provider's implementation.
typedef struct {
  int version;
  cudaq_status_t (*create)(cudaq_realtime_bridge_handle_t *, int, char **);
  cudaq_status_t (*destroy)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*get_transport_context)(cudaq_realtime_bridge_handle_t,
                                          cudaq_realtime_transport_context_t,
                                          void *);
  cudaq_status_t (*connect)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*launch)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*disconnect)(cudaq_realtime_bridge_handle_t);

  //--------------------------------------------------------------------------
  // Version 2 fields.  Read only when `version >= 2`; each may be NULL when
  // the provider does not support the capability (the API wrappers then
  // return CUDAQ_ERR_UNSUPPORTED).
  //--------------------------------------------------------------------------

  /// Fills *out with the ring data-plane the library's single-thread
  /// unified CPU loop drives. `out` must be set to NULL if the transport does
  /// not support the unified shape.
  cudaq_status_t (*get_cpu_dataplane)(cudaq_realtime_bridge_handle_t,
                                      cudaq_cpu_dataplane_t *out);

  /// One-line `key=value` endpoint description; see
  /// cudaq_bridge_get_endpoint_info.
  cudaq_status_t (*get_endpoint_info)(cudaq_realtime_bridge_handle_t, char *buf,
                                      size_t buf_len);

  /// Ring geometry (slot count / slot stride); see
  /// cudaq_bridge_get_ring_geometry.
  cudaq_status_t (*get_ring_geometry)(cudaq_realtime_bridge_handle_t,
                                      uint32_t *out_num_slots,
                                      uint32_t *out_slot_size);

} cudaq_realtime_bridge_interface_t;

#ifdef __cplusplus
}
#endif
