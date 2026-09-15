/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file bridge_impl.cpp
/// @brief UDP bridge provider (libcudaq-realtime-bridge-udp.so).
///
/// Wraps the loopback/Ethernet UDP ring transceiver (udp_wrapper.h) behind
/// the transport-provider interface (bridge_interface.h), so a consumer that
/// speaks only cudaq_bridge_* / cudaq_dispatcher_* runs over UDP with zero
/// transport-specific code.  Deliberately the simplest provider: no peer
/// rendezvous (UDP is connectionless; responses go to each request's source
/// address); CUDA is touched only when --pinned-rings requests GPU-pollable
/// ring memory.
///
/// Arguments accepted by create() (unrecognized arguments are ignored so
/// callers can forward their full transport argument list):
///   --port=N          UDP port to bind (0 = ephemeral, default; read the
///                     bound port back via get_endpoint_info)
///   --num-slots=N     ring slots on both rings            [default 8]
///   --slot-size=N     slot stride in bytes on both rings  [default 256]
///   --pinned-rings    allocate the rings as CUDA pinned+mapped host memory
///                     so a GPU consumer (e.g. a device-resident dispatch
///                     scheduler) can poll them directly; requires a CUDA
///                     device at create()
///   --unified         serve the single-thread unified shape instead of the
///                     3-thread ring shape: the transceiver starts no pump
///                     threads and the consumer's dispatch thread drives the
///                     wire through the rx_poll/tx_publish hooks returned by
///                     get_cpu_dataplane.  get_transport_context(UNIFIED) then
///                     answers OK with a NULL launch_fn -- "unified, but the
///                     loop is yours" -- which is what sends a consumer on to
///                     get_cpu_dataplane.  The two shapes are exclusive, so
///                     RING_BUFFER is refused in this mode, and UNIFIED plus
///                     get_cpu_dataplane are refused without it.
///
/// This file stays a thin adapter in both shapes: the transceiver owns the
/// socket, the rings, the peer address, and both data planes (udp_wrapper.h,
/// "Unified mode"), and the hooks below only translate its 1/0 returns into the
/// interface's status enums.
///
/// Lifecycle mapping:
///   create      transceiver construction + bind (port is known after this,
///               so get_endpoint_info is valid before connect/launch)
///   connect     no-op (connectionless)
///   launch      start the RX/TX ring threads.  Under --unified there are no
///               such threads, so this is harmless but unnecessary; a unified
///               consumer normally skips it (starting a second loop that
///               raced the data-plane hooks is precisely what that shape
///               forbids).
///   disconnect  close the socket and stop the threads.  Under --unified the
///               consumer MUST stop its dispatcher first: the hooks touch the
///               socket this closes.
///   destroy     free the transceiver

#include "cudaq/realtime/cpu_transport/udp_wrapper.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>

namespace {

struct UdpBridgeContext {
  cpu_udp_transceiver_t transceiver = nullptr;
  uint16_t requested_port = 0;
  uint32_t num_slots = 8;
  uint32_t slot_size = 256;
  bool pinned_rings = false;
  bool unified = false;
  // Owned pinned+mapped buffers when pinned_rings is set (freed AFTER the
  // transceiver is destroyed).
  void *pinned[4] = {nullptr, nullptr, nullptr, nullptr};
};

void free_pinned(UdpBridgeContext *ctx) {
  for (auto *&buffer : ctx->pinned) {
    if (buffer)
      cudaFreeHost(buffer);
    buffer = nullptr;
  }
}

bool starts_with(const std::string &s, const char *prefix) {
  const size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

// Describe the transceiver's four rings.  Shared by both shapes: the ring
// shape hands this out as the transport context, the unified shape embeds it
// in the data-plane alongside the hooks.
cudaq_status_t fill_ring(UdpBridgeContext *ctx, cudaq_ringbuffer_t *ring) {
  auto *rx_flags = reinterpret_cast<volatile uint64_t *>(
      cpu_udp_get_rx_ring_flag_addr(ctx->transceiver));
  auto *tx_flags = reinterpret_cast<volatile uint64_t *>(
      cpu_udp_get_tx_ring_flag_addr(ctx->transceiver));
  auto *rx_data = reinterpret_cast<uint8_t *>(
      cpu_udp_get_rx_ring_data_addr(ctx->transceiver));
  auto *tx_data = reinterpret_cast<uint8_t *>(
      cpu_udp_get_tx_ring_data_addr(ctx->transceiver));
  if (!rx_flags || !tx_flags || !rx_data || !tx_data)
    return CUDAQ_ERR_INTERNAL;

  // Plain host memory: the device-pointer and host-view fields are the same
  // addresses (consumers running CUDAQ_DISPATCH_PATH_HOST read the _host
  // fields; nothing dereferences these as device pointers).
  ring->rx_flags = rx_flags;
  ring->tx_flags = tx_flags;
  ring->rx_data = rx_data;
  ring->tx_data = tx_data;
  ring->rx_stride_sz = ctx->slot_size;
  ring->tx_stride_sz = ctx->slot_size;
  ring->rx_flags_host = rx_flags;
  ring->tx_flags_host = tx_flags;
  ring->rx_data_host = rx_data;
  ring->tx_data_host = tx_data;
  return CUDAQ_OK;
}

} // namespace

extern "C" {

static cudaq_status_t udp_bridge_create(cudaq_realtime_bridge_handle_t *handle,
                                        int argc, char **argv) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;

  auto *ctx = new UdpBridgeContext();
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i] ? argv[i] : "";
    try {
      if (starts_with(a, "--port="))
        ctx->requested_port = static_cast<uint16_t>(std::stoul(a.substr(7)));
      else if (starts_with(a, "--num-slots="))
        ctx->num_slots = static_cast<uint32_t>(std::stoul(a.substr(12)));
      else if (starts_with(a, "--slot-size="))
        ctx->slot_size = static_cast<uint32_t>(std::stoul(a.substr(12)));
      else if (a == "--pinned-rings")
        ctx->pinned_rings = true;
      else if (a == "--unified")
        ctx->unified = true;
      // Unrecognized arguments are ignored (callers forward their full
      // transport argument list).
    } catch (const std::exception &) {
      std::cerr << "ERROR: udp bridge: bad numeric value in '" << a << "'"
                << std::endl;
      delete ctx;
      return CUDAQ_ERR_INVALID_ARG;
    }
  }

  if (ctx->pinned_rings) {
    // Pinned+mapped rings: a GPU consumer polls the same allocation through
    // its device alias (identical pointer under UVA), while this transport's
    // socket threads fill/drain it from the host.
    const size_t sizes[4] = {
        ctx->num_slots * sizeof(uint64_t), ctx->num_slots * sizeof(uint64_t),
        static_cast<size_t>(ctx->num_slots) * ctx->slot_size,
        static_cast<size_t>(ctx->num_slots) * ctx->slot_size};
    for (int i = 0; i < 4; ++i) {
      if (cudaHostAlloc(&ctx->pinned[i], sizes[i], cudaHostAllocMapped) !=
          cudaSuccess) {
        std::cerr << "ERROR: udp bridge: pinned ring alloc failed "
                     "(--pinned-rings requires a CUDA device)"
                  << std::endl;
        free_pinned(ctx);
        delete ctx;
        return CUDAQ_ERR_INTERNAL;
      }
      std::memset(ctx->pinned[i], 0, sizes[i]);
    }
    ctx->transceiver = cpu_udp_create_transceiver_ext(
        ctx->slot_size, ctx->num_slots,
        static_cast<volatile uint64_t *>(ctx->pinned[0]),
        static_cast<volatile uint64_t *>(ctx->pinned[1]),
        static_cast<uint8_t *>(ctx->pinned[2]),
        static_cast<uint8_t *>(ctx->pinned[3]));
  } else {
    ctx->transceiver =
        cpu_udp_create_transceiver(ctx->slot_size, ctx->num_slots);
  }
  if (!ctx->transceiver) {
    std::cerr << "ERROR: udp bridge: transceiver create failed" << std::endl;
    free_pinned(ctx);
    delete ctx;
    return CUDAQ_ERR_INTERNAL;
  }
  // Before the transceiver is started, which is what the mode changes.
  if (ctx->unified && !cpu_udp_set_unified(ctx->transceiver, 1)) {
    std::cerr << "ERROR: udp bridge: could not select unified mode"
              << std::endl;
    cpu_udp_destroy_transceiver(ctx->transceiver);
    free_pinned(ctx);
    delete ctx;
    return CUDAQ_ERR_INTERNAL;
  }
  if (!cpu_udp_bind(ctx->transceiver, ctx->requested_port)) {
    std::cerr << "ERROR: udp bridge: bind(port=" << ctx->requested_port
              << ") failed" << std::endl;
    cpu_udp_destroy_transceiver(ctx->transceiver);
    free_pinned(ctx);
    delete ctx;
    return CUDAQ_ERR_INTERNAL;
  }

  *handle = ctx;
  return CUDAQ_OK;
}

static cudaq_status_t
udp_bridge_destroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<UdpBridgeContext *>(handle);
  if (ctx->transceiver)
    cpu_udp_destroy_transceiver(ctx->transceiver);
  free_pinned(ctx); // after destroy: the transceiver threads use the rings
  delete ctx;
  return CUDAQ_OK;
}

static cudaq_status_t udp_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t handle,
    cudaq_realtime_transport_context_t context_type, void *out_context) {
  if (!handle || !out_context)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<UdpBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;

  // A consumer asks here for whichever shape it means to serve, so both are
  // answerable.  Under --unified the answer is "unified, but the loop is
  // yours": no dispatcher override, and the hooks that drive it come from
  // get_cpu_dataplane.  Both fields are written rather than left alone, so a
  // caller that did not zero-initialize reads "no override" and not stack
  // garbage it would then try to call.
  if (context_type == UNIFIED) {
    if (!ctx->unified)
      return CUDAQ_ERR_UNSUPPORTED;
    auto *unified_ctx =
        reinterpret_cast<cudaq_unified_dispatch_ctx_t *>(out_context);
    unified_ctx->launch_fn = nullptr;
    unified_ctx->transport_ctx = nullptr;
    return CUDAQ_OK;
  }
  if (context_type != RING_BUFFER)
    return CUDAQ_ERR_UNSUPPORTED;
  // Report the rings only in the shape that uses them.  Handing them out under
  // --unified would let a consumer poll rx_flags that the unified data path
  // never sets, which looks like a transport silently dropping every request;
  // refusing turns that into a consumer-side wiring error naming the shape.
  if (ctx->unified)
    return CUDAQ_ERR_UNSUPPORTED;

  return fill_ring(ctx, reinterpret_cast<cudaq_ringbuffer_t *>(out_context));
}

// The unified data plane's two hooks, translating the transceiver's 1/0
// returns into the interface's status enums.  `ctx` is the transceiver handle
// itself, so they carry no state of their own -- the socket, the rings, the
// peer address and the RX cursor all live one layer down, shared with the
// threaded shape rather than duplicated here.
static cudaq_rx_status_t udp_dp_rx_poll(void *ctx, uint32_t *out_slot) {
  return cpu_udp_rx_poll(ctx, out_slot) ? CUDAQ_RX_READY : CUDAQ_RX_EMPTY;
}

static cudaq_status_t udp_dp_tx_publish(void *ctx, uint32_t slot) {
  return cpu_udp_tx_publish(ctx, slot) ? CUDAQ_OK : CUDAQ_ERR_INTERNAL;
}

// Serve the single-thread unified shape: the ring plus the two hooks that
// drive it.  Refused unless --unified was passed, so a consumer that asks for
// this shape against a ring-mode bridge gets a clean UNSUPPORTED rather than
// hooks that race the pump threads.
static cudaq_status_t
udp_bridge_get_cpu_dataplane(cudaq_realtime_bridge_handle_t handle,
                             cudaq_cpu_dataplane_t *out_dataplane) {
  if (!handle || !out_dataplane)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<UdpBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;
  if (!ctx->unified)
    return CUDAQ_ERR_UNSUPPORTED;

  const cudaq_status_t status = fill_ring(ctx, &out_dataplane->ring);
  if (status != CUDAQ_OK)
    return status;
  out_dataplane->ctx = ctx->transceiver;
  out_dataplane->rx_poll = udp_dp_rx_poll;
  out_dataplane->tx_publish = udp_dp_tx_publish;
  return CUDAQ_OK;
}

static cudaq_status_t
udp_bridge_connect(cudaq_realtime_bridge_handle_t handle) {
  // Connectionless: nothing to rendezvous with.
  return handle ? CUDAQ_OK : CUDAQ_ERR_INVALID_ARG;
}

static cudaq_status_t udp_bridge_launch(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<UdpBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;
  // No special case for --unified: cpu_udp_start starts no pump threads in
  // that mode, so there is nothing here that could race the hooks.
  if (!cpu_udp_start(ctx->transceiver)) {
    std::cerr << "ERROR: udp bridge: transceiver start failed" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  return CUDAQ_OK;
}

static cudaq_status_t
udp_bridge_disconnect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<UdpBridgeContext *>(handle);
  if (ctx->transceiver)
    cpu_udp_close(ctx->transceiver);
  return CUDAQ_OK;
}

static cudaq_status_t
udp_bridge_get_endpoint_info(cudaq_realtime_bridge_handle_t handle, char *buf,
                             size_t buf_len) {
  if (!handle || !buf || buf_len == 0)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<UdpBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;
  const int n =
      std::snprintf(buf, buf_len, "transport=udp port=%u",
                    static_cast<unsigned>(cpu_udp_get_port(ctx->transceiver)));
  return (n > 0 && static_cast<size_t>(n) < buf_len) ? CUDAQ_OK
                                                     : CUDAQ_ERR_INVALID_ARG;
}

static cudaq_status_t
udp_bridge_get_ring_geometry(cudaq_realtime_bridge_handle_t handle,
                             uint32_t *out_num_slots, uint32_t *out_slot_size) {
  if (!handle || !out_num_slots || !out_slot_size)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<UdpBridgeContext *>(handle);
  *out_num_slots = ctx->num_slots;
  *out_slot_size = ctx->slot_size;
  return CUDAQ_OK;
}

cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface() {
  static cudaq_realtime_bridge_interface_t cudaq_udp_bridge_interface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      udp_bridge_create,
      udp_bridge_destroy,
      udp_bridge_get_transport_context,
      udp_bridge_connect,
      udp_bridge_launch,
      udp_bridge_disconnect,
      udp_bridge_get_cpu_dataplane,
      udp_bridge_get_endpoint_info,
      udp_bridge_get_ring_geometry,
      /*set_function_table=*/nullptr, // no function table needed
  };
  return &cudaq_udp_bridge_interface;
}

} // extern "C"
