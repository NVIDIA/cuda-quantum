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
/// address), no CUDA at runtime.
///
/// Arguments accepted by create() (unrecognized arguments are ignored so
/// callers can forward their full transport argument list):
///   --port=N        UDP port to bind (0 = ephemeral, default; read the
///                   bound port back via get_endpoint_info)
///   --num-slots=N   ring slots on both rings            [default 8]
///   --slot-size=N   slot stride in bytes on both rings  [default 256]
///
/// Lifecycle mapping:
///   create      transceiver construction + bind (port is known after this,
///               so get_endpoint_info is valid before connect/launch)
///   connect     no-op (connectionless)
///   launch      start the RX/TX ring threads
///   disconnect  close the socket and stop the threads
///   destroy     free the transceiver

#include "cudaq/realtime/cpu_transport/udp_wrapper.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

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
};

bool starts_with(const std::string &s, const char *prefix) {
  const size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

} // namespace

extern "C" {

static cudaq_status_t
udp_bridge_create(cudaq_realtime_bridge_handle_t *handle, int argc,
                  char **argv) {
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
      // Unrecognized arguments are ignored (callers forward their full
      // transport argument list).
    } catch (const std::exception &) {
      std::cerr << "ERROR: udp bridge: bad numeric value in '" << a << "'"
                << std::endl;
      delete ctx;
      return CUDAQ_ERR_INVALID_ARG;
    }
  }

  ctx->transceiver =
      cpu_udp_create_transceiver(ctx->slot_size, ctx->num_slots);
  if (!ctx->transceiver) {
    std::cerr << "ERROR: udp bridge: transceiver create failed" << std::endl;
    delete ctx;
    return CUDAQ_ERR_INTERNAL;
  }
  if (!cpu_udp_bind(ctx->transceiver, ctx->requested_port)) {
    std::cerr << "ERROR: udp bridge: bind(port=" << ctx->requested_port
              << ") failed" << std::endl;
    cpu_udp_destroy_transceiver(ctx->transceiver);
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
  if (context_type != RING_BUFFER)
    return CUDAQ_ERR_UNSUPPORTED;

  auto *ring = reinterpret_cast<cudaq_ringbuffer_t *>(out_context);
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

static cudaq_status_t
udp_bridge_connect(cudaq_realtime_bridge_handle_t handle) {
  // Connectionless: nothing to rendezvous with.
  return handle ? CUDAQ_OK : CUDAQ_ERR_INVALID_ARG;
}

static cudaq_status_t
udp_bridge_launch(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<UdpBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;
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
      /*get_cpu_dataplane=*/nullptr, // ring path only (unified shape is a
                                     // follow-up; see udp_wrapper.h ring
                                     // contract)
      udp_bridge_get_endpoint_info,
      udp_bridge_get_ring_geometry,
  };
  return &cudaq_udp_bridge_interface;
}

} // extern "C"
