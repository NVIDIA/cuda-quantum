/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file bridge_impl.cpp
/// @brief CPU RoCE bridge provider (libcudaq-realtime-bridge-cpu-roce.so).
///
/// Wraps the CPU RoCE RDMA ring transceiver (roce_wrapper.h) behind the
/// transport-provider interface (bridge_interface.h), including BOTH queue-
/// pair exchange methods that previously lived inline in consumers:
///
///   --qp_config=rendezvous (default)
///     TCP QP/rkey rendezvous with a CpuRoceChannel caller.  The RendezvousInfo
///     wire struct here is the SERVICE end of CpuRoceChannel's exchange (the
///     server reads the caller's {qp, rkey, ip} first, then replies) and lives
///     next to the transceiver it describes, so the byte-for-byte contract is
///     maintained in one repository.
///
///   --qp_config=hsb_fpga
///     Holoscan-Sensor-Bridge FPGA method (the hsb_bridge_cpu.cpp precedent):
///     the peer QP is an argument (the FPGA's fixed data-plane QP, or the
///     emulator's), the transceiver is created one-shot with the peer already
///     known, and create() prints this end's QP / RKey / Buffer Addr in the
///     canonical bridge handshake format ("  KEY: VALUE", single space after
///     the colon -- hololink_bridge_common.h; orchestration scripts parse it
///     with strict regexes, same contract as the Hololink bridges), so the
///     banner precedes any consumer readiness line.  The provider performs
///     NO Hololink control-plane traffic; the playback tool alone programs
///     the FPGA.
///
/// tx_mode is always CPU_ROCE_TX_MODE_RDMA_SEND: this end Sends responses;
/// the peer (channel or FPGA) Writes requests into our RX ring.
///
/// Arguments accepted by create() (unrecognized arguments are ignored so
/// callers can forward their full transport argument list):
///   --device=NAME     RDMA device                        [default mlx5_0]
///   --local-ip=ADDR   local RoCE IPv4                    [default 10.0.0.2]
///   --port=N          rendezvous TCP port (0 = ephemeral; rendezvous only)
///   --num-slots=N     ring slots on both rings           [default 8; capped
///                     at 64 for hsb_fpga -- the HSB WQE depth]
///   --slot-size=N     slot stride in bytes               [default 256]
///   --qp_config=rendezvous|hsb_fpga                      [default rendezvous]
///   --peer-ip=ADDR    FPGA/emulator data-plane IPv4      (hsb_fpga: required)
///   --remote-qp=N     FPGA/emulator data-plane QP, decimal or 0x-hex
///                     (hsb_fpga only)                    [default 0x2]
///   --frame-size=N    TX SGE bytes (hsb_fpga only; 0 => slot-size)
///
/// Lifecycle mapping (both qp_configs):
///   create      transceiver bring-up to the point where this end's endpoint
///               identity is known -- rendezvous: setup() + TCP listen (the
///               rendezvous port is known); hsb_fpga: one-shot start() + the
///               canonical handshake banner (QP / RKey / Buffer Addr known).
///               get_endpoint_info is therefore valid BEFORE connect()
///               blocks on the peer.
///   connect     rendezvous: accept() + QP/rkey swap + connect() -- BLOCKS
///               until the caller channel dials in; hsb_fpga: no-op (the
///               FPGA is programmed out-of-band from the create() banner).
///   launch      start the blocking-monitor I/O thread
///   disconnect  close the transceiver and join the monitor thread
///   destroy     free the transceiver

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"
#include "cudaq/realtime/daemon/bridge/bridge_interface.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

namespace {

// Must match CpuRoceChannel's RendezvousInfo byte-for-byte (network order).
struct RendezvousInfo {
  uint32_t qp_number = 0;
  uint32_t rkey = 0;
  uint32_t roce_ipv4 = 0;
};

struct CpuRoceBridgeContext {
  // Parsed configuration.
  std::string device = "mlx5_0";
  std::string local_ip = "10.0.0.2";
  std::string qp_config = "rendezvous";
  std::string peer_ip;         // hsb_fpga: FPGA/emulator data-plane IPv4
  uint16_t tcp_port = 0;       // rendezvous TCP port (0 = ephemeral)
  uint32_t remote_qp = 0x2;    // hsb_fpga: FPGA data-plane QP
  uint32_t num_slots = 8;
  uint32_t slot_size = 256;
  size_t frame_size = 0;       // hsb_fpga TX SGE bytes; 0 => slot_size

  // Live state.
  cpu_roce_transceiver_t transceiver = nullptr;
  int listen_fd = -1;          // rendezvous: bound+listening after create()
  uint16_t bound_port = 0;     // rendezvous: actual TCP port after create()
  std::thread monitor;         // started by launch()
  bool connected = false;
};

bool starts_with(const std::string &s, const char *prefix) {
  const size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

bool write_all(int fd, const void *buf, size_t len) {
  const auto *p = static_cast<const uint8_t *>(buf);
  while (len > 0) {
    const ssize_t n = ::write(fd, p, len);
    if (n <= 0) {
      if (n < 0 && errno == EINTR)
        continue;
      return false;
    }
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

bool read_all(int fd, void *buf, size_t len) {
  auto *p = static_cast<uint8_t *>(buf);
  while (len > 0) {
    const ssize_t n = ::read(fd, p, len);
    if (n <= 0) {
      if (n < 0 && errno == EINTR)
        continue;
      return false;
    }
    p += n;
    len -= static_cast<size_t>(n);
  }
  return true;
}

void teardown(CpuRoceBridgeContext *ctx) {
  if (ctx->listen_fd >= 0) {
    ::close(ctx->listen_fd);
    ctx->listen_fd = -1;
  }
  if (ctx->transceiver) {
    cpu_roce_close(ctx->transceiver);
    if (ctx->monitor.joinable())
      ctx->monitor.join();
    cpu_roce_destroy_transceiver(ctx->transceiver);
    ctx->transceiver = nullptr;
  }
}

// Rendezvous-mode create: transceiver setup() (QP/rkey known, peer not yet)
// plus the TCP listen socket, so the rendezvous endpoint is publishable
// before connect() blocks in accept().
cudaq_status_t create_rendezvous(CpuRoceBridgeContext *ctx) {
  ctx->transceiver = cpu_roce_create_transceiver(
      ctx->device.c_str(), /*ib_port=*/1, /*tx_ibv_qp=*/0u,
      /*frame_size=*/ctx->slot_size, /*page_size=*/ctx->slot_size,
      ctx->num_slots, /*peer_ip=*/"0.0.0.0", /*forward=*/0, /*rx_only=*/0,
      /*tx_only=*/0, /*unified=*/0, CPU_ROCE_TX_MODE_RDMA_SEND,
      /*peer_rx_base_addr=*/0, /*peer_rx_rkey=*/0);
  if (!ctx->transceiver) {
    std::cerr << "ERROR: cpu_roce bridge: transceiver create failed"
              << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  cpu_roce_set_local_ip(ctx->transceiver, ctx->local_ip.c_str());
  if (!cpu_roce_setup(ctx->transceiver)) {
    std::cerr << "ERROR: cpu_roce bridge: transceiver setup() failed"
              << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  ctx->listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (ctx->listen_fd < 0) {
    std::cerr << "ERROR: cpu_roce bridge: rendezvous socket() failed"
              << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  int reuse = 1;
  ::setsockopt(ctx->listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse,
               sizeof(reuse));
  sockaddr_in srv{};
  srv.sin_family = AF_INET;
  srv.sin_addr.s_addr = htonl(INADDR_ANY);
  srv.sin_port = htons(ctx->tcp_port);
  if (::bind(ctx->listen_fd, reinterpret_cast<sockaddr *>(&srv),
             sizeof(srv)) != 0 ||
      ::listen(ctx->listen_fd, 1) != 0) {
    std::cerr << "ERROR: cpu_roce bridge: rendezvous bind/listen failed"
              << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  socklen_t srvlen = sizeof(srv);
  ::getsockname(ctx->listen_fd, reinterpret_cast<sockaddr *>(&srv), &srvlen);
  ctx->bound_port = ntohs(srv.sin_port);
  return CUDAQ_OK;
}

// hsb_fpga-mode create: one-shot bring-up with the peer already known (the
// hsb_bridge_cpu.cpp precedent).  QP / RKey / Buffer Addr are valid after
// this returns.
cudaq_status_t create_hsb_fpga(CpuRoceBridgeContext *ctx) {
  const size_t frame_size = ctx->frame_size ? ctx->frame_size : ctx->slot_size;

  // The HSB receive queue is WQE_NUM=64 deep; a deeper ring would alias two
  // slots per WQE and race RX against TX (same constraint as the Hololink
  // bridges).
  constexpr uint32_t kHsbWqeNum = 64;
  if (ctx->num_slots > kHsbWqeNum) {
    std::cerr << "WARNING: cpu_roce bridge: --num-slots=" << ctx->num_slots
              << " exceeds the HSB WQE depth; clamping to " << kHsbWqeNum
              << std::endl;
    ctx->num_slots = kHsbWqeNum;
  }

  std::cout << "HSB FPGA QP exchange:\n"
            << "  Device:     " << ctx->device << "\n"
            << "  Peer IP:    " << ctx->peer_ip << "\n"
            << "  Remote QP:  0x" << std::hex << ctx->remote_qp << std::dec
            << "\n"
            << "  Slots:      " << ctx->num_slots << "\n"
            << "  Slot size:  " << ctx->slot_size << " bytes\n"
            << "  Frame size: " << frame_size << " bytes" << std::endl;

  ctx->transceiver = cpu_roce_create_transceiver(
      ctx->device.c_str(), /*ib_port=*/1, /*tx_ibv_qp=*/ctx->remote_qp,
      frame_size, /*page_size=*/ctx->slot_size, ctx->num_slots,
      ctx->peer_ip.c_str(), /*forward=*/0, /*rx_only=*/0, /*tx_only=*/0,
      /*unified=*/0, CPU_ROCE_TX_MODE_RDMA_SEND, /*peer_rx_base_addr=*/0,
      /*peer_rx_rkey=*/0);
  if (!ctx->transceiver) {
    std::cerr << "ERROR: cpu_roce bridge: transceiver create failed"
              << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  if (!cpu_roce_start(ctx->transceiver)) {
    std::cerr << "ERROR: cpu_roce bridge: cpu_roce_start failed" << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }

  // Publish this end's identity in the canonical bridge handshake format
  // ("  KEY: VALUE", single space after the colon -- hololink_bridge_common.h)
  // for the orchestration script to relay to the playback tool, which alone
  // programs the FPGA over the Hololink control plane.  Printed from create()
  // (not connect()) so the banner precedes any consumer readiness line, the
  // same ordering hsb_bridge_cpu.cpp established.  Buffer Addr is 0 with an
  // iova=0 MR registration; the playback tool handles that.
  std::cout << "\n=== Bridge Ready ===" << std::endl;
  std::cout << "  QP Number: 0x" << std::hex
            << cpu_roce_get_qp_number(ctx->transceiver) << std::dec
            << std::endl;
  std::cout << "  RKey: " << cpu_roce_get_rkey(ctx->transceiver) << std::endl;
  std::cout << "  Buffer Addr: 0x" << std::hex
            << cpu_roce_get_buffer_addr(ctx->transceiver) << std::dec
            << std::endl;
  std::cout.flush();
  return CUDAQ_OK;
}

} // namespace

extern "C" {

static cudaq_status_t
cpu_roce_bridge_create(cudaq_realtime_bridge_handle_t *handle, int argc,
                       char **argv) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;

  auto *ctx = new CpuRoceBridgeContext();
  for (int i = 0; i < argc; ++i) {
    const std::string a = argv[i] ? argv[i] : "";
    try {
      if (starts_with(a, "--device="))
        ctx->device = a.substr(9);
      else if (starts_with(a, "--local-ip="))
        ctx->local_ip = a.substr(11);
      else if (starts_with(a, "--port="))
        ctx->tcp_port = static_cast<uint16_t>(std::stoul(a.substr(7)));
      else if (starts_with(a, "--num-slots="))
        ctx->num_slots = static_cast<uint32_t>(std::stoul(a.substr(12)));
      else if (starts_with(a, "--slot-size="))
        ctx->slot_size = static_cast<uint32_t>(std::stoul(a.substr(12)));
      else if (starts_with(a, "--qp_config="))
        ctx->qp_config = a.substr(12);
      else if (starts_with(a, "--peer-ip="))
        ctx->peer_ip = a.substr(10);
      else if (starts_with(a, "--remote-qp="))
        // base 0: accepts both decimal and 0x-prefixed hex (QP numbers are
        // conventionally printed in hex, e.g. the FPGA's fixed 0x2).
        ctx->remote_qp =
            static_cast<uint32_t>(std::stoul(a.substr(12), nullptr, 0));
      else if (starts_with(a, "--frame-size="))
        ctx->frame_size = std::stoull(a.substr(13));
      // Unrecognized arguments are ignored (callers forward their full
      // transport argument list).
    } catch (const std::exception &) {
      std::cerr << "ERROR: cpu_roce bridge: bad numeric value in '" << a << "'"
                << std::endl;
      delete ctx;
      return CUDAQ_ERR_INVALID_ARG;
    }
  }

  cudaq_status_t status;
  if (ctx->qp_config == "rendezvous") {
    status = create_rendezvous(ctx);
  } else if (ctx->qp_config == "hsb_fpga") {
    if (ctx->peer_ip.empty()) {
      std::cerr << "ERROR: cpu_roce bridge: --qp_config=hsb_fpga requires "
                   "--peer-ip=<FPGA or emulator IPv4>"
                << std::endl;
      delete ctx;
      return CUDAQ_ERR_INVALID_ARG;
    }
    status = create_hsb_fpga(ctx);
  } else {
    std::cerr << "ERROR: cpu_roce bridge: unknown --qp_config="
              << ctx->qp_config << " (expected rendezvous or hsb_fpga)"
              << std::endl;
    delete ctx;
    return CUDAQ_ERR_INVALID_ARG;
  }

  if (status != CUDAQ_OK) {
    teardown(ctx);
    delete ctx;
    return status;
  }
  *handle = ctx;
  return CUDAQ_OK;
}

static cudaq_status_t
cpu_roce_bridge_destroy(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<CpuRoceBridgeContext *>(handle);
  teardown(ctx);
  delete ctx;
  return CUDAQ_OK;
}

static cudaq_status_t cpu_roce_bridge_get_transport_context(
    cudaq_realtime_bridge_handle_t handle,
    cudaq_realtime_transport_context_t context_type, void *out_context) {
  if (!handle || !out_context)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<CpuRoceBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;
  if (context_type != RING_BUFFER)
    return CUDAQ_ERR_UNSUPPORTED;

  auto *ring = reinterpret_cast<cudaq_ringbuffer_t *>(out_context);
  auto *rx_flags = reinterpret_cast<volatile uint64_t *>(
      cpu_roce_get_rx_ring_flag_addr(ctx->transceiver));
  auto *tx_flags = reinterpret_cast<volatile uint64_t *>(
      cpu_roce_get_tx_ring_flag_addr(ctx->transceiver));
  auto *rx_data =
      reinterpret_cast<uint8_t *>(cpu_roce_get_rx_ring_data_addr(ctx->transceiver));
  auto *tx_data =
      reinterpret_cast<uint8_t *>(cpu_roce_get_tx_ring_data_addr(ctx->transceiver));
  if (!rx_flags || !tx_flags || !rx_data || !tx_data)
    return CUDAQ_ERR_INTERNAL;

  // Host memory (the CPU RoCE rings are host allocations the NIC DMAs into);
  // device-pointer and host-view fields are the same addresses.
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
cpu_roce_bridge_connect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<CpuRoceBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;
  if (ctx->connected)
    return CUDAQ_OK;

  if (ctx->qp_config == "hsb_fpga") {
    // No wire traffic: the FPGA side is programmed out-of-band by the
    // playback tool using the handshake banner create() already printed.
    ctx->connected = true;
    return CUDAQ_OK;
  }

  // Rendezvous: mirror of CpuRoceChannel::exchangeRendezvous (the server
  // reads the caller's {qp, rkey, ip} first, then replies).  Blocks in
  // accept() until the caller channel dials in.
  const int conn_fd = ::accept(ctx->listen_fd, nullptr, nullptr);
  ::close(ctx->listen_fd);
  ctx->listen_fd = -1;
  if (conn_fd < 0) {
    std::cerr << "ERROR: cpu_roce bridge: rendezvous accept() failed"
              << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  int one = 1;
  ::setsockopt(conn_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  RendezvousInfo peer{};
  in_addr local_addr{};
  if (::inet_pton(AF_INET, ctx->local_ip.c_str(), &local_addr) != 1) {
    std::cerr << "ERROR: cpu_roce bridge: invalid --local-ip '"
              << ctx->local_ip << "'" << std::endl;
    ::close(conn_fd);
    return CUDAQ_ERR_INVALID_ARG;
  }
  const RendezvousInfo self{htonl(cpu_roce_get_qp_number(ctx->transceiver)),
                            htonl(cpu_roce_get_rkey(ctx->transceiver)),
                            local_addr.s_addr};
  if (!read_all(conn_fd, &peer, sizeof(peer)) ||
      !write_all(conn_fd, &self, sizeof(self))) {
    std::cerr << "ERROR: cpu_roce bridge: rendezvous exchange failed"
              << std::endl;
    ::close(conn_fd);
    return CUDAQ_ERR_INTERNAL;
  }
  ::close(conn_fd);

  char peer_ip[INET_ADDRSTRLEN] = {0};
  in_addr peer_addr{};
  peer_addr.s_addr = peer.roce_ipv4;
  ::inet_ntop(AF_INET, &peer_addr, peer_ip, sizeof(peer_ip));
  // We Send responses (no RDMA Writes to the caller), so no peer rkey needed.
  if (!cpu_roce_connect(ctx->transceiver, ntohl(peer.qp_number), peer_ip,
                        /*peer_rx_rkey=*/0)) {
    std::cerr << "ERROR: cpu_roce bridge: transceiver connect() failed"
              << std::endl;
    return CUDAQ_ERR_INTERNAL;
  }
  ctx->connected = true;
  return CUDAQ_OK;
}

static cudaq_status_t
cpu_roce_bridge_launch(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<CpuRoceBridgeContext *>(handle);
  if (!ctx->transceiver || !ctx->connected)
    return CUDAQ_ERR_INTERNAL;
  if (ctx->monitor.joinable())
    return CUDAQ_OK;
  cpu_roce_transceiver_t xcvr = ctx->transceiver;
  ctx->monitor = std::thread([xcvr] { cpu_roce_blocking_monitor(xcvr); });
  return CUDAQ_OK;
}

static cudaq_status_t
cpu_roce_bridge_disconnect(cudaq_realtime_bridge_handle_t handle) {
  if (!handle)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<CpuRoceBridgeContext *>(handle);
  // Mark disconnected FIRST so a concurrent/subsequent launch() cannot spawn
  // a monitor over the closed transceiver; the rendezvous listen socket is
  // gone after connect(), so this bridge cannot be reconnected -- destroy
  // and re-create instead.
  ctx->connected = false;
  if (ctx->transceiver)
    cpu_roce_close(ctx->transceiver);
  if (ctx->monitor.joinable())
    ctx->monitor.join();
  return CUDAQ_OK;
}

static cudaq_status_t
cpu_roce_bridge_get_endpoint_info(cudaq_realtime_bridge_handle_t handle,
                                  char *buf, size_t buf_len) {
  if (!handle || !buf || buf_len == 0)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<CpuRoceBridgeContext *>(handle);
  if (!ctx->transceiver)
    return CUDAQ_ERR_INTERNAL;
  int n;
  if (ctx->qp_config == "hsb_fpga")
    n = std::snprintf(
        buf, buf_len,
        "transport=cpu_roce qp_config=hsb_fpga peer_ip=%s qp=0x%x rkey=%u "
        "buffer_addr=0x%llx",
        ctx->peer_ip.c_str(), cpu_roce_get_qp_number(ctx->transceiver),
        cpu_roce_get_rkey(ctx->transceiver),
        static_cast<unsigned long long>(
            cpu_roce_get_buffer_addr(ctx->transceiver)));
  else
    n = std::snprintf(buf, buf_len,
                      "transport=cpu_roce port=%u roce_ip=%s",
                      static_cast<unsigned>(ctx->bound_port),
                      ctx->local_ip.c_str());
  return (n > 0 && static_cast<size_t>(n) < buf_len) ? CUDAQ_OK
                                                     : CUDAQ_ERR_INVALID_ARG;
}

static cudaq_status_t
cpu_roce_bridge_get_ring_geometry(cudaq_realtime_bridge_handle_t handle,
                                  uint32_t *out_num_slots,
                                  uint32_t *out_slot_size) {
  if (!handle || !out_num_slots || !out_slot_size)
    return CUDAQ_ERR_INVALID_ARG;
  auto *ctx = reinterpret_cast<CpuRoceBridgeContext *>(handle);
  *out_num_slots = ctx->num_slots;
  *out_slot_size = ctx->slot_size;
  return CUDAQ_OK;
}

cudaq_realtime_bridge_interface_t *cudaq_realtime_get_bridge_interface() {
  static cudaq_realtime_bridge_interface_t cudaq_cpu_roce_bridge_interface = {
      CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION,
      cpu_roce_bridge_create,
      cpu_roce_bridge_destroy,
      cpu_roce_bridge_get_transport_context,
      cpu_roce_bridge_connect,
      cpu_roce_bridge_launch,
      cpu_roce_bridge_disconnect,
      /*get_cpu_dataplane=*/nullptr, // ring path only
      cpu_roce_bridge_get_endpoint_info,
      cpu_roce_bridge_get_ring_geometry,
  };
  return &cudaq_cpu_roce_bridge_interface;
}

} // extern "C"
