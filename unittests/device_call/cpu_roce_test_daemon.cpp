/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file cpu_roce_test_daemon.cpp
/// @brief Test-only service-side daemon for the cpu_roce DeviceCallChannel.
///
/// This is the *service* end of the device_call RDMA wire: it receives the
/// CpuRoceChannel's Sends, dispatches them through libcudaq-realtime's
/// CUDAQ_DISPATCH_HOST_CALL host dispatcher to plain-C++ handlers, and Writes
/// the responses back to the caller's rx ring with RDMA Write-With-Imm.  It is
/// the asymmetric-A counterpart of the channel:
///
///   channel --IBV_WR_SEND-->            daemon   (request)
///   channel <--IBV_WR_RDMA_WRITE_WITH_IMM-- daemon (response)
///
/// so it stands in for a real HSB-enabled FPGA service (FPGAs receive Sends and
/// transmit Writes) without the FPGA.  It is NOT a production deliverable; a
/// real daemon with dlopen-able handlers, logging, etc. is a future phase.
///
/// QP/rkey rendezvous: connected (UC) QPs need a bidirectional exchange, so the
/// daemon runs a tiny TCP rendezvous *server* (the mirror of the channel's
/// client): it setup()s its transceiver, listens, accepts the channel, reads
/// the channel's {qp, rkey, roce-ipv4}, replies with its own, then connect()s.
/// It prints `CPU_ROCE_DAEMON_READY port=<P> roce_ip=<IP>` on stdout so the
/// test fixture can hand the channel the rendezvous host/port.
///
/// Usage:
///   cpu_roce_test_daemon --device=mlx5_0 --local-ip=10.0.0.2 \
///                        --rendezvous-port=0 --num-pages=64 --page-size=384 \
///                        --timeout=60

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

using cudaq::realtime::fnv1a_hash;
using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPC_MAGIC_RESPONSE;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

namespace {

// Must match CpuRoceChannel's RendezvousInfo byte-for-byte (network order).
struct RendezvousInfo {
  std::uint32_t qp_number = 0;
  std::uint32_t rkey = 0;
  std::uint32_t roce_ipv4 = 0;
};

std::atomic<int> g_shutdown{0};
void on_signal(int) { g_shutdown.store(1, std::memory_order_release); }

// ---------------------------------------------------------------------------
// HOST_CALL handlers.  Signature: cudaq_host_rpc_fn_t(void *slot, size_t).
// The dispatcher pre-copies the rx slot into the tx slot and hands us the tx
// slot; we rewrite it in place: read RPCHeader+args, write RPCResponse+result.
// ---------------------------------------------------------------------------

// addThem(int64 a, int64 b) -> int64 (a + b).  Wire: args = [a:8][b:8];
// result = [sum:8].
extern "C" void addThemHost(void *slot, std::size_t slot_size) {
  auto *header = static_cast<RPCHeader *>(slot);
  if (header->magic != RPC_MAGIC_REQUEST)
    return;

  const std::uint32_t request_id = header->request_id;
  const std::uint64_t ptp = header->ptp_timestamp;
  const std::uint32_t arg_len = header->arg_len;

  std::int64_t sum = 0;
  std::int32_t status = 0;
  if (arg_len >= 2 * sizeof(std::int64_t) &&
      slot_size >= sizeof(RPCResponse) + sizeof(std::int64_t)) {
    std::int64_t a = 0, b = 0;
    auto *args = static_cast<std::uint8_t *>(slot) + sizeof(RPCHeader);
    std::memcpy(&a, args, sizeof(a));
    std::memcpy(&b, args + sizeof(a), sizeof(b));
    sum = a + b;
  } else {
    status = 1; // InvalidArgument
  }

  auto *response = static_cast<RPCResponse *>(slot);
  response->magic = RPC_MAGIC_RESPONSE;
  response->status = status;
  response->result_len = status == 0 ? sizeof(std::int64_t) : 0;
  response->request_id = request_id;
  response->ptp_timestamp = ptp;
  if (status == 0)
    std::memcpy(static_cast<std::uint8_t *>(slot) + sizeof(RPCResponse), &sum,
                sizeof(sum));
}

// noop() -> void.  Used for fire-and-forget; still writes a (zero-length)
// response because the host dispatcher always publishes a tx slot.  The caller
// channel ignores it on the fire-and-forget path.
extern "C" void noopHost(void *slot, std::size_t /*slot_size*/) {
  auto *header = static_cast<RPCHeader *>(slot);
  if (header->magic != RPC_MAGIC_REQUEST)
    return;
  const std::uint32_t request_id = header->request_id;
  const std::uint64_t ptp = header->ptp_timestamp;
  auto *response = static_cast<RPCResponse *>(slot);
  response->magic = RPC_MAGIC_RESPONSE;
  response->status = 0;
  response->result_len = 0;
  response->request_id = request_id;
  response->ptp_timestamp = ptp;
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
struct DaemonConfig {
  std::string device = "mlx5_0";
  std::string local_ip = "10.0.0.2";
  std::uint16_t rendezvous_port = 0; // 0 => ephemeral, printed on stdout
  unsigned num_pages = 64;
  std::size_t page_size = 384;
  unsigned payload_size = 16; // bytes after RPCHeader (two int64 for addThem)
  int timeout_sec = 60;
};

bool starts_with(const std::string &s, const char *prefix) {
  std::size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

bool parse_args(int argc, char **argv, DaemonConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::cout << "Usage: " << argv[0] << " [options]\n"
                << "  --device=NAME         IB device (default: mlx5_0)\n"
                << "  --local-ip=ADDR       Our RoCE IPv4 (default: 10.0.0.2)\n"
                << "  --rendezvous-port=N   TCP rendezvous port (0=ephemeral)\n"
                << "  --num-pages=N         Ring slots, power of two (def 64)\n"
                << "  --page-size=N         Per-slot stride bytes (def 384)\n"
                << "  --payload-size=N      RPC payload bytes (def 16)\n"
                << "  --timeout=N           Run timeout seconds (def 60)\n";
      return false;
    } else if (starts_with(a, "--device="))
      cfg.device = a.substr(9);
    else if (starts_with(a, "--local-ip="))
      cfg.local_ip = a.substr(11);
    else if (starts_with(a, "--rendezvous-port="))
      cfg.rendezvous_port =
          static_cast<std::uint16_t>(std::stoul(a.substr(18)));
    else if (starts_with(a, "--num-pages="))
      cfg.num_pages = static_cast<unsigned>(std::stoul(a.substr(12)));
    else if (starts_with(a, "--page-size="))
      cfg.page_size = std::stoull(a.substr(12));
    else if (starts_with(a, "--payload-size="))
      cfg.payload_size = static_cast<unsigned>(std::stoul(a.substr(15)));
    else if (starts_with(a, "--timeout="))
      cfg.timeout_sec = std::stoi(a.substr(10));
    else {
      std::cerr << "Unknown argument: " << a << "  (use --help)\n";
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// Blocking read/write helpers
// ---------------------------------------------------------------------------
bool write_all(int fd, const void *buf, std::size_t len) {
  const auto *p = static_cast<const std::uint8_t *>(buf);
  while (len > 0) {
    const ssize_t n = ::write(fd, p, len);
    if (n <= 0) {
      if (n < 0 && errno == EINTR)
        continue;
      return false;
    }
    p += n;
    len -= static_cast<std::size_t>(n);
  }
  return true;
}

bool read_all(int fd, void *buf, std::size_t len) {
  auto *p = static_cast<std::uint8_t *>(buf);
  while (len > 0) {
    const ssize_t n = ::read(fd, p, len);
    if (n <= 0) {
      if (n < 0 && errno == EINTR)
        continue;
      return false;
    }
    p += n;
    len -= static_cast<std::size_t>(n);
  }
  return true;
}

} // namespace

int main(int argc, char **argv) {
  DaemonConfig cfg;
  if (!parse_args(argc, argv, cfg))
    return 0;

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  const std::size_t frame_size = sizeof(RPCHeader) + cfg.payload_size;

  std::cout << "=== cpu_roce_test_daemon ===" << std::endl;
  std::cout << "Device:     " << cfg.device << std::endl;
  std::cout << "Local IP:   " << cfg.local_ip << std::endl;
  std::cout << "Pages:      " << cfg.num_pages << std::endl;
  std::cout << "Page size:  " << cfg.page_size << " bytes" << std::endl;

  // [1] Construct the service-end transceiver via the public C wrapper.
  //     tx_mode=WRITE_WITH_IMM_FOR_PEER: we Write responses back; we receive
  //     the channel's Sends.  peer_rx_base_addr=0 matches the channel's iova=0
  //     rx_data MR (it addresses slots by offset alone); the channel's rkey is
  //     learned at rendezvous and supplied to cpu_roce_connect().
  cpu_roce_transceiver_t xcvr = cpu_roce_create_transceiver(
      cfg.device.c_str(), /*ib_port=*/1, /*tx_ibv_qp=*/0u, frame_size,
      cfg.page_size, cfg.num_pages, /*peer_ip=*/"0.0.0.0", /*forward=*/0,
      /*rx_only=*/0, /*tx_only=*/0, /*unified=*/0,
      CPU_ROCE_TX_MODE_WRITE_WITH_IMM_FOR_PEER, /*peer_rx_base_addr=*/0,
      /*peer_rx_rkey=*/0);
  if (!xcvr) {
    std::cerr << "ERROR: transceiver create failed" << std::endl;
    return 1;
  }
  // Pin the source GID to our configured local address (right SGID on a
  // multi-IP port).
  cpu_roce_set_local_ip(xcvr, cfg.local_ip.c_str());
  if (!cpu_roce_setup(xcvr)) {
    std::cerr << "ERROR: transceiver setup() failed" << std::endl;
    return 1;
  }

  // [2] Stand up the TCP rendezvous server.
  int listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (listen_fd < 0) {
    std::cerr << "ERROR: rendezvous socket() failed" << std::endl;
    return 1;
  }
  int reuse = 1;
  ::setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
  sockaddr_in srv{};
  srv.sin_family = AF_INET;
  srv.sin_addr.s_addr = htonl(INADDR_ANY);
  srv.sin_port = htons(cfg.rendezvous_port);
  if (::bind(listen_fd, reinterpret_cast<sockaddr *>(&srv), sizeof(srv)) != 0 ||
      ::listen(listen_fd, 1) != 0) {
    std::cerr << "ERROR: rendezvous bind/listen failed" << std::endl;
    ::close(listen_fd);
    return 1;
  }
  // Resolve the actual (possibly ephemeral) port.
  socklen_t srvlen = sizeof(srv);
  ::getsockname(listen_fd, reinterpret_cast<sockaddr *>(&srv), &srvlen);
  const std::uint16_t actual_port = ntohs(srv.sin_port);

  // [3] Publish the rendezvous endpoint for the test fixture.
  std::cout << "CPU_ROCE_DAEMON_READY port=" << actual_port
            << " roce_ip=" << cfg.local_ip << std::endl;
  std::cout.flush();

  // [4] Accept the channel and perform the bidirectional {qp,rkey,ip} swap.
  //     Server reads the client's info first, then replies with its own (the
  //     mirror of CpuRoceChannel::exchangeRendezvous).
  int conn_fd = ::accept(listen_fd, nullptr, nullptr);
  ::close(listen_fd);
  if (conn_fd < 0) {
    std::cerr << "ERROR: rendezvous accept() failed" << std::endl;
    return 1;
  }
  int one = 1;
  ::setsockopt(conn_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

  RendezvousInfo peer{};
  if (!read_all(conn_fd, &peer, sizeof(peer))) {
    std::cerr << "ERROR: rendezvous read failed" << std::endl;
    ::close(conn_fd);
    return 1;
  }
  in_addr la{};
  ::inet_pton(AF_INET, cfg.local_ip.c_str(), &la);
  RendezvousInfo self{htonl(cpu_roce_get_qp_number(xcvr)),
                      htonl(cpu_roce_get_rkey(xcvr)), la.s_addr};
  if (!write_all(conn_fd, &self, sizeof(self))) {
    std::cerr << "ERROR: rendezvous write failed" << std::endl;
    ::close(conn_fd);
    return 1;
  }
  ::close(conn_fd);

  const std::uint32_t peer_qp = ntohl(peer.qp_number);
  const std::uint32_t peer_rkey = ntohl(peer.rkey);
  char peer_ip[INET_ADDRSTRLEN] = {0};
  in_addr pa{};
  pa.s_addr = peer.roce_ipv4;
  ::inet_ntop(AF_INET, &pa, peer_ip, sizeof(peer_ip));

  std::cout << "Rendezvous: peer qp=0x" << std::hex << peer_qp << " rkey=0x"
            << peer_rkey << std::dec << " ip=" << peer_ip << std::endl;

  // [5] connect(): we Write to the channel's rx ring, so we need its rkey.
  if (!cpu_roce_connect(xcvr, peer_qp, peer_ip, peer_rkey)) {
    std::cerr << "ERROR: transceiver connect() failed" << std::endl;
    return 1;
  }

  // [6] Build the HOST_CALL function table: addThem, noop.
  cudaq_function_entry_t entries[2];
  std::memset(entries, 0, sizeof(entries));
  entries[0].handler.host_fn = &addThemHost;
  entries[0].function_id = fnv1a_hash("addThem");
  entries[0].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  entries[0].schema.num_args = 1;
  entries[0].schema.num_results = 1;
  entries[0].schema.args[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
  entries[0].schema.results[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
  entries[1].handler.host_fn = &noopHost;
  entries[1].function_id = fnv1a_hash("noop");
  entries[1].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  entries[1].schema.num_args = 0;
  entries[1].schema.num_results = 0;

  // [7] Wire the host dispatcher to the transceiver rings (3-thread layout,
  //     identical to hsb_bridge_cpu but tx_mode=Write on the transceiver).
  std::atomic<int> dispatcher_shutdown{0};
  std::uint64_t packets_dispatched = 0;
  cudaq_host_dispatch_loop_ctx_t dctx{};
  dctx.ringbuffer.rx_flags_host = reinterpret_cast<volatile uint64_t *>(
      cpu_roce_get_rx_ring_flag_addr(xcvr));
  dctx.ringbuffer.tx_flags_host = reinterpret_cast<volatile uint64_t *>(
      cpu_roce_get_tx_ring_flag_addr(xcvr));
  dctx.ringbuffer.rx_data_host =
      reinterpret_cast<uint8_t *>(cpu_roce_get_rx_ring_data_addr(xcvr));
  dctx.ringbuffer.tx_data_host =
      reinterpret_cast<uint8_t *>(cpu_roce_get_tx_ring_data_addr(xcvr));
  dctx.ringbuffer.rx_stride_sz = cfg.page_size;
  dctx.ringbuffer.tx_stride_sz = cfg.page_size;
  dctx.config.num_slots = cfg.num_pages;
  dctx.config.slot_size = static_cast<uint32_t>(cfg.page_size);
  dctx.config.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
  dctx.config.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
  dctx.config.skip_tx_markers = 1;
  dctx.function_table.entries = entries;
  dctx.function_table.count = 2;
  dctx.shutdown_flag = &dispatcher_shutdown;
  dctx.stats_counter = &packets_dispatched;
  dctx.skip_stream_sweep = true;

  std::thread dispatcher_thread(
      [&dctx]() { cudaq_host_dispatcher_loop(&dctx); });
  std::thread xcvr_monitor([xcvr]() { cpu_roce_blocking_monitor(xcvr); });

  std::cout << "Daemon running (timeout=" << cfg.timeout_sec << "s)..."
            << std::endl;
  std::cout.flush();

  // [8] Run until signalled or timed out.
  auto t0 = std::chrono::steady_clock::now();
  while (g_shutdown.load(std::memory_order_acquire) == 0) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    if (elapsed > cfg.timeout_sec)
      break;
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  // [9] Orderly shutdown.
  dispatcher_shutdown.store(1, std::memory_order_release);
  if (dispatcher_thread.joinable())
    dispatcher_thread.join();
  cpu_roce_close(xcvr);
  if (xcvr_monitor.joinable())
    xcvr_monitor.join();
  cpu_roce_destroy_transceiver(xcvr);

  std::cout << "Packets dispatched: " << packets_dispatched << std::endl;
  std::cout << "Daemon done." << std::endl;
  return 0;
}
