/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hsb_bridge_cpu.cpp
/// @brief Phase 1 GPU-less HSB bridge using CpuRoceTransceiver +
///        CUDAQ_DISPATCH_HOST_CALL.
///
/// Replaces gpu_roce_bridge for the CPU-data-path test case.  No
/// libhololink dependency, no GPU, no DOCA — only libibverbs + libcudaq-
/// realtime + libcudaq-realtime-cpu-roce-transport.
///
/// The FPGA-side rendezvous (telling the FPGA our QP number and rkey) is
/// out-of-band: this binary prints them to stdout and the orchestration
/// script (hsb_test_cpu.sh) feeds them to the emulator / FPGA control
/// plane separately.
///
/// Usage:
///   hsb_bridge_cpu --device=mlx5_0 --peer-ip=192.168.0.2 \
///                  --remote-qp=2 --num-pages=64 --page-size=384 \
///                  --timeout=60 [--unified]

#include "cudaq/realtime/cpu_transport/roce_wrapper.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/daemon/dispatcher/graph_launch_engine.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

// Provided by init_rpc_increment_function_table_host.cpp.
extern "C" void
setup_rpc_increment_function_table_host(cudaq_function_entry_t *h_entries);

namespace {

// ============================================================================
// Argument parsing — small and self-contained; no shared parse_bridge_args.
// ============================================================================
struct CpuBridgeConfig {
  std::string device = "mlx5_0";
  std::string peer_ip = "192.168.0.2";
  unsigned remote_qp = 2;
  unsigned num_pages = 64;
  std::size_t page_size = 384;
  unsigned payload_size = 24; // bytes after RPCHeader; default matches FPGA
                              // emulator's increment-handler stimulus
  int timeout_sec = 60;
  bool unified = false;
  bool forward = false; // CpuRoceTransceiver forward mode: RX thread loops
                        // every incoming slot back to the peer; no dispatch,
                        // no HOST_CALL.  Wire-RTT baseline; mutually
                        // exclusive with --unified.
};

bool starts_with(const std::string &s, const char *prefix) {
  std::size_t n = std::strlen(prefix);
  return s.size() >= n && std::memcmp(s.data(), prefix, n) == 0;
}

bool parse_args(int argc, char **argv, CpuBridgeConfig &cfg) {
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--help" || a == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n\n"
          << "Phase 1 CPU-RoCE bridge for libcudaq-realtime HOST_CALL "
          << "dispatch.\n\n"
          << "Options:\n"
          << "  --device=NAME       IB device (default: mlx5_0)\n"
          << "  --peer-ip=ADDR      Peer IPv4 (FPGA / emulator) (default: "
             "192.168.0.2)\n"
          << "  --remote-qp=N       Remote QP number (default: 2)\n"
          << "  --num-pages=N       Ring slots, power of two (default: 64)\n"
          << "  --page-size=N       Per-slot stride in bytes (default: 384)\n"
          << "  --payload-size=N    RPC payload bytes (default: 24)\n"
          << "  --timeout=N         Run timeout in seconds (default: 60)\n"
          << "  --unified           Use single-thread unified RX+dispatch+TX "
             "(cudaq_host_unified_loop over the transceiver hooks)\n"
          << "  --forward           Echo every incoming slot back to peer "
             "(wire-RTT baseline, no dispatch)\n";
      return false;
    } else if (starts_with(a, "--device="))
      cfg.device = a.substr(9);
    else if (starts_with(a, "--peer-ip="))
      cfg.peer_ip = a.substr(10);
    else if (starts_with(a, "--remote-qp="))
      cfg.remote_qp =
          static_cast<unsigned>(std::stoul(a.substr(12), nullptr, 0));
    else if (starts_with(a, "--num-pages="))
      cfg.num_pages = static_cast<unsigned>(std::stoul(a.substr(12)));
    else if (starts_with(a, "--page-size="))
      cfg.page_size = std::stoull(a.substr(12));
    else if (starts_with(a, "--payload-size="))
      cfg.payload_size = static_cast<unsigned>(std::stoul(a.substr(15)));
    else if (starts_with(a, "--timeout="))
      cfg.timeout_sec = std::stoi(a.substr(10));
    else if (a == "--unified")
      cfg.unified = true;
    else if (a == "--forward")
      cfg.forward = true;
    else {
      std::cerr << "Unknown argument: " << a << "  (use --help)" << std::endl;
      return false;
    }
  }
  return true;
}

// ============================================================================
// Shutdown signal handling
// ============================================================================
std::atomic<int> g_shutdown{0};
void on_signal(int) { g_shutdown.store(1, std::memory_order_release); }

// ============================================================================
// Unified-mode data-plane hooks: translate the transceiver's 1/0 returns into
// the cudaq_cpu_dataplane_t status enums.  `ctx` is the transceiver handle.
// ============================================================================
cudaq_rx_status_t dp_rx_poll(void *ctx, uint32_t *out_slot) {
  return cpu_roce_rx_poll(ctx, out_slot) ? CUDAQ_RX_READY : CUDAQ_RX_EMPTY;
}

cudaq_status_t dp_tx_publish(void *ctx, uint32_t slot) {
  return cpu_roce_tx_publish(ctx, slot) ? CUDAQ_OK : CUDAQ_ERR_INTERNAL;
}

} // namespace

// ============================================================================
// main
// ============================================================================
int main(int argc, char **argv) {
  CpuBridgeConfig cfg;
  if (!parse_args(argc, argv, cfg))
    return 0;

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  if (cfg.unified && cfg.forward) {
    std::cerr << "ERROR: --unified and --forward are mutually exclusive"
              << std::endl;
    return 1;
  }

  // RPC frame = RPCHeader (24B) + payload.  Passed to the transceiver as
  // cu_frame_size, which is the SGE length on every TX (Send / Write-With-
  // Imm) so we don't transmit unused slot tail bytes.
  const std::size_t frame_size =
      sizeof(cudaq::realtime::RPCHeader) + cfg.payload_size;

  const char *mode_str = cfg.forward   ? "FORWARD"
                         : cfg.unified ? "UNIFIED"
                                       : "3-thread";

  std::cout << "=== HSB CPU Bridge (Phase 1) ===" << std::endl;
  std::cout << "Device:        " << cfg.device << std::endl;
  std::cout << "Peer IP:       " << cfg.peer_ip << std::endl;
  std::cout << "Remote QP:     0x" << std::hex << cfg.remote_qp << std::dec
            << std::endl;
  std::cout << "Pages:         " << cfg.num_pages << std::endl;
  std::cout << "Page size:     " << cfg.page_size << " bytes" << std::endl;
  std::cout << "Frame size:    " << frame_size << " bytes" << std::endl;
  std::cout << "Mode:          " << mode_str << std::endl;

  // ------------------------------------------------------------------------
  // [1] Create CpuRoceTransceiver.
  //     3-thread: cudaq_host_ring_dispatch_loop consumes RX flags / produces TX
  //               flags; transceiver's RX+TX threads do the wire I/O.
  //     unified:  cudaq_host_unified_loop does RX + dispatch + TX on one
  //               thread through the transceiver's rx_poll/tx_publish hooks;
  //               the transceiver runs no I/O threads.
  //     forward:  transceiver's forward_loop echoes every RX slot back to
  //               the peer; no host dispatcher needed.
  // ------------------------------------------------------------------------
  const int rx_only = 0;
  const int tx_only = 0;
  cpu_roce_transceiver_t xcvr = cpu_roce_create_transceiver(
      cfg.device.c_str(), /*ib_port=*/1, cfg.remote_qp, frame_size,
      cfg.page_size, cfg.num_pages, cfg.peer_ip.c_str(), cfg.forward ? 1 : 0,
      rx_only, tx_only, cfg.unified ? 1 : 0,
      /*tx_mode=*/CPU_ROCE_TX_MODE_RDMA_SEND,
      /*peer_rx_base_addr=*/0, /*peer_rx_rkey=*/0);
  if (!xcvr) {
    std::cerr << "ERROR: cpu_roce_create_transceiver failed" << std::endl;
    return 1;
  }

  if (!cpu_roce_start(xcvr)) {
    std::cerr << "ERROR: cpu_roce_start failed" << std::endl;
    cpu_roce_destroy_transceiver(xcvr);
    return 1;
  }

  const uint32_t our_qp = cpu_roce_get_qp_number(xcvr);
  const uint32_t our_rkey = cpu_roce_get_rkey(xcvr);
  const uint64_t our_buffer = cpu_roce_get_buffer_addr(xcvr); // always 0 with
                                                              // iova=0 MR
                                                              // registration

  // ------------------------------------------------------------------------
  // [2] Set up the HOST_CALL function table.  Skipped in forward mode (no
  //     dispatch happens there).
  // ------------------------------------------------------------------------
  cudaq_function_entry_t h_entries[1];
  if (!cfg.forward)
    setup_rpc_increment_function_table_host(h_entries);

  // ------------------------------------------------------------------------
  // [3] Mode-specific wiring.
  // ------------------------------------------------------------------------
  std::thread dispatcher_thread;
  volatile int dispatcher_shutdown = 0;
  cudaq_ringbuffer_t ring{};
  // Read by pointer for the whole run of cudaq_host_unified_loop, so it lives
  // at main scope rather than in the branch that fills it.
  cudaq_cpu_dataplane_t dataplane{};
  cudaq_function_table_t table{};
  cudaq_dispatcher_config_t dcfg{};
  uint64_t packets_dispatched = 0;
  if (!cfg.forward) {
    ring.rx_flags_host = reinterpret_cast<volatile uint64_t *>(
        cpu_roce_get_rx_ring_flag_addr(xcvr));
    ring.tx_flags_host = reinterpret_cast<volatile uint64_t *>(
        cpu_roce_get_tx_ring_flag_addr(xcvr));
    ring.rx_data_host =
        reinterpret_cast<uint8_t *>(cpu_roce_get_rx_ring_data_addr(xcvr));
    ring.tx_data_host =
        reinterpret_cast<uint8_t *>(cpu_roce_get_tx_ring_data_addr(xcvr));
    ring.rx_stride_sz = cfg.page_size;
    ring.tx_stride_sz = cfg.page_size;
    table.entries = h_entries;
    table.count = 1;
  }
  if (cfg.forward) {
    // Forward: the transceiver's forward_loop echoes every RX slot back
    // to the peer.  No dispatcher.  cu_frame_size determines the
    // bytes-on-wire.
  } else if (cfg.unified) {
    // Unified: cudaq_host_unified_loop drives the transceiver through its
    // hooks on this one thread.  A HOST_CALL-only table needs no GRAPH_LAUNCH
    // engine, so engine == NULL.
    dataplane.ctx = xcvr;
    dataplane.ring = ring;
    dataplane.rx_poll = dp_rx_poll;
    dataplane.tx_publish = dp_tx_publish;

    dispatcher_thread = std::thread([&]() {
      cudaq_host_unified_loop(&dataplane, &table, /*engine=*/nullptr,
                              &dispatcher_shutdown, &packets_dispatched);
    });
  } else {
    // 3-thread layout: spawn cudaq_host_ring_dispatch_loop on a dedicated
    // thread.  It busy-polls rx_flags_host (the transceiver's RX thread
    // publishes them), invokes our HOST_CALL handler synchronously, and
    // publishes tx_flags_host (the transceiver's TX thread consumes them).
    // A HOST_CALL-only table needs no GRAPH_LAUNCH engine, so engine == NULL
    // (the loop touches no graph workers).
    dcfg.num_slots = cfg.num_pages;
    dcfg.slot_size = static_cast<uint32_t>(cfg.page_size);
    dcfg.dispatch_path = CUDAQ_DISPATCH_PATH_HOST;
    dcfg.dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;
    dcfg.skip_tx_markers = 1; // we own the TX path; sentinel pattern
                              // (used to avoid GpuRoceTransceiver TX kernel
                              // confusion) is irrelevant here.

    dispatcher_thread = std::thread([&]() {
      cudaq_host_ring_dispatch_loop(&ring, &table, &dcfg, /*engine=*/nullptr,
                                    &dispatcher_shutdown, &packets_dispatched);
    });
  }

  // ------------------------------------------------------------------------
  // [4] Print rendezvous info for the orchestration script.
  // ------------------------------------------------------------------------
  // NOTE: output format MUST match gpu_roce_bridge_common.h exactly —
  // "  KEY: VALUE" with a single space after the colon — because the
  // orchestration script (hsb_test_cpu.sh, mirrored from gpu_roce_test.sh)
  // uses strict regexes like 'QP Number: 0x\K...' to parse it.
  std::cout << "\n=== Bridge Ready ===" << std::endl;
  std::cout << "  QP Number: 0x" << std::hex << our_qp << std::dec << std::endl;
  std::cout << "  RKey: " << our_rkey << std::endl;
  std::cout << "  Buffer Addr: 0x" << std::hex << our_buffer << std::dec
            << std::endl;
  std::cout << "\nWaiting (Ctrl+C to stop, timeout=" << cfg.timeout_sec
            << "s)..." << std::endl;
  std::cout.flush();

  // ------------------------------------------------------------------------
  // [5] Run the transceiver I/O threads until shutdown (none in unified
  //     mode: the dispatcher thread drives the wire itself).
  // ------------------------------------------------------------------------
  std::thread xcvr_monitor;
  if (!cfg.unified)
    xcvr_monitor = std::thread([xcvr]() { cpu_roce_blocking_monitor(xcvr); });

  // Timeout / signal wait loop.
  auto t0 = std::chrono::steady_clock::now();
  while (g_shutdown.load(std::memory_order_acquire) == 0) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    if (elapsed > cfg.timeout_sec) {
      std::cout << "\nTimeout reached (" << cfg.timeout_sec << "s)"
                << std::endl;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // ------------------------------------------------------------------------
  // [6] Orderly shutdown: signal the dispatcher, then the transceiver,
  //     then join both.
  // ------------------------------------------------------------------------
  std::cout << "\n=== Shutting down ===" << std::endl;
  // Every mode but forward runs a host-dispatcher thread, which must stop
  // before cpu_roce_close() releases the QP/CQs the unified hooks use.
  const bool runs_dispatcher = !cfg.forward;
  if (runs_dispatcher) {
    dispatcher_shutdown = 1;
    __sync_synchronize();
    if (dispatcher_thread.joinable())
      dispatcher_thread.join();
  }
  cpu_roce_close(xcvr);
  if (xcvr_monitor.joinable())
    xcvr_monitor.join();

  if (runs_dispatcher)
    std::cout << "Packets dispatched: " << packets_dispatched << std::endl;

  cpu_roce_destroy_transceiver(xcvr);
  std::cout << "Done." << std::endl;
  return 0;
}
