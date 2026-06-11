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
/// Replaces hololink_bridge for the CPU-data-path test case.  No
/// libhololink dependency, no GPU, no DOCA — only libibverbs + libcudaq-
/// realtime + libcudaq-realtime-cpu-transport.
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
#include "cudaq/realtime/daemon/dispatcher/host_dispatcher.h"

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

// Provided by init_rpc_increment_function_table_host.cpp via internal
// linkage; we need the same handler for unified mode's dispatch callback,
// so re-declare its C ABI here.
extern "C" void rpc_increment_handler_host(const void *rx_slot, void *tx_slot,
                                           std::size_t slot_size);

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
          << "  --unified           Use single-thread unified RX+dispatch+TX\n"
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
// Unified-mode dispatch callback.  Forwards directly to the host increment
// handler.  The two-pointer handler reads the request from rx_slot and writes
// the response into tx_slot, so no copy is needed here (the unified loop hands
// us both sides, matching the handler's RX/TX-pointer ABI).
// ============================================================================
std::size_t unified_dispatch_cb(void * /*context*/, const void *rx_slot,
                                void *tx_slot, std::size_t slot_size) {
  using cudaq::realtime::RPC_MAGIC_REQUEST;
  using cudaq::realtime::RPCHeader;
  const auto *header = static_cast<const RPCHeader *>(rx_slot);
  if (header->magic != RPC_MAGIC_REQUEST)
    return 0; // drop without sending — silent for non-RPC noise
  rpc_increment_handler_host(rx_slot, tx_slot, slot_size);
  // Response length: response header + payload bytes (use full slot to
  // match the three-thread / FPGA TX framing, since the FPGA expects a
  // fixed-size payload per slot).
  return slot_size;
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
  //     3-thread: cudaq_host_dispatcher_loop consumes RX flags / produces TX
  //               flags; transceiver's RX+TX threads do the wire I/O.
  //     unified:  transceiver's unified_loop does RX + dispatch + TX inline.
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
  std::atomic<int> dispatcher_shutdown{0};
  cudaq_host_dispatch_loop_ctx_t dctx{};
  uint64_t packets_dispatched = 0;
  if (cfg.forward) {
    // Forward: the transceiver's forward_loop echoes every RX slot back
    // to the peer.  No dispatcher, no callback to install.  cu_frame_size
    // determines the bytes-on-wire.
  } else if (cfg.unified) {
    // Unified: install the dispatch closure; the transceiver's unified
    // thread will invoke it.  No cudaq_host_dispatcher_loop needed.
    cpu_roce_set_unified_dispatch(xcvr, &unified_dispatch_cb,
                                  /*context=*/nullptr);
  } else {
    // 3-thread layout: spawn cudaq_host_dispatcher_loop on a dedicated
    // thread.  It busy-polls rx_flags_host (the transceiver's RX thread
    // publishes them), invokes our HOST_CALL handler synchronously, and
    // publishes tx_flags_host (the transceiver's TX thread consumes
    // them).  Worker-pool fields are zero/null because HOST_CALL bypasses
    // the worker pool; skip_stream_sweep prevents the dispatcher from
    // poking workers that don't exist.
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
    dctx.config.skip_tx_markers = 1; // we own the TX path; sentinel pattern
                                     // (used to avoid Hololink TX kernel
                                     // confusion) is irrelevant here.
    dctx.function_table.entries = h_entries;
    dctx.function_table.count = 1;
    dctx.shutdown_flag = &dispatcher_shutdown;
    dctx.stats_counter = &packets_dispatched;
    dctx.skip_stream_sweep = true; // no graph workers, no streams to sweep

    dispatcher_thread =
        std::thread([&dctx]() { cudaq_host_dispatcher_loop(&dctx); });
  }

  // ------------------------------------------------------------------------
  // [4] Print rendezvous info for the orchestration script.
  // ------------------------------------------------------------------------
  // NOTE: output format MUST match hololink_bridge_common.h exactly —
  // "  KEY: VALUE" with a single space after the colon — because the
  // orchestration script (hsb_test_cpu.sh, mirrored from hololink_test.sh)
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
  // [5] Run the transceiver I/O threads on the main thread until shutdown.
  // ------------------------------------------------------------------------
  std::thread xcvr_monitor([xcvr]() { cpu_roce_blocking_monitor(xcvr); });

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
  // Only the 3-thread mode runs a separate host-dispatcher thread.
  const bool runs_dispatcher = !cfg.unified && !cfg.forward;
  if (runs_dispatcher) {
    dispatcher_shutdown.store(1, std::memory_order_release);
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
