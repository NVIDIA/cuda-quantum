/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file hololink_bridge_common.h
/// @brief Header-only bridge skeleton for Hololink-based RPC dispatch.
///
/// Provides common infrastructure used by all Hololink bridge tools:
///   - Command-line argument parsing for IB device, peer IP, QP, etc.
///   - Hololink transceiver creation and QP connection
///   - Dispatch kernel wiring via the cudaq host API
///   - Main run loop with diagnostics
///   - Graceful shutdown
///
/// Each concrete bridge tool (generic increment, mock decoder, real decoder)
/// implements a small main() that:
///   1. Parses any tool-specific arguments
///   2. Sets up its RPC function table on the GPU
///   3. Calls bridge_run() with a BridgeConfig struct
///
/// This header is compiled by a standard C++ compiler; all CUDA and Hololink
/// calls go through C interfaces (cudaq_realtime.h, hololink_wrapper.h).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <cuda_runtime.h>

#include "cudaq/nvqlink/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_kernel_launch.h"

// Hololink C wrapper (link against hololink_wrapper_bridge static library)
#include "hololink_wrapper.h"

namespace cudaq::nvqlink {

//==============================================================================
// CUDA Error Checking
//==============================================================================

#ifndef BRIDGE_CUDA_CHECK
#define BRIDGE_CUDA_CHECK(call)                                                \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "    \
                << cudaGetErrorString(err) << std::endl;                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#endif

//==============================================================================
// Global Signal Handler
//==============================================================================

namespace detail {
inline std::atomic<bool> &bridge_shutdown_flag() {
  static std::atomic<bool> flag{false};
  return flag;
}
inline void bridge_signal_handler(int) { bridge_shutdown_flag() = true; }
} // namespace detail

//==============================================================================
// Bridge Configuration
//==============================================================================

/// @brief Configuration for the bridge's Hololink and dispatch kernel setup.
struct BridgeConfig {
  // IB / network
  std::string device = "rocep1s0f0"; ///< IB device name
  std::string peer_ip = "10.0.0.2";  ///< FPGA/emulator IP
  uint32_t remote_qp = 0x2;          ///< Remote QP number (FPGA default: 2)
  int gpu_id = 0;                     ///< GPU device ID
  int timeout_sec = 60;               ///< Runtime timeout in seconds

  // Ring buffer sizing
  size_t frame_size = 256;   ///< Minimum frame size (RPCHeader + payload)
  size_t page_size = 384;    ///< Ring buffer slot size (>= frame_size, 128-aligned)
  unsigned num_pages = 64;   ///< Number of ring buffer slots

  // QP exchange (emulator mode)
  bool exchange_qp = false;  ///< Use QP exchange protocol
  int exchange_port = 12345; ///< TCP port for QP exchange

  // Dispatch kernel config
  cudaq_function_entry_t *d_function_entries = nullptr; ///< GPU function table
  size_t func_count = 0;                                ///< Number of entries

  /// @brief Dispatch kernel grid configuration.
  /// Defaults match the regular (non-cooperative) kernel.
  cudaq_kernel_type_t kernel_type = CUDAQ_KERNEL_REGULAR;
  uint32_t num_blocks = 1;
  uint32_t threads_per_block = 32;

  /// @brief Pointer to the dispatch kernel launch function.
  /// Default: cudaq_launch_dispatch_kernel_regular
  cudaq_dispatch_launch_fn_t launch_fn = nullptr;

  /// @brief Optional cleanup callback invoked during shutdown.
  std::function<void()> cleanup_fn;
};

//==============================================================================
// Common Argument Parsing
//==============================================================================

/// @brief Parse common bridge arguments from the command line.
///
/// Recognised flags: --device=, --peer-ip=, --remote-qp=, --gpu=,
/// --timeout=, --page-size=, --num-pages=, --exchange-qp, --exchange-port=.
/// Unknown flags are silently ignored (so tool-specific flags can coexist).
///
/// @param argc Argument count
/// @param argv Argument vector
/// @param[out] config Bridge configuration to populate
inline void parse_bridge_args(int argc, char *argv[], BridgeConfig &config) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--device=") == 0)
      config.device = arg.substr(9);
    else if (arg.find("--peer-ip=") == 0)
      config.peer_ip = arg.substr(10);
    else if (arg.find("--remote-qp=") == 0)
      config.remote_qp = std::stoul(arg.substr(12), nullptr, 0);
    else if (arg.find("--gpu=") == 0)
      config.gpu_id = std::stoi(arg.substr(6));
    else if (arg.find("--timeout=") == 0)
      config.timeout_sec = std::stoi(arg.substr(10));
    else if (arg.find("--page-size=") == 0)
      config.page_size = std::stoull(arg.substr(12));
    else if (arg.find("--num-pages=") == 0)
      config.num_pages = std::stoul(arg.substr(12));
    else if (arg == "--exchange-qp")
      config.exchange_qp = true;
    else if (arg.find("--exchange-port=") == 0)
      config.exchange_port = std::stoi(arg.substr(16));
  }
}

//==============================================================================
// Bridge Run Function
//==============================================================================

/// @brief Run the Hololink bridge with the given configuration.
///
/// This function:
///   1. Initialises CUDA on the configured GPU
///   2. Creates the Hololink transceiver and connects the QP
///   3. Forces eager CUDA module loading
///   4. Wires the cudaq dispatch kernel to the Hololink ring buffers
///   5. Launches Hololink RX+TX kernels
///   6. Runs the main diagnostic loop until timeout or signal
///   7. Performs orderly shutdown
///
/// The caller must set config.d_function_entries and config.func_count
/// before calling this function.
///
/// @param config Fully-populated bridge configuration
/// @return 0 on success, non-zero on error
inline int bridge_run(BridgeConfig &config) {
  signal(SIGINT, detail::bridge_signal_handler);
  signal(SIGTERM, detail::bridge_signal_handler);

  auto &g_shutdown = detail::bridge_shutdown_flag();

  //============================================================================
  // [1] Initialize CUDA
  //============================================================================
  std::cout << "\n[1/5] Initializing CUDA..." << std::endl;
  BRIDGE_CUDA_CHECK(cudaSetDevice(config.gpu_id));

  cudaDeviceProp prop;
  BRIDGE_CUDA_CHECK(cudaGetDeviceProperties(&prop, config.gpu_id));
  std::cout << "  GPU: " << prop.name << std::endl;

  //============================================================================
  // [2] Create Hololink transceiver
  //============================================================================
  std::cout << "\n[2/5] Creating Hololink transceiver..." << std::endl;

  // Ensure page_size >= frame_size
  if (config.page_size < config.frame_size) {
    std::cout << "  Adjusting page_size from " << config.page_size << " to "
              << config.frame_size << " to fit frame" << std::endl;
    config.page_size = config.frame_size;
  }

  std::cout << "  Frame size: " << config.frame_size << " bytes" << std::endl;
  std::cout << "  Page size: " << config.page_size << " bytes" << std::endl;
  std::cout << "  Num pages: " << config.num_pages << std::endl;

  hololink_transceiver_t transceiver = hololink_create_transceiver(
      config.device.c_str(), 1, // ib_port
      config.frame_size, config.page_size, config.num_pages,
      "0.0.0.0", // deferred connection
      0,          // forward = false
      1,          // rx_only = true
      1           // tx_only = true
  );

  if (!transceiver) {
    std::cerr << "ERROR: Failed to create Hololink transceiver" << std::endl;
    return 1;
  }

  if (!hololink_start(transceiver)) {
    std::cerr << "ERROR: Failed to start Hololink transceiver" << std::endl;
    hololink_destroy_transceiver(transceiver);
    return 1;
  }

  // Connect QP to remote peer
  {
    uint8_t remote_gid[16] = {};
    remote_gid[10] = 0xff;
    remote_gid[11] = 0xff;
    inet_pton(AF_INET, config.peer_ip.c_str(), &remote_gid[12]);

    std::cout << "  Connecting QP to remote QP 0x" << std::hex
              << config.remote_qp << std::dec << " at " << config.peer_ip
              << "..." << std::endl;

    if (!hololink_reconnect_qp(transceiver, remote_gid, config.remote_qp)) {
      std::cerr << "ERROR: Failed to connect QP to remote peer" << std::endl;
      hololink_destroy_transceiver(transceiver);
      return 1;
    }
    std::cout << "  QP connected to remote peer" << std::endl;
  }

  uint32_t our_qp = hololink_get_qp_number(transceiver);
  uint32_t our_rkey = hololink_get_rkey(transceiver);
  uint64_t our_buffer = hololink_get_buffer_addr(transceiver);

  std::cout << "  QP Number: 0x" << std::hex << our_qp << std::dec
            << std::endl;
  std::cout << "  RKey: " << our_rkey << std::endl;
  std::cout << "  Buffer Addr: 0x" << std::hex << our_buffer << std::dec
            << std::endl;

  // Ring buffer pointers
  uint8_t *rx_ring_data =
      reinterpret_cast<uint8_t *>(hololink_get_rx_ring_data_addr(transceiver));
  uint64_t *rx_ring_flag = hololink_get_rx_ring_flag_addr(transceiver);
  uint8_t *tx_ring_data =
      reinterpret_cast<uint8_t *>(hololink_get_tx_ring_data_addr(transceiver));
  uint64_t *tx_ring_flag = hololink_get_tx_ring_flag_addr(transceiver);

  if (!rx_ring_data || !rx_ring_flag || !tx_ring_data || !tx_ring_flag) {
    std::cerr << "ERROR: Failed to get ring buffer pointers" << std::endl;
    hololink_destroy_transceiver(transceiver);
    return 1;
  }

  //============================================================================
  // [3] Force eager CUDA module loading
  //============================================================================
  std::cout << "\n[3/5] Forcing CUDA module loading..." << std::endl;
  {
    int dispatch_blocks = 0;
    cudaError_t occ_err =
        cudaq_dispatch_kernel_query_occupancy(&dispatch_blocks, 1);
    if (occ_err != cudaSuccess) {
      std::cerr << "ERROR: Dispatch kernel occupancy query failed: "
                << cudaGetErrorString(occ_err) << std::endl;
      return 1;
    }
    std::cout << "  Dispatch kernel occupancy: " << dispatch_blocks
              << " blocks/SM" << std::endl;

    if (!hololink_query_kernel_occupancy()) {
      std::cerr << "ERROR: Hololink kernel occupancy query failed" << std::endl;
      return 1;
    }
  }

  //============================================================================
  // [4] Wire dispatch kernel to Hololink ring buffers
  //============================================================================
  std::cout << "\n[4/5] Wiring dispatch kernel..." << std::endl;

  // Allocate control variables
  void *tmp_shutdown = nullptr;
  BRIDGE_CUDA_CHECK(
      cudaHostAlloc(&tmp_shutdown, sizeof(int), cudaHostAllocMapped));
  volatile int *shutdown_flag = static_cast<volatile int *>(tmp_shutdown);
  void *tmp_d_shutdown = nullptr;
  BRIDGE_CUDA_CHECK(
      cudaHostGetDevicePointer(&tmp_d_shutdown, tmp_shutdown, 0));
  volatile int *d_shutdown_flag =
      static_cast<volatile int *>(tmp_d_shutdown);
  *shutdown_flag = 0;
  int zero = 0;
  BRIDGE_CUDA_CHECK(cudaMemcpy(const_cast<int *>(d_shutdown_flag), &zero,
                                sizeof(int), cudaMemcpyHostToDevice));

  uint64_t *d_stats = nullptr;
  BRIDGE_CUDA_CHECK(cudaMalloc(&d_stats, sizeof(uint64_t)));
  BRIDGE_CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(uint64_t)));

  // Host API wiring
  cudaq_dispatch_manager_t *manager = nullptr;
  cudaq_dispatcher_t *dispatcher = nullptr;

  if (cudaq_dispatch_manager_create(&manager) != CUDAQ_OK) {
    std::cerr << "ERROR: Failed to create dispatch manager" << std::endl;
    return 1;
  }

  cudaq_dispatcher_config_t dconfig{};
  dconfig.device_id = config.gpu_id;
  dconfig.num_blocks = config.num_blocks;
  dconfig.threads_per_block = config.threads_per_block;
  dconfig.num_slots = static_cast<uint32_t>(config.num_pages);
  dconfig.slot_size = static_cast<uint32_t>(config.page_size);
  dconfig.vp_id = 0;
  dconfig.kernel_type = config.kernel_type;
  dconfig.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;

  if (cudaq_dispatcher_create(manager, &dconfig, &dispatcher) != CUDAQ_OK) {
    std::cerr << "ERROR: Failed to create dispatcher" << std::endl;
    return 1;
  }

  cudaq_ringbuffer_t ringbuffer{};
  ringbuffer.rx_flags = reinterpret_cast<volatile uint64_t *>(rx_ring_flag);
  ringbuffer.tx_flags = reinterpret_cast<volatile uint64_t *>(tx_ring_flag);
  ringbuffer.rx_data = rx_ring_data;
  ringbuffer.tx_data = tx_ring_data;
  ringbuffer.rx_stride_sz = config.page_size;
  ringbuffer.tx_stride_sz = config.page_size;

  if (cudaq_dispatcher_set_ringbuffer(dispatcher, &ringbuffer) != CUDAQ_OK) {
    std::cerr << "ERROR: Failed to set ringbuffer" << std::endl;
    return 1;
  }

  cudaq_function_table_t table{};
  table.entries = config.d_function_entries;
  table.count = config.func_count;
  if (cudaq_dispatcher_set_function_table(dispatcher, &table) != CUDAQ_OK) {
    std::cerr << "ERROR: Failed to set function table" << std::endl;
    return 1;
  }

  if (cudaq_dispatcher_set_control(dispatcher, d_shutdown_flag, d_stats) !=
      CUDAQ_OK) {
    std::cerr << "ERROR: Failed to set control" << std::endl;
    return 1;
  }

  // Use provided launch function, or default to regular dispatch
  cudaq_dispatch_launch_fn_t launch_fn = config.launch_fn;
  if (!launch_fn) {
    launch_fn = &cudaq_launch_dispatch_kernel_regular;
  }
  if (cudaq_dispatcher_set_launch_fn(dispatcher, launch_fn) != CUDAQ_OK) {
    std::cerr << "ERROR: Failed to set launch function" << std::endl;
    return 1;
  }

  if (cudaq_dispatcher_start(dispatcher) != CUDAQ_OK) {
    std::cerr << "ERROR: Failed to start dispatcher" << std::endl;
    return 1;
  }
  std::cout << "  Dispatch kernel launched" << std::endl;

  //============================================================================
  // [5] Launch Hololink kernels and run
  //============================================================================
  std::cout << "\n[5/5] Launching Hololink kernels..." << std::endl;

  std::thread hololink_thread(
      [transceiver]() { hololink_blocking_monitor(transceiver); });

  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  std::cout << "  Hololink RX+TX kernels started" << std::endl;

  // Print QP info for FPGA stimulus tool
  std::cout << "\n=== Bridge Ready ===" << std::endl;
  std::cout << "  QP Number: 0x" << std::hex << our_qp << std::dec
            << std::endl;
  std::cout << "  RKey: " << our_rkey << std::endl;
  std::cout << "  Buffer Addr: 0x" << std::hex << our_buffer << std::dec
            << std::endl;
  std::cout << "\nWaiting for data (Ctrl+C to stop, timeout="
            << config.timeout_sec << "s)..." << std::endl;

  //============================================================================
  // Main run loop
  //============================================================================
  cudaStream_t diag_stream = nullptr;
  BRIDGE_CUDA_CHECK(
      cudaStreamCreateWithFlags(&diag_stream, cudaStreamNonBlocking));

  auto start_time = std::chrono::steady_clock::now();
  uint64_t last_processed = 0;

  while (!g_shutdown) {
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                       std::chrono::steady_clock::now() - start_time)
                       .count();
    if (elapsed > config.timeout_sec) {
      std::cout << "\nTimeout reached (" << config.timeout_sec << "s)"
                << std::endl;
      break;
    }

    // Progress report every 5 seconds
    if (elapsed > 0 && elapsed % 5 == 0) {
      uint64_t processed = 0;
      cudaMemcpyAsync(&processed, d_stats, sizeof(uint64_t),
                      cudaMemcpyDeviceToHost, diag_stream);
      cudaStreamSynchronize(diag_stream);
      if (processed != last_processed) {
        std::cout << "  [" << elapsed << "s] Processed " << processed
                  << " packets" << std::endl;
        last_processed = processed;
      }
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  //============================================================================
  // Shutdown
  //============================================================================
  std::cout << "\n=== Shutting down ===" << std::endl;

  if (diag_stream) {
    cudaStreamDestroy(diag_stream);
    diag_stream = nullptr;
  }

  *shutdown_flag = 1;
  __sync_synchronize();
  cudaq_dispatcher_stop(dispatcher);

  uint64_t total_processed = 0;
  cudaq_dispatcher_get_processed(dispatcher, &total_processed);
  std::cout << "  Total packets processed: " << total_processed << std::endl;

  hololink_close(transceiver);
  if (hololink_thread.joinable())
    hololink_thread.join();

  cudaq_dispatcher_destroy(dispatcher);
  cudaq_dispatch_manager_destroy(manager);
  hololink_destroy_transceiver(transceiver);

  if (shutdown_flag)
    cudaFreeHost(const_cast<int *>(shutdown_flag));
  if (d_stats)
    cudaFree(d_stats);

  // Call tool-specific cleanup
  if (config.cleanup_fn)
    config.cleanup_fn();

  std::cout << "\n*** Bridge shutdown complete ***" << std::endl;
  return 0;
}

/// @brief Default dispatch kernel launch wrapper.
///
/// Matches cudaq_dispatch_launch_fn_t signature; delegates to
/// cudaq_launch_dispatch_kernel_regular from libcudaq-realtime.
inline void bridge_launch_dispatch_kernel(
    volatile std::uint64_t *rx_flags, volatile std::uint64_t *tx_flags,
    std::uint8_t *rx_data, std::uint8_t *tx_data, std::size_t rx_stride_sz,
    std::size_t tx_stride_sz, cudaq_function_entry_t *function_table,
    std::size_t func_count, volatile int *shutdown_flag, std::uint64_t *stats,
    std::size_t num_slots, std::uint32_t num_blocks,
    std::uint32_t threads_per_block, cudaStream_t stream) {
  cudaq_launch_dispatch_kernel_regular(
      rx_flags, tx_flags, rx_data, tx_data, rx_stride_sz, tx_stride_sz,
      function_table, func_count, shutdown_flag, stats, num_slots, num_blocks,
      threads_per_block, stream);
}

} // namespace cudaq::nvqlink
