/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file hololink_bridge.cpp
/// @brief Generic Hololink bridge tool for testing libcudaq-realtime dispatch.
///
/// Registers a simple increment RPC handler (adds 1 to each byte) and wires
/// it through the Hololink GPU-RoCE Transceiver.  No QEC or decoder dependency.
///
/// Usage:
///   ./hololink_bridge \
///       --device=rocep1s0f0 \
///       --peer-ip=10.0.0.2 \
///       --remote-qp=0x2 \
///       --gpu=0 \
///       --timeout=60

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

//==============================================================================
// Increment RPC Handler Function Table
//==============================================================================

// The actual __device__ rpc_increment_handler lives in
// init_rpc_increment_function_table.cu (compiled by nvcc).  We declare the
// host-callable setup function here so this .cpp can be compiled by g++.

extern "C" void
setup_rpc_increment_function_table(cudaq_function_entry_t *d_entries);

/// @brief Configuration for dispatch kernel setup.
struct DispatchConfig {
  int gpu_id = 0;          ///< GPU device ID
  int timeout_sec = 60;    ///< Runtime timeout in seconds
                           // Ring buffer sizing
  size_t frame_size = 256; ///< Minimum frame size (RPCHeader + payload)
  size_t page_size =
      384; ///< Ring buffer slot size (>= frame_size, 128-aligned)
  unsigned num_pages = 64; ///< Number of ring buffer slots
  /// @brief Dispatch kernel grid configuration.
  /// Defaults match the regular (non-cooperative) kernel.
  cudaq_kernel_type_t kernel_type = CUDAQ_KERNEL_REGULAR;
  uint32_t num_blocks = 1;
  uint32_t threads_per_block = 32;
};

inline void parse_bridge_args(int argc, char *argv[], DispatchConfig &config) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--gpu=") == 0)
      config.gpu_id = std::stoi(arg.substr(6));
    else if (arg.find("--timeout=") == 0)
      config.timeout_sec = std::stoi(arg.substr(10));
  }
}

#define HANDLE_CUDAQ_REALTIME_ERROR(x)                                         \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUDAQ_OK) {                                                     \
      std::stringstream ss;                                                    \
      ss << "CUDAQ realtime error at " << __FILE__ << ":" << __LINE__ << ": "  \
         << err << std::endl;                                                  \
      throw std::runtime_error(ss.str());                                      \
    }                                                                          \
  }

//==============================================================================
// CUDA Error Checking
//==============================================================================

#ifndef BRIDGE_CUDA_CHECK
#define BRIDGE_CUDA_CHECK(call)                                                \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "     \
                << cudaGetErrorString(err) << std::endl;                       \
      return 1;                                                                \
    }                                                                          \
  } while (0)
#endif

std::atomic<bool> &bridge_shutdown_flag() {
  static std::atomic<bool> flag{false};
  return flag;
}
void bridge_signal_handler(int) { bridge_shutdown_flag() = true; }

//==============================================================================
// Main
//==============================================================================

int main(int argc, char *argv[]) {
  // Check for help
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: " << argv[0] << " [options]\n"
          << "\n"
          << "Generic Hololink bridge for testing libcudaq-realtime dispatch.\n"
          << "Registers increment handler (adds 1 to each byte of the RPC "
             "payload).\n"
          << "\n"
          << "Options:\n"
          << "  --device=NAME         IB device (default: rocep1s0f0)\n"
          << "  --peer-ip=ADDR        FPGA/emulator IP (default: 10.0.0.2)\n"
          << "  --remote-qp=N         Remote QP number (default: 0x2)\n"
          << "  --gpu=N               GPU device ID (default: 0)\n"
          << "  --timeout=N           Timeout in seconds (default: 60)\n"
          << "  --page-size=N         Ring buffer slot size (default: 384)\n"
          << "  --num-pages=N         Number of ring buffer slots (default: "
             "64)\n"
          << "  --exchange-qp         Enable QP exchange protocol\n"
          << "  --exchange-port=N     TCP port for QP exchange (default: "
             "12345)\n";
      return 0;
    }
  }

  try {
    signal(SIGINT, bridge_signal_handler);
    signal(SIGTERM, bridge_signal_handler);
    auto &g_shutdown = bridge_shutdown_flag();
    std::cout << "=== Hololink Generic Bridge ===" << std::endl;

    //============================================================================
    // Set up the Hololink bridge
    //============================================================================
    cudaq_realtime_bridge_handle_t bridge_handle = nullptr;
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_create(
        &bridge_handle, CUDAQ_PROVIDER_HOLOLINK, argc, argv));

    std::cout << "Bridge created successfully. Connecting..." << std::endl;

    std::cout << "\n Forcing CUDA module loading..." << std::endl;
    {
      int dispatch_blocks = 0;
      BRIDGE_CUDA_CHECK(
          cudaq_dispatch_kernel_query_occupancy(&dispatch_blocks, 1));
      std::cout << "  Dispatch kernel occupancy: " << dispatch_blocks
                << " blocks/SM" << std::endl;
    }

    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_connect(bridge_handle));

    std::cout << "Bridge connected successfully. Retrieving the ring buffer ..."
              << std::endl;

    cudaq_ringbuffer_t ringbuffer{};
    HANDLE_CUDAQ_REALTIME_ERROR(
        cudaq_bridge_get_ringbuffer(bridge_handle, &ringbuffer));

    //============================================================================
    // Set up CUDA-Q realtime dispatch
    //============================================================================
    DispatchConfig config;
    parse_bridge_args(argc, argv, config);

    // Allocate control variables
    void *tmp_shutdown = nullptr;
    BRIDGE_CUDA_CHECK(
        cudaHostAlloc(&tmp_shutdown, sizeof(int), cudaHostAllocMapped));
    volatile int *shutdown_flag = static_cast<volatile int *>(tmp_shutdown);
    void *tmp_d_shutdown = nullptr;
    BRIDGE_CUDA_CHECK(
        cudaHostGetDevicePointer(&tmp_d_shutdown, tmp_shutdown, 0));
    volatile int *d_shutdown_flag = static_cast<volatile int *>(tmp_d_shutdown);
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

    // Create CUDA-Q dispatcher manager 
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_dispatch_manager_create(&manager));      
    cudaq_dispatcher_config_t dconfig{};
    dconfig.device_id = config.gpu_id;
    dconfig.num_blocks = config.num_blocks;
    dconfig.threads_per_block = config.threads_per_block;
    dconfig.num_slots = static_cast<uint32_t>(config.num_pages);
    dconfig.slot_size = static_cast<uint32_t>(config.page_size);
    dconfig.vp_id = 0;
    dconfig.kernel_type = config.kernel_type;
    dconfig.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;
    // Create dispatcher with the above config
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_dispatcher_create(manager, &dconfig, &dispatcher));
    
    // Set the ring buffer retrieved from the bridge
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_dispatcher_set_ringbuffer(dispatcher, &ringbuffer));

    // Set up the function table with the increment handler entries
    // Populate the GPU function table with the increment handler entry
    cudaq_function_entry_t *d_function_entries = nullptr;
    BRIDGE_CUDA_CHECK(
        cudaMalloc(&d_function_entries, sizeof(cudaq_function_entry_t)));
    setup_rpc_increment_function_table(d_function_entries);
    // Create a function table struct to pass to the dispatcher
    cudaq_function_table_t table{};
    table.entries = d_function_entries;
    table.count = 1; // Only one handler (increment)
    // Set the function table for the dispatcher
    HANDLE_CUDAQ_REALTIME_ERROR(
        cudaq_dispatcher_set_function_table(dispatcher, &table));
    // Set the control variables (shutdown flag and stats pointer) for the
    // dispatcher
    HANDLE_CUDAQ_REALTIME_ERROR(
        cudaq_dispatcher_set_control(dispatcher, d_shutdown_flag, d_stats));
    // Set the dispatch kernel launch function for the dispatcher
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_dispatcher_set_launch_fn(
        dispatcher, &cudaq_launch_dispatch_kernel_regular));

    // Start the dispatcher (launches the dispatch kernel)
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_dispatcher_start(dispatcher));
    std::cout << "  Dispatch kernel launched" << std::endl;

    // Launch Hololink kernels and run
    std::cout << "\n[5/5] Launching Hololink kernels...\n";
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_launch(bridge_handle));
    std::cout << "  Hololink RX+TX kernels started\n";
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

    std::cout << "  Disconnecting bridge..." << std::endl;
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_disconnect(bridge_handle));
    std::cout << "  Destroying bridge..." << std::endl;
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_destroy(bridge_handle));
    std::cout << "Bridge shut down successfully." << std::endl;
    // Clean up
    cudaFree(d_function_entries);
    cudaFree(d_stats);
    cudaq_dispatcher_destroy(dispatcher);
    cudaq_dispatch_manager_destroy(manager);
    if (shutdown_flag)
      cudaFreeHost(const_cast<int *>(shutdown_flag));
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
