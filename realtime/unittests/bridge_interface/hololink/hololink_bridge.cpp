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
///   ./hololink_app \
///       --device=mlx5_1 \
///       --peer-ip=10.0.0.2 \
///       --remote-qp=0x2 \
///       --gpu=0 \
///       --timeout=60

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/bridge/hololink/hololink_doca_transport_ctx.h"
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
  // Forward mode: use Hololink's built-in forward kernel (echo) instead of
  // separate RX + dispatch + TX kernels.  Useful for baseline latency testing.
  bool forward = false;

  // Unified dispatch mode: single kernel combines RDMA RX, RPC dispatch, and
  // RDMA TX via direct DOCA verbs calls.  Eliminates the inter-kernel flag
  // handoff overhead of the 3-kernel path.  Regular handlers only.
  bool unified = false;
};

void parse_bridge_args(int argc, char *argv[], DispatchConfig &config) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg.find("--gpu=") == 0)
      config.gpu_id = std::stoi(arg.substr(6));
    else if (arg.find("--timeout=") == 0)
      config.timeout_sec = std::stoi(arg.substr(10));
    else if (arg == "--forward")
      config.forward = true;
    else if (arg == "--unified")
      config.unified = true;
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
          << "  --payload-size=N      RPC payload size in bytes (default: 8)\n"
          << "  --page-size=N         Ring buffer slot size (default: 384)\n"
          << "  --num-pages=N         Number of ring buffer slots (default: "
             "64)\n"
          << "  --exchange-qp         Enable QP exchange protocol\n"
          << "  --exchange-port=N     TCP port for QP exchange (default: "
             "12345)\n"
          << "  --forward             Use Hololink forward kernel (echo) "
             "instead of dispatch\n"
          << "  --unified             Use unified dispatch kernel (RX + "
             "dispatch + TX in one kernel)\n";
      return 0;
    }
  }

  try {
    signal(SIGINT, bridge_signal_handler);
    signal(SIGTERM, bridge_signal_handler);
    auto &g_shutdown = bridge_shutdown_flag();
    std::cout << "=== Hololink Generic Bridge ===" << std::endl;
    // Allocate control variables (shutdown flag)
    void *tmp_shutdown = nullptr;
    BRIDGE_CUDA_CHECK(
        cudaHostAlloc(&tmp_shutdown, sizeof(int), cudaHostAllocMapped));
    void *tmp_d_shutdown = nullptr;
    BRIDGE_CUDA_CHECK(
        cudaHostGetDevicePointer(&tmp_d_shutdown, tmp_shutdown, 0));
    volatile int *d_shutdown_flag = static_cast<volatile int *>(tmp_d_shutdown);
    volatile int *shutdown_flag = static_cast<volatile int *>(tmp_shutdown);

    // CUDA-Q realtime variables
    uint64_t *d_stats = nullptr;
    BRIDGE_CUDA_CHECK(cudaMalloc(&d_stats, sizeof(uint64_t)));
    BRIDGE_CUDA_CHECK(cudaMemset(d_stats, 0, sizeof(uint64_t)));
    cudaq_function_entry_t *d_function_entries = nullptr;
    cudaq_dispatch_manager_t *manager = nullptr;
    cudaq_dispatcher_t *dispatcher = nullptr;

    //============================================================================
    // Parse configurations from args
    //============================================================================
    DispatchConfig config;
    parse_bridge_args(argc, argv, config);
    //============================================================================
    // Set up the Hololink bridge
    //============================================================================
    cudaq_realtime_bridge_handle_t bridge_handle = nullptr;
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_create(
        &bridge_handle, CUDAQ_PROVIDER_HOLOLINK, argc, argv));
    std::cout << "Bridge created successfully. Connecting..." << std::endl;

    if (!config.forward) {
      if (!config.unified) {
        int dispatch_blocks = 0;
        cudaError_t occ_err;
        if (config.kernel_type == CUDAQ_KERNEL_COOPERATIVE) {
          occ_err = cudaq_dispatch_kernel_cooperative_query_occupancy(
              &dispatch_blocks, config.threads_per_block);
        } else {
          occ_err = cudaq_dispatch_kernel_query_occupancy(&dispatch_blocks, 1);
        }
        if (occ_err != cudaSuccess) {
          std::cerr << "ERROR: Dispatch kernel occupancy query failed: "
                    << cudaGetErrorString(occ_err) << std::endl;
          return 1;
        }
        std::cout << "  Dispatch kernel occupancy: " << dispatch_blocks
                  << " blocks/SM" << std::endl;
      }

      HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_connect(bridge_handle));

      std::cout << "\nWiring dispatch kernel ("
                << (config.unified ? "unified" : "3-kernel") << ")..."
                << std::endl;

      *shutdown_flag = 0;
      int zero = 0;
      BRIDGE_CUDA_CHECK(cudaMemcpy(const_cast<int *>(d_shutdown_flag), &zero,
                                   sizeof(int), cudaMemcpyHostToDevice));

      // Create CUDA-Q dispatcher manager
      HANDLE_CUDAQ_REALTIME_ERROR(cudaq_dispatch_manager_create(&manager));
      cudaq_dispatcher_config_t dconfig{};
      dconfig.device_id = config.gpu_id;
      dconfig.vp_id = 0;
      dconfig.dispatch_mode = CUDAQ_DISPATCH_DEVICE_CALL;

      if (config.unified) {
        dconfig.kernel_type = CUDAQ_KERNEL_UNIFIED;
        dconfig.num_blocks = 1;
        dconfig.threads_per_block = 1;
        dconfig.num_slots = 0;
        dconfig.slot_size = 0;
      } else {
        dconfig.kernel_type = config.kernel_type;
        dconfig.num_blocks = config.num_blocks;
        dconfig.threads_per_block = config.threads_per_block;
        dconfig.num_slots = static_cast<uint32_t>(config.num_pages);
        dconfig.slot_size = static_cast<uint32_t>(config.page_size);
      }

      // Create dispatcher with the above config
      HANDLE_CUDAQ_REALTIME_ERROR(
          cudaq_dispatcher_create(manager, &dconfig, &dispatcher));

      // Transport context for unified mode (must outlive the dispatcher)
      cudaq_unified_dispatch_ctx_t unified_dispatch{};

      if (config.unified) {
        std::cout << "Retrieving the unified dispatch context ..." << std::endl;
        HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_get_transport_context(
            bridge_handle, UNIFIED, &unified_dispatch));
        if (cudaq_dispatcher_set_unified_launch(
                dispatcher, unified_dispatch.launch_fn,
                unified_dispatch.transport_ctx) != CUDAQ_OK) {
          std::cerr << "ERROR: Failed to set unified launch function"
                    << std::endl;
          return 1;
        }
      } else {
        std::cout << "Retrieving the ring buffer ..." << std::endl;

        cudaq_ringbuffer_t ringbuffer{};
        HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_get_transport_context(
            bridge_handle, RING_BUFFER, &ringbuffer));
        if (cudaq_dispatcher_set_ringbuffer(dispatcher, &ringbuffer) !=
            CUDAQ_OK) {
          std::cerr << "ERROR: Failed to set ringbuffer" << std::endl;
          return 1;
        }

        if (cudaq_dispatcher_set_launch_fn(
                dispatcher, &cudaq_launch_dispatch_kernel_regular) !=
            CUDAQ_OK) {
          std::cerr << "ERROR: Failed to set launch function" << std::endl;
          return 1;
        }
      }

      // Set up the function table with the increment handler entries
      // Populate the GPU function table with the increment handler entry
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

      // Start the dispatcher (launches the dispatch kernel)
      HANDLE_CUDAQ_REALTIME_ERROR(cudaq_dispatcher_start(dispatcher));
      std::cout << "  Dispatch kernel launched" << std::endl;
    } else {
      std::cout << "\n[4/5] Forward mode -- skipping dispatch kernel"
                << std::endl;
    }

    // Launch Hololink kernels and run
    std::cout << "\n[5/5] Launching Hololink kernels...\n";
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_launch(bridge_handle));
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
      if (!config.forward && d_stats && elapsed > 0 && elapsed % 5 == 0) {
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
    if (!config.forward) {
      *shutdown_flag = 1;
      __sync_synchronize();
      cudaq_dispatcher_stop(dispatcher);

      uint64_t total_processed = 0;
      cudaq_dispatcher_get_processed(dispatcher, &total_processed);
      std::cout << "  Total packets processed (dispatch RX): "
                << total_processed << std::endl;
    }
    std::cout << "  Disconnecting bridge..." << std::endl;
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_disconnect(bridge_handle));
    // Clean up
    BRIDGE_CUDA_CHECK(cudaFree(d_function_entries));
    BRIDGE_CUDA_CHECK(cudaFree(d_stats));
    if (dispatcher)
      cudaq_dispatcher_destroy(dispatcher);
    if (manager)
      cudaq_dispatch_manager_destroy(manager);
    std::cout << "  Destroying bridge..." << std::endl;
    HANDLE_CUDAQ_REALTIME_ERROR(cudaq_bridge_destroy(bridge_handle));
    std::cout << "Bridge shut down successfully." << std::endl;
    if (shutdown_flag)
      BRIDGE_CUDA_CHECK(cudaFreeHost(const_cast<int *>(shutdown_flag)));
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
