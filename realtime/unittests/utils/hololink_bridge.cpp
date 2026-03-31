/*******************************************************************************
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

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/realtime/hololink_bridge_common.h"

//==============================================================================
// Increment RPC Handler Function Table
//==============================================================================

// The actual __device__ rpc_increment_handler lives in
// init_rpc_increment_function_table.cu (compiled by nvcc).  We declare the
// host-callable setup function here so this .cpp can be compiled by g++.

extern "C" void
setup_rpc_increment_function_table(cudaq_function_entry_t *d_entries);

//==============================================================================
// Increment RPC Handler Graph Function Table
//==============================================================================
extern "C" void setup_rpc_graph_increment_function_table(
    cudaq_function_entry_t *h_entries, void **d_mailbox_bank,
    cudaGraph_t *graph_out, cudaGraphExec_t *exec_out);

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
             "dispatch + TX in one kernel)\n"
          << "  --cpu                 Use host dispatch with CUDA graph "
             "launch.\n";
      return 0;
    }
  }

  try {
    std::cout << "=== Hololink Generic Bridge ===" << std::endl;

    // Parse common bridge args
    cudaq::realtime::BridgeConfig config;
    cudaq::realtime::parse_bridge_args(argc, argv, config);

    std::cout << "Device: " << config.device << std::endl;
    std::cout << "Peer IP: " << config.peer_ip << std::endl;
    std::cout << "Remote QP: 0x" << std::hex << config.remote_qp << std::dec
              << std::endl;
    std::cout << "GPU: " << config.gpu_id << std::endl;

    // Initialize CUDA early to allocate function table
    cudaError_t err = cudaSetDevice(config.gpu_id);
    if (err != cudaSuccess) {
      std::cerr << "ERROR: cudaSetDevice failed: " << cudaGetErrorString(err)
                << std::endl;
      return 1;
    }

    if (config.dispatch_path != CUDAQ_DISPATCH_PATH_HOST) {
      // Normal device dispatch path: set up increment RPC function table on GPU
      cudaq_function_entry_t *d_function_entries = nullptr;
      err = cudaMalloc(&d_function_entries, sizeof(cudaq_function_entry_t));
      if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed: " << cudaGetErrorString(err)
                  << std::endl;
        return 1;
      }
      setup_rpc_increment_function_table(d_function_entries);

      config.d_function_entries = d_function_entries;
      config.func_count = 1;
      config.launch_fn = &cudaq::realtime::bridge_launch_dispatch_kernel;
      config.cleanup_fn = [d_function_entries]() {
        cudaFree(d_function_entries);
      };
    } else {
      // Host dispatch path: set up increment RPC function table on host. 
      err =
          cudaHostAlloc(&config.h_mailbox, sizeof(void *), cudaHostAllocMapped);
      if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaHostAlloc failed: " << cudaGetErrorString(err)
                  << std::endl;
        return 1;
      }
      std::memset(config.h_mailbox, 0, sizeof(void *));
      err = cudaHostGetDevicePointer((void **)&config.d_mailbox,
                                     config.h_mailbox, 0);
      if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaHostGetDevicePointer failed: "
                  << cudaGetErrorString(err) << std::endl;
        return 1;
      }
      constexpr std::size_t num_func = 1;
      cudaGraph_t graph = nullptr;
      cudaGraphExec_t graph_exec = nullptr;
      cudaq_function_entry_t *h_function_entries =
          new cudaq_function_entry_t[num_func];
      setup_rpc_graph_increment_function_table(
          h_function_entries, config.d_mailbox, &graph, &graph_exec);

      config.h_function_entries = h_function_entries;
      config.h_func_count = num_func;
      config.cleanup_fn = [graph, graph_exec, h_function_entries,
                           h_mailbox_bank = config.h_mailbox]() {
        delete[] h_function_entries;
        cudaGraphExecDestroy(graph_exec);
        cudaGraphDestroy(graph);
        cudaFreeHost(h_mailbox_bank);
      };
    }

    return cudaq::realtime::bridge_run(config);

  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << std::endl;
    return 1;
  }
}
