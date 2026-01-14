/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "doca_udp_control.h"
#include "cudaq/nvqlink/daemon/config.h"
#include "cudaq/nvqlink/daemon/daemon.h"
#include "cudaq/nvqlink/network/channels/doca/doca_channel.h"
#include "cudaq/nvqlink/network/channels/doca/doca_channel_config.h"

#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

using namespace cudaq::nvqlink;

//==============================================================================
// Example RPC Function: Add Two Numbers
//==============================================================================

// GPU function is defined in gpu_functions.cu (compiled with nvcc)
// This external function retrieves the proper device function pointer
extern "C" void *get_gpu_add_function_ptr();

//==============================================================================
// Signal Handling
//==============================================================================

static volatile bool running = true;

void signal_handler(int signal) {
  std::cout << "\nReceived signal " << signal << ", shutting down...\n";
  running = false;
}

//==============================================================================
// Main Example
//==============================================================================

int main(int argc, char *argv[]) {
  std::cout << "=================================================\n"
            << "NVQLink DOCA Daemon Example\n"
            << "=================================================\n"
            << std::endl;

  // Set up signal handling
  std::signal(SIGINT, signal_handler);
  std::signal(SIGTERM, signal_handler);

  // Parse simple arguments
  std::string nic_device = "mlx5_1";
  std::string peer_ip;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--nic" && i + 1 < argc)
      nic_device = argv[++i];
    else if (arg == "--peer" && i + 1 < argc)
      peer_ip = argv[++i];
  }

  try {
    std::cout << "Step 1: Creating DOCA channel..." << std::endl;

    // Create DOCA channel (from Plan lines 975-980)
    DOCAChannelConfig doca_config;
    doca_config.nic_device = nic_device;
    doca_config.peer_ip = peer_ip;
    doca_config.page_size = 4096;
    doca_config.num_pages = 1024;
    doca_config.max_rpc_size =
        4096; // Must be non-zero for RDMA WRITE WITH IMMEDIATE

    auto channel = std::make_unique<DOCAChannel>(doca_config);
    channel->initialize();

    std::cout << "  ✓ Channel initialized" << std::endl;

    //===--------------------------------------------------------------------===
    // Exchange connection parameters via UDP control plane
    //===--------------------------------------------------------------------===

    std::cout << "\nStep 2: Exchanging connection parameters..." << std::endl;

    DOCAUDPControlServer control_server(9999);
    control_server.start();

    if (!control_server.exchange_params(channel.get(), &running)) {
      std::cerr << "Failed to exchange connection parameters\n";
      return 1;
    }

    std::cout << "Connection parameters exchanged\n";

    // Verify QP is connected
    if (!channel->is_connected()) {
      std::cerr << "Channel not connected after parameter exchange!\n";
      return 1;
    }
    std::cout << "QP is in connected state (RTS)\n";

    //===--------------------------------------------------------------------===
    // Create daemon with GPU datapath
    //===--------------------------------------------------------------------===

    std::cout << "\nStep 3: Creating daemon with GPU datapath...\n";

    DaemonConfig daemon_config;
    daemon_config.id = "qec_decoder";
    daemon_config.datapath_mode = DatapathMode::GPU;
    daemon_config.compute.gpu_device_id = 0;

    std::cout << "  Daemon ID:   " << daemon_config.id << "\n";
    std::cout << "  Datapath:    GPU\n";

    Daemon daemon(daemon_config, std::move(channel));

    //===--------------------------------------------------------------------===
    // Register functions (from Plan lines 992)
    //===--------------------------------------------------------------------===

    std::cout << "\nStep 4: Registering RPC functions...\n";

    // Register GPU function
    // Function ID must match client's FUNCTION_ADD = 2
    //
    // CRITICAL: Use get_gpu_add_function_ptr() to get the DEVICE function
    // pointer! The GPU function is defined in gpu_functions.cu and compiled
    // with nvcc. Taking &device_function directly from host code gives a HOST
    // address, which causes undefined behavior when the GPU tries to call it.
    void *device_func_ptr = get_gpu_add_function_ptr();
    if (!device_func_ptr) {
      throw std::runtime_error("Failed to get GPU function pointer");
    }

    FunctionMetadata gpu_add_meta;
    gpu_add_meta.function_id = 2; // Must match client's FUNCTION_ADD
    gpu_add_meta.name = "gpu_add_numbers";
    gpu_add_meta.type = FunctionType::GPU;
    gpu_add_meta.max_result_size = 128;
    gpu_add_meta.gpu_function = device_func_ptr;

    daemon.register_function(gpu_add_meta);
    std::cout << "Registered: gpu_add_numbers (GPU, function_id=2)\n";

    //===--------------------------------------------------------------------===
    // Start daemon
    //===--------------------------------------------------------------------===

    std::cout << "\nStep 5: Starting daemon...\n";
    daemon.start();

    std::cout << "\n=================================================\n"
              << "Daemon is running!\n"
              << "=================================================\n\n";

    std::cout << "The GPU kernel is now polling for RPC requests.\n"
              << "GPU has direct control of the NIC (zero CPU involvement).\n\n"
              << "Press Ctrl+C to stop the daemon.\n\n";

    //===--------------------------------------------------------------------===
    // Run Until Interrupted
    //===--------------------------------------------------------------------===

    while (running) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    //===--------------------------------------------------------------------===
    // Shutdown
    //===--------------------------------------------------------------------===

    std::cout << "\nShutting down daemon...\n";
    daemon.stop();
    std::cout << "Daemon stopped\n";

    std::cout << "\n=================================================\n"
              << "Example completed successfully!\n"
              << "=================================================\n\n";

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "\nError: " << e.what() << "\n";
    return 1;
  }
}
