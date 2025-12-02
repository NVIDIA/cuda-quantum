/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/roce/roce_channel.h"
#include "udp_control_server.h"
#include "utils.h"
#include "cudaq/nvqlink/network/serialization/input_stream.h"
#include "cudaq/nvqlink/network/serialization/output_stream.h"
#include "cudaq/nvqlink/network/steering/verbs_flow_switch.h"
#include "cudaq/nvqlink/utils/instrumentation/profiler.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>

using namespace cudaq::nvqlink;

// Atomic flag to control worker threads
static std::atomic<bool> g_running{true};

// Signal handler for clean shutdown
static void signal_handler(int) {
  g_running.store(false);
  std::cout << "\n[SIGNAL] Shutdown requested..." << std::endl;
}

//===----------------------------------------------------------------------===//
// Worker thread functions
//===----------------------------------------------------------------------===//

static void add_worker_thread(Channel *channel) {
  std::cout << "[ADD Worker] Thread started (queue-based routing)\n";

  // Create persistent streams for this thread
  InputStream in(*channel);
  OutputStream out(*channel);

  while (g_running.load()) {
    NVQLINK_TRACE_USER_RANGE("add_worker_poll");

    // Check if input is available
    if (in.available()) {
      try {
        NVQLINK_TRACE_USER_RANGE("add_function_processing");

        // Read payload header: [arg_len]
        uint32_t arg_len = in.read<uint32_t>();

        // Read the two integers
        int32_t a = in.read<int32_t>();
        int32_t b = in.read<int32_t>();

        std::cout << "[ADD Worker] Processing: " << a << " + " << b << " = "
                  << (a + b) << std::endl;

        // Compute the result
        int32_t result = a + b;

        // Write result back
        out.write(result);
        out.flush(); // Send the response packet

      } catch (const std::exception &e) {
        std::cerr << "[ADD Worker] Error: " << e.what() << std::endl;
        NVQLINK_TRACE_MARK_ERROR(DOMAIN_USER, "ADD_WORKER_ERROR");
      }
    }

    // Small sleep to avoid busy-waiting (can be removed for production)
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }

  std::cout << "[ADD Worker] Thread stopped\n";
}

static void multiply_worker_thread(Channel *channel) {
  std::cout << "[MULTIPLY Worker] Thread started (queue-based routing)\n";

  // Create persistent streams for this thread
  InputStream in(*channel);
  OutputStream out(*channel);

  while (g_running.load()) {
    NVQLINK_TRACE_USER_RANGE("multiply_worker_poll");

    // Check if input is available
    if (in.available()) {
      try {
        NVQLINK_TRACE_USER_RANGE("multiply_function_processing");

        // Read payload header: [arg_len]
        uint32_t arg_len = in.read<uint32_t>();

        // Read the two integers
        int32_t a = in.read<int32_t>();
        int32_t b = in.read<int32_t>();

        std::cout << "[MULTIPLY Worker] Processing: " << a << " * " << b
                  << " = " << (a * b) << std::endl;

        // Compute the result
        int32_t result = a * b;

        // Write result back
        out.write(result);
        out.flush(); // Send the response packet

      } catch (const std::exception &e) {
        std::cerr << "[MULTIPLY Worker] Error: " << e.what() << std::endl;
        NVQLINK_TRACE_MARK_ERROR(DOMAIN_USER, "MULTIPLY_WORKER_ERROR");
      }
    }

    // Small sleep to avoid busy-waiting
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }

  std::cout << "[MULTIPLY Worker] Thread stopped\n";
}

static void echo_worker_thread(Channel *channel) {
  std::cout << "[ECHO Worker] Thread started (queue-based routing)\n";

  // Create persistent streams for this thread
  InputStream in(*channel);
  OutputStream out(*channel);

  while (g_running.load()) {
    NVQLINK_TRACE_USER_RANGE("echo_worker_poll");

    // Check if input is available
    if (in.available()) {
      try {
        NVQLINK_TRACE_USER_RANGE("echo_function_processing");

        // No function_id needed - queue selection IS the routing!
        // Read the string length first (client must send length prefix)
        uint32_t str_len = in.read<uint32_t>();

        // Read the string data
        std::vector<uint8_t> data(str_len);
        in.read_bytes(data.data(), str_len);
        std::string msg(data.begin(), data.end());

        std::cout << "[ECHO Worker] Processing: \"" << msg << "\"" << std::endl;

        // Echo back the data
        out.write_bytes(data.data(), data.size());
        out.flush();

      } catch (const std::exception &e) {
        std::cerr << "[ECHO Worker] Error: " << e.what() << std::endl;
        NVQLINK_TRACE_MARK_ERROR(DOMAIN_USER, "ECHO_WORKER_ERROR");
      }
    }

    // Small sleep to avoid busy-waiting
    std::this_thread::sleep_for(std::chrono::microseconds(10));
  }

  std::cout << "[ECHO Worker] Thread stopped\n";
}

//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  std::cout << "============================================================\n";
  std::cout << "       RoCE Channel Example with Worker Threads\n";
  std::cout << "============================================================\n";
  std::cout << "This example demonstrates using Channel API with RoCE\n";
  std::cout << "backend and multiple worker threads polling for data.\n";
  std::cout << "============================================================\n";

  // Setup signal handler
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  try {
    // Parse command line arguments
    std::string nic_device = "rxe0"; // Default to rxe0 (Soft-RoCE)
    if (argc > 1)
      nic_device = argv[1];

    std::string roce_iface = get_interface_name(nic_device);
    std::string server_mac = get_interface_mac(roce_iface);

    std::cout << "\n========================================================\n";
    std::cout << "  Network Configuration\n";
    std::cout << "========================================================\n";
    std::cout << "RoCE device:  " << nic_device << std::endl;
    std::cout << "RoCE iface:   " << roce_iface << std::endl;
    std::cout << "RoCE MAC:     " << server_mac << std::endl;
    std::cout << "========================================================\n";

    // Build backend configuration - one Channel per worker thread
    std::cout
        << "\n[1] Creating RoCE channels (one per worker, shared NIC)...\n";

    // Create FlowSwitch for traffic steering
    auto flow_switch = std::make_shared<VerbsFlowSwitch>();

    // Create channels with independent contexts
    // Each channel listens on a different UDP port for flow steering
    auto add_channel =
        std::make_unique<RoCEChannel>(nic_device, 9000, flow_switch);
    auto mult_channel =
        std::make_unique<RoCEChannel>(nic_device, 9001, flow_switch);
    auto echo_channel =
        std::make_unique<RoCEChannel>(nic_device, 9002, flow_switch);

    auto *add_channel_raw = add_channel.get();
    auto *mult_channel_raw = mult_channel.get();
    auto *echo_channel_raw = echo_channel.get();

    // Initialize all channels (sets up InfiniBand resources)
    add_channel_raw->initialize();
    mult_channel_raw->initialize();
    echo_channel_raw->initialize();

    std::cout << "    ✓ Three RoCE channels created (independent contexts)\n";
    std::cout << "    ✓ ADD channel: UDP port 9000\n";
    std::cout << "    ✓ MULTIPLY channel: UDP port 9001\n";
    std::cout << "    ✓ ECHO channel: UDP port 9002\n";

    // Start UDP control server
    std::cout << "\n[3] Starting UDP control server...\n";
    UDPControlServer control_server(9999);
    control_server.start();

    std::cout << "    ✓ UDP control server listening on port 9999\n";

    // Wait for client connection and exchange params
    std::cout << "\n========================================================\n";
    std::cout << "  Waiting for Client Connection\n";
    std::cout << "========================================================\n";
    std::cout << "TCP Control Port: 9999\n\n";
    std::cout << "Press Ctrl+C to cancel...\n\n";

    std::cout << "To connect from client:\n";
    std::cout << "  sudo ./roce_full_client " << nic_device << " <server_ip>\n";
    std::cout << "========================================================\n";

    volatile bool running_for_udp = true;

    // Use multi-channel handler to set up all three channels
    std::vector<RoCEChannel *> all_channels = {
        add_channel_raw, mult_channel_raw, echo_channel_raw};

    if (!control_server.exchange_multi_channel(all_channels,
                                               &running_for_udp)) {
      std::cout << "\nFailed to exchange channel parameters.\n";
      return 1;
    }

    std::cout << "\n✅ All 3 UC QPs ready for data path.\n";

    // Start worker threads - each with dedicated Channel
    std::cout << "\n[4] Starting worker threads...\n";
    g_running.store(true);

    std::thread add_thread(add_worker_thread, add_channel_raw);
    std::thread multiply_thread(multiply_worker_thread, mult_channel_raw);
    std::thread echo_thread(echo_worker_thread, echo_channel_raw);

    std::cout << "    ✓ ADD worker thread started (queue 0) [ACTIVE]\n";
    std::cout << "    ✓ MULTIPLY worker thread started (queue 1) [ACTIVE]\n";
    std::cout << "    ✓ ECHO worker thread started (queue 2) [ACTIVE]\n";

    // Give threads time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "\n========================================================\n";
    std::cout << "  RoCE Channel Ready!\n";
    std::cout << "========================================================\n\n";
    std::cout << "Channel-to-Function mapping:\n";
    std::cout << "  Channel 0 (queue 0): ADD\n";
    std::cout << "  Channel 1 (queue 1): MULTIPLY\n";
    std::cout << "  Channel 2 (queue 2): ECHO\n\n";

    std::cout << "Press Ctrl+C to stop\n";
    std::cout << "========================================================\n";

    // Main loop - print periodic status
    auto last_print = std::chrono::steady_clock::now();
    int status_count = 0;
    while (g_running.load()) {
      std::this_thread::sleep_for(std::chrono::seconds(1));

      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::seconds>(now - last_print)
              .count() >= 10) {
        status_count++;
        std::cout << "[Status] Workers running (" << (status_count * 10)
                  << "s elapsed)..." << std::endl;
        last_print = now;
      }
    }

    // Stop worker threads
    std::cout << "\n[5] Stopping worker threads...\n";
    g_running.store(false);

    add_thread.join();
    multiply_thread.join();
    echo_thread.join();

    std::cout << "    ✓ All worker threads stopped\n";

    std::cout << "\n[6] Cleaning up...\n";
    add_channel.reset();
    mult_channel.reset();
    echo_channel.reset();
    std::cout << "    ✓ All channels destroyed\n";

    std::cout << "\n========================================================\n";
    std::cout << "RoCE Channel Example completed successfully!\n";
    std::cout << "========================================================\n";

  } catch (const std::exception &e) {
    std::cerr << "\n❌ Error: " << e.what() << std::endl;
    std::cerr << "\nTroubleshooting:\n";
    std::cerr << "  1. Check if Soft-RoCE is installed: rdma link show\n";
    std::cerr << "  2. Create RoCE device if needed:\n";
    std::cerr << "     sudo rdma link add rxe0 type rxe netdev lo\n";
    std::cerr << "  3. Check device status: ibv_devices\n";
    return 1;
  }

  return 0;
}
