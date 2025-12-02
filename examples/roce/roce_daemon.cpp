/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "udp_control_server.h"
#include "utils.h"
#include "cudaq/nvqlink/daemon/daemon.h"
#include "cudaq/nvqlink/daemon/registry/function_traits.h"
#include "cudaq/nvqlink/daemon/registry/function_wrapper.h"
#include "cudaq/nvqlink/network/config.h"
#include "cudaq/nvqlink/network/roce/roce_channel.h"
#include "cudaq/nvqlink/network/steering/verbs_flow_switch.h"

#include <csignal>
#include <iostream>
#include <unistd.h>

using namespace cudaq::nvqlink;

//===----------------------------------------------------------------------===//
// Utility
//===----------------------------------------------------------------------===//

// Global flag for clean shutdown
static volatile bool running = true;

static void signal_handler(int) { running = false; }

static void print_network_config(const ChannelConfig &cfg) {
  // Get network interface info
  std::string roce_iface = get_interface_name(cfg.nic_device);
  std::string roce_iface_ip = get_interface_ip(roce_iface);
  std::string server_mac = get_interface_mac(roce_iface);
  std::string udp_ip = roce_iface_ip.empty() ? get_primary_ip() : roce_iface_ip;

  std::cout << "\n========================================================\n";
  std::cout << "  Network Configuration\n";
  std::cout << "========================================================\n";
  std::cout << "RoCE device:  " << cfg.nic_device << "\n";
  std::cout << "RoCE iface:   " << roce_iface << "\n";
  std::cout << "RoCE MAC:     " << server_mac << "\n";
  if (!roce_iface_ip.empty()) {
    std::cout << "RoCE IP:      " << roce_iface_ip << "\n";
    std::cout << "UDP IP:       " << roce_iface_ip << " (same as RoCE)\n";
  } else {
    std::cout << "RoCE IP:      (none - layer 2 only)\n";
    std::cout << "UDP IP:       " << udp_ip << " (primary system IP)\n";
  }
  std::cout << "ROCE_PORT:    4791\n";
  std::cout << "========================================================\n\n";
}

//===----------------------------------------------------------------------===//
// RPC Functions
//===----------------------------------------------------------------------===//

namespace rpc {

static int32_t add(int32_t a, int32_t b) {
  int32_t result = a + b;
  std::cout << "[RPC] Add: " << a << " + " << b << " = " << result << std::endl;
  return result;
}

/// Echo function - demonstrates variable-length data handling.
/// Note: This uses the stream-based API since it needs dynamic sizing.
static int echo_function(InputStream &in, OutputStream &out) {
  // Read entire input
  std::vector<char> data;
  while (in.available() > 0) {
    data.push_back(in.read<char>());
  }

  // Echo back
  out.write_bytes(data.data(), data.size());

  std::cout << "[RPC] Echo: " << std::string(data.begin(), data.end())
            << std::endl;
  return 0;
}

} // namespace rpc

//===----------------------------------------------------------------------===//

int main(int argc, char **argv) {
  std::cout << "\n========================================================\n";
  std::cout << "  NVQLink Daemon Example " << std::endl;
  std::cout << "========================================================\n";

  // Setup signal handler
  signal(SIGINT, signal_handler);
  signal(SIGTERM, signal_handler);

  try {
    // 1. Configure backend (Soft-RoCE device)
    ChannelConfig backend_cfg;
    backend_cfg.nic_device = (argc > 1) ? argv[1] : "rxe0";
    backend_cfg.queue_id = 0;
    backend_cfg.pool_size_bytes = 64 * 1024 * 1024;

    print_network_config(backend_cfg);

    // 2. Create RoCE backend in UC mode
    std::cout << "\nCreating RoCE backend (UC mode)...\n";

    // Create FlowSwitch for traffic steering
    auto flow_switch = std::make_shared<VerbsFlowSwitch>();

    // Create channel (independent context, listens on UDP port 9000)
    auto channel = std::make_unique<RoCEChannel>(backend_cfg.nic_device, 9000,
                                                 flow_switch);

    // Initialize the channel (sets up InfiniBand resources)
    channel->initialize();
    std::cout << "RoCE channel initialized successfully\n";

    // 3. Start UDP control server
    std::cout << "\nStarting UDP control server...\n";
    UDPControlServer control_server(9999);
    control_server.start();

    // 4. Wait for client and exchange params
    std::cout << "\n========================================================\n";
    std::cout << "  Waiting for Client Connection\n";
    std::cout << "========================================================\n";
    std::cout << "UDP Control Port: 9999\n";
    std::cout << "Press Ctrl+C to cancel...\n";

    // Single channel exchange
    auto *channel_raw = channel.get();
    std::vector<RoCEChannel *> channels = {channel_raw};
    if (!control_server.exchange_multi_channel(channels, &running)) {
      std::cout << "\nFailed to exchange channel parameters.\n";
      return 1;
    }
    std::cout << "\nUC QP ready for data path.\n";

    // 5. Create Daemon with RPC functions
    std::cout << "\nCreating Daemon...\n";
    DaemonConfig daemon_config = DaemonConfigBuilder()
                                     .set_id("roce_daemon_0")
                                     .set_datapath_mode(DatapathMode::CPU)
                                     .set_cpu_cores({0})
                                     .build();

    auto daemon = std::make_unique<Daemon>(daemon_config, std::move(channel));

    // Register RPC functions
    //   - Method 1: Supports variable-length data
    FunctionMetadata echo_meta = {.function_id = 1,
                                  .name = "echo",
                                  .type = FunctionType::CPU,
                                  .max_result_size = 1024,
                                  .cpu_function = rpc::echo_function};
    daemon->register_function(echo_meta);

    //   - Method 2: Just pass the function. The `NVQLINK_RPC_HANDLE` macro
    //               extracts the function name automatically and generates a
    //               stable hash-based ID.
    daemon->register_function(NVQLINK_RPC_HANDLE(rpc::add));

    std::cout << "Registered RPC functions:\n";
    std::cout << "  ID 1: echo\n";
    std::cout << "  ID 0x" << std::hex << hash_name("rpc::add") << std::dec
              << ": rpc::add\n";

    daemon->start();

    // Run until interrupted
    std::cout << "\n========================================================\n";
    std::cout << "  Server running. Press Ctrl+C to stop.\n";
    std::cout << "========================================================\n";

    while (running) {
      sleep(1);

      // Periodically print stats
      static int counter = 0;
      if (++counter % 10 == 0) {
        auto stats = daemon->get_stats();
        std::cout << "[Stats] Packets received: " << stats.packets_received
                  << ", sent: " << stats.packets_sent << "\n";
      }
    }

    std::cout << "\nShutting down...\n";
    daemon->stop();

    auto stats = daemon->get_stats();
    std::cout << "\nFinal stats:\n";
    std::cout << "  Packets received: " << stats.packets_received << "\n";
    std::cout << "  Packets sent: " << stats.packets_sent << "\n";

    return 0;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
