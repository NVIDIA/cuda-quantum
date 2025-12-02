/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "udp_control_server.h"

#include <arpa/inet.h>
#include <cstring>
#include <errno.h>
#include <iostream>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

UDPControlServer::UDPControlServer(uint16_t port)
    : port_(port), sock_fd_(-1), client_connected_(false) {
  std::memset(&client_addr_, 0, sizeof(client_addr_));
}

UDPControlServer::~UDPControlServer() {
  if (sock_fd_ >= 0)
    close(sock_fd_);
}

void UDPControlServer::start() {
  // Create UDP socket
  sock_fd_ = socket(AF_INET, SOCK_DGRAM, 0);
  if (sock_fd_ < 0)
    throw std::runtime_error("Failed to create UDP socket");

  // Allow address reuse
  int opt = 1;
  setsockopt(sock_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  // Bind to port
  struct sockaddr_in addr = {};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = INADDR_ANY;
  addr.sin_port = htons(port_);

  if (bind(sock_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0)
    throw std::runtime_error("Failed to bind UDP socket");

  std::cout << "[UDP] Control server listening on port " << port_ << "\n";
}

bool UDPControlServer::exchange_multi_channel(
    const std::vector<cudaq::nvqlink::RoCEChannel *> &channels,
    volatile bool *running_flag) {

  std::cout << "[UDP] Waiting for client...\n";

  // Set receive timeout to allow checking running flag
  struct timeval tv;
  tv.tv_sec = 1;
  tv.tv_usec = 0;
  setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  try {
    // Exchange parameters for each channel
    for (std::size_t i = 0; i < channels.size(); i++) {
      std::cout << "\n[UDP] === Channel " << i << " exchange ===\n";

      auto *channel = channels[i];

      // Build server's connection params
      auto params = channel->get_connection_params();
      nlohmann::json j;
      j["channel_idx"] = i;
      j["qpn"] = params.qpn;
      j["psn"] = params.psn;
      j["gid"] = gid_to_string(params.gid);
      j["vaddr"] = params.vaddr;
      j["rkey"] = params.rkey;
      j["lid"] = params.lid;
      j["num_slots"] = params.num_slots;
      j["slot_size"] = params.slot_size;

      std::string msg = j.dump();

      // Send channel params to client (retry until ACK received)
      bool ack_received = false;
      int retry_count = 0;
      const int max_retries = 5;

      while (!ack_received && retry_count < max_retries && *running_flag) {
        // If first channel, we don't know client address yet
        // Send to broadcast or wait for client to send first packet
        if (i == 0 && !client_connected_) {
          // Wait for initial packet from client (discovery)
          char buf[4096];
          socklen_t len = sizeof(client_addr_);

          std::cout << "  [WAIT] Waiting for client discovery packet..."
                    << std::endl;

          ssize_t n = recvfrom(sock_fd_, buf, sizeof(buf), 0,
                               (struct sockaddr *)&client_addr_, &len);

          if (n < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
              continue; // Timeout, check running flag
            throw std::runtime_error("Failed to receive discovery packet");
          }

          // Got discovery packet
          std::cout << "  ✓ Client discovered: "
                    << inet_ntoa(client_addr_.sin_addr) << ":"
                    << ntohs(client_addr_.sin_port) << "\n";
          client_connected_ = true;
        }

        // Send params to client
        ssize_t sent =
            sendto(sock_fd_, msg.c_str(), msg.length(), 0,
                   (struct sockaddr *)&client_addr_, sizeof(client_addr_));

        if (sent < 0)
          throw std::runtime_error("Failed to send channel params");

        std::cout << "  Server → Client: Channel " << i << " params sent\n";
        std::cout << "    QPN: " << params.qpn << "\n";
        std::cout << "    GID: " << gid_to_string(params.gid) << "\n";
        std::cout << "    vaddr: 0x" << std::hex << params.vaddr << std::dec
                  << "\n";
        std::cout << "    rkey: 0x" << std::hex << params.rkey << std::dec
                  << "\n";

        // Wait for ACK with client params
        char buf[4096];
        socklen_t len = sizeof(client_addr_);

        ssize_t n = recvfrom(sock_fd_, buf, sizeof(buf), 0,
                             (struct sockaddr *)&client_addr_, &len);

        if (n < 0) {
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // Timeout, retry
            retry_count++;
            std::cout << "  [RETRY] No ACK, retrying (" << retry_count << "/"
                      << max_retries << ")...\n";
            continue;
          }
          throw std::runtime_error("Failed to receive ACK");
        }

        // Parse ACK
        try {
          std::string ack_str(buf, n);
          auto client_j = nlohmann::json::parse(ack_str);

          uint32_t ack_channel_idx = client_j["channel_idx"];
          if (ack_channel_idx != i) {
            std::cerr << "  [WARN] ACK for wrong channel (expected " << i
                      << ", got " << ack_channel_idx << ")\n";
            continue;
          }

          uint32_t client_qpn = client_j["qpn"];
          std::string client_gid_str = client_j["gid"];

          std::cout << "  Client → Server: ACK received\n";
          std::cout << "    QPN: " << client_qpn << "\n";
          std::cout << "    GID: " << client_gid_str << "\n";

          // Configure QP with client params
          union ibv_gid client_gid = string_to_gid(client_gid_str);
          channel->set_remote_qp(client_qpn, client_gid);

          std::cout << "  ✓ Channel " << i << " UC QP setup complete\n";

          ack_received = true;

        } catch (const std::exception &e) {
          std::cerr << "  [WARN] Failed to parse ACK: " << e.what() << "\n";
          retry_count++;
          continue;
        }
      }

      if (!ack_received) {
        std::cerr << "[UDP] Failed to receive ACK for channel " << i
                  << " after " << max_retries << " retries\n";
        return false;
      }
    }

    std::cout << "\n[UDP] All " << channels.size() << " channels exchanged!\n";

    // Step 3: Send START message with function routing
    // In Channel mode, routing is done by queue selection (no function_id
    // needed)
    std::cout << "[UDP] Sending START message to client...\n";
    nlohmann::json start_msg;
    start_msg["cmd"] = "START";
    start_msg["num_channels"] = channels.size();

    // Function routing: map function_id → channel_idx
    // This tells the client which queue to send each function to
    nlohmann::json routing;
    routing["1"] = 2; // ECHO (function_id=1) → channel 2
    routing["2"] = 0; // ADD (function_id=2) → channel 0
    routing["3"] = 1; // MULTIPLY (function_id=3) → channel 1
    start_msg["function_routing"] = routing;

    std::string msg = start_msg.dump();
    ssize_t sent =
        sendto(sock_fd_, msg.c_str(), msg.length(), 0,
               (struct sockaddr *)&client_addr_, sizeof(client_addr_));

    if (sent < 0) {
      std::cerr << "[UDP] Warning: Failed to send START message\n";
    } else {
      std::cout << "[UDP] ✓ START message sent - client can begin RPC\n";
      std::cout << "       Function routing:\n";
      std::cout << "         ECHO (1) → channel 2\n";
      std::cout << "         ADD (2) → channel 0\n";
      std::cout << "         MULTIPLY (3) → channel 1\n";
    }

    return true;

  } catch (const std::exception &e) {
    std::cerr << "[UDP] Error: " << e.what() << "\n";
    return false;
  }
}

std::string UDPControlServer::gid_to_string(const union ibv_gid &gid) {
  char buf[64];
  inet_ntop(AF_INET6, gid.raw, buf, sizeof(buf));
  return buf;
}

union ibv_gid UDPControlServer::string_to_gid(const std::string &gid_str) {
  union ibv_gid gid;
  inet_pton(AF_INET6, gid_str.c_str(), gid.raw);
  return gid;
}
