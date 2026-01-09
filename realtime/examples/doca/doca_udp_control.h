/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/channels/doca/doca_channel.h"

#include <arpa/inet.h>
#include <cstdint>
#include <cstring>
#include <errno.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

/// @brief UDP Control Server for DOCA Channel connection setup
///
/// Adapted from RoCE UDPControlServer for DOCAChannel.
/// Server-driven protocol: Server sends channel params, waits for client ACK.
///
/// Protocol flow:
///   1. Client → Server: Discovery packet (UDP)
///   2. Server → Client: Connection params (JSON)
///   3. Client → Server: ACK with client params (JSON)
///   4. Server → Client: START message (JSON)
///
class DOCAUDPControlServer {
public:
  DOCAUDPControlServer(uint16_t port)
      : port_(port), sock_fd_(-1), client_connected_(false) {
    std::memset(&client_addr_, 0, sizeof(client_addr_));
  }

  ~DOCAUDPControlServer() {
    if (sock_fd_ >= 0)
      close(sock_fd_);
  }

  /// @brief Start listening on the control port (UDP)
  void start() {
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

  /// @brief Exchange connection parameters with client
  ///
  /// Protocol:
  ///   1. Wait for client discovery packet
  ///   2. Send server params (QPN, GID, vaddr, rkey, num_slots, slot_size)
  ///   3. Wait for client ACK with client params (QPN, GID)
  ///   4. Configure QP with client params
  ///   5. Send START message
  ///
  /// @param channel DOCA channel to configure
  /// @param running_flag Pointer to flag for checking if server should keep
  /// running
  /// @return true if exchange succeeded, false otherwise
  bool exchange_params(cudaq::nvqlink::DOCAChannel *channel,
                       volatile bool *running_flag) {
    std::cout << "[UDP] Waiting for client...\n";

    // Set receive timeout to allow checking running flag
    struct timeval tv;
    tv.tv_sec = 1;
    tv.tv_usec = 0;
    setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    try {
      // Step 1: Wait for client discovery packet
      if (!client_connected_) {
        char buf[4096];
        socklen_t len = sizeof(client_addr_);

        std::cout << "  [WAIT] Waiting for client discovery packet..."
                  << std::endl;

        while (*running_flag) {
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
          break;
        }

        if (!client_connected_)
          return false;
      }

      // Step 2: Send server's connection params
      auto params = channel->get_connection_params();
      nlohmann::json j;
      j["qpn"] = params.qpn;
      j["gid"] = gid_to_string(params.gid);
      j["vaddr"] = params.buffer_addr;
      j["rkey"] = params.rkey;
      j["num_slots"] = params.num_slots;
      j["slot_size"] = params.slot_size;

      std::string msg = j.dump();

      ssize_t sent =
          sendto(sock_fd_, msg.c_str(), msg.length(), 0,
                 (struct sockaddr *)&client_addr_, sizeof(client_addr_));

      if (sent < 0)
        throw std::runtime_error("Failed to send channel params");

      std::cout << "  Server → Client: Connection params sent\n";
      std::cout << "    QPN: " << params.qpn << "\n";
      std::cout << "    GID: " << gid_to_string(params.gid) << "\n";
      std::cout << "    vaddr: 0x" << std::hex << params.buffer_addr << std::dec
                << "\n";
      std::cout << "    rkey: 0x" << std::hex << params.rkey << std::dec
                << "\n";
      std::cout << "    Ring buffer: " << params.num_slots << " slots x "
                << params.slot_size << " bytes\n";

      // Step 3: Wait for client ACK with client params
      bool ack_received = false;
      int retry_count = 0;
      const int max_retries = 5;

      while (!ack_received && retry_count < max_retries && *running_flag) {
        char buf[4096];
        socklen_t len = sizeof(client_addr_);

        ssize_t n = recvfrom(sock_fd_, buf, sizeof(buf), 0,
                             (struct sockaddr *)&client_addr_, &len);

        if (n < 0) {
          if (errno == EAGAIN || errno == EWOULDBLOCK) {
            // Timeout, retry sending params
            retry_count++;
            std::cout << "  [RETRY] No ACK, retrying (" << retry_count << "/"
                      << max_retries << ")...\n";
            sendto(sock_fd_, msg.c_str(), msg.length(), 0,
                   (struct sockaddr *)&client_addr_, sizeof(client_addr_));
            continue;
          }
          throw std::runtime_error("Failed to receive ACK");
        }

        // Parse ACK
        try {
          std::string ack_str(buf, n);
          auto client_j = nlohmann::json::parse(ack_str);

          uint32_t client_qpn = client_j["qpn"];
          std::string client_gid_str = client_j["gid"];

          std::cout << "  Client → Server: ACK received\n";
          std::cout << "    QPN: " << client_qpn << "\n";
          std::cout << "    GID: " << client_gid_str << "\n";

          // Configure QP with client params
          auto client_gid = string_to_gid(client_gid_str);

          std::cout << "  [DEBUG] Calling set_remote_qp()...\n";
          try {
            channel->set_remote_qp(client_qpn, client_gid);
            std::cout << "  [DEBUG] set_remote_qp() completed\n";
          } catch (const std::exception &e) {
            std::cerr << "  [ERROR] set_remote_qp() failed: " << e.what()
                      << "\n";
            throw;
          }

          std::cout << "  ✓ QP setup complete\n";

          ack_received = true;

        } catch (const std::exception &e) {
          std::cerr << "  [WARN] Failed to parse ACK: " << e.what() << "\n";
          retry_count++;
          continue;
        }
      }

      if (!ack_received) {
        std::cerr << "[UDP] Failed to receive ACK after " << max_retries
                  << " retries\n";
        return false;
      }

      // Step 4: Send START message
      std::cout << "[UDP] Sending START message to client...\n";
      nlohmann::json start_msg;
      start_msg["cmd"] = "START";

      std::string start_str = start_msg.dump();
      sent = sendto(sock_fd_, start_str.c_str(), start_str.length(), 0,
                    (struct sockaddr *)&client_addr_, sizeof(client_addr_));

      if (sent < 0) {
        std::cerr << "[UDP] Warning: Failed to send START message\n";
        return false;
      }

      std::cout << "[UDP] ✓ START message sent - client can begin RDMA\n";

      return true;

    } catch (const std::exception &e) {
      std::cerr << "[UDP] Error: " << e.what() << "\n";
      return false;
    }
  }

private:
  uint16_t port_;
  int sock_fd_;
  struct sockaddr_in client_addr_;
  bool client_connected_;

  std::string gid_to_string(const std::array<uint8_t, 16> &gid) {
    char buf[64];
    inet_ntop(AF_INET6, gid.data(), buf, sizeof(buf));
    return buf;
  }

  std::array<uint8_t, 16> string_to_gid(const std::string &gid_str) {
    std::array<uint8_t, 16> gid;
    inet_pton(AF_INET6, gid_str.c_str(), gid.data());
    return gid;
  }
};
