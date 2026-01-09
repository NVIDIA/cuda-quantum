/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <infiniband/verbs.h>
#include <nlohmann/json.hpp>
#include <string>

namespace cudaq::nvqlink {

class RoCEChannel;

/// @brief UDP server for out-of-band control communication with QCS
///
/// Handles the control plane protocol between RTH (Real-time Host) and QCS
/// (Quantum Control System) for RDMA connection setup and execution control.
///
/// Protocol:
/// 1. QCS → RTH: DISCOVER packet
/// 2. RTH → QCS: RDMA params {qpn, gid, vaddr, rkey, num_slots, slot_size}
/// 3. QCS → RTH: ACK with {qpn, gid}
/// 4. RTH → QCS: START command (triggers quantum program execution)
/// 5. ... data flows over RDMA (data plane) ...
/// 6. RTH → QCS: STOP command (or QCS → RTH: COMPLETE)
class ControlServer {
public:
  /// @brief Create a control server listening on the specified UDP port
  /// @param port UDP port number for control plane communication
  explicit ControlServer(uint16_t port);
  ~ControlServer();

  // Delete copy/move
  ControlServer(const ControlServer &) = delete;
  ControlServer &operator=(const ControlServer &) = delete;
  ControlServer(ControlServer &&) = delete;
  ControlServer &operator=(ControlServer &&) = delete;

  /// @brief Start listening on UDP port
  void start();

  /// @brief Stop server and close socket
  void stop();

  /// @brief Wait for QCS to send DISCOVER, exchange RDMA params, configure
  /// channel
  ///
  /// This performs the RDMA handshake:
  /// 1. Wait for QCS DISCOVER packet
  /// 2. Send channel's connection params (QPN, GID, vaddr, rkey)
  /// 3. Receive QCS's connection params (QPN, GID)
  /// 4. Configure the channel with remote QP info
  ///
  /// @param channel The RoCE channel to configure with remote QP info
  /// @return true if connection established successfully
  bool exchange_connection_params(RoCEChannel *channel);

  /// @brief Send a command to QCS
  /// @param cmd Command string (e.g., "START", "STOP", "ABORT")
  /// @param params Optional JSON parameters
  void send_command(const std::string &cmd, const nlohmann::json &params = {});

  /// @brief Wait for response from QCS
  /// @param timeout Maximum time to wait for response
  /// @return JSON response from QCS
  nlohmann::json wait_for_response(std::chrono::milliseconds timeout);

  /// @brief Check if QCS client is connected
  bool is_client_connected() const { return client_connected_; }

  /// @brief Get client address (for diagnostics)
  std::string get_client_address() const;

private:
  uint16_t port_;
  int sock_fd_{-1};
  struct sockaddr_in client_addr_;
  bool client_connected_{false};

  // Helper methods
  std::string gid_to_string(const union ibv_gid &gid);
  union ibv_gid string_to_gid(const std::string &gid_str);
};

} // namespace cudaq::nvqlink
