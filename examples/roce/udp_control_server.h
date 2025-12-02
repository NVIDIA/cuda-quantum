/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/roce/roce_channel.h"

#include <arpa/inet.h>
#include <cstdint>
#include <string>
#include <vector>

/// @brief UDP Control Server for RoCE connection setup
///
/// Server-driven protocol: Server sends channel params, waits for client ACK.
/// No initial handshake - server drives by sending params for each channel.
///
class UDPControlServer {
public:
  UDPControlServer(uint16_t port);
  ~UDPControlServer();

  /// @brief Start listening on the control port (UDP)
  void start();

  /// @brief Exchange connection parameters for multiple channels
  ///
  /// For each channel:
  ///   1. Server → Client: Channel params (JSON)
  ///   2. Client → Server: ACK with client params (JSON)
  ///   3. Server configures QP
  ///
  /// @param channels Vector of RoCE channels to configure
  /// @param running_flag Pointer to flag for checking if server should keep
  /// running
  /// @return true if all channels setup succeeded, false otherwise
  bool exchange_multi_channel(
      const std::vector<cudaq::nvqlink::RoCEChannel *> &channels,
      volatile bool *running_flag);

private:
  uint16_t port_;
  int sock_fd_;
  struct sockaddr_in client_addr_;
  bool client_connected_;

  std::string gid_to_string(const union ibv_gid &gid);
  union ibv_gid string_to_gid(const std::string &gid_str);
};
