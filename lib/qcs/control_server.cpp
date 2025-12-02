/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/qcs/control_server.h"
#include "cudaq/nvqlink/network/roce/roce_channel.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"

#include <arpa/inet.h>
#include <cstring>
#include <errno.h>
#include <stdexcept>
#include <sys/socket.h>
#include <unistd.h>

using namespace cudaq::nvqlink;

ControlServer::ControlServer(uint16_t port)
    : port_(port), sock_fd_(-1), client_connected_(false) {
  std::memset(&client_addr_, 0, sizeof(client_addr_));
}

ControlServer::~ControlServer() { stop(); }

void ControlServer::start() {
  if (sock_fd_ >= 0)
    return; // Already started

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

  if (bind(sock_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    int err = errno;
    close(sock_fd_);
    sock_fd_ = -1;
    throw std::runtime_error("Failed to bind UDP socket to port " +
                             std::to_string(port_) + ": " + strerror(err));
  }

  // Set receive timeout (1 second)
  struct timeval tv;
  tv.tv_sec = 1;
  tv.tv_usec = 0;
  setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  NVQLINK_LOG_INFO(DOMAIN_NETWORK, "Control server listening on UDP port {}",
                   port_);
}

void ControlServer::stop() {
  if (sock_fd_ >= 0) {
    close(sock_fd_);
    sock_fd_ = -1;
  }
  client_connected_ = false;
}

bool ControlServer::exchange_connection_params(RoCEChannel *channel) {
  if (!channel)
    throw std::invalid_argument("Channel cannot be null");

  if (sock_fd_ < 0)
    throw std::runtime_error("Control server not started");

  // Wait for DISCOVER packet from client
  if (!client_connected_) {
    NVQLINK_LOG_INFO(DOMAIN_NETWORK, "Waiting for QCS discovery packet...");

    char buf[4096];
    socklen_t len = sizeof(client_addr_);

    // Poll with timeout to allow for cancellation
    int max_attempts = 60; // 60 seconds max wait
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
      ssize_t n = recvfrom(sock_fd_, buf, sizeof(buf), 0,
                           (struct sockaddr *)&client_addr_, &len);

      if (n < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK)
          continue; // Timeout, retry
        throw std::runtime_error("Failed to receive discovery packet");
      }

      // Got discovery packet
      NVQLINK_LOG_INFO(DOMAIN_NETWORK, "QCS discovered: {}:{}",
                       inet_ntoa(client_addr_.sin_addr),
                       ntohs(client_addr_.sin_port));
      client_connected_ = true;
      break;
    }

    if (!client_connected_) {
      throw std::runtime_error("Timeout waiting for QCS discovery");
    }
  }

  // Build server's connection params
  auto params = channel->get_connection_params();
  nlohmann::json j;
  j["qpn"] = params.qpn;
  j["psn"] = params.psn;
  j["gid"] = gid_to_string(params.gid);
  j["vaddr"] = params.vaddr;
  j["rkey"] = params.rkey;
  j["lid"] = params.lid;
  j["num_slots"] = params.num_slots;
  j["slot_size"] = params.slot_size;

  std::string msg = j.dump();

  // Send params to client (with retries)
  const int max_retries = 5;
  for (int retry = 0; retry < max_retries; ++retry) {
    ssize_t sent =
        sendto(sock_fd_, msg.c_str(), msg.length(), 0,
               (struct sockaddr *)&client_addr_, sizeof(client_addr_));

    if (sent < 0)
      throw std::runtime_error("Failed to send channel params");

    NVQLINK_LOG_INFO(
        DOMAIN_NETWORK,
        "Sent channel params: QPN={}, GID={}, vaddr=0x{:x}, rkey=0x{:x}",
        params.qpn, gid_to_string(params.gid), params.vaddr, params.rkey);

    // Wait for ACK with client params
    char buf[4096];
    socklen_t len = sizeof(client_addr_);

    ssize_t n = recvfrom(sock_fd_, buf, sizeof(buf), 0,
                         (struct sockaddr *)&client_addr_, &len);

    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        NVQLINK_LOG_WARNING(DOMAIN_NETWORK, "No ACK received, retrying ({}/{})",
                            retry + 1, max_retries);
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

      NVQLINK_LOG_INFO(DOMAIN_NETWORK, "Received ACK: QPN={}, GID={}",
                       client_qpn, client_gid_str);

      // Configure QP with client params
      union ibv_gid client_gid = string_to_gid(client_gid_str);
      channel->set_remote_qp(client_qpn, client_gid);

      NVQLINK_LOG_INFO(DOMAIN_NETWORK,
                       "RDMA connection established successfully");
      return true;

    } catch (const std::exception &e) {
      NVQLINK_LOG_WARNING(DOMAIN_NETWORK, "Failed to parse ACK: {}", e.what());
      continue;
    }
  }

  NVQLINK_LOG_ERROR(DOMAIN_NETWORK,
                    "Failed to establish connection after {} retries",
                    max_retries);
  return false;
}

void ControlServer::send_command(const std::string &cmd,
                                 const nlohmann::json &params) {
  if (sock_fd_ < 0)
    throw std::runtime_error("Control server not started");

  if (!client_connected_)
    throw std::runtime_error("No client connected");

  nlohmann::json msg;
  msg["cmd"] = cmd;
  if (!params.empty())
    msg["params"] = params;

  std::string msg_str = msg.dump();
  ssize_t sent = sendto(sock_fd_, msg_str.c_str(), msg_str.length(), 0,
                        (struct sockaddr *)&client_addr_, sizeof(client_addr_));

  if (sent < 0)
    throw std::runtime_error("Failed to send command: " + cmd);

  NVQLINK_LOG_INFO(DOMAIN_NETWORK, "Sent command: {}", cmd);
}

nlohmann::json
ControlServer::wait_for_response(std::chrono::milliseconds timeout) {
  if (sock_fd_ < 0)
    throw std::runtime_error("Control server not started");

  // Set timeout
  struct timeval tv;
  tv.tv_sec = timeout.count() / 1000;
  tv.tv_usec = (timeout.count() % 1000) * 1000;
  setsockopt(sock_fd_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

  char buf[4096];
  socklen_t len = sizeof(client_addr_);

  ssize_t n = recvfrom(sock_fd_, buf, sizeof(buf), 0,
                       (struct sockaddr *)&client_addr_, &len);

  if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK)
      throw std::runtime_error("Timeout waiting for response");
    throw std::runtime_error("Failed to receive response");
  }

  std::string response_str(buf, n);
  return nlohmann::json::parse(response_str);
}

std::string ControlServer::get_client_address() const {
  if (!client_connected_)
    return "not connected";

  char buf[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &client_addr_.sin_addr, buf, sizeof(buf));
  return std::string(buf) + ":" + std::to_string(ntohs(client_addr_.sin_port));
}

std::string ControlServer::gid_to_string(const union ibv_gid &gid) {
  char buf[64];
  inet_ntop(AF_INET6, gid.raw, buf, sizeof(buf));
  return buf;
}

union ibv_gid ControlServer::string_to_gid(const std::string &gid_str) {
  union ibv_gid gid;
  inet_pton(AF_INET6, gid_str.c_str(), gid.raw);
  return gid;
}
