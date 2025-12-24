/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/qcs/qcs_device.h"
#include "cudaq/nvqlink/network/channels/roce/roce_channel.h"
#include "cudaq/nvqlink/qcs/control_server.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"

#include <stdexcept>

using namespace cudaq::nvqlink;

QCSDevice::QCSDevice(const QCSDeviceConfig &config) : config_(config) {
  if (!config_.is_valid())
    throw std::invalid_argument("Invalid QCSDeviceConfig");

  // Create control server with UDP socket
  control_server_ = std::make_unique<ControlServer>(config_.control_port);

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "QCSDevice created: name={}, control_port={}", config_.name,
                   config_.control_port);
}

QCSDevice::~QCSDevice() {
  if (connected_) {
    try {
      disconnect();
    } catch (...) {
      // Don't throw from destructor
    }
  }
}

void QCSDevice::establish_connection(RoCEChannel *channel) {
  if (!channel)
    throw std::invalid_argument("Channel cannot be null");

  if (connected_)
    throw std::runtime_error("Already connected");

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Establishing connection to QCS '{}'...",
                   config_.name);

  // Start UDP control server
  control_server_->start();

  // Exchange RDMA connection parameters
  bool success = control_server_->exchange_connection_params(channel);
  if (!success)
    throw std::runtime_error("Failed to exchange RDMA parameters with QCS");

  // Save connection info from channel
  auto params = channel->get_connection_params();
  connection_info_.local_qpn = params.qpn;
  connection_info_.ring_buffer_addr = params.vaddr;
  connection_info_.rkey = params.rkey;

  // Copy GID
  std::memcpy(connection_info_.local_gid.data(), params.gid.raw,
              sizeof(params.gid.raw));

  connected_ = true;

  NVQLINK_LOG_INFO(
      DOMAIN_CHANNEL, "QCS connection established: QPN={}, vaddr=0x{:x}",
      connection_info_.local_qpn, connection_info_.ring_buffer_addr);
}

void QCSDevice::disconnect() {
  if (!connected_)
    return;

  try {
    // Send STOP command to QCS
    control_server_->send_command("STOP");
    NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Sent STOP command to QCS");
  } catch (const std::exception &e) {
    NVQLINK_LOG_WARNING(DOMAIN_CHANNEL, "Failed to send STOP command: {}",
                        e.what());
  }

  control_server_->stop();
  connected_ = false;

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Disconnected from QCS");
}

void QCSDevice::upload_program(const std::vector<std::byte> &binary) {
  if (!connected_)
    throw std::runtime_error("Not connected to QCS");

  // TODO: Implement program upload
  // This is a stub - actual implementation depends on:
  // 1. Program format (LLVM IR, custom ISA, etc.)
  // 2. Transfer mechanism (UDP chunks, separate channel, etc.)
  //
  // Possible approaches:
  // - Chunk large binaries and send via UDP
  // - Use separate TCP connection for bulk transfer
  // - Use RDMA WRITE for direct memory transfer
  //
  // For now, just log a warning
  NVQLINK_LOG_WARNING(DOMAIN_CHANNEL,
                      "Program upload not yet implemented (size={} bytes)",
                      binary.size());

  // For a simple implementation, could send metadata via UDP
  nlohmann::json msg;
  msg["cmd"] = "UPLOAD_PROGRAM";
  msg["size"] = binary.size();
  // In real implementation, would transfer the actual binary here

  control_server_->send_command("UPLOAD_PROGRAM", msg);
}

void QCSDevice::trigger() {
  if (!connected_)
    throw std::runtime_error("Not connected to QCS");

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Triggering QCS program execution");

  // Send START command
  control_server_->send_command("START");

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "START command sent - QCS should begin execution");
}

bool QCSDevice::is_complete() const {
  if (!connected_)
    return false;

  // TODO: Implement completion check
  // This is a stub - actual implementation would:
  // 1. Poll for COMPLETE message from QCS via UDP
  // 2. Or check a shared status flag in RDMA memory
  //
  // For now, return false (not complete)
  return false;
}

void QCSDevice::abort() {
  if (!connected_)
    throw std::runtime_error("Not connected to QCS");

  NVQLINK_LOG_WARNING(DOMAIN_CHANNEL, "Aborting QCS program execution");

  control_server_->send_command("ABORT");

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "ABORT command sent to QCS");
}
