/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/qcs/config.h"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace cudaq::nvqlink {

class ControlServer;
class RoCEChannel;

/// @brief Represents a connection to a Quantum Control System (QCS/FPGA)
///
/// Handles the CONTROL PLANE via UDP (out-of-band communication):
/// - Exchange RDMA connection parameters (QPN, GID, vaddr, rkey)
/// - Program upload
/// - Execution trigger (START/STOP commands)
///
/// The DATA PLANE (device_call handling) is managed by Daemon + Channel.
/// QCSDevice CONFIGURES the Channel but does NOT own it - ownership is
/// transferred to the Daemon for data plane operations.
///
/// Architecture:
/// ```
/// QCSDevice (UDP control)  ──configures──>  Channel (RoCE data)
///                                              │
///                                              │ owned by
///                                              ▼
///                                           Daemon (RPC server)
/// ```
class QCSDevice {
public:
  /// @brief Create QCSDevice with its own UDP control socket
  /// @param config QCS device configuration (name, control port)
  explicit QCSDevice(const QCSDeviceConfig &config);
  ~QCSDevice();

  // Delete copy/move
  QCSDevice(const QCSDevice &) = delete;
  QCSDevice &operator=(const QCSDevice &) = delete;
  QCSDevice(QCSDevice &&) = delete;
  QCSDevice &operator=(QCSDevice &&) = delete;

  //--- Connection Management (UDP) ---

  /// @brief Start UDP control server and wait for QCS to connect
  ///
  /// Performs RDMA parameter exchange:
  /// 1. Waits for QCS DISCOVER packet
  /// 2. Sends channel's connection params (QPN, GID, vaddr, rkey)
  /// 3. Receives QCS's connection params (QPN, GID)
  /// 4. Configures the channel with remote QP info
  ///
  /// @param channel The RoCE channel to configure (not owned by QCSDevice)
  /// @throws std::runtime_error if connection fails
  void establish_connection(RoCEChannel *channel);

  /// @brief Check if QCS is connected
  bool is_connected() const { return connected_; }

  /// @brief Disconnect from QCS (sends STOP command)
  void disconnect();

  /// @brief Get connection parameters (for diagnostics)
  struct ConnectionInfo {
    uint32_t local_qpn{0};
    uint32_t remote_qpn{0};
    std::array<uint8_t, 16> local_gid{};
    std::array<uint8_t, 16> remote_gid{};
    uint64_t ring_buffer_addr{0};
    uint32_t rkey{0};
  };
  ConnectionInfo get_connection_info() const { return connection_info_; }

  //--- Program Management (via UDP) ---

  /// @brief Upload compiled quantum program to QCS
  ///
  /// Note: Currently a stub - actual implementation depends on:
  /// - Program format (LLVM IR, custom ISA, etc.)
  /// - Transfer mechanism (UDP chunks, separate channel, etc.)
  ///
  /// @param binary Compiled program binary
  void upload_program(const std::vector<std::byte> &binary);

  //--- Execution Control (UDP commands) ---

  /// @brief Send START command to QCS - begins quantum program execution
  ///
  /// After this call, the QCS will start executing the uploaded program.
  /// device_call packets will start arriving via the RDMA data plane.
  void trigger();

  /// @brief Check if program execution is complete
  ///
  /// Note: Currently a stub - actual implementation would poll QCS
  /// status or wait for COMPLETE message over UDP.
  bool is_complete() const;

  /// @brief Send ABORT command to QCS - stops program execution
  void abort();

private:
  QCSDeviceConfig config_;
  std::unique_ptr<ControlServer> control_server_;
  ConnectionInfo connection_info_;
  bool connected_{false};
};

} // namespace cudaq::nvqlink

