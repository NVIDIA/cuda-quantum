/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <infiniband/verbs.h>

#include <cstdint>
#include <string>

namespace cudaq::nvqlink {

/// Shared InfiniBand Verbs context for memory sharing across channels.
///
/// Holds a shared `ibv_context` and `ibv_pd` that can be used by multiple
/// `RoCEChannel` instances to enable zero-copy data sharing. When multiple
/// channels share the same `VerbsContext`, they can register memory once
/// and access it across all channels.
///
/// Optional: Only needed when data sharing between channels is required.
/// Channels can also operate independently with their own contexts.
///
class VerbsContext {
public:
  /// Create a shared verbs context for the specified device.
  ///
  /// @param device_name Device name (e.g., "mlx5_0") or index (e.g., "0")
  /// @throws std::runtime_error if device not found or initialization fails
  ///
  explicit VerbsContext(const std::string &device_name);

  ~VerbsContext();

  // Non-copyable, non-movable
  VerbsContext(const VerbsContext &) = delete;
  VerbsContext &operator=(const VerbsContext &) = delete;

  /// Get the InfiniBand Verbs context
  struct ibv_context *get_context() const { return context_; }

  /// Get the Protection Domain (for memory registration and QP creation)
  struct ibv_pd *get_protection_domain() const { return pd_; }

  /// Get the device name
  const std::string &get_device_name() const { return device_name_; }

  /// Get the port number (default: 1)
  std::uint8_t get_port_num() const { return port_num_; }

private:
  void open_device();
  void create_protection_domain();

  std::string device_name_;
  struct ibv_context *context_{nullptr};
  struct ibv_pd *pd_{nullptr};
  std::uint8_t port_num_{1};
};

} // namespace cudaq::nvqlink
