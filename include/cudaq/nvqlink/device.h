/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <cstddef>
#include <limits>

namespace cudaq::nvqlink {

/// @brief Memory pointer abstraction (kept for reference, not used by new architecture)
/// @details Represents a pointer to memory allocated on a quantum processing
/// unit (controller or classical device). Encapsulates device memory management
/// details including location and size.
///
/// NOTE: This struct is kept for backward compatibility and reference,
/// but is not actively used by the new Daemon-based architecture.
/// New code should use native C++ types with automatic serialization.
struct device_ptr {

  /// @brief Opaque handle to device memory block
  std::size_t handle = std::numeric_limits<std::size_t>::max();

  /// @brief Size of allocated memory in bytes
  std::size_t size = 0;

  /// @brief Physical device identifier
  std::size_t device_id = std::numeric_limits<std::size_t>::max();

  /// @brief Pointer to host memory when referencing local memory
  void *host_shadow = nullptr;

  /// @brief Default constructor
  device_ptr() = default;

  /// @brief Constructor with handle and size
  device_ptr(std::size_t h, std::size_t s) : handle(h), size(s) {}

  /// @brief Constructor with handle, size, and device ID
  device_ptr(std::size_t h, std::size_t s, std::size_t dev)
      : handle(h), size(s), device_id(dev) {}

  /// @brief Check if this device_ptr is valid
  bool is_valid() const {
    return handle != std::numeric_limits<std::size_t>::max();
  }

  /// @brief Check if this represents local/host memory
  bool is_local() const {
    return device_id == std::numeric_limits<std::size_t>::max();
  }
};

} // namespace cudaq::nvqlink
