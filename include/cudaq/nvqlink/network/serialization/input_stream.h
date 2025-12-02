/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/memory/buffer.h"

#include <cstddef>
#include <cstring>
#include <string>

namespace cudaq::nvqlink {

// Forward declaration
class Channel;

/// Unified input stream for deserializing data.
///
/// Works in two modes:
/// 1. Bound to Channel (persistent, manages packet lifecycle)
/// 2. Bound to raw buffer (temporary, for Daemon RPC functions)
///
/// @example Backend mode (persistent):
/// ```cpp
/// Backend* backend = create_backend(...);
/// InputStream in(channel);
///
/// while (running) {
///   if (in.available()) {
///     int a = in.read<int>();
///     process(a);
///   }
/// }
/// ```
///
/// @example Daemon mode (temporary):
/// ```cpp
/// int rpc_function(InputStream& in, OutputStream& out) {
///   int a = in.read<int>();  // Works the same!
///   return 0;
/// }
/// ```
///
class InputStream {
public:
  /// Construct from Backend (persistent mode).
  ///
  /// Stream manages packet lifecycle, fetching new packets as needed.
  ///
  /// @param backend Backend to receive packets from
  explicit InputStream(Channel &channel);

  /// Construct from raw buffer (temporary mode).
  ///
  /// Stream wraps existing buffer, does not manage lifecycle.
  /// Used by Daemon dispatcher for RPC functions.
  ///
  /// @param data Pointer to buffer data
  /// @param size Size of buffer in bytes
  InputStream(const void *data, std::size_t size);

  ~InputStream();

  // Disable copy/move
  InputStream(const InputStream &) = delete;
  InputStream &operator=(const InputStream &) = delete;
  InputStream(InputStream &&) = delete;
  InputStream &operator=(InputStream &&) = delete;

  /// Check if data is available (non-blocking).
  ///
  /// Only meaningful in Backend mode. In buffer mode, returns true if not at
  /// end.
  ///
  /// @return true if data can be read
  bool available();

  /// Read a value of type T from the stream.
  ///
  /// @tparam T Type to read (must be trivially copyable)
  /// @return The read value
  /// @throws std::runtime_error if insufficient data
  template <typename T>
  T read() {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable");

    ensure_data(sizeof(T));

    T value;
    std::memcpy(&value, read_ptr_, sizeof(T));
    read_ptr_ += sizeof(T);
    bytes_read_ += sizeof(T);
    return value;
  }

  /// Read raw bytes from the stream.
  ///
  /// @param dest Destination buffer
  /// @param len Number of bytes to read
  void read_bytes(void *dest, std::size_t len);

  /// Read a length-prefixed string.
  ///
  /// @return The read string
  std::string read_string();

  /// Peek at the next value without consuming it.
  ///
  /// @tparam T Type to peek at
  /// @return The peeked value
  template <typename T>
  T peek() {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable");

    ensure_data(sizeof(T));

    T value;
    std::memcpy(&value, read_ptr_, sizeof(T));
    return value;
  }

  /// Skip bytes in the stream.
  ///
  /// @param len Number of bytes to skip
  void skip(std::size_t len);

  /// Get number of bytes remaining in current packet/buffer.
  ///
  /// @return Bytes remaining
  std::size_t remaining() const;

  /// Check if current packet/buffer is fully consumed.
  ///
  /// @return true if at end
  bool at_end() const;

  /// Get total bytes read.
  ///
  /// @return Total bytes read
  std::size_t bytes_read() const { return bytes_read_; }

  /// Get pointer to current read position (advanced usage).
  ///
  /// @return Pointer to current position
  const void *current_position() const { return read_ptr_; }

private:
  void fetch_next_packet();
  void ensure_data(std::size_t needed);

  // Backend mode (persistent)
  Channel *channel_{nullptr};
  Buffer *current_packet_{nullptr};

  // Both modes
  const char *read_ptr_{nullptr};
  const char *read_end_{nullptr};
  std::size_t bytes_read_{0};
};

} // namespace cudaq::nvqlink
