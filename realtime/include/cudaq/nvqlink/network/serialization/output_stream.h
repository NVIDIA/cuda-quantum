/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/memory/buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace cudaq::nvqlink {

// Forward declaration
class Channel;

/// Unified output stream for serializing data.
///
/// Works in two modes:
/// 1. Bound to Channel (persistent, manages buffer lifecycle)
/// 2. Bound to raw buffer (temporary, for Daemon RPC functions)
///
/// @example Channel mode (persistent):
/// ```cpp
/// Channel channel(std::move(backend));
/// OutputStream out(channel);
///
/// out.write<int>(42);
/// out.flush();  // Send packet
/// ```
///
/// @example Daemon mode (temporary):
/// ```cpp
/// int rpc_function(InputStream& in, OutputStream& out) {
///   out.write<int>(100);  // Works the same!
///   return 0;
/// }
/// ```
///
class OutputStream {
public:
  /// Construct from Backend (persistent mode).
  ///
  /// Stream manages buffer lifecycle, allocating new buffers as needed.
  ///
  /// @param backend Backend to send packets to
  explicit OutputStream(Channel &channel);

  /// Construct from raw buffer (temporary mode).
  ///
  /// Stream wraps existing buffer, does not manage lifecycle.
  /// Used by Daemon dispatcher for RPC functions.
  ///
  /// @param data Pointer to writable buffer
  /// @param capacity Buffer capacity in bytes
  OutputStream(void *data, size_t capacity);

  ~OutputStream();

  // Disable copy/move
  OutputStream(const OutputStream &) = delete;
  OutputStream &operator=(const OutputStream &) = delete;
  OutputStream(OutputStream &&) = delete;
  OutputStream &operator=(OutputStream &&) = delete;

  /// Write a value of type T to the stream.
  ///
  /// @tparam T Type to write (must be trivially copyable)
  /// @param value Value to write
  template <typename T>
  void write(const T &value) {
    static_assert(std::is_trivially_copyable<T>::value,
                  "Type must be trivially copyable");

    ensure_space(sizeof(T));

    std::memcpy(write_ptr_, &value, sizeof(T));
    write_ptr_ += sizeof(T);
    bytes_written_ += sizeof(T);
  }

  /// Write raw bytes to the stream.
  ///
  /// @param src Source buffer
  /// @param len Number of bytes to write
  void write_bytes(const void *src, size_t len);

  /// Write a length-prefixed string.
  ///
  /// @param str String to write
  void write_string(const std::string &str);

  /// Flush the current buffer (send packet).
  ///
  /// Only meaningful in Channel mode. In buffer mode, this is a no-op.
  void flush();

  /// Get bytes written to current buffer.
  ///
  /// @return Bytes written
  uint32_t bytes_written() const { return bytes_written_; }

  /// Get remaining capacity in current buffer.
  ///
  /// @return Remaining bytes available
  size_t remaining_capacity() const;

  /// Check if buffer is full.
  ///
  /// @return true if no space remaining
  bool is_full() const;

  /// Get pointer to start of written data (advanced usage).
  ///
  /// @return Pointer to buffer start
  const void *data() const { return buffer_start_; }

  /// Get pointer to current write position (advanced usage).
  ///
  /// @return Pointer to current position
  void *current_position() const { return write_ptr_; }

private:
  void acquire_buffer();
  void ensure_space(size_t needed);

  // Backend mode (persistent)
  Channel *channel_{nullptr};
  Buffer *current_buffer_{nullptr};

  // Both modes (buffer_start_ must be before write_ptr_ and write_end_ due to
  // initialization order)
  char *buffer_start_{nullptr};
  char *write_ptr_{nullptr};
  const char *write_end_{nullptr};
  uint32_t bytes_written_{0};
};

} // namespace cudaq::nvqlink
