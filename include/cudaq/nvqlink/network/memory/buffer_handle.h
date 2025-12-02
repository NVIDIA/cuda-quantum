/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/memory/buffer.h"

namespace cudaq::nvqlink {

// Forward declaration
class Channel;

/// @brief RAII wrapper for Buffer ownership (optional convenience wrapper)
///
/// Enforces ownership semantics:
/// - Non-copyable (ownership is exclusive)
/// - Move-only (ownership transfer is explicit)
/// - Auto-releases on destruction (unless released to send)
///
/// Example:
/// ```cpp
/// BufferHandle buf(channel, channel->acquire_buffer());
/// write_to(buf.get());
/// channel->send(buf.release());  // Ownership transferred
/// ```
///
/// @note For hot path, use raw Buffer* directly. This is for convenience.
///
class BufferHandle {
public:
  BufferHandle() = default;

  explicit BufferHandle(Channel *channel, Buffer *buffer)
      : channel_(channel), buffer_(buffer) {}

  ~BufferHandle() {
    if (buffer_ && channel_) {
      channel_->release_buffer(buffer_);
    }
  }

  // Non-copyable (exclusive ownership)
  BufferHandle(const BufferHandle &) = delete;
  BufferHandle &operator=(const BufferHandle &) = delete;

  // Move-only (explicit ownership transfer)
  BufferHandle(BufferHandle &&other) noexcept
      : channel_(other.channel_), buffer_(other.buffer_) {
    other.buffer_ = nullptr;
    other.channel_ = nullptr;
  }

  BufferHandle &operator=(BufferHandle &&other) noexcept {
    if (this != &other) {
      // Release current buffer
      if (buffer_ && channel_) {
        channel_->release_buffer(buffer_);
      }
      // Transfer ownership
      channel_ = other.channel_;
      buffer_ = other.buffer_;
      other.buffer_ = nullptr;
      other.channel_ = nullptr;
    }
    return *this;
  }

  /// Get raw pointer (does NOT transfer ownership)
  Buffer *get() const { return buffer_; }

  /// Release ownership (caller takes ownership of raw pointer)
  Buffer *release() {
    Buffer *buf = buffer_;
    buffer_ = nullptr;
    return buf;
  }

  /// Check if valid
  explicit operator bool() const { return buffer_ != nullptr; }

private:
  Channel *channel_{nullptr};
  Buffer *buffer_{nullptr};
};

} // namespace cudaq::nvqlink
