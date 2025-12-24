/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace cudaq::nvqlink {

// Forward declaration
class GPUChannel;

/// GPU-side output stream for serializing data in CUDA kernels.
///
/// Works in two modes:
/// 1. Bound to raw device memory (buffer mode)
/// 2. Bound to GPU channel (channel mode - persistent)
///
/// @example Buffer mode:
/// ```cuda
/// __global__ void generate_response(
///     char* output_buffer,
///     size_t buffer_capacity,
///     int result) {
///
///   GPUOutputStream out(output_buffer, buffer_capacity);
///
///   out.write<int>(result);
///   out.write<int>(42);
///
///   // bytes_written() can be used to set packet length
/// }
/// ```
///
/// @example Channel mode:
/// ```cuda
/// __global__ void persistent_kernel(GPUChannel* channel) {
///   GPUOutputStream out(*channel);
///
///   out.write<int>(42);
///   out.flush();  // Send packet
/// }
/// ```
///
/// @note All methods are __device__ only - this class is for GPU use only.
/// @note This is a lightweight, header-only implementation for zero overhead.
///
class GPUOutputStream {
public:
  /// Construct from GPUChannel (persistent mode).
  ///
  /// Stream will automatically allocate buffers and send packets.
  ///
  /// @param channel GPU channel to send packets to
  __device__ GPUOutputStream(GPUChannel &channel);

  /// Construct from raw device memory (buffer mode).
  ///
  /// @param data Pointer to writable device memory
  /// @param capacity Buffer capacity in bytes
  __device__ GPUOutputStream(void *data, size_t capacity)
      : channel_(nullptr), current_buffer_(nullptr),
        buffer_start_(static_cast<char *>(data)), write_ptr_(buffer_start_),
        write_end_(write_ptr_ + capacity), bytes_written_(0) {}

  /// Write a value of type T to the stream.
  ///
  /// @tparam T Type to write (must be trivially copyable)
  /// @param value Value to write
  template <typename T>
  __device__ void write(const T &value) {
    // Check bounds
    if (write_ptr_ + sizeof(T) > write_end_) {
      error_ = true;
      return;
    }

    // Use direct memory access
    T *ptr = reinterpret_cast<T *>(write_ptr_);
    *ptr = value;

    write_ptr_ += sizeof(T);
    bytes_written_ += sizeof(T);
  }

  /// Write raw bytes to the stream.
  ///
  /// @param src Source buffer (device memory)
  /// @param len Number of bytes to write
  __device__ void write_bytes(const void *src, std::size_t len);

  /// Flush the current buffer (send packet).
  ///
  /// Only meaningful in Channel mode. In buffer mode, this is a no-op.
  __device__ void flush();

  /// Get bytes written to buffer.
  ///
  /// @return Bytes written
  __device__ std::uint32_t bytes_written() const { return bytes_written_; }

  /// Get remaining capacity in buffer.
  ///
  /// @return Remaining bytes available
  __device__ std::size_t remaining_capacity() const {
    if (write_ptr_ >= write_end_) {
      return 0;
    }
    return static_cast<std::size_t>(write_end_ - write_ptr_);
  }

  /// Check if buffer is full.
  ///
  /// @return true if no space remaining
  __device__ bool is_full() const { return write_ptr_ >= write_end_; }

  /// Check if an error occurred (bounds check failure).
  ///
  /// @return true if error
  __device__ bool has_error() const { return error_; }

  /// Reset error flag.
  __device__ void clear_error() { error_ = false; }

  /// Get pointer to start of written data.
  ///
  /// @return Pointer to buffer start (device memory)
  __device__ const void *data() const { return buffer_start_; }

  /// Get pointer to current write position.
  ///
  /// @return Pointer to current position (device memory)
  __device__ void *current_position() const { return write_ptr_; }

  /// Reset stream to write from beginning.
  ///
  /// Useful for reusing the same buffer.
  __device__ void reset() {
    write_ptr_ = buffer_start_;
    bytes_written_ = 0;
    error_ = false;
  }

private:
  /// Allocate a new buffer (channel mode only)
  __device__ void allocate_buffer();

  /// Ensure sufficient space is available
  __device__ void ensure_space(std::size_t needed);

  // Channel mode (persistent)
  GPUChannel *channel_;
  void *current_buffer_;

  // Both modes
  char *buffer_start_;
  char *write_ptr_;
  const char *write_end_;
  std::uint32_t bytes_written_;
  bool error_{false};
};

} // namespace cudaq::nvqlink
