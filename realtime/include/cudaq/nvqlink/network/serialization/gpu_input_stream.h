/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace cudaq::nvqlink {

// Forward declaration
class GPUChannel;

/// GPU-side input stream for deserializing data in CUDA kernels.
///
/// Works in two modes:
/// 1. Bound to raw device memory (buffer mode)
/// 2. Bound to GPU channel (channel mode - persistent)
///
/// @example Buffer mode:
/// ```cuda
/// __global__ void process_packets(
///     const char* packet_data,
///     size_t packet_size,
///     int* results) {
///
///   GPUInputStream in(packet_data, packet_size);
///
///   int a = in.read<int>();
///   int b = in.read<int>();
///
///   results[threadIdx.x] = a + b;
/// }
/// ```
///
/// @example Channel mode:
/// ```cuda
/// __global__ void persistent_kernel(GPUChannel* channel) {
///   GPUInputStream in(*channel);
///
///   while (channel->is_running()) {
///     if (in.available()) {
///       int a = in.read<int>();
///       process(a);
///     }
///   }
/// }
/// ```
///
/// @note All methods are __device__ only - this class is for GPU use only.
/// @note This is a lightweight, header-only implementation for zero overhead.
///
class GPUInputStream {
public:
  /// Construct from GPUChannel (persistent mode).
  ///
  /// Stream will automatically fetch packets from the channel.
  ///
  /// @param channel GPU channel to receive packets from
  __device__ GPUInputStream(GPUChannel &channel);

  /// Construct from raw device memory (buffer mode).
  ///
  /// @param data Pointer to device memory
  /// @param size Size of buffer in bytes
  __device__ GPUInputStream(const void *data, std::size_t size)
      : channel_(nullptr), current_packet_(nullptr),
        read_ptr_(static_cast<const char *>(data)), read_end_(read_ptr_ + size),
        bytes_read_(0) {}

  /// Read a value of type T from the stream.
  ///
  /// @tparam T Type to read (must be trivially copyable)
  /// @return The read value
  template <typename T>
  __device__ T read() {
    // Check bounds
    if (read_ptr_ + sizeof(T) > read_end_) {
      // In device code, we can't throw exceptions
      // Return zero-initialized value and set error flag
      error_ = true;
      return T{};
    }

    T value;
    // Use direct memory access (no memcpy in device code)
    const T *ptr = reinterpret_cast<const T *>(read_ptr_);
    value = *ptr;

    read_ptr_ += sizeof(T);
    bytes_read_ += sizeof(T);
    return value;
  }

  /// Read raw bytes from the stream.
  ///
  /// @param dest Destination buffer (device memory)
  /// @param len Number of bytes to read
  __device__ void read_bytes(void *dest, std::size_t len) {
    if (read_ptr_ + len > read_end_) {
      error_ = true;
      return;
    }

    // Manual copy in device code
    char *dst = static_cast<char *>(dest);
    for (std::size_t i = 0; i < len; ++i) {
      dst[i] = read_ptr_[i];
    }

    read_ptr_ += len;
    bytes_read_ += len;
  }

  /// Peek at the next value without consuming it.
  ///
  /// @tparam T Type to peek at
  /// @return The peeked value
  template <typename T>
  __device__ T peek() const {
    if (read_ptr_ + sizeof(T) > read_end_) {
      return T{};
    }

    const T *ptr = reinterpret_cast<const T *>(read_ptr_);
    return *ptr;
  }

  /// Skip bytes in the stream.
  ///
  /// @param len Number of bytes to skip
  __device__ void skip(std::size_t len) {
    if (read_ptr_ + len > read_end_) {
      error_ = true;
      return;
    }
    read_ptr_ += len;
    bytes_read_ += len;
  }

  /// Get number of bytes remaining.
  ///
  /// @return Bytes remaining
  __device__ std::size_t remaining() const {
    if (read_ptr_ >= read_end_) {
      return 0;
    }
    return static_cast<std::size_t>(read_end_ - read_ptr_);
  }

  /// Check if at end of stream.
  ///
  /// @return true if at end
  __device__ bool at_end() const { return read_ptr_ >= read_end_; }

  /// Get total bytes read.
  ///
  /// @return Total bytes read
  __device__ std::size_t bytes_read() const { return bytes_read_; }

  /// Check if an error occurred (bounds check failure).
  ///
  /// @return true if error
  __device__ bool has_error() const { return error_; }

  /// Reset error flag.
  __device__ void clear_error() { error_ = false; }

  /// Check if data is available (non-blocking).
  ///
  /// In channel mode, tries to fetch next packet.
  /// In buffer mode, checks if more data exists.
  ///
  /// @return true if data can be read
  __device__ bool available();

  /// Get pointer to current read position.
  ///
  /// @return Pointer to current position (device memory)
  __device__ const void *current_position() const { return read_ptr_; }

private:
  /// Fetch next packet from channel (channel mode only)
  __device__ void fetch_next_packet();

  /// Ensure sufficient data is available
  __device__ void ensure_data(std::size_t needed);

  // Channel mode (persistent)
  GPUChannel *channel_;
  void *current_packet_;

  // Both modes
  const char *read_ptr_;
  const char *read_end_;
  std::size_t bytes_read_;
  bool error_{false};
};

} // namespace cudaq::nvqlink
