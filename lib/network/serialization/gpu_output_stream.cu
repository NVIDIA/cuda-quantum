/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/gpu_channel.h"
#include "cudaq/nvqlink/network/serialization/gpu_output_stream.h"

namespace cudaq::nvqlink {

// Channel mode constructor
__device__ GPUOutputStream::GPUOutputStream(GPUChannel &channel)
    : channel_(&channel), current_buffer_(nullptr), buffer_start_(nullptr),
      write_ptr_(nullptr), write_end_(nullptr), bytes_written_(0),
      error_(false) {
  // Allocate first buffer
  allocate_buffer();
}

__device__ void GPUOutputStream::write_bytes(const void *src, size_t len) {
  ensure_space(len);

  if (error_) {
    return;
  }

  // Manual copy in device code
  const char *s = static_cast<const char *>(src);
  for (size_t i = 0; i < len; ++i) {
    write_ptr_[i] = s[i];
  }

  write_ptr_ += len;
  bytes_written_ += len;
}

__device__ void GPUOutputStream::flush() {
  if (!channel_) {
    return; // Buffer mode - no flushing
  }

  if (!current_buffer_ || bytes_written_ == 0) {
    return; // Nothing to flush
  }

  // Send packet via channel
  bool sent = channel_->send_packet(current_buffer_, bytes_written_);

  if (!sent) {
    error_ = true;
    return;
  }

  // Reset state (buffer is now owned by channel)
  current_buffer_ = nullptr;
  write_ptr_ = nullptr;
  write_end_ = nullptr;
  buffer_start_ = nullptr;
  bytes_written_ = 0;

  // Allocate new buffer for next write
  allocate_buffer();
}

__device__ void GPUOutputStream::allocate_buffer() {
  if (!channel_) {
    return; // Buffer mode - no allocation
  }

  // Request buffer from channel
  current_buffer_ = channel_->allocate_buffer(2048); // Default size

  if (!current_buffer_) {
    error_ = true;
    return;
  }

  buffer_start_ = static_cast<char *>(current_buffer_);
  write_ptr_ = buffer_start_;
  write_end_ = write_ptr_ + 2048; // TODO: Get actual capacity from channel
  bytes_written_ = 0;
}

__device__ void GPUOutputStream::ensure_space(size_t needed) {
  // Check if we have enough space in current buffer
  if (write_ptr_ && write_ptr_ + needed <= write_end_) {
    return; // Have enough space
  }

  // In Channel mode, flush and get new buffer
  if (channel_ && bytes_written_ > 0) {
    flush(); // This allocates a new buffer
  }

  // Check again after flushing
  if (!write_ptr_ || write_ptr_ + needed > write_end_) {
    error_ = true;
  }
}

} // namespace cudaq::nvqlink
