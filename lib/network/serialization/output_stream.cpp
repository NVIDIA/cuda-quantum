/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/serialization/output_stream.h"
#include "cudaq/nvqlink/network/channel.h"

#include <cstring>

namespace cudaq::nvqlink {

// Backend mode constructor (persistent)
OutputStream::OutputStream(Channel &channel) : channel_(&channel) {
  // Lazy buffer acquisition - only acquire when writing
  // acquire_buffer() will be called on first write
}

// Buffer mode constructor (temporary, for Daemon RPC)
OutputStream::OutputStream(void *data, std::size_t capacity)
    : buffer_start_(static_cast<char *>(data)), write_ptr_(buffer_start_),
      write_end_(write_ptr_ + capacity) {}

OutputStream::~OutputStream() {
  // Flush any pending data in Backend mode
  if (channel_) {
    try {
      if (current_buffer_ && bytes_written_ > 0)
        flush();
    } catch (...) {
      // Ignore exceptions in destructor
    }

    // Release buffer if still held (we still own it)
    if (current_buffer_)
      channel_->release_buffer(current_buffer_);
  }
}

void OutputStream::write_bytes(const void *src, std::size_t len) {
  ensure_space(len);

  std::memcpy(write_ptr_, src, len);
  write_ptr_ += len;
  bytes_written_ += len;
}

void OutputStream::write_string(const std::string &str) {
  // Write length prefix
  std::uint32_t len = static_cast<std::uint32_t>(str.size());
  write(len);

  // Write string data
  if (len > 0) {
    ensure_space(len);
    std::memcpy(write_ptr_, str.data(), len);
    write_ptr_ += len;
    bytes_written_ += len;
  }
}

void OutputStream::flush() {
  if (!channel_)
    return; // Buffer mode - no flushing

  if (!current_buffer_ || bytes_written_ == 0)
    return; // Nothing to flush

  // Set buffer length
  current_buffer_->set_data_length(bytes_written_);

  // Send packet - send_burst takes ownership of buffer (always)
  channel_->send_burst(&current_buffer_, 1);

  // Reset state (buffer ownership transferred to send_burst)
  current_buffer_ = nullptr;
  write_ptr_ = nullptr;
  write_end_ = nullptr;
  buffer_start_ = nullptr;
  bytes_written_ = 0;

  // Acquire new buffer for next write
  acquire_buffer();
}

std::size_t OutputStream::remaining_capacity() const {
  if (!write_ptr_ || !write_end_)
    return 0;
  return static_cast<std::size_t>(write_end_ - write_ptr_);
}

bool OutputStream::is_full() const { return remaining_capacity() == 0; }

void OutputStream::acquire_buffer() {
  if (!channel_)
    return; // Buffer mode - no acquisition

  current_buffer_ = channel_->acquire_buffer();

  if (!current_buffer_)
    throw std::runtime_error("OutputStream: failed to acquire buffer");

  buffer_start_ = static_cast<char *>(current_buffer_->get_data());
  write_ptr_ = buffer_start_;
  write_end_ = write_ptr_ + current_buffer_->get_total_size();
  bytes_written_ = 0;
}

void OutputStream::ensure_space(std::size_t needed) {
  // Check if we have enough space in current buffer
  if (write_ptr_ && write_ptr_ + needed <= write_end_)
    return; // Have enough space

  // In Channel mode, acquire buffer if we don't have one yet
  if (channel_ && !current_buffer_) {
    acquire_buffer();
    if (write_ptr_ && write_ptr_ + needed <= write_end_)
      return; // Now have enough space
  }

  // In Channel mode, flush and get new buffer if current is full
  if (channel_ && bytes_written_ > 0)
    flush(); // This acquires a new buffer

  // Check again after flushing
  if (!write_ptr_ || write_ptr_ + needed > write_end_) {
    throw std::runtime_error("OutputStream: insufficient space (need " +
                             std::to_string(needed) + " bytes, have " +
                             std::to_string(remaining_capacity()) + ")");
  }
}

} // namespace cudaq::nvqlink
