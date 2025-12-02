/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/serialization/input_stream.h"
#include "cudaq/nvqlink/network/channel.h"

#include <cstring>

namespace cudaq::nvqlink {

// Backend mode constructor (persistent)
InputStream::InputStream(Channel &channel) : channel_(&channel) {
  // Don't fetch packet yet - lazy initialization
}

// Buffer mode constructor (temporary, for Daemon RPC)
InputStream::InputStream(const void *data, std::size_t size)
    : read_ptr_(static_cast<const char *>(data)), read_end_(read_ptr_ + size) {}

InputStream::~InputStream() {
  // Release current packet if in Backend mode
  if (channel_ && current_packet_)
    channel_->release_buffer(current_packet_);
}

bool InputStream::available() {
  // Buffer mode: check if we have data
  if (!channel_)
    return read_ptr_ < read_end_;

  // Backend mode: check if we have data in current packet
  if (current_packet_ && read_ptr_ < read_end_)
    return true;

  // Try to get a new packet (non-blocking)
  fetch_next_packet();

  return current_packet_ != nullptr;
}

void InputStream::read_bytes(void *dest, std::size_t len) {
  ensure_data(len);

  std::memcpy(dest, read_ptr_, len);
  read_ptr_ += len;
  bytes_read_ += len;
}

std::string InputStream::read_string() {
  // Read length prefix
  std::uint32_t len = read<std::uint32_t>();

  if (len == 0)
    return std::string();

  ensure_data(len);

  std::string result(read_ptr_, read_ptr_ + len);
  read_ptr_ += len;
  bytes_read_ += len;

  return result;
}

void InputStream::skip(std::size_t len) {
  ensure_data(len);
  read_ptr_ += len;
  bytes_read_ += len;
}

std::size_t InputStream::remaining() const {
  if (!read_ptr_ || read_ptr_ >= read_end_)
    return 0;
  return static_cast<std::size_t>(read_end_ - read_ptr_);
}

bool InputStream::at_end() const {
  return !read_ptr_ || read_ptr_ >= read_end_;
}

void InputStream::fetch_next_packet() {
  if (!channel_)
    return; // Buffer mode - no fetching

  // Release old packet if any
  if (current_packet_) {
    channel_->release_buffer(current_packet_);
    current_packet_ = nullptr;
    read_ptr_ = nullptr;
    read_end_ = nullptr;
  }

  // Try to receive new packet (non-blocking)
  Buffer *buffers[1];
  uint32_t received = channel_->receive_burst(buffers, 1);
  current_packet_ = (received > 0) ? buffers[0] : nullptr;

  if (current_packet_) {
    read_ptr_ = static_cast<const char *>(current_packet_->get_data());
    read_end_ = read_ptr_ + current_packet_->get_data_length();
  }
}

void InputStream::ensure_data(std::size_t needed) {
  // Check if we have enough data in current buffer/packet
  if (read_ptr_ && read_ptr_ + needed <= read_end_)
    return; // Have enough data

  // In Backend mode, try to fetch next packet if at end
  if (channel_ && (!current_packet_ || read_ptr_ >= read_end_))
    fetch_next_packet();

  // Check again after fetching
  if (!read_ptr_ || read_ptr_ + needed > read_end_) {
    throw std::runtime_error("InputStream: insufficient data (need " +
                             std::to_string(needed) + " bytes, have " +
                             std::to_string(remaining()) + ")");
  }
}

} // namespace cudaq::nvqlink
