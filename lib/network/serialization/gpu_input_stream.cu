/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/gpu_channel.h"
#include "cudaq/nvqlink/network/serialization/gpu_input_stream.h"

namespace cudaq::nvqlink {

// Channel mode constructor
__device__ GPUInputStream::GPUInputStream(GPUChannel &channel)
    : channel_(&channel), current_packet_(nullptr), read_ptr_(nullptr),
      read_end_(nullptr), bytes_read_(0), error_(false) {
  // Don't fetch packet yet - lazy initialization
}

__device__ bool GPUInputStream::available() {
  // Buffer mode: check if we have data
  if (!channel_) {
    return read_ptr_ < read_end_;
  }

  // Channel mode: check if we have data in current packet
  if (current_packet_ && read_ptr_ < read_end_) {
    return true;
  }

  // Try to get a new packet (non-blocking)
  fetch_next_packet();

  return current_packet_ != nullptr;
}

__device__ void GPUInputStream::fetch_next_packet() {
  if (!channel_) {
    return; // Buffer mode - no fetching
  }

  // Release old packet if any
  if (current_packet_) {
    channel_->release_buffer(current_packet_);
    current_packet_ = nullptr;
    read_ptr_ = nullptr;
    read_end_ = nullptr;
  }

  // Try to receive new packet (non-blocking)
  void *packet_data = nullptr;
  size_t packet_size = 0;

  if (channel_->receive_packet(&packet_data, &packet_size)) {
    current_packet_ = packet_data;
    read_ptr_ = static_cast<const char *>(packet_data);
    read_end_ = read_ptr_ + packet_size;
  }
}

__device__ void GPUInputStream::ensure_data(size_t needed) {
  // Check if we have enough data in current buffer/packet
  if (read_ptr_ && read_ptr_ + needed <= read_end_) {
    return; // Have enough data
  }

  // In Channel mode, try to fetch next packet if at end
  if (channel_ && (!current_packet_ || read_ptr_ >= read_end_)) {
    fetch_next_packet();
  }

  // Check again after fetching
  if (!read_ptr_ || read_ptr_ + needed > read_end_) {
    error_ = true; // Not enough data
  }
}

} // namespace cudaq::nvqlink
