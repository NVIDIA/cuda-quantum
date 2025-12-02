/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "loopback_channel.h"

#include <cstring>
#include <stdexcept>

using namespace cudaq::nvqlink;
using namespace cudaq::nvqlink::test;

LoopbackChannel::LoopbackChannel(size_t buffer_size, size_t num_buffers)
    : buffer_size_(buffer_size), num_buffers_(num_buffers) {}

LoopbackChannel::~LoopbackChannel() {
  if (initialized_)
    cleanup();
}

void LoopbackChannel::initialize() {
  if (initialized_)
    return;

  // Allocate memory pool
  size_t total_size = buffer_size_ * num_buffers_;
  memory_pool_.resize(total_size);

  // Create buffer objects
  uint8_t *ptr = memory_pool_.data();
  for (size_t i = 0; i < num_buffers_; ++i) {
    auto buffer = std::make_unique<Buffer>(ptr, buffer_size_, 256, 64);
    free_buffers_.push(buffer.get());
    buffers_.push_back(std::move(buffer));
    ptr += buffer_size_;
  }

  initialized_ = true;
}

void LoopbackChannel::cleanup() {
  if (!initialized_)
    return;

  std::lock_guard<std::mutex> lock(buffer_mutex_);

  // Clear all queues
  {
    std::lock_guard<std::mutex> rx_lock(rx_mutex_);
    std::queue<std::vector<uint8_t>>().swap(rx_queue_);
  }
  {
    std::lock_guard<std::mutex> tx_lock(tx_mutex_);
    std::queue<std::vector<uint8_t>>().swap(tx_queue_);
  }

  // Return all buffers to free queue
  std::queue<Buffer *>().swap(free_buffers_);
  for (auto &buffer : buffers_)
    free_buffers_.push(buffer.get());

  initialized_ = false;
}

Buffer *LoopbackChannel::acquire_buffer() {
  if (!initialized_)
    throw std::runtime_error("LoopbackChannel not initialized");

  std::lock_guard<std::mutex> lock(buffer_mutex_);
  if (free_buffers_.empty())
    return nullptr; // Pool exhausted

  Buffer *buffer = free_buffers_.front();
  free_buffers_.pop();
  buffer->set_data_length(0); // Reset data length
  return buffer;
}

void LoopbackChannel::release_buffer(Buffer *buffer) {
  if (!buffer)
    return;

  std::lock_guard<std::mutex> lock(buffer_mutex_);
  buffer->set_data_length(0); // Reset data length
  free_buffers_.push(buffer);
}

uint32_t LoopbackChannel::receive_burst(Buffer **buffers, uint32_t max) {
  if (!initialized_)
    return 0;

  std::lock_guard<std::mutex> rx_lock(rx_mutex_);
  
  uint32_t received = 0;
  while (received < max && !rx_queue_.empty()) {
    auto &packet = rx_queue_.front();
    
    // Get a buffer
    Buffer *buffer = acquire_buffer();
    if (!buffer)
      break; // No buffers available

    // Copy packet data to buffer
    size_t capacity = buffer->get_total_size() - buffer->get_headroom() - buffer->get_tailroom();
    if (packet.size() > capacity) {
      release_buffer(buffer);
      rx_queue_.pop(); // Drop oversized packet
      continue;
    }

    std::memcpy(buffer->get_data(), packet.data(), packet.size());
    buffer->set_data_length(packet.size());
    
    buffers[received++] = buffer;
    rx_queue_.pop();
  }

  return received;
}

uint32_t LoopbackChannel::send_burst(Buffer **buffers, uint32_t count) {
  if (!initialized_)
    return 0;

  std::lock_guard<std::mutex> tx_lock(tx_mutex_);
  
  for (uint32_t i = 0; i < count; ++i) {
    Buffer *buffer = buffers[i];
    if (!buffer)
      continue;

    // Copy buffer data to TX queue
    uint8_t *data_start = static_cast<uint8_t*>(buffer->get_data());
    std::vector<uint8_t> packet(data_start, data_start + buffer->get_data_length());
    tx_queue_.push(std::move(packet));
  }

  return count;
}

void LoopbackChannel::inject_rx_packet(const void *data, size_t len) {
  std::lock_guard<std::mutex> lock(rx_mutex_);
  
  std::vector<uint8_t> packet(static_cast<const uint8_t *>(data),
                              static_cast<const uint8_t *>(data) + len);
  rx_queue_.push(std::move(packet));
}

std::vector<uint8_t> LoopbackChannel::pop_tx_packet() {
  std::lock_guard<std::mutex> lock(tx_mutex_);
  
  if (tx_queue_.empty())
    return {};

  std::vector<uint8_t> packet = std::move(tx_queue_.front());
  tx_queue_.pop();
  return packet;
}

bool LoopbackChannel::has_tx_packets() const {
  std::lock_guard<std::mutex> lock(tx_mutex_);
  return !tx_queue_.empty();
}

size_t LoopbackChannel::rx_queue_size() const {
  std::lock_guard<std::mutex> lock(rx_mutex_);
  return rx_queue_.size();
}

size_t LoopbackChannel::tx_queue_size() const {
  std::lock_guard<std::mutex> lock(tx_mutex_);
  return tx_queue_.size();
}

