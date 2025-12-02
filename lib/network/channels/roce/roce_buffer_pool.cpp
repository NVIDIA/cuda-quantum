/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/roce/roce_buffer_pool.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"

#include <cstring>
#include <stdexcept>

namespace cudaq::nvqlink {

RoCEBufferPool::RoCEBufferPool(std::size_t pool_size) {
  // Pre-allocate Buffer wrapper objects
  buffer_pool_.reserve(pool_size);
  free_buffers_.reserve(pool_size);

  for (std::size_t i = 0; i < pool_size; i++) {
    // Create Buffer with dummy parameters (will be reset when wrapping received
    // data)
    auto buffer = std::make_unique<Buffer>(nullptr, 0, 0, 0);
    free_buffers_.push_back(buffer.get());
    buffer_pool_.push_back(std::move(buffer));
  }

  NVQLINK_LOG_INFO(DOMAIN_CHANNEL,
                   "RoCEBufferPool: Created pool with {} buffer wrappers",
                   pool_size);
}

RoCEBufferPool::~RoCEBufferPool() { wr_id_map_.clear(); }

Buffer *RoCEBufferPool::wrap_ring_buffer_data(void *addr, std::uint32_t length,
                                              std::uint64_t slot_id) {
  if (free_buffers_.empty())
    throw std::runtime_error("RoCEBufferPool: No free buffer wrappers");

  // Get a free buffer wrapper
  Buffer *buf = free_buffers_.back();
  free_buffers_.pop_back();

  // Ring buffer slot layout:
  // [SlotHeader (16 bytes)] [Payload (RPC data)]
  //                         ^ addr points here
  //
  // The poller already skipped the SlotHeader, but we need to tell Buffer
  // that there IS headroom available (for prepending RPCResponse header)

  constexpr std::size_t SLOT_HEADER_SIZE = 16; // sizeof(SlotHeader)

  char *payload_start = static_cast<char *>(addr);
  char *slot_start = payload_start - SLOT_HEADER_SIZE; // Back up to SlotHeader

  // Reset buffer:
  // - base_ptr: slot start (includes SlotHeader)
  // - data_ptr: payload start (after SlotHeader)
  // - headroom: SLOT_HEADER_SIZE (so prepend() can use it)
  buf->reset(slot_start,                // base pointer (slot start)
             payload_start,             // data pointer (RPC payload)
             length + SLOT_HEADER_SIZE, // total size (includes SlotHeader)
             length,           // current data length (RPC payload only)
             SLOT_HEADER_SIZE, // headroom available
             0);               // no tailroom

  // Store slot ID (for tracking)
  wr_id_map_[buf] = slot_id;

  return buf;
}

void RoCEBufferPool::return_buffer(Buffer *buffer) {
  if (!buffer)
    return;

  // Clean up mappings
  wr_id_map_.erase(buffer);

  // Return to free pool
  free_buffers_.push_back(buffer);
}

std::uint64_t RoCEBufferPool::get_wr_id(Buffer *buffer) const {
  auto it = wr_id_map_.find(buffer);
  if (it == wr_id_map_.end())
    throw std::runtime_error("RoCEBufferPool: Buffer has no WR ID");
  return it->second;
}

void RoCEBufferPool::set_wr_id(Buffer *buffer, std::uint64_t wr_id) {
  wr_id_map_[buffer] = wr_id;
}

Buffer *RoCEBufferPool::get_free_buffer() {
  // NO allocation - just return a pre-allocated buffer wrapper
  if (free_buffers_.empty())
    return nullptr;

  Buffer *buf = free_buffers_.back();
  free_buffers_.pop_back();
  return buf;
}

} // namespace cudaq::nvqlink
