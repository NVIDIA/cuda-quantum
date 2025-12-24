/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace cudaq::nvqlink {

/// @brief Shared memory ring buffer layout for RDMA WRITE operations
///
/// This implements a lock-free, single-producer single-consumer ring buffer
/// optimized for RDMA WRITE operations.
///
/// Memory Layout:
/// ┌────────────────────────────────────┐
/// │ RingBufferHeader (metadata)        │
/// ├────────────────────────────────────┤
/// │ Slot 0: [seqn | len | payload]     │
/// │ Slot 1: [seqn | len | payload]     │
/// │ ...                                │
/// │ Slot N-1: [seqn | len | payload]   │
/// └────────────────────────────────────┘
///
/// Producer (RDMA WRITE client):
/// - Writes to slots in circular order
/// - Increments sequence number for each write
/// - Updates head pointer after write
///
/// Consumer (Server memory polling):
/// - Polls tail slot's sequence number
/// - When seq_num changes → new packet available
/// - Advances tail pointer after processing
///

/// Ring buffer control structure (at start of registered memory)
struct alignas(64) RingBufferHeader {
  std::atomic<std::uint64_t> head; ///< Next slot to write (producer)
  std::atomic<std::uint64_t> tail; ///< Next slot to read (consumer)
  std::uint32_t num_slots;         ///< Total number of slots
  std::uint32_t slot_size;         ///< Size of each slot in bytes
  std::uint64_t padding[6];        ///< Cache line padding
};

/// Per-slot header (at start of each slot)
struct alignas(16) SlotHeader {
  std::atomic<std::uint64_t> seqn; ///< Sequence number (0 = empty)
  std::uint32_t payload_len;       ///< Payload length in bytes
  std::uint32_t reserved;          ///< Reserved for future use
};

/// Configuration for ring buffer
struct RingBufferConfig {
  std::uint32_t num_slots = 1024; ///< Number of slots (must be power of 2)
  std::uint32_t slot_size = 2048; ///< Size per slot (including header)

  std::size_t total_size() const {
    return sizeof(RingBufferHeader) + (num_slots * slot_size);
  }

  std::size_t get_slot_offset(std::uint32_t index) const {
    return sizeof(RingBufferHeader) + (index * slot_size);
  }

  std::uint32_t get_max_payload_size() const {
    return slot_size - sizeof(SlotHeader);
  }
};

/// Helper class for managing ring buffer on server side
class RingBufferPoller {
public:
  RingBufferPoller(void *base_addr, const RingBufferConfig &config)
      : base_addr_(static_cast<char *>(base_addr)), config_(config),
        header_(reinterpret_cast<RingBufferHeader *>(base_addr)) {
    // Initialize header
    header_->head.store(0, std::memory_order_relaxed);
    header_->tail.store(0, std::memory_order_relaxed);
    header_->num_slots = config.num_slots;
    header_->slot_size = config.slot_size;

    // Clear all slot sequence numbers
    for (std::uint32_t i = 0; i < config.num_slots; i++) {
      SlotHeader *slot = get_slot_header(i);
      slot->seqn.store(0, std::memory_order_relaxed);
    }
  }

  /// @brief Poll for next packet (non-blocking)
  /// @param[out] data Pointer to packet data (if available)
  /// @param[out] len Length of packet data
  /// @return true if packet available, false otherwise
  ///
  bool poll_next(char **data, std::uint32_t *len) {
    std::uint64_t tail = header_->tail.load(std::memory_order_acquire);
    std::uint32_t slot_idx = tail % config_.num_slots;

    SlotHeader *slot = get_slot_header(slot_idx);
    std::uint64_t seq = slot->seqn.load(std::memory_order_acquire);

    // Check if new packet is available
    if (seq != expected_seqn_)
      return false; // No new packet

    // Packet available!
    *len = slot->payload_len;
    *data = base_addr_ + config_.get_slot_offset(slot_idx) + sizeof(SlotHeader);

    // Advance tail and expected sequence
    header_->tail.store(tail + 1, std::memory_order_release);
    expected_seqn_++;

    // Mark slot as consumed (optional, for debugging)
    slot->seqn.store(0, std::memory_order_release);

    return true;
  }

  /// @brief Poll for next burst of packets (non-blocking)
  /// @param[out] data Array of pointers to packet data (if available)
  /// @param[out] lens Array of lengths of packet data
  /// @param[in] max_count Maximum number of packets to poll
  /// @return Number of packets polled
  ///
  std::uint32_t poll_burst(char **data, std::uint32_t *lens,
                           std::uint32_t max_count) {
    std::uint64_t tail = header_->tail.load(std::memory_order_acquire);
    std::uint32_t count = 0;

    for (std::uint32_t i = 0; i < max_count; i++) {
      std::uint32_t slot_idx = (tail + i) % config_.num_slots;
      SlotHeader *slot = get_slot_header(slot_idx);

      // Check if packet is available
      std::uint64_t seqn = slot->seqn.load(std::memory_order_acquire);
      if (seqn != expected_seqn_ + i)
        break; // No more packets

      // Extract packet data
      lens[i] = slot->payload_len;
      data[i] =
          base_addr_ + config_.get_slot_offset(slot_idx) + sizeof(SlotHeader);

      // Mark slot as consumed (optional, for debugging)
      slot->seqn.store(0, std::memory_order_release);

      count++;
    }

    // Update tail pointer once for entire burst
    if (count > 0) {
      header_->tail.store(tail + count, std::memory_order_release);
      expected_seqn_ += count;
    }

    return count;
  }

  std::uint64_t get_packets_received() const { return expected_seqn_ - 1; }

private:
  SlotHeader *get_slot_header(std::uint32_t index) {
    char *slot_addr = base_addr_ + config_.get_slot_offset(index);
    return reinterpret_cast<SlotHeader *>(slot_addr);
  }

  char *base_addr_;
  RingBufferConfig config_;
  RingBufferHeader *header_;
  std::uint64_t expected_seqn_{1}; // Start from 1 (0 means empty)
};

} // namespace cudaq::nvqlink
