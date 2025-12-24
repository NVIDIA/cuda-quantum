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

namespace cudaq::nvqlink {

/// Zero-copy packet buffer with headroom/tailroom for header manipulation.
/// Layout: [Headroom | Data | Tailroom]
///
class Buffer {
public:
  Buffer(void *base_addr, std::size_t total_size, std::size_t headroom,
         std::size_t tailroom);

  //===--------------------------------------------------------------------===//
  // Data access
  //===--------------------------------------------------------------------===//

  void *get_data() const { return data_ptr_; }

  std::size_t get_data_length() const { return data_len_; }

  void set_data_length(std::size_t len) { data_len_ = len; }

  /// Reset buffer to point to new memory (used by buffer pools for zero-copy).
  ///
  /// @param base Base address of buffer
  /// @param data Data pointer (may be offset from base for headroom)
  /// @param total Total buffer size
  /// @param datalen Current data length
  /// @param headroom_size Available headroom
  /// @param tailroom_size Available tailroom
  void reset(void *base, void *data, std::size_t total, std::size_t datalen,
             std::size_t headroom_size, std::size_t tailroom_size) {
    base_addr_ = base;
    data_ptr_ = data;
    total_size_ = total;
    data_len_ = datalen;
    headroom_ = headroom_size;
    tailroom_ = tailroom_size;
  }

  // Headroom manipulation (for prepending headers)
  void *prepend(std::size_t bytes);

  // Raw buffer info
  void *get_base_address() const { return base_addr_; }
  std::size_t get_total_size() const { return total_size_; }
  std::size_t get_headroom() const { return headroom_; }
  std::size_t get_tailroom() const { return tailroom_; }

  // Metadata for NIC
  std::uint64_t dma_addr{0}; // Physical/device address for NIC
  std::uint32_t queue_id{0};

private:
  void *base_addr_;        // Start of entire buffer
  void *data_ptr_;         // Start of actual packet data
  std::size_t total_size_; // Total buffer size
  std::size_t headroom_;
  std::size_t tailroom_;
  std::size_t data_len_; // Current data length
};

} // namespace cudaq::nvqlink
