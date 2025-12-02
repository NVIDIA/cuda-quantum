/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/memory/buffer.h"

#include <infiniband/verbs.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace cudaq::nvqlink {

/// Buffer pool for RoCE/RDMA buffer management.
///
/// Manages Buffer wrappers for RDMA buffers using memory polling architecture.
///
/// Used exclusively with UC (Unreliable Connection) mode with RDMA WRITE
/// and memory polling .
///
class RoCEBufferPool {
public:
  explicit RoCEBufferPool(std::size_t pool_size);
  ~RoCEBufferPool();

  /// Wrap data from ring buffer (memory polling).
  ///
  /// @param addr Base address of RPC payload (no Ethernet header)
  /// @param length Length of RPC data
  /// @param slot_id Slot ID for tracking
  /// @return Buffer wrapper pointing to the data
  Buffer *wrap_ring_buffer_data(void *addr, std::uint32_t length,
                                std::uint64_t slot_id);

  /// Return buffer to pool after processing.
  void return_buffer(Buffer *buffer);

  /// Get Work Request ID for a buffer (slot ID for ring buffer tracking).
  std::uint64_t get_wr_id(Buffer *buffer) const;

  /// Set Work Request ID for a buffer (for TX slot tracking).
  void set_wr_id(Buffer *buffer, std::uint64_t wr_id);

  /// Get a free buffer wrapper for TX allocation (no memory allocation!).
  Buffer *get_free_buffer();

private:
  std::vector<std::unique_ptr<Buffer>> buffer_pool_;
  std::vector<Buffer *> free_buffers_;

  // Map buffer to Work Request ID (slot ID for ring buffer)
  std::unordered_map<Buffer *, std::uint64_t> wr_id_map_;
};

} // namespace cudaq::nvqlink
