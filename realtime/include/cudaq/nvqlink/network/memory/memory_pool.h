/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/daemon/config.h"
#include "cudaq/nvqlink/network/config.h"
#include "cudaq/nvqlink/network/memory/buffer.h"

#include <mutex>
#include <vector>

namespace cudaq::nvqlink {

/// Zero-copy memory pool for packet buffers.
/// CPU mode: Uses hugepage-backed memory
/// GPU mode: Uses GPU device memory
///
class MemoryPool {
public:
  explicit MemoryPool(const MemoryConfig &config, DatapathMode mode);
  ~MemoryPool();

  // Buffer allocation
  Buffer *allocate();
  void deallocate(Buffer *buffer);

  // Statistics
  std::size_t get_total_buffers() const { return total_buffers_; }
  std::size_t get_available_buffers() const;

  // Memory base address getters
  void *get_cpu_base_address() const { return base_addr_; }
  void *get_gpu_base_address() const { return gpu_base_addr_; }

private:
  void allocate_cpu_memory();
  void allocate_gpu_memory();

  MemoryConfig config_;
  DatapathMode mode_;

  void *base_addr_{nullptr};
  void *gpu_base_addr_{nullptr}; // For GPU mode
  bool using_hugepages_{false};  // Track allocation type for proper cleanup

  std::size_t total_buffers_{0};
  std::vector<Buffer *> free_buffers_;
  mutable std::mutex mutex_;
};

} // namespace cudaq::nvqlink
