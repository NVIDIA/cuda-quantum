/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/network/memory/buffer.h"

#include <cuda.h>
#include <doca_dev.h>
#include <doca_gpunetio.h>
#include <infiniband/verbs.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <queue>
#include <vector>

namespace cudaq::nvqlink {

/// @brief GPU-registered buffer pool for DOCA Channel
///
/// Key features
/// - Allocates GPU memory via doca_gpu_mem_alloc()
/// - dmabuf registration: doca_gpu_dmabuf_fd() + ibv_reg_dmabuf_mr()
/// - Fallback: ibv_reg_mr_iova() if dmabuf registration fails
///
class DOCABufferPool {
public:
  DOCABufferPool(doca_gpu *gpu_device, doca_dev *nic_device,
                 ibv_pd *protection_domain, std::size_t total_size,
                 std::size_t page_size, unsigned num_pages);

  ~DOCABufferPool();

  DOCABufferPool(const DOCABufferPool &) = delete;
  DOCABufferPool &operator=(const DOCABufferPool &) = delete;

  /// @brief Initialize the buffer pool
  bool initialize();

  std::uint8_t *get_gpu_buffer() const { return gpu_buffer_; }
  std::uint32_t get_rkey() const { return mr_ ? mr_->rkey : 0; }
  std::uint32_t get_lkey() const { return mr_ ? mr_->lkey : 0; }

  /// @brief Get memory key in big-endian format for DOCA WQEs
  std::uint32_t get_mkey_be() const;

  std::uint64_t get_external_address() const { return external_addr_; }

  // For Channel interface compatibility
  Buffer *acquire();
  void release(Buffer *buffer);

  std::size_t get_page_size() const { return page_size_; }
  unsigned get_num_pages() const { return num_pages_; }

private:
  doca_gpu *gpu_device_;
  doca_dev *nic_device_;
  ibv_pd *protection_domain_;

  std::size_t total_size_;
  std::size_t page_size_;
  unsigned num_pages_;

  std::uint8_t *gpu_buffer_ = nullptr;
  ibv_mr *mr_ = nullptr;
  std::uint64_t external_addr_ = 0;
  int dmabuf_fd_ = -1;

  std::vector<std::unique_ptr<Buffer>> buffers_;
  std::queue<Buffer *> free_list_;
  std::mutex mutex_;
};

} // namespace cudaq::nvqlink
