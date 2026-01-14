/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/channels/doca/doca_buffer_pool.h"
#include "cudaq/nvqlink/utils/instrumentation/domains.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"

#include <cerrno>
#include <cstring>
#include <unistd.h>

namespace cudaq::nvqlink {

static std::size_t get_page_size() {
  long ret = sysconf(_SC_PAGESIZE);
  if (ret == -1)
    return 4096;
  return static_cast<std::size_t>(ret);
}

static std::uint32_t doca_gpu_dev_verbs_bswap32(std::uint32_t x) {
  return __builtin_bswap32(x);
}

DOCABufferPool::DOCABufferPool(doca_gpu *gpu_device, doca_dev *nic_device,
                               ibv_pd *protection_domain,
                               std::size_t total_size, std::size_t page_size,
                               unsigned num_pages)
    : gpu_device_(gpu_device), nic_device_(nic_device),
      protection_domain_(protection_domain), total_size_(total_size),
      page_size_(page_size), num_pages_(num_pages) {}

DOCABufferPool::~DOCABufferPool() {
  if (mr_)
    ibv_dereg_mr(mr_);
  if (dmabuf_fd_ >= 0)
    close(dmabuf_fd_);
  if (gpu_buffer_)
    doca_gpu_mem_free(gpu_device_, gpu_buffer_);
}

bool DOCABufferPool::initialize() {
  doca_error_t result;

  // Allocate GPU memory
  result = doca_gpu_mem_alloc(gpu_device_, total_size_, get_page_size(),
                              DOCA_GPU_MEM_TYPE_GPU,
                              reinterpret_cast<void **>(&gpu_buffer_), nullptr);
  if (result != DOCA_SUCCESS) {
    NVQLINK_LOG_ERROR(DOMAIN_CHANNEL, "Failed to allocate GPU buffer: {}",
                      doca_error_get_descr(result));
    return false;
  }

  // CRITICAL: The external address must be 0 because we register the memory
  // with ibv_reg_mr_iova(..., iova=0, ...) or ibv_reg_dmabuf_mr(..., iova=0,
  // ...). The rkey is only valid for addresses starting at 0, not the GPU
  // virtual address. Clients write to offsets from 0, and the NIC translates
  // 0+offset to gpu_buffer_+offset.
  external_addr_ = 0;

  // Try dmabuf registration first
  result = doca_gpu_dmabuf_fd(gpu_device_, static_cast<void *>(gpu_buffer_),
                              total_size_, &dmabuf_fd_);
  if (result == DOCA_SUCCESS) {
    // To use ibv_reg_dmabuf_mr() we must ensure application provides
    // address and size aligned with the system page size
    mr_ = ibv_reg_dmabuf_mr(protection_domain_, 0, total_size_,
                            0, // Use offset 0
                            dmabuf_fd_,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ |
                                IBV_ACCESS_RELAXED_ORDERING);

    if (mr_) {
      NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Buffer registered via dmabuf, rkey={}",
                       mr_->rkey);
    }
  }

  // Fallback to iova registration
  if (mr_ == nullptr) {
    mr_ = ibv_reg_mr_iova(protection_domain_, static_cast<void *>(gpu_buffer_),
                          total_size_, 0,
                          IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
    if (mr_ == nullptr) {
      NVQLINK_LOG_ERROR(
          DOMAIN_CHANNEL,
          "Cannot register memory region ptr={} size={}, errno={}",
          reinterpret_cast<std::uint64_t>(gpu_buffer_), total_size_, errno);
      return false;
    }

    NVQLINK_LOG_INFO(DOMAIN_CHANNEL, "Buffer registered via iova, rkey={}",
                     mr_->rkey);
  }

  // Initialize buffer free list
  // Buffer constructor: (base_addr, total_size, headroom, tailroom)
  // For DOCA GPU buffers, we use the full page with no headroom/tailroom
  buffers_.reserve(num_pages_);
  for (unsigned i = 0; i < num_pages_; ++i) {
    auto buffer =
        std::make_unique<Buffer>(gpu_buffer_ + i * page_size_, page_size_,
                                 0,  // headroom
                                 0); // tailroom
    free_list_.push(buffer.get());
    buffers_.push_back(std::move(buffer));
  }

  return true;
}

std::uint32_t DOCABufferPool::get_mkey_be() const {
  return mr_ ? doca_gpu_dev_verbs_bswap32(mr_->lkey) : 0;
}

Buffer *DOCABufferPool::acquire() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (free_list_.empty())
    return nullptr;

  Buffer *buffer = free_list_.front();
  free_list_.pop();
  return buffer;
}

void DOCABufferPool::release(Buffer *buffer) {
  std::lock_guard<std::mutex> lock(mutex_);
  free_list_.push(buffer);
}

} // namespace cudaq::nvqlink
