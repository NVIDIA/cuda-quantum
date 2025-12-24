/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/network/memory/memory_pool.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"
#include "cudaq/nvqlink/utils/instrumentation/profiler.h"

#include <cassert>
#include <cerrno>
#include <cstring>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <stdexcept>
#include <sys/mman.h>
#include <unistd.h>

using namespace cudaq::nvqlink;

MemoryPool::MemoryPool(const MemoryConfig &config, DatapathMode mode)
    : config_(config), mode_(mode) {
  NVQLINK_TRACE_FULL(DOMAIN_MEMORY, "MemoryPool::constructor");

  total_buffers_ = config_.pool_size_bytes / config_.buffer_size_bytes;

  NVQLINK_TRACE_COUNTER("pool_size_bytes", config_.pool_size_bytes);
  NVQLINK_TRACE_COUNTER("buffer_size_bytes", config_.buffer_size_bytes);
  NVQLINK_TRACE_COUNTER("total_buffers", total_buffers_);

  if (mode_ == DatapathMode::CPU) {
    allocate_cpu_memory();
  } else {
#ifdef __CUDACC__
    allocate_gpu_memory();
#else
    throw std::runtime_error("GPU mode requested but CUDA is not available");
#endif
  }
}

MemoryPool::~MemoryPool() {
  NVQLINK_TRACE_FULL(DOMAIN_MEMORY, "MemoryPool::destructor");

  // Delete all Buffer objects first (allocated with 'new')
  for (Buffer *buffer : free_buffers_)
    buffer->~Buffer();

  if (mode_ == DatapathMode::CPU) {
    if (base_addr_) {
      if (using_hugepages_)
        munmap(base_addr_, config_.pool_size_bytes);
      else
        std::free(base_addr_);
    }
    return;
  }

  assert(mode_ == DatapathMode::GPU);
#ifdef __CUDACC__
  if (gpu_base_addr_)
    cudaFree(gpu_base_addr_);
#endif
}

void MemoryPool::allocate_cpu_memory() {
  // Try to allocate hugepage-backed memory for optimal performance
  // Hugepages provide:
  // 1. Better TLB efficiency (fewer page table walks)
  // 2. Physical memory contiguity (important for DMA)
  // 3. Reduced page fault overhead

  NVQLINK_LOG_INFO(
      DOMAIN_MEMORY,
      "MemoryPool: Attempting to allocate {} MB using hugepages...",
      config_.pool_size_bytes / (1024 * 1024));

  base_addr_ =
      mmap(nullptr, config_.pool_size_bytes, PROT_READ | PROT_WRITE,
           MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_POPULATE, -1, 0);

  if (base_addr_ == MAP_FAILED) {
    // Hugepage allocation failed, fall back to regular allocation
    NVQLINK_LOG_WARNING(DOMAIN_MEMORY,
                        "Warning: Hugepage allocation failed (errno={}). "
                        "Falling back to regular pages.",
                        errno);
    NVQLINK_LOG_WARNING(DOMAIN_MEMORY, "  Possible causes:");
    NVQLINK_LOG_WARNING(DOMAIN_MEMORY,
                        "    - Insufficient hugepages configured");
    NVQLINK_LOG_WARNING(
        DOMAIN_MEMORY,
        "    - Run: echo 1024 > "
        "/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages");
    NVQLINK_LOG_WARNING(DOMAIN_MEMORY,
                        "    - Or add 'hugepagesz=2M hugepages=1024' to kernel "
                        "boot parameters");

    base_addr_ = std::aligned_alloc(4096, config_.pool_size_bytes);
    if (!base_addr_) {
      throw std::runtime_error(
          "Failed to allocate CPU memory pool (regular pages also failed)");
    }
    using_hugepages_ = false;
    NVQLINK_LOG_INFO(
        DOMAIN_MEMORY,
        "MemoryPool: Using regular 4KB pages (performance will be reduced)");
  } else {
    using_hugepages_ = true;
    // Advise kernel not to fork this memory (performance optimization)
    madvise(base_addr_, config_.pool_size_bytes, MADV_DONTFORK);
    NVQLINK_LOG_INFO(
        DOMAIN_MEMORY,
        "MemoryPool: Successfully allocated {} MB using 2MB hugepages",
        config_.pool_size_bytes / (1024 * 1024));
  }

  std::memset(base_addr_, 0, config_.pool_size_bytes);

  // Create buffer objects
  char *ptr = static_cast<char *>(base_addr_);
  for (std::size_t i = 0; i < total_buffers_; ++i) {
    auto *buffer = new Buffer(ptr, config_.buffer_size_bytes,
                              config_.headroom_bytes, config_.tailroom_bytes);
    free_buffers_.push_back(buffer);
    ptr += config_.buffer_size_bytes;
  }

  NVQLINK_LOG_INFO(DOMAIN_MEMORY,
                   "MemoryPool: Created {} buffers of {} bytes each",
                   total_buffers_, config_.buffer_size_bytes);
}

#ifdef __CUDACC__
void MemoryPool::allocate_gpu_memory() {
  cudaError_t err = cudaMalloc(&gpu_base_addr_, config_.pool_size_bytes);
  if (err != cudaSuccess) {
    throw std::runtime_error("Failed to allocate GPU memory pool");
  }

  cudaMemset(gpu_base_addr_, 0, config_.pool_size_bytes);

  // Create buffer objects (metadata on host, data on GPU)
  char *ptr = static_cast<char *>(gpu_base_addr_);
  for (std::size_t i = 0; i < total_buffers_; ++i) {
    auto *buffer = new Buffer(ptr, config_.buffer_size_bytes,
                              config_.headroom_bytes, config_.tailroom_bytes);
    free_buffers_.push_back(buffer);
    ptr += config_.buffer_size_bytes;
  }
}
#endif

Buffer *MemoryPool::allocate() {
  NVQLINK_TRACE_MEMORY("allocate");

  std::lock_guard<std::mutex> lock(mutex_);
  if (free_buffers_.empty()) {
    NVQLINK_TRACE_MARK_ERROR(DOMAIN_MEMORY, "POOL_EXHAUSTED");
    return nullptr; // Pool exhausted
  }
  Buffer *buffer = free_buffers_.back();
  free_buffers_.pop_back();
  return buffer;
}

void MemoryPool::deallocate(Buffer *buffer) {
  NVQLINK_TRACE_MEMORY("deallocate");

  std::lock_guard<std::mutex> lock(mutex_);
  free_buffers_.push_back(buffer);

  // Track pool pressure: low availability = high pressure
  size_t available = free_buffers_.size();
  uint64_t pressure = (total_buffers_ - available) * 100 / total_buffers_;
  NVQLINK_TRACE_COUNTER("pool_pressure_percent", pressure);
}

std::size_t MemoryPool::get_available_buffers() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return free_buffers_.size();
}
