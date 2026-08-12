/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

/// \file cuda_memory.h
/// \brief Minimal RAII device-memory wrapper used by the dynamics
/// matrix-exponential support and propagator cache.
///
/// Allocations go through cudaq::dynamics::DeviceAllocator so they share the
/// dynamics backend allocation scheme / performance tracking, and errors are
/// reported through the shared HANDLE_CUDA_ERROR helper.

#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatUtils.h"

#include <algorithm>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <utility>

namespace cudaq::detail {

/// \brief RAII wrapper for CUDA device memory (move-only, 2x growth).
///
/// Existing content is not preserved on reallocate.
template <typename T>
class CudaDeviceMemory {
public:
  CudaDeviceMemory() = default;
  explicit CudaDeviceMemory(size_t count) { reallocate(count); }

  ~CudaDeviceMemory() {
    if (ptr_)
      cudaq::dynamics::DeviceAllocator::free(ptr_);
  }

  CudaDeviceMemory(const CudaDeviceMemory &) = delete;
  CudaDeviceMemory &operator=(const CudaDeviceMemory &) = delete;

  CudaDeviceMemory(CudaDeviceMemory &&other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        capacity_(std::exchange(other.capacity_, 0)) {}

  CudaDeviceMemory &operator=(CudaDeviceMemory &&other) noexcept {
    if (this != &other) {
      if (ptr_)
        cudaq::dynamics::DeviceAllocator::free(ptr_);
      ptr_ = std::exchange(other.ptr_, nullptr);
      size_ = std::exchange(other.size_, 0);
      capacity_ = std::exchange(other.capacity_, 0);
    }
    return *this;
  }

  /// \brief Reallocate to hold \p new_count elements (2x growth, no preserve).
  void reallocate(size_t new_count) {
    if (new_count <= capacity_) {
      size_ = new_count;
      return;
    }
    if (ptr_) {
      cudaq::dynamics::DeviceAllocator::free(ptr_);
      ptr_ = nullptr;
    }
    capacity_ = std::max(new_count, capacity_ * 2);
    size_ = new_count;
    if (capacity_ > 0)
      ptr_ = static_cast<T *>(
          cudaq::dynamics::DeviceAllocator::allocate(capacity_ * sizeof(T)));
  }

  [[nodiscard]] T *get() noexcept { return ptr_; }
  [[nodiscard]] const T *get() const noexcept { return ptr_; }
  [[nodiscard]] size_t size() const noexcept { return size_; }
  [[nodiscard]] size_t size_bytes() const noexcept { return size_ * sizeof(T); }
  [[nodiscard]] explicit operator bool() const noexcept {
    return ptr_ != nullptr;
  }

  void copy_from_host(const T *host_ptr, size_t count) {
    if (count > size_)
      throw std::runtime_error("copy_from_host: count exceeds allocated size");
    HANDLE_CUDA_ERROR(
        cudaMemcpy(ptr_, host_ptr, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copy_to_host(T *host_ptr, size_t count) const {
    if (count > size_)
      throw std::runtime_error("copy_to_host: count exceeds allocated size");
    HANDLE_CUDA_ERROR(
        cudaMemcpy(host_ptr, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  void zero() {
    if (ptr_)
      HANDLE_CUDA_ERROR(cudaMemset(ptr_, 0, size_ * sizeof(T)));
  }

private:
  T *ptr_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
};

using CudaComplexMemory = CudaDeviceMemory<cuDoubleComplex>;

} // namespace cudaq::detail
