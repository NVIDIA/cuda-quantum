/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#pragma once

#include "cudaq/gf2/core.h"
#include "cudaq/gf2/memory.h"
#include <cstddef>
#include <cstdint>

namespace cudaq_gf2_private {
inline bool valid_allocator(const cudaq_gf2_allocator &resource) {
  return resource.allocate && resource.deallocate;
}
inline bool empty_owner(const cudaq_gf2_owned_buffer &owner) {
  return !owner.data && !owner.size && !owner.deallocate && !owner.context;
}
inline cudaq_gf2_allocator scratch_allocator(const cudaq_gf2_memory *memory) {
  return memory ? memory->scratch : cudaq_gf2_default_allocator();
}
inline cudaq_gf2_allocator result_allocator(const cudaq_gf2_memory *memory) {
  return memory ? memory->result : cudaq_gf2_default_allocator();
}

// Each allocation captures its callbacks, not a pointer to temporary options.
// A failed operation cleans up ownership; monotonic resources need not reclaim.
class allocation {
  cudaq_gf2_allocator resource_;
  void *data_ = nullptr;
  uint64_t bytes_ = 0;
  uint32_t alignment_ = 0;

public:
  explicit allocation(cudaq_gf2_allocator resource) : resource_(resource) {}
  allocation(const allocation &) = delete;
  allocation &operator=(const allocation &) = delete;
  ~allocation() { reset(); }
  cudaq_gf2_status reserve(uint64_t bytes, uint32_t alignment) {
    if (!valid_allocator(resource_) || bytes > SIZE_MAX || !alignment ||
        (alignment & (alignment - 1)))
      return CUDAQ_GF2_INVALID_ARGUMENT;
    bytes_ = bytes;
    alignment_ = alignment;
    if (!bytes)
      return CUDAQ_GF2_OK;
    data_ = resource_.allocate(resource_.context, bytes, alignment);
    if (!data_)
      return CUDAQ_GF2_ALLOCATION_FAILED;
    if (reinterpret_cast<uintptr_t>(data_) % alignment)
      return CUDAQ_GF2_INVALID_ARGUMENT;
    return CUDAQ_GF2_OK;
  }
  void reset() {
    if (data_) {
      void *data = data_;
      data_ = nullptr;
      resource_.deallocate(resource_.context, data, bytes_, alignment_);
    }
  }
  void *data() const { return data_; }
  cudaq_gf2_owned_buffer relinquish() {
    const cudaq_gf2_owned_buffer owner{data_, bytes_, resource_.deallocate,
                                       resource_.context};
    data_ = nullptr;
    return owner;
  }
};
} // namespace cudaq_gf2_private
