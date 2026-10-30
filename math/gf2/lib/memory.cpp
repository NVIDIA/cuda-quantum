/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma GCC visibility push(default)
#include "cudaq/gf2/memory.h"
#pragma GCC visibility pop
#include "cudaq/gf2/format.h"
#include <cstddef>
#include <cstdlib>

namespace {
void *heap_allocate(void *, uint64_t bytes, uint32_t alignment) {
  if (!bytes || bytes > SIZE_MAX || !alignment ||
      (alignment & (alignment - 1)) || alignment > alignof(std::max_align_t))
    return nullptr;
  return std::malloc(static_cast<size_t>(bytes));
}
void heap_deallocate(void *, void *pointer, uint64_t, uint32_t) {
  std::free(pointer);
}
} // namespace

cudaq_gf2_allocator cudaq_gf2_default_allocator(void) {
  return {nullptr, heap_allocate, heap_deallocate};
}
void cudaq_gf2_owned_buffer_release(cudaq_gf2_owned_buffer *buffer) {
  if (!buffer)
    return;
  const auto owned = *buffer;
  *buffer = {};
  if (owned.data && owned.deallocate)
    owned.deallocate(owned.context, const_cast<void *>(owned.data), owned.size,
                     CUDAQ_GF2_PACKED_ALIGNMENT);
}
