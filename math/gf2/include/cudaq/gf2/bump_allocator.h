/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#ifndef CUDAQ_GF2_BUMP_ALLOCATOR_H
#define CUDAQ_GF2_BUMP_ALLOCATOR_H
#include "cudaq/gf2/memory.h"
#include <stddef.h>

/* Initialize as {data, capacity, 0}. Allocation includes alignment padding and
 * never falls back. Deallocation does not reclaim capacity. Reset only after
 * every allocation in this arena is dead. Separate scratch/result arenas allow
 * scratch reset after each setup call while results survive. Shared arenas
 * require external synchronization. */
typedef struct cudaq_gf2_bump_allocator {
  void *data;
  uint64_t capacity;
  uint64_t used;
} cudaq_gf2_bump_allocator;

static inline void *cudaq_gf2_bump_allocate(void *context, uint64_t bytes,
                                            uint32_t alignment) {
  cudaq_gf2_bump_allocator *arena = (cudaq_gf2_bump_allocator *)context;
  if (!arena || !arena->data || !alignment || (alignment & (alignment - 1)) ||
      arena->used > arena->capacity)
    return NULL;
  uintptr_t base = (uintptr_t)arena->data;
  if (arena->used > UINTPTR_MAX - base)
    return NULL;
  uintptr_t current = base + (uintptr_t)arena->used;
  uint64_t padding = (uint64_t)((-(uintptr_t)current) & (alignment - 1));
  uint64_t available = arena->capacity - arena->used;
  if (padding > available || bytes > available - padding ||
      padding > UINTPTR_MAX - current ||
      bytes > UINTPTR_MAX - current - padding)
    return NULL;
  void *result = (unsigned char *)arena->data + arena->used + padding;
  arena->used += padding + bytes;
  return result;
}
static inline void cudaq_gf2_bump_deallocate(void *context, void *pointer,
                                             uint64_t bytes,
                                             uint32_t alignment) {
  (void)context;
  (void)pointer;
  (void)bytes;
  (void)alignment;
}
static inline cudaq_gf2_allocator
cudaq_gf2_bump_resource(cudaq_gf2_bump_allocator *arena) {
  cudaq_gf2_allocator result = {arena, cudaq_gf2_bump_allocate,
                                cudaq_gf2_bump_deallocate};
  return result;
}
static inline void cudaq_gf2_bump_reset(cudaq_gf2_bump_allocator *arena) {
  if (arena)
    arena->used = 0;
}
#endif
