/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

/* Test-only resources. Production headers do not depend on this file. */

#pragma once

#include "cudaq/gf2/bump_allocator.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TA_CHECK(condition)                                                    \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      abort();                                                                 \
    }                                                                          \
  } while (0)

enum { TEST_MAX_LIVE_ALLOCATIONS = 32 };
typedef struct test_allocation {
  void *pointer;
  uint64_t bytes;
  uint32_t alignment;
} test_allocation;
typedef struct test_resource {
  uint64_t attempts, allocations, releases, live_bytes, peak_bytes;
  uint64_t fail_on_attempt, largest_request;
  int bounded;
  cudaq_gf2_bump_allocator arena;
  test_allocation active[TEST_MAX_LIVE_ALLOCATIONS];
} test_resource;

static inline void test_resource_init(test_resource *resource) {
  memset(resource, 0, sizeof(*resource));
}
static inline void test_resource_bump(test_resource *resource, void *data,
                                      uint64_t capacity) {
  test_resource_init(resource);
  resource->bounded = 1;
  resource->arena = (cudaq_gf2_bump_allocator){data, capacity, 0};
}
static inline void *test_resource_allocate(void *context, uint64_t bytes,
                                           uint32_t alignment) {
  test_resource *resource = context;
  ++resource->attempts;
  if (resource->attempts == resource->fail_on_attempt)
    return NULL;
  TA_CHECK(bytes && alignment && !(alignment & (alignment - 1)));
  TA_CHECK(alignment <= _Alignof(max_align_t) && bytes <= SIZE_MAX);
  unsigned slot = 0;
  while (slot < TEST_MAX_LIVE_ALLOCATIONS && resource->active[slot].pointer)
    ++slot;
  TA_CHECK(slot < TEST_MAX_LIVE_ALLOCATIONS);
  void *pointer = resource->bounded ? cudaq_gf2_bump_allocate(&resource->arena,
                                                              bytes, alignment)
                                    : malloc((size_t)bytes);
  if (!pointer)
    return NULL;
  TA_CHECK((uintptr_t)pointer % alignment == 0);
  resource->active[slot] = (test_allocation){pointer, bytes, alignment};
  ++resource->allocations;
  resource->live_bytes += bytes;
  if (resource->live_bytes > resource->peak_bytes)
    resource->peak_bytes = resource->live_bytes;
  if (bytes > resource->largest_request)
    resource->largest_request = bytes;
  return pointer;
}
static inline void test_resource_deallocate(void *context, void *pointer,
                                            uint64_t bytes,
                                            uint32_t alignment) {
  test_resource *resource = context;
  unsigned slot = 0;
  while (slot < TEST_MAX_LIVE_ALLOCATIONS &&
         resource->active[slot].pointer != pointer)
    ++slot;
  TA_CHECK(slot < TEST_MAX_LIVE_ALLOCATIONS && pointer);
  TA_CHECK(resource->active[slot].bytes == bytes &&
           resource->active[slot].alignment == alignment);
  resource->active[slot] = (test_allocation){0};
  ++resource->releases;
  resource->live_bytes -= bytes;
  if (resource->bounded)
    cudaq_gf2_bump_deallocate(&resource->arena, pointer, bytes, alignment);
  else {
    memset(pointer, 0xdd, (size_t)bytes);
    free(pointer);
  }
}
static inline cudaq_gf2_allocator
test_resource_allocator(test_resource *resource) {
  return (cudaq_gf2_allocator){resource, test_resource_allocate,
                               test_resource_deallocate};
}
static inline cudaq_gf2_memory test_memory(test_resource *scratch,
                                           test_resource *result) {
  return (cudaq_gf2_memory){test_resource_allocator(scratch),
                            test_resource_allocator(result)};
}
static inline void test_resource_balanced(const test_resource *resource) {
  TA_CHECK(!resource->live_bytes &&
           resource->allocations == resource->releases);
  for (unsigned i = 0; i < TEST_MAX_LIVE_ALLOCATIONS; ++i)
    TA_CHECK(!resource->active[i].pointer);
}
static inline void test_resource_reset(test_resource *resource) {
  test_resource_balanced(resource);
  TA_CHECK(resource->bounded);
  memset(resource->arena.data, 0xa5, (size_t)resource->arena.capacity);
  cudaq_gf2_bump_reset(&resource->arena);
}
static inline int test_same_owner(cudaq_gf2_owned_buffer a,
                                  cudaq_gf2_owned_buffer b) {
  return a.data == b.data && a.size == b.size && a.deallocate == b.deallocate &&
         a.context == b.context;
}
static inline void test_release_sentinel(void *context, void *pointer,
                                         uint64_t bytes, uint32_t alignment) {
  (void)context;
  (void)pointer;
  (void)bytes;
  (void)alignment;
  TA_CHECK(0);
}
