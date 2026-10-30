/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#ifndef CUDAQ_GF2_MEMORY_H
#define CUDAQ_GF2_MEMORY_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef void *(*cudaq_gf2_allocate_fn)(void *context, uint64_t bytes,
                                       uint32_t alignment);
typedef void (*cudaq_gf2_deallocate_fn)(void *context, void *pointer,
                                        uint64_t bytes, uint32_t alignment);
typedef struct cudaq_gf2_allocator {
  void *context;
  cudaq_gf2_allocate_fn allocate;
  cudaq_gf2_deallocate_fn deallocate;
} cudaq_gf2_allocator;

/* Allocate returns storage aligned as requested, or NULL on failure. Returned
 * allocations must be disjoint from live input/output storage. Deallocate is
 * called with the exact pointer, size and alignment used for that allocation.
 * Library requests use the fixed eight-byte packed alignment. Callbacks must
 * not reenter an operation using overlapping storage. */

/* NULL policy selects malloc/free. Explicit resources need both callbacks and
 * never fall back to the heap. Both slots may use the same provider. Read,
 * check and validate use scratch only; constructors also use result. Temporary
 * allocations are released before return. Callback code and the result context
 * must outlive all returned owners. Callbacks must not throw across C.
 * Deallocate may reclaim nothing: failure does not roll back a bump arena.
 */
typedef struct cudaq_gf2_memory {
  cudaq_gf2_allocator scratch;
  cudaq_gf2_allocator result;
} cudaq_gf2_memory;

/* Unique ownership of immutable packed bytes, aligned to the fixed eight-byte
 * format constant. This storage record is not a semantic handle or a general
 * exchange envelope. Deallocator/context are copied, with no hidden prefix or
 * retained options pointer. Move by copying and clearing the source; shallow
 * copies must not both be released. Checked views borrow until release.
 * Callback code/context must stay alive; release runs on the calling thread.
 */
typedef struct cudaq_gf2_owned_buffer {
  const void *data;
  uint64_t size;
  cudaq_gf2_deallocate_fn deallocate;
  void *context;
} cudaq_gf2_owned_buffer;

/* Does not itself allocate; useful when only one slot uses a custom resource.
 */
cudaq_gf2_allocator cudaq_gf2_default_allocator(void);
/* Clears the record, then calls deallocate(context, data, size, 8).
 * NULL and a zero-initialized record are harmless. Clearing one alias does not
 * make other accidental shallow copies safe. */
void cudaq_gf2_owned_buffer_release(cudaq_gf2_owned_buffer *buffer);
#ifdef __cplusplus
}
#endif
#endif
