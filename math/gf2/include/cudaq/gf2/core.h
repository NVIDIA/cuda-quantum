/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#ifndef CUDAQ_GF2_CORE_H
#define CUDAQ_GF2_CORE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Numeric compatibility revision for v0.x development, not a major release. */
#define CUDAQ_GF2_ABI_VERSION 1

/* Statuses are returned by value only, so they stay a distinct enum type.
 * MAX_ENUM forces a 32-bit enum, including under `-fshort-enums`; it is never
 * returned. Values stored in structs or the packed format are uint32_t. */
typedef enum cudaq_gf2_status {
  CUDAQ_GF2_OK = 0,
  CUDAQ_GF2_INVALID_ARGUMENT = 1,
  CUDAQ_GF2_INVALID_CONTENT = 2,
  CUDAQ_GF2_INSUFFICIENT_CAPACITY = 3,
  CUDAQ_GF2_UNSUPPORTED_VERSION = 4,
  CUDAQ_GF2_ALLOCATION_FAILED = 5,
  CUDAQ_GF2_STATUS_MAX_ENUM = 0x7FFFFFFF
} cudaq_gf2_status;

/* Canonical compressed sparse columns: sorted, duplicate-free row indices in
 * each column. All entries are one. Empty shapes and empty columns are valid.
 * Offsets has columns + 1 elements, including the terminal nonzero count.
 * Views borrow all arrays; no library function retains them or allocates. */
typedef struct cudaq_gf2_csc_view {
  uint32_t rows;
  uint32_t columns;
  uint64_t nonzeros;
  const uint64_t *column_offsets;
  const uint32_t *row_indices;
} cudaq_gf2_csc_view;

typedef struct cudaq_gf2_entry {
  uint32_t row;
  uint32_t column;
} cudaq_gf2_entry;

uint32_t cudaq_gf2_abi_version(void);
const char *cudaq_gf2_status_string(cudaq_gf2_status status);
cudaq_gf2_status cudaq_gf2_csc_validate(const cudaq_gf2_csc_view *matrix);

/* Construct from unordered entries; repeated entries cancel modulo two.
 * Caller capacities: columns + 1 offsets, entry_count row indices, and
 * entry_count * sizeof(cudaq_gf2_entry) scratch bytes aligned for that type.
 * All buffers must be disjoint. The returned view borrows the output arrays.
 * On failure, result is unchanged. Scratch may be modified. */
cudaq_gf2_status cudaq_gf2_csc_construct(
    uint32_t rows, uint32_t columns, const cudaq_gf2_entry *entries,
    uint64_t entry_count, uint64_t *column_offsets, uint64_t offsets_capacity,
    uint32_t *row_indices, uint64_t indices_capacity, void *scratch,
    uint64_t scratch_bytes, cudaq_gf2_csc_view *result);

/* Evaluate a validated matrix. Vectors contain one byte per bit, must be
 * disjoint, and must have the exact shape.
 * offset may be NULL (zero). Content validation is a separate setup operation.
 */
cudaq_gf2_status cudaq_gf2_csc_apply(const cudaq_gf2_csc_view *matrix,
                                     const uint8_t *input, uint64_t input_size,
                                     const uint8_t *offset, uint8_t *output,
                                     uint64_t output_size);

#ifdef __cplusplus
}
#endif

#endif
