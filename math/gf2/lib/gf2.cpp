/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#pragma GCC visibility push(default)
#include "cudaq/gf2/core.h"
#pragma GCC visibility pop

#include "csc_xor.h"

#include <algorithm>
#include <span>
#include <stddef.h>
#include <string.h>

uint32_t cudaq_gf2_abi_version(void) { return CUDAQ_GF2_ABI_VERSION; }

cudaq_gf2_status cudaq_gf2_csc_validate(const cudaq_gf2_csc_view *m) {
  if (!m || !m->column_offsets || (m->nonzeros && !m->row_indices))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  if (m->nonzeros > SIZE_MAX / sizeof(uint32_t) ||
      (uint64_t)m->columns + 1 > SIZE_MAX / sizeof(uint64_t))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  if (m->column_offsets[0] != 0 || m->column_offsets[m->columns] != m->nonzeros)
    return CUDAQ_GF2_INVALID_CONTENT;
  for (uint64_t c = 0; c < m->columns; ++c) {
    uint64_t begin = m->column_offsets[c], end = m->column_offsets[c + 1];
    if (begin > end || end > m->nonzeros)
      return CUDAQ_GF2_INVALID_CONTENT;
    for (uint64_t p = begin; p < end; ++p)
      if (m->row_indices[p] >= m->rows ||
          (p > begin && m->row_indices[p - 1] >= m->row_indices[p]))
        return CUDAQ_GF2_INVALID_CONTENT;
  }
  return CUDAQ_GF2_OK;
}

static bool entry_less(cudaq_gf2_entry a, cudaq_gf2_entry b) noexcept {
  return a.column < b.column || (a.column == b.column && a.row < b.row);
}

cudaq_gf2_status cudaq_gf2_csc_construct(
    uint32_t rows, uint32_t columns, const cudaq_gf2_entry *entries,
    uint64_t count, uint64_t *offsets, uint64_t offsets_capacity,
    uint32_t *indices, uint64_t indices_capacity, void *scratch,
    uint64_t scratch_bytes, cudaq_gf2_csc_view *result) {
  if (!result || !offsets || (count && (!entries || !indices || !scratch)) ||
      count > SIZE_MAX / sizeof(cudaq_gf2_entry) ||
      (uint64_t)columns + 1 > SIZE_MAX / sizeof(uint64_t) ||
      (scratch && (uintptr_t)scratch % alignof(cudaq_gf2_entry)))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  if (offsets_capacity < (uint64_t)columns + 1 || indices_capacity < count ||
      scratch_bytes < count * sizeof(cudaq_gf2_entry))
    return CUDAQ_GF2_INSUFFICIENT_CAPACITY;
  for (uint64_t i = 0; i < count; ++i)
    if (entries[i].row >= rows || entries[i].column >= columns)
      return CUDAQ_GF2_INVALID_CONTENT;
  cudaq_gf2_entry *ordered = (cudaq_gf2_entry *)scratch;
  if (count)
    memcpy(ordered, entries, count * sizeof(*ordered));
  if (count > 1) {
    std::span entries_to_sort(ordered, static_cast<size_t>(count));
    std::sort(entries_to_sort.begin(), entries_to_sort.end(), entry_less);
  }
  uint64_t p = 0, nonzeros = 0;
  for (uint64_t c = 0; c < columns; ++c) {
    offsets[c] = nonzeros;
    while (p < count && ordered[p].column == c) {
      uint64_t begin = p++;
      while (p < count && ordered[p].column == c &&
             ordered[p].row == ordered[begin].row)
        ++p;
      if ((p - begin) % 2)
        indices[nonzeros++] = ordered[begin].row;
    }
  }
  offsets[columns] = nonzeros;
  *result = cudaq_gf2_csc_view{rows, columns, nonzeros, offsets, indices};
  return CUDAQ_GF2_OK;
}

cudaq_gf2_status cudaq_gf2_csc_apply(const cudaq_gf2_csc_view *m,
                                     const uint8_t *input, uint64_t input_size,
                                     const uint8_t *offset, uint8_t *output,
                                     uint64_t output_size) {
  if (!m || !m->column_offsets || (m->nonzeros && !m->row_indices) ||
      input_size != m->columns || output_size != m->rows ||
      (input_size && !input) || (output_size && !output) ||
      (input_size && input == output))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  for (uint64_t i = 0; i < input_size; ++i)
    if (input[i] > 1)
      return CUDAQ_GF2_INVALID_ARGUMENT;
  for (uint64_t i = 0; i < output_size; ++i)
    if (offset && offset[i] > 1)
      return CUDAQ_GF2_INVALID_ARGUMENT;
  for (uint64_t i = 0; i < output_size; ++i)
    output[i] = offset ? offset[i] : 0;
  cudaq_gf2_csc_xor_kernel(m->columns, m->column_offsets, m->row_indices, input,
                           output);
  return CUDAQ_GF2_OK;
}

const char *cudaq_gf2_status_string(cudaq_gf2_status status) {
  switch (status) {
  case CUDAQ_GF2_OK:
    return "success";
  case CUDAQ_GF2_INVALID_ARGUMENT:
    return "invalid argument or buffer alignment";
  case CUDAQ_GF2_INVALID_CONTENT:
    return "invalid GF(2) content";
  case CUDAQ_GF2_INSUFFICIENT_CAPACITY:
    return "insufficient capacity";
  case CUDAQ_GF2_UNSUPPORTED_VERSION:
    return "unsupported format, kind, byte order or required section";
  case CUDAQ_GF2_ALLOCATION_FAILED:
    return "allocation failed";
  case CUDAQ_GF2_STATUS_MAX_ENUM:
    break;
  }
  return "unknown status";
}
