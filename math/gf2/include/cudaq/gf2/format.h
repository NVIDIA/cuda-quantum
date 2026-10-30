/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#ifndef CUDAQ_GF2_FORMAT_H
#define CUDAQ_GF2_FORMAT_H
#include <stdint.h>

/* Concrete v0.x development revision. All multibyte fields are little-endian.
 * Base, payloads and trailing directory are 8-byte aligned. All padding is
 * zero. References inside a section are relative to that section's start. */
#define CUDAQ_GF2_FORMAT_VERSION 1
#define CUDAQ_GF2_PACKED_ALIGNMENT 8
#define CUDAQ_GF2_OPTIONAL_SECTION_MIN UINT32_C(0x80000000)
#define CUDAQ_GF2_EXTENSION_SECTION_MIN UINT32_C(0x00010000)

typedef enum cudaq_gf2_kind { CUDAQ_GF2_KIND_LINEAR_MAP = 1 } cudaq_gf2_kind;
typedef enum cudaq_gf2_section_flags {
  CUDAQ_GF2_SECTION_OPTIONAL = 0,
  CUDAQ_GF2_SECTION_REQUIRED = 1
} cudaq_gf2_section_flags;
typedef enum cudaq_gf2_section_id {
  CUDAQ_GF2_SECTION_TYPES = 1,
  CUDAQ_GF2_SECTION_INPUTS = 2,
  CUDAQ_GF2_SECTION_OUTPUTS = 3,
  CUDAQ_GF2_SECTION_BLOCKS = 4,
  CUDAQ_GF2_SECTION_MATRICES = 5
} cudaq_gf2_section_id;

typedef struct cudaq_gf2_packed_header {
  uint8_t magic[8];
  uint32_t version, kind;
  uint64_t size;
  uint32_t alignment, index_bits;
  uint64_t directory_offset, directory_count, directory_bytes, reserved;
} cudaq_gf2_packed_header;
typedef struct cudaq_gf2_packed_section {
  uint32_t id, flags;
  uint64_t count, bytes, offset;
} cudaq_gf2_packed_section;
typedef struct cudaq_gf2_packed_type {
  uint64_t name, name_size;
  uint32_t role, wire_count;
} cudaq_gf2_packed_type;
typedef struct cudaq_gf2_packed_bundle {
  uint32_t type, reserved;
  uint64_t widths, offsets;
} cudaq_gf2_packed_bundle;
typedef struct cudaq_gf2_packed_block {
  uint32_t output_bundle, output_wire, input_bundle, input_wire;
  uint32_t matrix, reserved;
} cudaq_gf2_packed_block;
typedef struct cudaq_gf2_packed_matrix {
  uint32_t rows, columns;
  uint64_t nonzeros, column_offsets, row_indices;
} cudaq_gf2_packed_matrix;
#endif
