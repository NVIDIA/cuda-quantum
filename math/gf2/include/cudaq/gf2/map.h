/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#ifndef CUDAQ_GF2_MAP_H
#define CUDAQ_GF2_MAP_H
#include "cudaq/gf2/core.h"
#include "cudaq/gf2/format.h"
#include "cudaq/gf2/memory.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Side of a map: wires of input bundles index columns; wires of output
 * bundles index rows. */
typedef uint32_t cudaq_gf2_wire_direction;
enum { CUDAQ_GF2_INPUT = 0, CUDAQ_GF2_OUTPUT = 1 };
/* Names are nonempty UTF-8, compared byte-for-byte, without normalization.
 * One name denotes one (role, wire_count) schema. Local type IDs are not
 * identities. Interface bundle compatibility requires matching type schemas
 * and wire widths. Free bundles are not connection endpoints.
 */
typedef struct cudaq_gf2_bundle_type {
  const char *name;
  uint64_t name_size;
  cudaq_gf2_bundle_role role;
  uint32_t wire_count;
} cudaq_gf2_bundle_type;
typedef struct cudaq_gf2_bundle {
  uint32_t type, reserved;
  const uint32_t *widths;
} cudaq_gf2_bundle;
typedef struct cudaq_gf2_wire {
  uint32_t bundle, wire;
} cudaq_gf2_wire;
typedef struct cudaq_gf2_map_matrix_block {
  cudaq_gf2_wire output, input;
  cudaq_gf2_csc_view matrix;
} cudaq_gf2_map_matrix_block;
/* Extension payloads are already canonical, with section-relative references.
 * IDs below OPTIONAL_SECTION_MIN require REQUIRED; IDs above require OPTIONAL.
 * Only IDs >= EXTENSION_SECTION_MIN are available to extensions. Optional
 * sections must not change the map's meaning. Generic map validation checks
 * extension bounds, not their domain semantics. Domain checks must run before
 * using a value as that domain.
 */
typedef struct cudaq_gf2_section {
  cudaq_gf2_section_id id;
  cudaq_gf2_section_flags flags;
  uint64_t count, bytes;
  const void *data;
} cudaq_gf2_section;
typedef struct cudaq_gf2_map_description {
  uint32_t type_count, input_bundle_count, output_bundle_count, section_count;
  uint64_t block_count;
  const cudaq_gf2_bundle_type *types;
  const cudaq_gf2_bundle *inputs, *outputs;
  const cudaq_gf2_map_matrix_block *blocks;
  const cudaq_gf2_section *sections;
} cudaq_gf2_map_description;
/* Trusted borrowed token, initialized only by successful construct/read in this
 * ABI. Backing bytes must stay alive and unchanged. Getters/apply do not
 * revalidate content or make fabricated/stale tokens safe. Only packed bytes
 * cross packages.
 */
typedef struct cudaq_gf2_map_view {
  const void *data;
  uint64_t size;
} cudaq_gf2_map_view;
typedef struct cudaq_gf2_map_shape {
  uint32_t type_count, input_bundle_count, output_bundle_count, section_count;
  uint64_t block_count, matrix_count;
} cudaq_gf2_map_shape;
/* Declares required extension schemas understood by the caller. NULL means no
 * required extensions. Unknown required sections are rejected as unsupported;
 * unknown optional sections are bounded and skipped. Recognition does not run
 * semantic checks: the caller must validate recognized domain payloads.
 */
typedef struct cudaq_gf2_schema {
  const uint32_t *required_sections;
  uint32_t required_section_count, reserved;
} cudaq_gf2_schema;

/* Stateless setup uses per-call resources (NULL selects the heap). Explicit
 * resources never fall back. Temporary allocations are released before return.
 * Description arrays and output records must be disjoint. Failure preserves
 * owner/view outputs; allocator state is not rolled back. Construction returns
 * owned bytes and an already checked borrowed view, retaining no scratch.
 * The owner must be zero-initialized and empty before construction.
 * Type names and blocks may be unordered; bundle order is preserved. Packing
 * sorts types and blocks, remaps IDs, drops zero blocks and interns matrices.
 * Duplicate names or block endpoints are invalid, including duplicate zeros.
 */
cudaq_gf2_status
cudaq_gf2_map_validate(const cudaq_gf2_map_description *description,
                       const cudaq_gf2_memory *memory);
cudaq_gf2_status
cudaq_gf2_map_construct(const cudaq_gf2_map_description *description,
                        const cudaq_gf2_memory *memory,
                        cudaq_gf2_owned_buffer *owner,
                        cudaq_gf2_map_view *result);
/* Zero-copy checked read borrows the exact-sized, PACKED_ALIGNMENT-aligned
 * buffer. Only temporary scratch is allocated, using memory->scratch. */
cudaq_gf2_status cudaq_gf2_map_read(const void *buffer, uint64_t bytes,
                                    const cudaq_gf2_schema *schema,
                                    const cudaq_gf2_memory *memory,
                                    cudaq_gf2_map_view *result);
cudaq_gf2_status cudaq_gf2_map_get_shape(const cudaq_gf2_map_view *value,
                                         cudaq_gf2_map_shape *shape);
cudaq_gf2_status cudaq_gf2_map_get_type(const cudaq_gf2_map_view *value,
                                        uint32_t index,
                                        cudaq_gf2_bundle_type *type);
cudaq_gf2_status cudaq_gf2_map_get_bundle(const cudaq_gf2_map_view *value,
                                          cudaq_gf2_wire_direction direction,
                                          uint32_t index,
                                          cudaq_gf2_bundle *bundle);
cudaq_gf2_status cudaq_gf2_map_get_wire(const cudaq_gf2_map_view *value,
                                        cudaq_gf2_wire_direction direction,
                                        cudaq_gf2_wire wire, uint32_t *width,
                                        uint64_t *offset);
cudaq_gf2_status
cudaq_gf2_map_get_matrix_block(const cudaq_gf2_map_view *value, uint64_t index,
                               cudaq_gf2_map_matrix_block *block);
cudaq_gf2_status cudaq_gf2_map_get_section(const cudaq_gf2_map_view *value,
                                           cudaq_gf2_section_id id,
                                           cudaq_gf2_section *section);
cudaq_gf2_status cudaq_gf2_map_vector_sizes(const cudaq_gf2_map_view *value,
                                            uint64_t *inputs,
                                            uint64_t *outputs);
/* Read-only runtime operation. Disjoint exact-sized vectors, one byte per bit.
 * No workspace. Invalid arguments leave output unchanged. */
cudaq_gf2_status cudaq_gf2_map_apply(const cudaq_gf2_map_view *value,
                                     const uint8_t *input, uint64_t input_size,
                                     uint8_t *output, uint64_t output_size);
#ifdef __cplusplus
}
#endif
#endif
