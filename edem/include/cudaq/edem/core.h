/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#ifndef CUDAQ_EDEM_CORE_H
#define CUDAQ_EDEM_CORE_H

#include "cudaq/gf2/map.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Numeric compatibility revisions for v0.x development, not major releases. */
#define CUDAQ_EDEM_ABI_VERSION 1
#define CUDAQ_EDEM_FORMAT_VERSION CUDAQ_GF2_FORMAT_VERSION
#define CUDAQ_EDEM_PACKED_ALIGNMENT CUDAQ_GF2_PACKED_ALIGNMENT

/* Compatibility allowance on exclusive-group totals only. Individual
 * probabilities remain in [0, 1]; accepted values are never renormalized. */
#define CUDAQ_EDEM_EXCLUSIVE_TOTAL_TOLERANCE 1e-6

typedef enum cudaq_edem_status {
  CUDAQ_EDEM_OK = 0,
  CUDAQ_EDEM_INVALID_ARGUMENT = 1,
  CUDAQ_EDEM_INVALID_CONTENT = 2,
  CUDAQ_EDEM_INSUFFICIENT_CAPACITY = 3,
  CUDAQ_EDEM_UNSUPPORTED_VERSION = 4,
  CUDAQ_EDEM_ALLOCATION_FAILED = 5
} cudaq_edem_status;

typedef enum cudaq_edem_port_kind {
  CUDAQ_EDEM_FAULT = 0,
  CUDAQ_EDEM_VIRTUAL = 1,
  CUDAQ_EDEM_SYNDROME = 2,
  CUDAQ_EDEM_LOGICAL = 3,
  CUDAQ_EDEM_DETECTOR = 4,
  CUDAQ_EDEM_OBSERVABLE = 5,
  CUDAQ_EDEM_SUCCESS = 6
} cudaq_edem_port_kind;

/* V, S and L indices identify code-block ordinals, independently on each side.
 * Other indices identify partitions of that kind. Directions are implicit in
 * the containing block: output first, input second. */
typedef struct cudaq_edem_endpoint {
  uint32_t kind;
  uint32_t index;
} cudaq_edem_endpoint;

typedef struct cudaq_edem_code_presentation {
  const char *name;
  uint64_t name_size;
  uint32_t generators;
  uint32_t logical_qubits;
} cudaq_edem_code_presentation;

typedef enum cudaq_edem_group_kind {
  CUDAQ_EDEM_INDEPENDENT = 0,
  CUDAQ_EDEM_EXCLUSIVE = 1
} cudaq_edem_group_kind;

/* Sorted original indicator IDs and one probability per ID. All indicators
 * occur in exactly one group. There is at most one Independent group;
 * Exclusive groups have at least two members. Their compensated sum must be
 * <= 1 + CUDAQ_EDEM_EXCLUSIVE_TOTAL_TOLERANCE. Each probability must be finite
 * and within [0, 1], without an individual tolerance.
 * Fault coordinates are original indicator IDs. */
typedef struct cudaq_edem_fault_group {
  uint32_t kind;
  uint32_t size;
  const uint32_t *fault_ids;
  const double *probabilities;
} cudaq_edem_fault_group;

typedef struct cudaq_edem_fault_distribution {
  uint32_t size;
  uint32_t group_count;
  const cudaq_edem_fault_group *groups;
} cudaq_edem_fault_distribution;

typedef struct cudaq_edem_block {
  cudaq_edem_endpoint output;
  cudaq_edem_endpoint input;
  cudaq_gf2_csc_view matrix;
} cudaq_edem_block;

typedef struct cudaq_edem_shape {
  uint32_t code_presentations;
  uint32_t input_code_blocks;
  uint32_t output_code_blocks;
  uint32_t fault_ports;
  uint32_t detector_ports;
  uint32_t observable_ports;
  uint32_t success_ports;
  uint64_t blocks;
} cudaq_edem_shape;

/* Semantic construction input. Arrays have the counts in shape. Code blocks
 * name presentation IDs; V/S widths are generators, L widths are 2*k. A block
 * names semantic endpoints, so producers never assemble flattened offsets or
 * port-binding tables. Omitted blocks are zero; duplicate endpoint pairs are
 * rejected. Blocks and presentations may be supplied in any order. Packing
 * orders them canonically, removes zero blocks and interns equal matrices.
 * These borrowed arrays and output storage must be disjoint. */
typedef struct cudaq_edem_description {
  cudaq_edem_shape shape;
  const cudaq_edem_code_presentation *codes;
  const uint32_t *input_codes;
  const uint32_t *output_codes;
  const cudaq_edem_fault_distribution *faults;
  const uint32_t *detector_sizes;
  const uint32_t *observable_sizes;
  const uint32_t *success_sizes;
  const cudaq_edem_block *blocks;
} cudaq_edem_description;

/* Trusted borrowed EDEM token, initialized only by successful EDEM construct,
 * read, or check in this ABI. The backing bytes must remain alive and
 * unchanged. Getters assume that provenance and do not make fabricated/stale
 * tokens safe. The distinct type prevents accidentally passing an unchecked GF2
 * map to EDEM getters. Generic inspection and arithmetic use the .map member
 * directly. Only packed bytes and their storage ownership cross packages. */
typedef struct cudaq_edem_view {
  cudaq_gf2_map_view map;
} cudaq_edem_view;

uint32_t cudaq_edem_abi_version(void);
const char *cudaq_edem_status_string(cudaq_edem_status);
/* One-call setup. NULL memory selects the heap; explicit resources never fall
 * back. Validation uses scratch only; construction also allocates through the
 * result resource and copies its deallocator/context into the owner. No options
 * pointer or scratch survives the call. Failure preserves owner/view outputs,
 * not arena consumption. Construction requires a zero-initialized empty owner.
 * Construction orders inputs F then code bundles; outputs D,O,Success then
 * code bundles. Code wires are V,S,L. */
cudaq_edem_status cudaq_edem_validate(const cudaq_edem_description *description,
                                      const cudaq_gf2_memory *memory);
cudaq_edem_status
cudaq_edem_construct(const cudaq_edem_description *description,
                     const cudaq_gf2_memory *memory,
                     cudaq_gf2_owned_buffer *owner, cudaq_edem_view *result);

/* Safe zero-copy loading validates GF2 and EDEM before publishing a view.
 * The exact-sized, eight-byte-aligned input stays borrowed and immutable.
 * Temporary scratch uses memory->scratch and is released before return.
 * Input and result must be disjoint. Failure preserves the result. */
cudaq_edem_status cudaq_edem_read(const void *buffer, uint64_t bytes,
                                  const cudaq_gf2_memory *memory,
                                  cudaq_edem_view *result);

/* Advanced path for an already validated GF2 map: check its EDEM sections
 * and publish a checked EDEM view. Check results are unchanged on failure.
 * EDEM maps flips; affine measurement offsets belong to separate contracts. */
cudaq_gf2_schema cudaq_edem_schema(void);
cudaq_edem_status cudaq_edem_check(const cudaq_gf2_map_view *value,
                                   const cudaq_gf2_memory *memory,
                                   cudaq_edem_view *result);
cudaq_edem_status cudaq_edem_get_shape(const cudaq_edem_view *value,
                                       cudaq_edem_shape *shape);
/* Presentation indices enumerate the retained, name-sorted presentation table.
 */
cudaq_edem_status cudaq_edem_get_code(const cudaq_edem_view *value,
                                      uint32_t index,
                                      cudaq_edem_code_presentation *code);
/* Resolve semantic endpoints to GF2 wires, or translate a wire back. */
cudaq_edem_status cudaq_edem_resolve_endpoint(const cudaq_edem_view *value,
                                              uint32_t output,
                                              cudaq_edem_endpoint endpoint,
                                              cudaq_gf2_wire *wire);
cudaq_edem_status cudaq_edem_describe_wire(const cudaq_edem_view *value,
                                           uint32_t output, cudaq_gf2_wire wire,
                                           cudaq_edem_endpoint *endpoint);
/* Distributions are addressed by their input bundle, not by type ID. */
cudaq_edem_status cudaq_edem_get_fault(const cudaq_edem_view *value,
                                       uint32_t input_bundle, uint32_t *size,
                                       uint32_t *groups);
cudaq_edem_status cudaq_edem_get_group(const cudaq_edem_view *value,
                                       uint32_t input_bundle, uint32_t group,
                                       cudaq_edem_fault_group *result);
#ifdef __cplusplus
}
#endif
#endif
