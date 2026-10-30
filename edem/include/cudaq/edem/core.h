/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#pragma once

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

/* Statuses are returned by value only, so they stay a distinct enum type.
 * Values 0-5 equal the cudaq_gf2_status values. MAX_ENUM forces a 32-bit enum,
 * including under `-fshort-enums`; it is never returned. */
typedef enum cudaq_edem_status {
  CUDAQ_EDEM_OK = 0,
  CUDAQ_EDEM_INVALID_ARGUMENT = 1,
  CUDAQ_EDEM_INVALID_CONTENT = 2,
  CUDAQ_EDEM_INSUFFICIENT_CAPACITY = 3,
  CUDAQ_EDEM_UNSUPPORTED_VERSION = 4,
  CUDAQ_EDEM_ALLOCATION_FAILED = 5,
  CUDAQ_EDEM_UNSUPPORTED_DISTRIBUTION = 6,
  CUDAQ_EDEM_STATUS_MAX_ENUM = 0x7FFFFFFF
} cudaq_edem_status;

/* Values stored in structs are fixed-width so the ABI does not depend on the
 * compiler's enum size. */
typedef uint32_t cudaq_edem_port_kind;
enum {
  CUDAQ_EDEM_FAULT = 0,
  CUDAQ_EDEM_VIRTUAL = 1,
  CUDAQ_EDEM_SYNDROME = 2,
  CUDAQ_EDEM_LOGICAL = 3,
  CUDAQ_EDEM_DETECTOR = 4,
  CUDAQ_EDEM_OBSERVABLE = 5,
  CUDAQ_EDEM_SUCCESS = 6
};

/* Side of the EDEM map: inputs carry F and input code bundles; outputs carry
 * D, O, Success and output code bundles. Values are the GF2 wire directions. */
typedef cudaq_gf2_wire_direction cudaq_edem_endpoint_direction;
enum {
  CUDAQ_EDEM_INPUT = CUDAQ_GF2_INPUT,
  CUDAQ_EDEM_OUTPUT = CUDAQ_GF2_OUTPUT
};

/* An endpoint and an endpoint direction together describe a port. A code
 * bundle is the (V,S,L) port triple of one code on one side. V, S and L
 * indices identify code-bundle ordinals, independently on each side. Other
 * indices identify ports of that kind. Directions are implicit in the
 * containing matrix block: output first, input second. */
typedef struct cudaq_edem_endpoint {
  cudaq_edem_port_kind kind;
  uint32_t index;
} cudaq_edem_endpoint;

/* The name identifies a code presentation and is unique within a code table.
 * Presentations with equal names must have equal counts. */
typedef struct cudaq_edem_code_presentation {
  const char *name;
  uint64_t name_size;
  uint32_t syndrome_count;
  uint32_t virtual_outcome_count;
  uint32_t logical_qubit_count;
} cudaq_edem_code_presentation;

/* Construction input: names are nonempty UTF-8 and unique. */
typedef struct cudaq_edem_code_table {
  uint32_t code_count;
  const cudaq_edem_code_presentation *codes;
} cudaq_edem_code_table;

typedef uint32_t cudaq_edem_fault_group_kind;
enum { CUDAQ_EDEM_INDEPENDENT = 0, CUDAQ_EDEM_EXCLUSIVE = 1 };

/* Sorted original indicator IDs and one probability per ID. All indicators
 * occur in exactly one group. There is at most one Independent group;
 * Exclusive groups have at least two members. Their compensated sum must be
 * <= 1 + CUDAQ_EDEM_EXCLUSIVE_TOTAL_TOLERANCE. Each probability must be finite
 * and within [0, 1], without an individual tolerance.
 * Fault coordinates are original indicator IDs. */
typedef struct cudaq_edem_fault_group {
  cudaq_edem_fault_group_kind kind;
  uint32_t size;
  const uint32_t *fault_ids;
  const double *probabilities;
} cudaq_edem_fault_group;

typedef struct cudaq_edem_fault_distribution {
  uint32_t size;
  uint32_t group_count;
  const cudaq_edem_fault_group *groups;
} cudaq_edem_fault_distribution;

typedef struct cudaq_edem_matrix_block {
  cudaq_edem_endpoint output;
  cudaq_edem_endpoint input;
  cudaq_gf2_csc_view matrix;
} cudaq_edem_matrix_block;

/* Port counts of each kind; code counts are code bundles per side. */
typedef struct cudaq_edem_shape {
  uint32_t input_code_count;
  uint32_t output_code_count;
  uint32_t fault_ports_count;
  uint32_t detector_ports_count;
  uint32_t observable_ports_count;
  uint32_t success_ports_count;
} cudaq_edem_shape;

/* Semantic construction input. Port arrays have the counts in shape. Code
 * bundles name indices into code_table, which construction checks; the packed
 * EDEM retains only the presentations its code bundles use. V widths are
 * virtual outcome counts, S widths are syndrome counts, L widths are 2*k. A
 * block names semantic endpoints, so producers never assemble flattened offsets
 * or port-binding tables. Omitted blocks are zero; duplicate endpoint pairs are
 * rejected. Blocks and presentations may be supplied in any order. Packing
 * orders them canonically, removes zero blocks and interns equal matrices.
 * These borrowed arrays and output storage must be disjoint. */
typedef struct cudaq_edem_description {
  cudaq_edem_shape shape;
  cudaq_edem_code_table code_table;
  const uint32_t *input_code_indices;
  const uint32_t *output_code_indices;
  const cudaq_edem_fault_distribution *fault_ports_distributions;
  const uint32_t *detector_ports_sizes;
  const uint32_t *observable_ports_sizes;
  const uint32_t *success_ports_sizes;
  uint64_t block_count;
  const cudaq_edem_matrix_block *blocks;
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
/* Presentation indices enumerate the retained, name-sorted presentations used
 * by this EDEM's code bundles. The count is of presentations, not of code
 * bundles (see shape); the matrix block count is in the GF2 map shape. */
cudaq_edem_status cudaq_edem_get_code_count(const cudaq_edem_view *value,
                                            uint32_t *count);
cudaq_edem_status cudaq_edem_get_code(const cudaq_edem_view *value,
                                      uint32_t index,
                                      cudaq_edem_code_presentation *code);
/* Resolve semantic endpoints to GF2 wires, or translate a wire back. */
cudaq_edem_status
cudaq_edem_resolve_endpoint(const cudaq_edem_view *value,
                            cudaq_edem_endpoint_direction direction,
                            cudaq_edem_endpoint endpoint, cudaq_gf2_wire *wire);
cudaq_edem_status
cudaq_edem_describe_wire(const cudaq_edem_view *value,
                         cudaq_edem_endpoint_direction direction,
                         cudaq_gf2_wire wire, cudaq_edem_endpoint *endpoint);
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
