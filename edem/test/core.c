/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#include "cudaq/edem/core.h"
#include "cudaq/edem/format.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);          \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define OK(call) CHECK((int)(call) == 0)
#define INVALID(call) CHECK((call) == CUDAQ_EDEM_INVALID_CONTENT)

static void test_open_round(void) {
  const cudaq_edem_code_presentation codes[] = {{"repetition-3", 12, 2, 1}};
  const uint32_t boundary[] = {0}, sizes[] = {2}, ids[] = {0, 1};
  const double probabilities[] = {0.01, 0.02};
  const cudaq_edem_fault_group group = {CUDAQ_EDEM_INDEPENDENT, 2, ids,
                                        probabilities};
  const cudaq_edem_fault_distribution fault = {2, 1, &group};
  const uint64_t identity_offsets[] = {0, 1, 2},
                 measurement_offsets[] = {0, 1, 1}, data_offsets[] = {0, 0, 1},
                 zero_offsets[] = {0, 0, 0};
  const uint32_t identity_rows[] = {0, 1}, row[] = {0};
  const cudaq_gf2_csc_view identity = {2, 2, 2, identity_offsets,
                                       identity_rows};
  const cudaq_gf2_csc_view measurement = {2, 2, 1, measurement_offsets, row};
  const cudaq_gf2_csc_view data = {2, 2, 1, data_offsets, row};
  const cudaq_gf2_csc_view zero = {2, 2, 0, zero_offsets, NULL};
  cudaq_edem_block blocks[] = {
      {{CUDAQ_EDEM_LOGICAL, 0}, {CUDAQ_EDEM_LOGICAL, 0}, identity},
      {{CUDAQ_EDEM_SYNDROME, 0}, {CUDAQ_EDEM_FAULT, 0}, data},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_FAULT, 0}, measurement},
      {{CUDAQ_EDEM_VIRTUAL, 0}, {CUDAQ_EDEM_SYNDROME, 0}, identity},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_SYNDROME, 0}, identity},
      {{CUDAQ_EDEM_SYNDROME, 0}, {CUDAQ_EDEM_SYNDROME, 0}, identity},
      {{CUDAQ_EDEM_VIRTUAL, 0}, {CUDAQ_EDEM_FAULT, 0}, measurement},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_VIRTUAL, 0}, identity},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_LOGICAL, 0}, zero},
  };
  cudaq_edem_description d = {{1, 1, 1, 1, 1, 0, 0, 9},
                              codes,
                              boundary,
                              boundary,
                              &fault,
                              sizes,
                              NULL,
                              NULL,
                              blocks};
  cudaq_gf2_owned_buffer owner = {0}, reordered = {0};
  CHECK(cudaq_edem_abi_version() == CUDAQ_EDEM_ABI_VERSION);
  OK(cudaq_edem_validate(&d, NULL));
  cudaq_edem_view value;
  OK(cudaq_edem_construct(&d, NULL, &owner, &value));
  uint64_t bytes = owner.size;
  const uint64_t *packed = owner.data;
  uint64_t *copy = malloc((size_t)bytes);
  CHECK(copy && bytes % 8 == 0 && value.map.data == owner.data);
  memcpy(copy, packed, (size_t)bytes);
  cudaq_edem_view relocated;
  OK(cudaq_edem_read(copy, bytes, NULL, &relocated));
  cudaq_gf2_map_shape shape;
  OK(cudaq_gf2_map_get_shape(&relocated.map, &shape));
  CHECK(shape.block_count == 8 && shape.input_bundle_count == 2 &&
        shape.output_bundle_count == 2);
  cudaq_edem_code_presentation code;
  OK(cudaq_edem_get_code(&relocated, 0, &code));
  CHECK(code.generators == 2 && code.logical_qubits == 1 &&
        code.name_size == 12);
  uint32_t width;
  cudaq_gf2_bundle bundle;
  cudaq_gf2_wire wire;
  uint64_t offset, inputs, outputs;
  OK(cudaq_gf2_map_get_bundle(&relocated.map, 1, 1, &bundle));
  CHECK(bundle.type == 0);
  OK(cudaq_edem_resolve_endpoint(
      &relocated, 0, (cudaq_edem_endpoint){CUDAQ_EDEM_LOGICAL, 0}, &wire));
  OK(cudaq_gf2_map_get_wire(&relocated.map, 0, wire, &width, &offset));
  CHECK(width == 2 && offset == 6);
  OK(cudaq_gf2_map_vector_sizes(&relocated.map, &inputs, &outputs));
  CHECK(inputs == 8 && outputs == 8);
  /* Exhaust every assignment, including combined faults and nonzero seams. */
  for (uint32_t assignment = 0; assignment < 256; ++assignment) {
    uint8_t x[8], y[8];
    for (uint32_t i = 0; i < 8; ++i)
      x[i] = (assignment >> i) & 1;
    uint8_t expected[] = {x[2] ^ x[4] ^ x[0], x[3] ^ x[5], x[4] ^ x[0], x[5],
                          x[4] ^ x[1],        x[5],        x[6],        x[7]};
    OK(cudaq_gf2_map_apply(&relocated.map, x, 8, y, 8));
    CHECK(!memcmp(y, expected, 8));
  }
  cudaq_gf2_map_block a, b;
  OK(cudaq_gf2_map_get_block(&relocated.map, 0, &a));
  CHECK(a.output.bundle == 0 && a.output.wire == 0 && a.input.bundle == 0 &&
        a.input.wire == 0);
  OK(cudaq_gf2_map_get_block(&relocated.map, 1, &a));
  OK(cudaq_gf2_map_get_block(&relocated.map, 2, &b));
  CHECK(a.matrix.column_offsets ==
        b.matrix.column_offsets); /* shared identity */
  cudaq_edem_fault_group read_group;
  OK(cudaq_edem_get_group(&relocated, 0, 0, &read_group));
  CHECK(read_group.kind == CUDAQ_EDEM_INDEPENDENT &&
        read_group.probabilities[1] == 0.02);
  /* Construction order does not affect canonical bytes. */
  for (uint32_t i = 0; i < 4; ++i) {
    cudaq_edem_block tmp = blocks[i];
    blocks[i] = blocks[8 - i];
    blocks[8 - i] = tmp;
  }
  OK(cudaq_edem_construct(&d, NULL, &reordered, &value));
  CHECK(reordered.size == bytes &&
        !memcmp(reordered.data, packed, (size_t)bytes));
  cudaq_gf2_owned_buffer_release(&reordered);
  value = (cudaq_edem_view){{packed, bytes}};
  /* Publication is atomic on invalid content. */
  cudaq_edem_block saved = blocks[0];
  blocks[0] = blocks[1];
  INVALID(cudaq_edem_construct(&d, NULL, &reordered, &value));
  CHECK(!reordered.data && value.map.data == packed && value.map.size == bytes);
  blocks[0] = saved;
  blocks[0] = (cudaq_edem_block){
      {CUDAQ_EDEM_SYNDROME, 0}, {CUDAQ_EDEM_VIRTUAL, 0}, identity};
  INVALID(cudaq_edem_validate(&d, NULL));
  blocks[0] = saved;
  /* Trust-boundary checks, including every truncated input length. */
  for (uint64_t n = 0; n < bytes; ++n)
    CHECK(cudaq_edem_read(packed, n, NULL, &value) != CUDAQ_EDEM_OK);
  CHECK(cudaq_edem_read((uint8_t *)packed + 1, bytes - 1, NULL, &value) ==
        CUDAQ_EDEM_INVALID_ARGUMENT);
  cudaq_gf2_packed_header *h = (cudaq_gf2_packed_header *)copy;
  cudaq_gf2_packed_section *directory =
      (cudaq_gf2_packed_section *)((uint8_t *)copy + h->directory_offset);
  h->version++;
  CHECK(cudaq_edem_read(copy, bytes, NULL, &value) ==
        CUDAQ_EDEM_UNSUPPORTED_VERSION);
  memcpy(copy, packed, (size_t)bytes);
  directory[0].offset = UINT64_MAX;
  INVALID(cudaq_edem_read(copy, bytes, NULL, &value));
  memcpy(copy, packed, (size_t)bytes);
  cudaq_gf2_packed_matrix *matrices =
      (cudaq_gf2_packed_matrix *)((uint8_t *)copy + directory[4].offset);
  matrices[0].nonzeros = UINT64_MAX;
  INVALID(cudaq_edem_read(copy, bytes, NULL, &value));
  memcpy(copy, packed, (size_t)bytes);
  cudaq_gf2_packed_block *packed_blocks =
      (cudaq_gf2_packed_block *)((uint8_t *)copy + directory[3].offset);
  packed_blocks[0].matrix = UINT32_MAX;
  INVALID(cudaq_edem_read(copy, bytes, NULL, &value));
  /* Exercise the reader's nested length/offset checks under sanitizers. A
   * mutation may describe another valid value; every accepted value must be
   * evaluable, and rejected values must leave the previous view untouched. */
  for (uint64_t i = 0; i < bytes; ++i) {
    memcpy(copy, packed, (size_t)bytes);
    ((uint8_t *)copy)[i] ^= 0xff;
    cudaq_edem_view candidate = {{packed, bytes}};
    if (cudaq_edem_read(copy, bytes, NULL, &candidate) == CUDAQ_EDEM_OK) {
      uint8_t x[8] = {0}, y[8];
      uint64_t in_size, out_size;
      OK(cudaq_gf2_map_vector_sizes(&candidate.map, &in_size, &out_size));
      if (in_size == 8 && out_size == 8)
        OK(cudaq_gf2_map_apply(&candidate.map, x, 8, y, 8));
    } else
      CHECK(candidate.map.data == packed && candidate.map.size == bytes);
  }
  free(copy);
  cudaq_gf2_owned_buffer_release(&owner);
}

static void test_distributions_and_names(void) {
  uint32_t ids[] = {0, 1}, boundary = 0;
  double probabilities[] = {0.25, 0.75};
  cudaq_edem_fault_group g = {CUDAQ_EDEM_EXCLUSIVE, 2, ids, probabilities};
  cudaq_edem_fault_distribution f = {2, 1, &g};
  cudaq_edem_code_presentation codes[] = {{"z", 1, 0, 1}, {"a", 1, 0, 1}};
  cudaq_edem_description d = {{2, 1, 0, 1, 0, 0, 0, 0},
                              codes,
                              &boundary,
                              NULL,
                              &f,
                              NULL,
                              NULL,
                              NULL,
                              NULL};
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_edem_view value;
  OK(cudaq_edem_construct(&d, NULL, &owner, &value));
  OK(cudaq_edem_read(owner.data, owner.size, NULL, &value));
  cudaq_gf2_bundle bundle;
  OK(cudaq_gf2_map_get_bundle(&value.map, 0, 1, &bundle));
  CHECK(bundle.type == 1); /* sorted names remap presentation IDs */
  probabilities[1] = 0.76;
  INVALID(cudaq_edem_validate(&d, NULL));
  probabilities[1] = NAN;
  INVALID(cudaq_edem_validate(&d, NULL));
  probabilities[1] = 0.75;
  ids[1] = 0;
  INVALID(cudaq_edem_validate(&d, NULL));
  ids[1] = 1;
  codes[0].name = "a";
  INVALID(cudaq_edem_validate(&d, NULL));
  codes[0].name = "\xff";
  INVALID(cudaq_edem_validate(&d, NULL));
  cudaq_gf2_owned_buffer_release(&owner);
  /* Empty EDEMs still retain their vocabulary and required sections. */
  cudaq_edem_description empty = {0};
  OK(cudaq_edem_validate(&empty, NULL));
  OK(cudaq_edem_construct(&empty, NULL, &owner, &value));
  OK(cudaq_edem_read(owner.data, owner.size, NULL, &value));
  OK(cudaq_gf2_map_apply(&value.map, NULL, 0, NULL, 0));
  cudaq_gf2_owned_buffer_release(&owner);
}

static void test_domain_checks(void) {
  cudaq_edem_code_presentation code = {"test", 4, 1, 1};
  uint32_t boundary = 0, id = 0, row = 0;
  double probability = 0.1;
  uint64_t offsets[] = {0, 1};
  cudaq_edem_fault_group group = {CUDAQ_EDEM_INDEPENDENT, 1, &id, &probability};
  cudaq_edem_fault_distribution fault = {1, 1, &group};
  cudaq_edem_block block = {{CUDAQ_EDEM_SYNDROME, 0},
                            {CUDAQ_EDEM_FAULT, 0},
                            {1, 1, 1, offsets, &row}};
  cudaq_edem_description d = {{1, 1, 1, 1, 0, 0, 0, 1},
                              &code,
                              &boundary,
                              &boundary,
                              &fault,
                              NULL,
                              NULL,
                              NULL,
                              &block};
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_edem_view value;
  OK(cudaq_edem_construct(&d, NULL, &owner, &value));
  uint64_t bytes = owner.size;
  const uint8_t *packed = owner.data;
  uint8_t *copy = malloc((size_t)bytes);
  CHECK(copy);
  cudaq_gf2_schema schema = cudaq_edem_schema();
  cudaq_gf2_section cs, fs, bs, ts;
  OK(cudaq_gf2_map_get_section(&value.map, CUDAQ_EDEM_SECTION_PRESENTATIONS,
                               &cs));
  OK(cudaq_gf2_map_get_section(&value.map, CUDAQ_EDEM_SECTION_DISTRIBUTIONS,
                               &fs));
  OK(cudaq_gf2_map_get_section(&value.map, CUDAQ_GF2_SECTION_BLOCKS, &bs));
  OK(cudaq_gf2_map_get_section(&value.map, CUDAQ_GF2_SECTION_TYPES, &ts));
  uint64_t co = (const uint8_t *)cs.data - packed;
  uint64_t fo = (const uint8_t *)fs.data - packed;
  uint64_t bo = (const uint8_t *)bs.data - packed;
  uint64_t to = (const uint8_t *)ts.data - packed;
  cudaq_gf2_map_view unchecked;
  /* Every change below is a valid GF2 map but violates EDEM semantics. */
  for (unsigned trial = 0; trial < 5; ++trial) {
    memcpy(copy, packed, (size_t)bytes);
    if (trial == 0)
      ((cudaq_edem_packed_code *)(copy + co))->generators = 2;
    if (trial == 1)
      ((cudaq_gf2_packed_type *)(copy + to))->role = CUDAQ_GF2_FREE;
    if (trial == 2)
      ((cudaq_gf2_packed_block *)(copy + bo))->input_bundle = 1;
    if (trial == 3)
      ((cudaq_edem_packed_fault *)(copy + fo))->bundle = 1;
    if (trial == 4) {
      cudaq_edem_packed_fault *f = (cudaq_edem_packed_fault *)(copy + fo);
      cudaq_edem_packed_group *g =
          (cudaq_edem_packed_group *)(copy + fo + f->groups);
      *(double *)(copy + fo + g->probabilities) = -0.0;
    }
    OK(cudaq_gf2_map_read(copy, bytes, &schema, NULL, &unchecked));
    INVALID(cudaq_edem_check(&unchecked, NULL, &value));
    CHECK(value.map.data == packed && value.map.size == bytes);
    INVALID(cudaq_edem_read(copy, bytes, NULL, &value));
    CHECK(value.map.data == packed && value.map.size == bytes);
  }
  memcpy(copy, packed, (size_t)bytes);
  ((cudaq_edem_packed_fault *)(copy + fo))->groups = UINT64_MAX;
  OK(cudaq_gf2_map_read(copy, bytes, &schema, NULL, &unchecked));
  INVALID(cudaq_edem_check(&unchecked, NULL, &value));
  INVALID(cudaq_edem_read(copy, bytes, NULL, &value));
  CHECK(value.map.data == packed && value.map.size == bytes);
  free(copy);
  cudaq_gf2_owned_buffer_release(&owner);
}

static void test_safe_read(void) {
  /* Semantic coverage needs its own scratch even when GF2's metadata is tiny.
   */
  const uint32_t n = 2048;
  uint32_t *ids = malloc(n * sizeof(*ids));
  double *probabilities = malloc(n * sizeof(*probabilities));
  CHECK(ids && probabilities);
  for (uint32_t i = 0; i < n; ++i) {
    ids[i] = i;
    probabilities[i] = 0.001;
  }
  cudaq_edem_fault_group group = {CUDAQ_EDEM_INDEPENDENT, n, ids,
                                  probabilities};
  cudaq_edem_fault_distribution fault = {n, 1, &group};
  cudaq_edem_description d = {0};
  d.shape.fault_ports = 1;
  d.faults = &fault;
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_edem_view original, result, from_map;
  OK(cudaq_edem_construct(&d, NULL, &owner, &original));
  OK(cudaq_edem_read(owner.data, owner.size, NULL, &result));
  CHECK(result.map.data == owner.data && result.map.size == owner.size);
  uint32_t size, groups;
  OK(cudaq_edem_get_fault(&result, 0, &size, &groups));
  CHECK(size == n && groups == 1);
  OK(cudaq_edem_check(&result.map, NULL, &from_map));
  CHECK(from_map.map.data == owner.data && from_map.map.size == owner.size);
  CHECK(cudaq_edem_read(owner.data, owner.size, NULL, NULL) ==
        CUDAQ_EDEM_INVALID_ARGUMENT);
  cudaq_gf2_owned_buffer_release(&owner);
  free(probabilities);
  free(ids);

  /* A valid bare GF2 map cannot publish an EDEM token. */
  cudaq_gf2_map_description bare_description = {0};
  cudaq_gf2_map_view bare_map;
  OK(cudaq_gf2_map_construct(&bare_description, NULL, &owner, &bare_map));
  result = (cudaq_edem_view){{NULL, 0}};
  INVALID(cudaq_edem_read(owner.data, owner.size, NULL, &result));
  CHECK(!result.map.data && !result.map.size);
  cudaq_gf2_owned_buffer_release(&owner);
}

static void probability_case(const double *probabilities, uint32_t count,
                             uint32_t kind, cudaq_edem_status expected) {
  uint32_t ids[9];
  double zeros[9] = {0};
  CHECK(count <= 9);
  for (uint32_t i = 0; i < count; ++i)
    ids[i] = i;
  cudaq_edem_fault_group group = {kind, count, ids, probabilities};
  cudaq_edem_fault_distribution fault = {count, 1, &group};
  cudaq_edem_description description = {{0, 0, 0, 1, 0, 0, 0, 0},
                                        NULL,
                                        NULL,
                                        NULL,
                                        &fault,
                                        NULL,
                                        NULL,
                                        NULL,
                                        NULL};
  CHECK(cudaq_edem_validate(&description, NULL) == expected);
  cudaq_gf2_owned_buffer constructed = {0};
  cudaq_edem_view value = {{NULL, 0}};
  CHECK(cudaq_edem_construct(&description, NULL, &constructed, &value) ==
        expected);
  if (expected != CUDAQ_EDEM_OK)
    CHECK(!constructed.data && !value.map.data);

  /* Change only probabilities in a canonical zero-valued packed distribution.
   * This tests the reader independently of construction's validation. */
  group.probabilities = zeros;
  cudaq_gf2_owned_buffer packed = {0};
  OK(cudaq_edem_construct(&description, NULL, &packed, &value));
  cudaq_edem_fault_group stored;
  OK(cudaq_edem_get_group(&value, 0, 0, &stored));
  memcpy((void *)stored.probabilities, probabilities, count * sizeof(double));
  value = (cudaq_edem_view){{NULL, 0}};
  CHECK(cudaq_edem_read(packed.data, packed.size, NULL, &value) == expected);
  if (expected == CUDAQ_EDEM_OK) {
    OK(cudaq_edem_get_group(&value, 0, 0, &stored));
    CHECK(memcmp(stored.probabilities, probabilities, count * sizeof(double)) ==
          0);
    CHECK(constructed.size == packed.size);
    CHECK(memcmp(constructed.data, packed.data, packed.size) == 0);
  } else
    CHECK(!value.map.data);
  cudaq_gf2_schema schema = cudaq_edem_schema();
  cudaq_gf2_map_view map;
  OK(cudaq_gf2_map_read(packed.data, packed.size, &schema, NULL, &map));
  cudaq_edem_view checked = {{NULL, 0}};
  CHECK(cudaq_edem_check(&map, NULL, &checked) == expected);
  if (expected != CUDAQ_EDEM_OK)
    CHECK(!checked.map.data);
  cudaq_gf2_owned_buffer_release(&constructed);
  cudaq_gf2_owned_buffer_release(&packed);
}

static void test_probability_validation(void) {
  double probabilities[9];
  for (uint32_t i = 0; i < 9; ++i)
    probabilities[i] = 1.0 / 9.0;
  probability_case(probabilities, 9, CUDAQ_EDEM_EXCLUSIVE, CUDAQ_EDEM_OK);

  double limit = 1.0 + CUDAQ_EDEM_EXCLUSIVE_TOTAL_TOLERANCE;
  const double totals[] = {nextafter(limit, 0), limit,
                           nextafter(limit, INFINITY)};
  for (uint32_t i = 0; i < 3; ++i) {
    const double pair[] = {0.5, totals[i] - 0.5};
    probability_case(pair, 2, CUDAQ_EDEM_EXCLUSIVE,
                     i == 2 ? CUDAQ_EDEM_INVALID_CONTENT : CUDAQ_EDEM_OK);
  }

  /* Naive summation loses these terms and would accept a total over the limit.
   */
  probabilities[0] = 0.5;
  probabilities[1] = limit - 0.5;
  for (uint32_t i = 2; i < 9; ++i)
    probabilities[i] = 3e-17;
  probability_case(probabilities, 9, CUDAQ_EDEM_EXCLUSIVE,
                   CUDAQ_EDEM_INVALID_CONTENT);
  /* Repeated rounding up would instead reject this admissible total. */
  double below = limit;
  for (uint32_t i = 0; i < 4; ++i)
    below = nextafter(below, 0);
  probabilities[1] = below - 0.5;
  for (uint32_t i = 2; i < 9; ++i)
    probabilities[i] = 1.2e-16;
  probability_case(probabilities, 9, CUDAQ_EDEM_EXCLUSIVE, CUDAQ_EDEM_OK);

  const double invalid[] = {-1e-12, 1 + 1e-12, NAN, INFINITY, -INFINITY};
  for (uint32_t i = 0; i < sizeof(invalid) / sizeof(*invalid); ++i) {
    const double pair[] = {0.25, invalid[i]};
    probability_case(pair, 2, CUDAQ_EDEM_EXCLUSIVE, CUDAQ_EDEM_INVALID_CONTENT);
  }
  for (uint32_t i = 0; i < 9; ++i)
    probabilities[i] = 0.9;
  probability_case(probabilities, 9, CUDAQ_EDEM_INDEPENDENT, CUDAQ_EDEM_OK);
}

int main(void) {
  test_open_round();
  test_distributions_and_names();
  test_probability_validation();
  test_domain_checks();
  test_safe_read();
  return 0;
}
