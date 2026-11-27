/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

/* This executable deliberately links GF2 alone. */
#include "cudaq/gf2/map.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c)                                                               \
  do {                                                                         \
    if (!(c)) {                                                                \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c);                  \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)
#define OK(c) CHECK((c) == CUDAQ_GF2_OK)
static uint8_t *construct(const cudaq_gf2_map_description *d,
                          cudaq_gf2_map_view *v,
                          cudaq_gf2_owned_buffer *owner) {
  OK(cudaq_gf2_map_construct(d, NULL, owner, v));
  CHECK(v->data == owner->data && v->size == owner->size);
  return (uint8_t *)owner->data;
}
static cudaq_gf2_status read_map(const void *p, uint64_t n,
                                 const cudaq_gf2_schema *schema,
                                 cudaq_gf2_map_view *v) {
  return cudaq_gf2_map_read(p, n, schema, NULL, v);
}
static cudaq_gf2_status validate(const cudaq_gf2_map_description *d) {
  return cudaq_gf2_map_validate(d, NULL);
}
static cudaq_gf2_packed_section *directory(uint8_t *p) {
  return (cudaq_gf2_packed_section *)(p + ((cudaq_gf2_packed_header *)p)
                                              ->directory_offset);
}
static void test_empty(void) {
  cudaq_gf2_map_description d = {0};
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_gf2_map_view v;
  uint8_t *data = construct(&d, &v, &owner);
  CHECK(owner.size == 64 + 5 * 32);
  OK(cudaq_gf2_map_read(data, owner.size, NULL, NULL, &v));
  OK(cudaq_gf2_map_apply(&v, NULL, 0, NULL, 0));
  cudaq_gf2_map_shape shape;
  OK(cudaq_gf2_map_get_shape(&v, &shape));
  CHECK(shape.type_count == 0 && shape.section_count == 0);
  cudaq_gf2_owned_buffer_release(&owner);
}
static void test_maps(void) {
  cudaq_gf2_bundle_type types[] = {{"z/iface", 7, CUDAQ_GF2_INTERFACE, 3},
                                   {"a/free", 6, CUDAQ_GF2_FREE, 1}};
  uint32_t one = 1, widths[] = {1, 2, 0};
  cudaq_gf2_bundle inputs[] = {{1, 0, &one}, {0, 0, widths}, {1, 0, &one}},
                   outputs[] = {{0, 0, widths}, {1, 0, &one}};
  uint64_t identity_offsets[] = {0, 1, 2}, zero_offsets[] = {0, 0};
  uint32_t rows[] = {0, 1};
  cudaq_gf2_csc_view identity = {1, 1, 1, identity_offsets, rows},
                     identity2 = {2, 2, 2, identity_offsets, rows},
                     zero = {2, 1, 0, zero_offsets, NULL};
  cudaq_gf2_map_matrix_block blocks[] = {{{1, 0}, {2, 0}, identity},
                                         {{0, 1}, {1, 1}, identity2},
                                         {{0, 0}, {1, 0}, identity},
                                         {{0, 0}, {0, 0}, identity},
                                         {{0, 1}, {0, 0}, zero}};
  cudaq_gf2_map_description d = {2,     3,      2,       0,      5,
                                 types, inputs, outputs, blocks, NULL};
  cudaq_gf2_map_view v;
  cudaq_gf2_owned_buffer owner = {0}, copy_owner = {0};
  uint8_t *data = construct(&d, &v, &owner);
  OK(read_map(data, v.size, NULL, &v));
  cudaq_gf2_map_shape shape;
  OK(cudaq_gf2_map_get_shape(&v, &shape));
  CHECK(shape.block_count == 4 && shape.matrix_count == 2 &&
        shape.input_bundle_count == 3);
  cudaq_gf2_bundle_type type;
  OK(cudaq_gf2_map_get_type(&v, 0, &type));
  CHECK(type.name_size == 6 && !memcmp(type.name, "a/free", 6));
  cudaq_gf2_bundle bundle;
  OK(cudaq_gf2_map_get_bundle(&v, CUDAQ_GF2_INPUT, 1, &bundle));
  CHECK(bundle.type == 1);
  uint32_t width;
  uint64_t offset, ni, no;
  OK(cudaq_gf2_map_get_wire(&v, CUDAQ_GF2_INPUT, (cudaq_gf2_wire){1, 2}, &width,
                            &offset));
  CHECK(width == 0 && offset == 4);
  OK(cudaq_gf2_map_get_wire(&v, CUDAQ_GF2_INPUT, (cudaq_gf2_wire){2, 0}, &width,
                            &offset));
  CHECK(width == 1 && offset == 4);
  OK(cudaq_gf2_map_vector_sizes(&v, &ni, &no));
  CHECK(ni == 5 && no == 4);
  for (unsigned bits = 0; bits < 32; ++bits) {
    uint8_t x[5], y[4];
    for (unsigned j = 0; j < 5; ++j)
      x[j] = (bits >> j) & 1;
    uint8_t expected[] = {x[0] ^ x[1], x[2], x[3], x[4]};
    OK(cudaq_gf2_map_apply(&v, x, 5, y, 4));
    CHECK(!memcmp(y, expected, 4));
  }
  uint8_t bad[] = {0, 0, 2, 0, 0}, unchanged[] = {7, 7, 7, 7};
  CHECK(cudaq_gf2_map_apply(&v, bad, 5, unchanged, 4) ==
        CUDAQ_GF2_INVALID_ARGUMENT);
  CHECK(unchanged[0] == 7 && unchanged[3] == 7);
  cudaq_gf2_map_matrix_block a, b;
  OK(cudaq_gf2_map_get_matrix_block(&v, 0, &a));
  OK(cudaq_gf2_map_get_matrix_block(&v, 1, &b));
  CHECK(a.matrix.column_offsets == b.matrix.column_offsets);
  /* Remapping local type IDs and presentation order preserves the value. */
  cudaq_gf2_bundle_type temp = types[0];
  types[0] = types[1];
  types[1] = temp;
  for (unsigned i = 0; i < 3; ++i)
    inputs[i].type = 1 - inputs[i].type;
  for (unsigned i = 0; i < 2; ++i)
    outputs[i].type = 1 - outputs[i].type;
  for (unsigned i = 0; i < 2; ++i) {
    cudaq_gf2_map_matrix_block t = blocks[i];
    blocks[i] = blocks[4 - i];
    blocks[4 - i] = t;
  }
  cudaq_gf2_map_view other;
  uint8_t *copy = construct(&d, &other, &copy_owner);
  CHECK(v.size == other.size && !memcmp(data, copy, (size_t)v.size));
  cudaq_gf2_owned_buffer_release(&copy_owner);
  /* Duplicate names are invalid even if their roles differ. */
  temp = types[1];
  types[1].name = types[0].name;
  types[1].name_size = types[0].name_size;
  CHECK(validate(&d) == CUDAQ_GF2_INVALID_CONTENT);
  types[1] = temp;
  types[0].role = 2;
  CHECK(validate(&d) == CUDAQ_GF2_INVALID_CONTENT);
  types[0].role = CUDAQ_GF2_FREE;
  outputs[0].reserved = 1;
  CHECK(validate(&d) == CUDAQ_GF2_INVALID_CONTENT);
  outputs[0].reserved = 0;
  widths[1] = 3;
  CHECK(validate(&d) == CUDAQ_GF2_INVALID_CONTENT);
  widths[1] = 2;
  cudaq_gf2_map_matrix_block saved = blocks[0];
  blocks[0] = blocks[1];
  CHECK(validate(&d) == CUDAQ_GF2_INVALID_CONTENT);
  blocks[0] = saved;
  saved = blocks[0];
  blocks[0].input.wire = UINT32_MAX;
  CHECK(validate(&d) == CUDAQ_GF2_INVALID_CONTENT);
  blocks[0] = saved;
  temp = types[0];
  types[0].name = "\xff";
  types[0].name_size = 1;
  CHECK(validate(&d) == CUDAQ_GF2_INVALID_CONTENT);
  types[0] = temp;
  copy = malloc((size_t)v.size);
  CHECK(copy);
  other = v;
  CHECK(cudaq_gf2_map_construct(&d, NULL, NULL, &other) ==
        CUDAQ_GF2_INVALID_ARGUMENT);
  CHECK(other.data == v.data && other.size == v.size);
  /* Each single-byte mutation exercises nested offsets/counts under ASan/UBSan.
   */
  for (uint64_t i = 0; i < v.size; ++i) {
    memcpy(copy, data, (size_t)v.size);
    copy[i] ^= 0xff;
    other = v;
    if (read_map(copy, v.size, NULL, &other) == CUDAQ_GF2_OK) {
      OK(cudaq_gf2_map_vector_sizes(&other, &ni, &no));
      if (ni == 5 && no == 4) {
        uint8_t x[5] = {0}, y[4];
        OK(cudaq_gf2_map_apply(&other, x, 5, y, 4));
      }
    } else
      CHECK(other.data == v.data && other.size == v.size);
  }
  free(copy);
  cudaq_gf2_owned_buffer_release(&owner);
}
static void test_directory(void) {
  cudaq_gf2_bundle_type type = {"opaque", 6, CUDAQ_GF2_FREE, 1};
  uint32_t width = 1;
  cudaq_gf2_bundle bundle = {0, 0, &width};
  cudaq_gf2_map_description d = {1,     1,       0,    0,    0,
                                 &type, &bundle, NULL, NULL, NULL};
  cudaq_gf2_map_view plain, enriched;
  cudaq_gf2_owned_buffer owner = {0}, optional_owner = {0},
                         required_owner = {0};
  uint8_t *data = construct(&d, &plain, &owner);
  uint8_t metadata[] = {7, 8, 9};
  cudaq_gf2_section ext = {CUDAQ_GF2_OPTIONAL_SECTION_MIN + 7,
                           CUDAQ_GF2_SECTION_OPTIONAL, 3, 3, metadata};
  d.section_count = 1;
  d.sections = &ext;
  uint8_t *optional = construct(&d, &enriched, &optional_owner);
  OK(read_map(optional, enriched.size, NULL, &enriched));
  cudaq_gf2_packed_section *a = directory(data), *b = directory(optional);
  for (unsigned i = 0; i < 5; ++i) {
    CHECK(a[i].id == b[i].id && a[i].flags == b[i].flags &&
          a[i].count == b[i].count && a[i].bytes == b[i].bytes &&
          a[i].offset == b[i].offset);
    CHECK(!memcmp(data + a[i].offset, optional + b[i].offset,
                  (size_t)a[i].bytes));
  }
  cudaq_gf2_section found;
  OK(cudaq_gf2_map_get_section(&enriched, ext.id, &found));
  CHECK(found.bytes == 3 && !memcmp(found.data, metadata, 3));
  uint64_t n = enriched.size;
  uint8_t *broken = malloc((size_t)n);
  CHECK(broken);
  for (uint64_t i = 0; i < n; ++i) {
    cudaq_gf2_map_view candidate = plain;
    CHECK(read_map(optional, i, NULL, &candidate) != CUDAQ_GF2_OK);
    CHECK(candidate.data == plain.data);
  }
  cudaq_gf2_map_view candidate = plain;
  CHECK(read_map(optional + 1, n - 1, NULL, &candidate) ==
        CUDAQ_GF2_INVALID_ARGUMENT);
  memcpy(broken, optional, (size_t)n);
  ((cudaq_gf2_packed_header *)broken)->alignment = 64;
  CHECK(read_map(broken, n, NULL, &candidate) == CUDAQ_GF2_INVALID_CONTENT);
  memcpy(broken, optional, (size_t)n);
  ((cudaq_gf2_packed_header *)broken)->directory_offset += 8;
  CHECK(read_map(broken, n, NULL, &candidate) == CUDAQ_GF2_INVALID_CONTENT);
  memcpy(broken, optional, (size_t)n);
  directory(broken)[1].id = directory(broken)[0].id;
  CHECK(read_map(broken, n, NULL, &candidate) == CUDAQ_GF2_INVALID_CONTENT);
  memcpy(broken, optional, (size_t)n);
  directory(broken)[5].offset = directory(broken)[0].offset;
  CHECK(read_map(broken, n, NULL, &candidate) == CUDAQ_GF2_INVALID_CONTENT);
  memcpy(broken, optional, (size_t)n);
  directory(broken)[5].bytes = UINT64_MAX;
  CHECK(read_map(broken, n, NULL, &candidate) == CUDAQ_GF2_INVALID_CONTENT);
  memcpy(broken, optional, (size_t)n);
  directory(broken)[5].flags = CUDAQ_GF2_SECTION_REQUIRED;
  CHECK(read_map(broken, n, NULL, &candidate) == CUDAQ_GF2_INVALID_CONTENT);
  memcpy(broken, optional, (size_t)n);
  broken[directory(broken)[5].offset + 3] = 1;
  CHECK(read_map(broken, n, NULL, &candidate) == CUDAQ_GF2_INVALID_CONTENT);
  ext.id = CUDAQ_GF2_EXTENSION_SECTION_MIN + 9;
  ext.flags = CUDAQ_GF2_SECTION_REQUIRED;
  cudaq_gf2_map_view required;
  uint8_t *required_data = construct(&d, &required, &required_owner);
  CHECK(read_map(required_data, required.size, NULL, &candidate) ==
        CUDAQ_GF2_UNSUPPORTED_VERSION);
  uint32_t id = ext.id;
  cudaq_gf2_schema schema = {&id, 1, 0};
  OK(read_map(required_data, required.size, &schema, &candidate));
  /* Unknown required core sections are unsupported too. */
  directory(required_data)[5].id = 6;
  CHECK(read_map(required_data, required.size, NULL, &candidate) ==
        CUDAQ_GF2_UNSUPPORTED_VERSION);
  ext.flags = CUDAQ_GF2_SECTION_OPTIONAL;
  CHECK(validate(&d) == CUDAQ_GF2_INVALID_CONTENT);
  cudaq_gf2_owned_buffer_release(&required_owner);
  free(broken);
  cudaq_gf2_owned_buffer_release(&optional_owner);
  cudaq_gf2_owned_buffer_release(&owner);
}
int main(void) {
  test_empty();
  test_maps();
  test_directory();
  return 0;
}
