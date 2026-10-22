/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#include "cudaq/edem/noise.h"
#include "../../math/gf2/test/test_allocator.h"
#include "cudaq/edem/format.h"
#include <math.h>

/* Checked readers accept GF2 bundle orders beyond semantic construction's
 * default. All output kinds participate in response equality. */
static void reordered_outputs(void) {
  uint32_t ids[] = {0, 1, 2}, rows[] = {0, 0, 0}, width = 1;
  uint64_t offsets[][4] = {{0, 1, 2, 3}, {0, 0, 1, 1}, {0, 0, 0, 1}};
  double probabilities[] = {.1, .2, .3};
  cudaq_edem_fault_group group = {CUDAQ_EDEM_INDEPENDENT, 3, ids,
                                  probabilities};
  cudaq_edem_fault_distribution fault = {3, 1, &group};
  cudaq_edem_block blocks[3];
  for (uint32_t i = 0; i < 3; ++i)
    blocks[i] = (cudaq_edem_block){{CUDAQ_EDEM_DETECTOR + i, 0},
                                   {CUDAQ_EDEM_FAULT, 0},
                                   {1, 3, offsets[i][3], offsets[i], rows}};
  cudaq_edem_description description = {0};
  description.shape = (cudaq_edem_shape){0, 0, 0, 1, 1, 1, 1, 3};
  description.faults = &fault;
  description.detector_sizes = description.observable_sizes =
      description.success_sizes = &width;
  description.blocks = blocks;
  cudaq_gf2_owned_buffer original = {0}, reordered = {0}, converted = {0},
                         reduced = {0}, mapping = {0};
  cudaq_edem_view source;
  TA_CHECK(cudaq_edem_construct(&description, 0, &original, &source) ==
           CUDAQ_EDEM_OK);
  cudaq_gf2_map_shape shape;
  TA_CHECK(cudaq_gf2_map_get_shape(&source.map, &shape) == CUDAQ_GF2_OK);
  TA_CHECK(shape.type_count == 4 && shape.output_bundle_count == 3 &&
           shape.input_bundle_count == 1 && shape.section_count == 2);
  cudaq_gf2_bundle_type types[4];
  cudaq_gf2_bundle input, outputs[3];
  cudaq_gf2_map_block reordered_blocks[3];
  for (uint32_t i = 0; i < 4; ++i)
    TA_CHECK(cudaq_gf2_map_get_type(&source.map, i, types + i) == CUDAQ_GF2_OK);
  TA_CHECK(cudaq_gf2_map_get_bundle(&source.map, 0, 0, &input) == CUDAQ_GF2_OK);
  for (uint32_t i = 0; i < 3; ++i) {
    TA_CHECK(cudaq_gf2_map_get_bundle(&source.map, 1, 2 - i, outputs + i) ==
             CUDAQ_GF2_OK);
    TA_CHECK(cudaq_gf2_map_get_block(&source.map, i, reordered_blocks + i) ==
             CUDAQ_GF2_OK);
    reordered_blocks[i].output.bundle = 2 - reordered_blocks[i].output.bundle;
  }
  cudaq_gf2_section sections[2];
  TA_CHECK(cudaq_gf2_map_get_section(&source.map,
                                     CUDAQ_EDEM_SECTION_PRESENTATIONS,
                                     sections) == CUDAQ_GF2_OK);
  TA_CHECK(cudaq_gf2_map_get_section(&source.map,
                                     CUDAQ_EDEM_SECTION_DISTRIBUTIONS,
                                     sections + 1) == CUDAQ_GF2_OK);
  cudaq_gf2_map_description packed = {
      4, 1, 3, 2, 3, types, &input, outputs, reordered_blocks, sections};
  cudaq_gf2_map_view map;
  TA_CHECK(cudaq_gf2_map_construct(&packed, 0, &reordered, &map) ==
           CUDAQ_GF2_OK);
  cudaq_edem_dem_view input_view, output_view;
  TA_CHECK(cudaq_edem_read(map.data, map.size, 0, &input_view.edem) ==
           CUDAQ_EDEM_OK);
  TA_CHECK(cudaq_edem_factorize_dem(&input_view, 0, &converted, &output_view) ==
           CUDAQ_EDEM_OK);
  TA_CHECK(converted.size == original.size &&
           !memcmp(converted.data, original.data, (size_t)original.size));
  TA_CHECK(cudaq_edem_coalesce_dem_columns(&input_view, 0, &reduced,
                                           &output_view,
                                           &mapping) == CUDAQ_EDEM_OK);
  const uint64_t *indices = mapping.data;
  for (uint32_t c = 0; c < 3; ++c) {
    uint8_t in[3] = {0}, out[3];
    in[c] = 1;
    TA_CHECK(indices[c] == c);
    TA_CHECK(cudaq_gf2_map_apply(&output_view.edem.map, in, 3, out, 3) ==
             CUDAQ_GF2_OK);
    TA_CHECK(out[0] == 1 && out[1] == (c == 1) && out[2] == (c == 2));
  }
  cudaq_gf2_owned_buffer_release(&original);
  cudaq_gf2_owned_buffer_release(&reordered);
  cudaq_gf2_owned_buffer_release(&converted);
  cudaq_gf2_owned_buffer_release(&reduced);
  cudaq_gf2_owned_buffer_release(&mapping);
}

static void failure_sites(const cudaq_edem_dem_view *source, int coalesce) {
  test_resource resource;
  test_resource_init(&resource);
  cudaq_gf2_memory memory = test_memory(&resource, &resource);
  cudaq_gf2_owned_buffer owner = {0}, mapping = {0};
  cudaq_edem_dem_view result = *source;
  cudaq_edem_status status =
      coalesce ? cudaq_edem_coalesce_dem_columns(source, &memory, &owner,
                                                 &result, &mapping)
               : cudaq_edem_factorize_dem(source, &memory, &owner, &result);
  TA_CHECK(status == CUDAQ_EDEM_OK);
  uint64_t attempts = resource.attempts;
  cudaq_gf2_owned_buffer_release(&owner);
  cudaq_gf2_owned_buffer_release(&mapping);
  test_resource_balanced(&resource);
  for (uint64_t fail = 1; fail <= attempts; ++fail) {
    test_resource_init(&resource);
    resource.fail_on_attempt = fail;
    result = *source;
    status = coalesce
                 ? cudaq_edem_coalesce_dem_columns(source, &memory, &owner,
                                                   &result, &mapping)
                 : cudaq_edem_factorize_dem(source, &memory, &owner, &result);
    TA_CHECK(status == CUDAQ_EDEM_ALLOCATION_FAILED);
    TA_CHECK(test_same_owner(owner, (cudaq_gf2_owned_buffer){0}));
    TA_CHECK(test_same_owner(mapping, (cudaq_gf2_owned_buffer){0}));
    TA_CHECK(result.edem.map.data == source->edem.map.data &&
             result.edem.map.size == source->edem.map.size);
    test_resource_balanced(&resource);
  }
}

static uint64_t channels(uint32_t count, int bounded) {
  uint32_t columns = count * 3;
  uint32_t *ids = malloc(columns * sizeof(*ids));
  double *probabilities = malloc(columns * sizeof(*probabilities));
  cudaq_edem_fault_group *groups = malloc(count * sizeof(*groups));
  uint64_t *offsets = malloc(((size_t)columns + 1) * sizeof(*offsets));
  uint32_t *rows = calloc(count * 2, sizeof(*rows));
  TA_CHECK(ids && probabilities && groups && offsets && rows);
  for (uint32_t i = 0; i < columns; ++i) {
    ids[i] = i;
    probabilities[i] = .001 / 3;
    offsets[i] = (i / 3) * 2 + (i % 3);
  }
  offsets[columns] = count * 2;
  for (uint32_t i = 0; i < count; ++i)
    groups[i] = (cudaq_edem_fault_group){CUDAQ_EDEM_EXCLUSIVE, 3, ids + 3 * i,
                                         probabilities + 3 * i};
  cudaq_edem_fault_distribution fault = {columns, count, groups};
  uint32_t width = 1;
  cudaq_edem_block block = {{CUDAQ_EDEM_DETECTOR, 0},
                            {CUDAQ_EDEM_FAULT, 0},
                            {1, columns, count * 2, offsets, rows}};
  cudaq_edem_description description = {0};
  description.shape.fault_ports = 1;
  description.shape.detector_ports = 1;
  description.shape.blocks = 1;
  description.faults = &fault;
  description.detector_sizes = &width;
  description.blocks = &block;
  cudaq_gf2_owned_buffer original = {0}, independent = {0}, reduced = {0},
                         map = {0};
  cudaq_edem_view value;
  TA_CHECK(cudaq_edem_construct(&description, 0, &original, &value) ==
           CUDAQ_EDEM_OK);
  free(ids);
  free(probabilities);
  free(groups);
  free(offsets);
  free(rows);
  cudaq_edem_dem_view source = {value}, converted, result = {value};
  test_resource scratch, output;
  test_resource_init(&scratch);
  test_resource_init(&output);
  cudaq_gf2_memory memory = test_memory(&scratch, &output);
  TA_CHECK(cudaq_edem_factorize_dem(&source, &memory, &independent,
                                    &converted) == CUDAQ_EDEM_OK);
  uint64_t peak = scratch.peak_bytes + output.peak_bytes;
  if (count == 1) {
    failure_sites(&source, 0);
    failure_sites(&converted, 1);
    TA_CHECK(
        cudaq_edem_coalesce_dem_columns(&source, 0, &reduced, &result, &map) ==
        CUDAQ_EDEM_UNSUPPORTED_DISTRIBUTION);
    TA_CHECK(!reduced.data && !map.data &&
             result.edem.map.data == value.map.data);
    TA_CHECK(cudaq_edem_coalesce_dem_columns(&converted, 0, &reduced, &result,
                                             &reduced) ==
             CUDAQ_EDEM_INVALID_ARGUMENT);
  }
  if (bounded) {
    _Alignas(8) unsigned char scratch_bytes[65536], result_bytes[65536];
    test_resource arena_scratch, arena_result;
    test_resource_bump(&arena_scratch, scratch_bytes, sizeof(scratch_bytes));
    test_resource_bump(&arena_result, result_bytes, sizeof(result_bytes));
    cudaq_gf2_memory arenas = test_memory(&arena_scratch, &arena_result);
    cudaq_gf2_owned_buffer bounded_factor = {0};
    cudaq_edem_dem_view bounded_view;
    TA_CHECK(cudaq_edem_factorize_dem(&source, &arenas, &bounded_factor,
                                      &bounded_view) == CUDAQ_EDEM_OK);
    cudaq_gf2_owned_buffer_release(&bounded_factor);
    test_resource_reset(&arena_scratch);
    test_resource_reset(&arena_result);
    TA_CHECK(cudaq_edem_coalesce_dem_columns(&converted, &arenas, &reduced,
                                             &result, &map) == CUDAQ_EDEM_OK);
    cudaq_gf2_owned_buffer_release(&reduced);
    cudaq_gf2_owned_buffer_release(&map);
    test_resource_balanced(&arena_scratch);
    test_resource_balanced(&arena_result);
    test_resource_bump(&arena_result, result_bytes, 8);
    TA_CHECK(cudaq_edem_coalesce_dem_columns(&converted, &arenas, &reduced,
                                             &result, &map) ==
             CUDAQ_EDEM_ALLOCATION_FAILED);
    TA_CHECK(!reduced.data && !map.data);
    test_resource_balanced(&arena_scratch);
    test_resource_balanced(&arena_result);
  }
  TA_CHECK(cudaq_edem_coalesce_dem_columns(&converted, &memory, &reduced,
                                           &result, &map) == CUDAQ_EDEM_OK);
  cudaq_gf2_owned_buffer_release(&original);
  cudaq_gf2_owned_buffer_release(&independent);
  uint32_t size, group_count;
  TA_CHECK(cudaq_edem_get_fault(&result.edem, 0, &size, &group_count) ==
           CUDAQ_EDEM_OK);
  TA_CHECK(size == 1 && group_count == 1 &&
           map.size == columns * sizeof(uint64_t));
  const uint64_t *mapping = map.data;
  for (uint32_t i = 0; i < columns; ++i)
    TA_CHECK(mapping[i] == (i % 3 == 2 ? UINT64_MAX : 0));
  cudaq_edem_fault_group law;
  TA_CHECK(cudaq_edem_get_group(&result.edem, 0, 0, &law) == CUDAQ_EDEM_OK);
  TA_CHECK(law.kind == CUDAQ_EDEM_INDEPENDENT);
  double expected = -.5 * expm1(count * log1p(-4 * .001 / 3));
  TA_CHECK(fabs(law.probabilities[0] - expected) < 1e-13);
  cudaq_gf2_owned_buffer_release(&reduced);
  cudaq_gf2_owned_buffer_release(&map);
  test_resource_balanced(&scratch);
  test_resource_balanced(&output);
  uint64_t coalesce_peak = scratch.peak_bytes + output.peak_bytes;
  if (coalesce_peak > peak)
    peak = coalesce_peak;
  TA_CHECK(peak < (uint64_t)columns * 256 + 65536);
  return peak;
}

int main(void) {
  reordered_outputs();
  (void)channels(1, 1);
  uint64_t small = channels(2000, 0), large = channels(20000, 0);
  TA_CHECK(large < small * 12);
  return 0;
}
