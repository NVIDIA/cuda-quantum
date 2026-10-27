/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#include "cudaq/edem/composition.h"
#include "../../math/gf2/test/test_allocator.h"
#include "cudaq/edem/format.h"

/* Repack through the public GF2 API to obtain checked EDEMs with arbitrary
 * free/interface interleaving. Orders map new bundle IDs to old bundle IDs;
 * fault bundles keep their relative order so their section payload order stays
 * canonical. No trusted token is fabricated or mutated. */
static void reorder_bundles(const cudaq_edem_view *source,
                            const uint32_t *input_order,
                            const uint32_t *output_order,
                            cudaq_gf2_owned_buffer *owner,
                            cudaq_edem_view *result) {
  enum { MAX_BUNDLES = 8, MAX_TYPES = 8, MAX_BLOCKS = 32 };
  cudaq_gf2_map_shape shape;
  TA_CHECK(cudaq_gf2_map_get_shape(&source->map, &shape) == CUDAQ_GF2_OK);
  TA_CHECK(shape.input_bundle_count <= MAX_BUNDLES &&
           shape.output_bundle_count <= MAX_BUNDLES &&
           shape.type_count <= MAX_TYPES && shape.block_count <= MAX_BLOCKS &&
           shape.section_count == 2);
  cudaq_gf2_bundle_type types[MAX_TYPES];
  cudaq_gf2_bundle inputs[MAX_BUNDLES], outputs[MAX_BUNDLES];
  uint32_t input_ids[MAX_BUNDLES], output_ids[MAX_BUNDLES];
  cudaq_gf2_map_matrix_block blocks[MAX_BLOCKS];
  for (uint32_t i = 0; i < shape.type_count; ++i)
    TA_CHECK(cudaq_gf2_map_get_type(&source->map, i, types + i) ==
             CUDAQ_GF2_OK);
  for (uint32_t side = 0; side < 2; ++side) {
    uint32_t count =
        side ? shape.output_bundle_count : shape.input_bundle_count;
    const uint32_t *order = side ? output_order : input_order;
    uint32_t *ids = side ? output_ids : input_ids;
    cudaq_gf2_bundle *bundles = side ? outputs : inputs;
    for (uint32_t i = 0; i < count; ++i) {
      TA_CHECK(order[i] < count);
      ids[order[i]] = i;
      TA_CHECK(cudaq_gf2_map_get_bundle(&source->map, side, order[i],
                                        bundles + i) == CUDAQ_GF2_OK);
    }
  }
  for (uint32_t i = 0; i < shape.block_count; ++i) {
    TA_CHECK(cudaq_gf2_map_get_matrix_block(&source->map, i, blocks + i) ==
             CUDAQ_GF2_OK);
    blocks[i].input.bundle = input_ids[blocks[i].input.bundle];
    blocks[i].output.bundle = output_ids[blocks[i].output.bundle];
  }
  cudaq_gf2_section sections[2];
  TA_CHECK(cudaq_gf2_map_get_section(&source->map,
                                     CUDAQ_EDEM_SECTION_PRESENTATIONS,
                                     sections) == CUDAQ_GF2_OK);
  TA_CHECK(cudaq_gf2_map_get_section(&source->map,
                                     CUDAQ_EDEM_SECTION_DISTRIBUTIONS,
                                     sections + 1) == CUDAQ_GF2_OK);
  void *distributions = malloc((size_t)sections[1].bytes);
  TA_CHECK(distributions || !sections[1].bytes);
  if (sections[1].bytes)
    memcpy(distributions, sections[1].data, (size_t)sections[1].bytes);
  cudaq_edem_packed_fault *faults = distributions;
  for (uint32_t i = 0; i < sections[1].count; ++i) {
    faults[i].bundle = input_ids[faults[i].bundle];
    TA_CHECK(!i || faults[i - 1].bundle < faults[i].bundle);
  }
  sections[1].data = distributions;
  cudaq_gf2_map_description description = {shape.type_count,
                                           shape.input_bundle_count,
                                           shape.output_bundle_count,
                                           2,
                                           shape.block_count,
                                           types,
                                           inputs,
                                           outputs,
                                           blocks,
                                           sections};
  cudaq_gf2_map_view map;
  TA_CHECK(cudaq_gf2_map_construct(&description, 0, owner, &map) ==
           CUDAQ_GF2_OK);
  free(distributions);
  TA_CHECK(cudaq_edem_check(&map, 0, result) == CUDAQ_EDEM_OK);
  TA_CHECK(cudaq_edem_read(owner->data, owner->size, 0, result) ==
           CUDAQ_EDEM_OK);
}

static void same_value(const cudaq_edem_view *a, const cudaq_edem_view *b) {
  TA_CHECK(a->map.size == b->map.size &&
           memcmp(a->map.data, b->map.data, (size_t)a->map.size) == 0);
}

static void failure_sites(const cudaq_edem_view *sources, uint32_t count,
                          int project) {
  cudaq_edem_view source = sources[0];
  test_resource resource;
  test_resource_init(&resource);
  cudaq_gf2_memory memory = test_memory(&resource, &resource);
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_edem_view view = source;
  cudaq_edem_dem_view dem = {source};
  cudaq_edem_status status =
      project
          ? cudaq_edem_project_dem(&source, &memory, &owner, &dem)
          : cudaq_edem_compose_chain(sources, count, &memory, &owner, &view);
  TA_CHECK(status == CUDAQ_EDEM_OK);
  uint64_t attempts = resource.attempts;
  TA_CHECK(attempts > 0 && owner.data != source.map.data);
  cudaq_gf2_owned_buffer_release(&owner);
  test_resource_balanced(&resource);
  for (uint64_t fail = 1; fail <= attempts; ++fail) {
    test_resource_init(&resource);
    resource.fail_on_attempt = fail;
    memory = test_memory(&resource, &resource);
    view = source;
    dem.edem = source;
    status = project ? cudaq_edem_project_dem(&source, &memory, &owner, &dem)
                     : cudaq_edem_compose_chain(sources, count, &memory, &owner,
                                                &view);
    TA_CHECK(status == CUDAQ_EDEM_ALLOCATION_FAILED);
    TA_CHECK(test_same_owner(owner, (cudaq_gf2_owned_buffer){0}));
    TA_CHECK(view.map.data == source.map.data &&
             view.map.size == source.map.size);
    TA_CHECK(dem.edem.map.data == source.map.data &&
             dem.edem.map.size == source.map.size);
    test_resource_balanced(&resource);
  }
}

static void boundaries(void) {
  const cudaq_edem_code_presentation codes[] = {{"left", 4, 0, 0, 0},
                                                {"right", 5, 0, 0, 0}};
  const cudaq_edem_code_table table = {2, codes};
  uint32_t code = 0;
  cudaq_edem_description description = {0};
  description.shape.output_code_count = 1;
  description.code_table = table;
  description.output_code_indices = &code;
  cudaq_gf2_owned_buffer prep_owner = {0}, mz_owner = {0}, result_owner = {0};
  cudaq_edem_view prep, mz, result;
  TA_CHECK(cudaq_edem_construct(&description, 0, &prep_owner, &prep) ==
           CUDAQ_EDEM_OK);
  uint32_t closed = 7;
  TA_CHECK(cudaq_edem_is_dem(&prep, &closed) == CUDAQ_EDEM_OK && !closed);
  cudaq_edem_dem_view dem = {prep};
  TA_CHECK(cudaq_edem_project_dem(&prep, 0, &result_owner, &dem) ==
           CUDAQ_EDEM_INVALID_CONTENT);
  TA_CHECK(!result_owner.data && dem.edem.map.data == prep.map.data);

  description.shape.output_code_count = 0;
  description.shape.input_code_count = 1;
  description.output_code_indices = 0;
  description.input_code_indices = &code;
  code = 1;
  TA_CHECK(cudaq_edem_construct(&description, 0, &mz_owner, &mz) ==
           CUDAQ_EDEM_OK);
  TA_CHECK(cudaq_edem_is_dem(&mz, &closed) == CUDAQ_EDEM_OK && !closed);
  TA_CHECK(cudaq_edem_project_dem(&mz, 0, &result_owner, &dem) ==
           CUDAQ_EDEM_INVALID_CONTENT);
  cudaq_edem_view sequence[] = {prep, mz};
  result = prep;
  TA_CHECK(cudaq_edem_compose_chain(sequence, 2, 0, &result_owner, &result) ==
           CUDAQ_EDEM_INVALID_CONTENT);
  TA_CHECK(!result_owner.data && result.map.data == prep.map.data);
  // Matching names identify the code; mismatched counts are still rejected.
  cudaq_gf2_owned_buffer_release(&mz_owner);
  const cudaq_edem_code_presentation wider = {"left", 4, 1, 1, 0};
  description.code_table = (cudaq_edem_code_table){1, &wider};
  code = 0;
  TA_CHECK(cudaq_edem_construct(&description, 0, &mz_owner, &mz) ==
           CUDAQ_EDEM_OK);
  sequence[1] = mz;
  TA_CHECK(cudaq_edem_compose_chain(sequence, 2, 0, &result_owner, &result) ==
           CUDAQ_EDEM_INVALID_CONTENT);
  TA_CHECK(!result_owner.data && result.map.data == prep.map.data);
  cudaq_gf2_owned_buffer_release(&mz_owner);
  description.code_table = table;
  TA_CHECK(cudaq_edem_construct(&description, 0, &mz_owner, &mz) ==
           CUDAQ_EDEM_OK);
  sequence[1] = mz;
  TA_CHECK(cudaq_edem_compose_chain(sequence, 2, 0, &result_owner, &result) ==
           CUDAQ_EDEM_OK);
  TA_CHECK(cudaq_edem_is_dem(&result, &closed) == CUDAQ_EDEM_OK && closed);
  cudaq_gf2_owned_buffer_release(&result_owner);
  // A zero-width code bundle remains a boundary and cannot disappear merely
  // because it has no bits. Repeating preparation therefore mismatches counts.
  sequence[1] = prep;
  TA_CHECK(cudaq_edem_compose_chain(sequence, 2, 0, &result_owner, &result) ==
           CUDAQ_EDEM_INVALID_CONTENT);
  cudaq_gf2_owned_buffer_release(&prep_owner);
  cudaq_gf2_owned_buffer_release(&mz_owner);
}

static void make_gadget(unsigned kind, cudaq_gf2_owned_buffer *owner,
                        cudaq_edem_view *view) {
  const cudaq_edem_code_presentation code = {"test", 4, 1, 1, 1};
  const uint32_t code_id = 0, width = 1;
  const uint32_t exclusive_ids[] = {0, 1}, independent_ids[] = {2};
  const double exclusive_p[] = {0.1, 0.2}, independent_p[] = {0.3};
  const cudaq_edem_fault_group groups[] = {
      {CUDAQ_EDEM_EXCLUSIVE, 2, exclusive_ids, exclusive_p},
      {CUDAQ_EDEM_INDEPENDENT, 1, independent_ids, independent_p}};
  const cudaq_edem_fault_distribution faults = {3, 2, groups};
  const uint64_t one_offsets[] = {0, 1}, identity_offsets[] = {0, 1, 2};
  const uint32_t zero[] = {0}, identity_rows[] = {0, 1};
  const uint64_t v_offsets[] = {0, 1, 1, 2}, s_offsets[] = {0, 0, 1, 2},
                 l_offsets[] = {0, 1, 2, 4}, f0_offsets[] = {0, 1, 1, 1},
                 f1_offsets[] = {0, 0, 1, 1}, f2_offsets[] = {0, 0, 0, 1},
                 f02_offsets[] = {0, 1, 1, 2}, all_offsets[] = {0, 1, 2, 3},
                 l0_offsets[] = {0, 1, 1};
  const uint32_t two_zeros[] = {0, 0}, three_zeros[] = {0, 0, 0},
                 l_rows[] = {0, 1, 0, 1};
  const cudaq_gf2_csc_view one = {1, 1, 1, one_offsets, zero},
                           identity = {2, 2, 2, identity_offsets,
                                       identity_rows},
                           v = {1, 3, 2, v_offsets, two_zeros},
                           s = {1, 3, 2, s_offsets, two_zeros},
                           l = {2, 3, 4, l_offsets, l_rows},
                           f0 = {1, 3, 1, f0_offsets, zero},
                           f1 = {1, 3, 1, f1_offsets, zero},
                           f2 = {1, 3, 1, f2_offsets, zero},
                           l_f2 = {2, 3, 1, f2_offsets, zero},
                           f02 = {1, 3, 2, f02_offsets, two_zeros},
                           all = {1, 3, 3, all_offsets, three_zeros},
                           l0 = {1, 2, 1, l0_offsets, zero},
                           l01 = {1, 2, 2, identity_offsets, two_zeros};
  cudaq_edem_matrix_block blocks[16];
  uint32_t n = 0;
#define BLOCK(out, in, m)                                                      \
  blocks[n++] = (cudaq_edem_matrix_block) {                                    \
    {CUDAQ_EDEM_##out, 0}, {CUDAQ_EDEM_##in, 0}, m                             \
  }
  if (kind == 0) {
    BLOCK(VIRTUAL, FAULT, v);
    BLOCK(SYNDROME, FAULT, s);
    BLOCK(LOGICAL, FAULT, l);
    BLOCK(DETECTOR, FAULT, f1);
  } else if (kind == 1) {
    BLOCK(VIRTUAL, VIRTUAL, one);
    BLOCK(SYNDROME, SYNDROME, one);
    BLOCK(LOGICAL, LOGICAL, identity);
    BLOCK(VIRTUAL, FAULT, f0);
    BLOCK(SYNDROME, FAULT, f1);
    BLOCK(LOGICAL, FAULT, l_f2);
    BLOCK(DETECTOR, VIRTUAL, one);
    BLOCK(DETECTOR, SYNDROME, one);
    BLOCK(DETECTOR, FAULT, f02);
    BLOCK(OBSERVABLE, LOGICAL, l0);
    BLOCK(SUCCESS, SYNDROME, one);
    BLOCK(SUCCESS, FAULT, f1);
  } else {
    BLOCK(DETECTOR, VIRTUAL, one);
    BLOCK(DETECTOR, SYNDROME, one);
    BLOCK(DETECTOR, FAULT, all);
    BLOCK(OBSERVABLE, LOGICAL, l01);
    BLOCK(OBSERVABLE, FAULT, f1);
    BLOCK(SUCCESS, SYNDROME, one);
    BLOCK(SUCCESS, FAULT, f2);
  }
#undef BLOCK
  cudaq_edem_description description = {
      {kind != 0, kind != 2, 1, 1, kind != 0, kind != 0},
      {1, &code},
      &code_id,
      &code_id,
      &faults,
      &width,
      &width,
      &width,
      n,
      blocks};
  TA_CHECK(cudaq_edem_construct(&description, 0, owner, view) == CUDAQ_EDEM_OK);
}

static unsigned parity(unsigned bits) {
  unsigned result = 0;
  while (bits) {
    result ^= bits & 1;
    bits >>= 1;
  }
  return result;
}

static void check_laws(const cudaq_edem_view *view, uint32_t count) {
  for (uint32_t p = 0; p < count; ++p) {
    uint32_t width, groups;
    TA_CHECK(cudaq_edem_get_fault(view, p, &width, &groups) == CUDAQ_EDEM_OK);
    TA_CHECK(width == 3 && groups == 2);
    unsigned seen = 0;
    for (uint32_t g = 0; g < groups; ++g) {
      cudaq_edem_fault_group law;
      TA_CHECK(cudaq_edem_get_group(view, p, g, &law) == CUDAQ_EDEM_OK);
      if (law.kind == CUDAQ_EDEM_EXCLUSIVE) {
        TA_CHECK(law.size == 2 && law.fault_ids[0] == 0 &&
                 law.fault_ids[1] == 1 && law.probabilities[0] == 0.1 &&
                 law.probabilities[1] == 0.2);
        seen |= 1;
      } else {
        TA_CHECK(law.kind == CUDAQ_EDEM_INDEPENDENT && law.size == 1 &&
                 law.fault_ids[0] == 2 && law.probabilities[0] == 0.3);
        seen |= 2;
      }
    }
    TA_CHECK(seen == 3);
  }
}

static void reordered_chain(void) {
  cudaq_gf2_owned_buffer owners[3] = {{0}}, reordered_owners[3] = {{0}},
                         canonical_owner = {0}, composed_owner = {0};
  cudaq_edem_view values[3], reordered[3], canonical, composed;
  const uint32_t prep_inputs[] = {0}, prep_outputs[] = {1, 0},
                 round_inputs[] = {1, 0}, round_outputs[] = {3, 2, 0, 1},
                 readout_outputs[] = {2, 1, 0};
  for (unsigned i = 0; i < 3; ++i) {
    make_gadget(i, owners + i, values + i);
    reorder_bundles(values + i, i ? round_inputs : prep_inputs,
                    i == 0   ? prep_outputs
                    : i == 1 ? round_outputs
                             : readout_outputs,
                    reordered_owners + i, reordered + i);
    /* One occurrence also checks exposed boundaries on both sides. */
    TA_CHECK(cudaq_edem_compose_chain(reordered + i, 1, 0, &composed_owner,
                                      &composed) == CUDAQ_EDEM_OK);
    same_value(values + i, &composed);
    cudaq_gf2_owned_buffer_release(&composed_owner);
  }
  const cudaq_edem_view sequence[] = {values[0], values[1], values[1],
                                      values[2]};
  const cudaq_edem_view reordered_sequence[] = {reordered[0], reordered[1],
                                                reordered[1], reordered[2]};
  const cudaq_edem_view mixed_sequence[] = {values[0], reordered[1], values[1],
                                            reordered[2]};
  TA_CHECK(cudaq_edem_compose_chain(sequence, 4, 0, &canonical_owner,
                                    &canonical) == CUDAQ_EDEM_OK);
  TA_CHECK(cudaq_edem_compose_chain(reordered_sequence, 4, 0, &composed_owner,
                                    &composed) == CUDAQ_EDEM_OK);
  same_value(&canonical, &composed);
  cudaq_gf2_owned_buffer_release(&composed_owner);
  TA_CHECK(cudaq_edem_compose_chain(mixed_sequence, 4, 0, &composed_owner,
                                    &composed) == CUDAQ_EDEM_OK);
  same_value(&canonical, &composed);
  failure_sites(reordered_sequence, 4, 0);
  for (unsigned i = 0; i < 3; ++i) {
    cudaq_gf2_owned_buffer_release(owners + i);
    cudaq_gf2_owned_buffer_release(reordered_owners + i);
  }
  check_laws(&composed, 4);
  same_value(&canonical, &composed);
  cudaq_edem_dem_view projected;
  cudaq_gf2_owned_buffer projected_owner = {0};
  TA_CHECK(cudaq_edem_project_dem(&composed, 0, &projected_owner, &projected) ==
           CUDAQ_EDEM_OK);
  cudaq_gf2_owned_buffer_release(&composed_owner);
  check_laws(&projected.edem, 4);
  same_value(&canonical, &projected.edem);
  cudaq_gf2_owned_buffer_release(&projected_owner);
  cudaq_gf2_owned_buffer_release(&canonical_owner);
}

static void make_multiport_gadget(cudaq_gf2_owned_buffer *owner,
                                  cudaq_edem_view *view) {
  const cudaq_edem_code_presentation codes[] = {{"one", 3, 1, 1, 1},
                                                {"two", 3, 2, 2, 0}};
  const uint32_t code_ids[] = {0, 1}, detectors[] = {1, 2}, one = 1,
                 ids[] = {0, 1}, zero[] = {0}, last_row[] = {1},
                 both_rows[] = {0, 0};
  const double exclusive_p[] = {0.1, 0.2}, independent_p[] = {0.3};
  const cudaq_edem_fault_group exclusive = {CUDAQ_EDEM_EXCLUSIVE, 2, ids,
                                            exclusive_p},
                               independent = {CUDAQ_EDEM_INDEPENDENT, 1, zero,
                                              independent_p};
  const cudaq_edem_fault_distribution faults[] = {{2, 1, &exclusive},
                                                  {1, 1, &independent}};
  const uint64_t one_offsets[] = {0, 1}, identity_offsets[] = {0, 1, 2},
                 first_offsets[] = {0, 1, 1};
  const cudaq_gf2_csc_view identity_one = {1, 1, 1, one_offsets, zero},
                           identity_two = {2, 2, 2, identity_offsets, ids},
                           first = {1, 2, 1, first_offsets, zero},
                           both = {1, 2, 2, identity_offsets, both_rows},
                           last = {2, 1, 1, one_offsets, last_row};
  const cudaq_edem_matrix_block blocks[] = {
      {{CUDAQ_EDEM_VIRTUAL, 0}, {CUDAQ_EDEM_VIRTUAL, 0}, identity_one},
      {{CUDAQ_EDEM_SYNDROME, 0}, {CUDAQ_EDEM_SYNDROME, 0}, identity_one},
      {{CUDAQ_EDEM_LOGICAL, 0}, {CUDAQ_EDEM_LOGICAL, 0}, identity_two},
      {{CUDAQ_EDEM_VIRTUAL, 1}, {CUDAQ_EDEM_VIRTUAL, 1}, identity_two},
      {{CUDAQ_EDEM_SYNDROME, 1}, {CUDAQ_EDEM_SYNDROME, 1}, identity_two},
      {{CUDAQ_EDEM_SYNDROME, 0}, {CUDAQ_EDEM_FAULT, 1}, identity_one},
      {{CUDAQ_EDEM_LOGICAL, 0}, {CUDAQ_EDEM_FAULT, 1}, last},
      {{CUDAQ_EDEM_VIRTUAL, 0}, {CUDAQ_EDEM_FAULT, 0}, first},
      {{CUDAQ_EDEM_SYNDROME, 1}, {CUDAQ_EDEM_FAULT, 0}, identity_two},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_FAULT, 0}, both},
      {{CUDAQ_EDEM_DETECTOR, 1}, {CUDAQ_EDEM_FAULT, 1}, last},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_VIRTUAL, 0}, identity_one},
      {{CUDAQ_EDEM_DETECTOR, 1}, {CUDAQ_EDEM_SYNDROME, 1}, identity_two},
      {{CUDAQ_EDEM_OBSERVABLE, 0}, {CUDAQ_EDEM_LOGICAL, 0}, both},
      {{CUDAQ_EDEM_SUCCESS, 0}, {CUDAQ_EDEM_FAULT, 1}, identity_one},
      {{CUDAQ_EDEM_SUCCESS, 0}, {CUDAQ_EDEM_SYNDROME, 0}, identity_one}};
  const cudaq_edem_description description = {{2, 2, 2, 2, 1, 1},
                                              {2, codes},
                                              code_ids,
                                              code_ids,
                                              faults,
                                              detectors,
                                              &one,
                                              &one,
                                              sizeof(blocks) /
                                                  sizeof(blocks[0]),
                                              blocks};
  TA_CHECK(cudaq_edem_construct(&description, 0, owner, view) == CUDAQ_EDEM_OK);
}

static void interleaved_multiport_chain(void) {
  cudaq_gf2_owned_buffer source_owner = {0}, reordered_owners[2] = {{0}},
                         canonical_owner = {0}, composed_owner = {0};
  cudaq_edem_view source, reordered[2], canonical, composed;
  make_multiport_gadget(&source_owner, &source);
  const uint32_t inputs_a[] = {2, 0, 3, 1}, inputs_b[] = {0, 2, 1, 3},
                 outputs_a[] = {3, 4, 2, 0, 5, 1},
                 outputs_b[] = {4, 0, 3, 5, 2, 1};
  reorder_bundles(&source, inputs_a, outputs_a, reordered_owners, reordered);
  reorder_bundles(&source, inputs_b, outputs_b, reordered_owners + 1,
                  reordered + 1);
  const cudaq_edem_view sequence[] = {source, source, source};
  const cudaq_edem_view reordered_sequence[] = {reordered[0], reordered[1],
                                                reordered[0]};
  TA_CHECK(cudaq_edem_compose_chain(sequence, 3, 0, &canonical_owner,
                                    &canonical) == CUDAQ_EDEM_OK);
  TA_CHECK(cudaq_edem_compose_chain(reordered_sequence, 3, 0, &composed_owner,
                                    &composed) == CUDAQ_EDEM_OK);
  same_value(&canonical, &composed);
  cudaq_edem_shape shape;
  TA_CHECK(cudaq_edem_get_shape(&composed, &shape) == CUDAQ_EDEM_OK);
  TA_CHECK(shape.input_code_count == 2 && shape.output_code_count == 2 &&
           shape.fault_ports_count == 6 && shape.detector_ports_count == 6 &&
           shape.observable_ports_count == 3 && shape.success_ports_count == 3);
  failure_sites(reordered_sequence, 3, 0);
  cudaq_gf2_owned_buffer_release(&source_owner);
  for (unsigned i = 0; i < 2; ++i)
    cudaq_gf2_owned_buffer_release(reordered_owners + i);
  /* Every occurrence keeps both original fault ports and their exact laws,
   * independently of definition reuse or source allocation lifetime. */
  for (uint32_t p = 0; p < 6; ++p) {
    uint32_t width, count;
    cudaq_edem_fault_group group;
    TA_CHECK(cudaq_edem_get_fault(&composed, p, &width, &count) ==
             CUDAQ_EDEM_OK);
    TA_CHECK(count == 1 && width == (p % 2 ? 1 : 2));
    TA_CHECK(cudaq_edem_get_group(&composed, p, 0, &group) == CUDAQ_EDEM_OK);
    TA_CHECK(group.size == width && group.fault_ids[0] == 0);
    if (p % 2)
      TA_CHECK(group.kind == CUDAQ_EDEM_INDEPENDENT &&
               group.probabilities[0] == 0.3);
    else
      TA_CHECK(group.kind == CUDAQ_EDEM_EXCLUSIVE && group.fault_ids[1] == 1 &&
               group.probabilities[0] == 0.1 && group.probabilities[1] == 0.2);
  }
  same_value(&canonical, &composed);
  uint8_t input[17], actual[23], expected[23];
  uint64_t input_size, output_size;
  TA_CHECK(cudaq_gf2_map_vector_sizes(&composed.map, &input_size,
                                      &output_size) == CUDAQ_GF2_OK);
  TA_CHECK(input_size == sizeof(input) && output_size == sizeof(actual));
  for (unsigned bit = 0; bit < sizeof(input); ++bit) {
    memset(input, 0, sizeof(input));
    input[bit] = 1;
    TA_CHECK(cudaq_gf2_map_apply(&composed.map, input, sizeof(input), actual,
                                 sizeof(actual)) == CUDAQ_GF2_OK);
    TA_CHECK(cudaq_gf2_map_apply(&canonical.map, input, sizeof(input), expected,
                                 sizeof(expected)) == CUDAQ_GF2_OK);
    TA_CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
  }
  cudaq_gf2_owned_buffer_release(&canonical_owner);
  cudaq_gf2_owned_buffer_release(&composed_owner);
}

static void serial_chain(void) {
  cudaq_gf2_owned_buffer owners[3] = {{0}}, composed_owner = {0};
  cudaq_edem_view values[3], composed;
  for (unsigned i = 0; i < 3; ++i)
    make_gadget(i, owners + i, values + i);
  const cudaq_edem_view sequence[] = {values[0], values[1], values[1],
                                      values[2]};
  TA_CHECK(cudaq_edem_compose_chain(sequence, 4, 0, &composed_owner,
                                    &composed) == CUDAQ_EDEM_OK);
  cudaq_edem_shape shape;
  TA_CHECK(cudaq_edem_get_shape(&composed, &shape) == CUDAQ_EDEM_OK);
  TA_CHECK(shape.fault_ports_count == 4 && shape.detector_ports_count == 4 &&
           shape.observable_ports_count == 3 &&
           shape.success_ports_count == 3 && !shape.input_code_count &&
           !shape.output_code_count);
  check_laws(&composed, 4);
  // Hand-derived equations, in D/O/Success port order, for all 12 original
  // indicators. The repeated round definition has two distinct fault ports.
#define BIT(i) (1u << (i))
  const unsigned equations[] = {BIT(1),
                                BIT(0) ^ BIT(1) ^ BIT(3) ^ BIT(5),
                                BIT(0) ^ BIT(1) ^ BIT(3) ^ BIT(4) ^ BIT(6) ^
                                    BIT(8),
                                BIT(0) ^ BIT(1) ^ BIT(3) ^ BIT(4) ^ BIT(6) ^
                                    BIT(7) ^ BIT(9) ^ BIT(10) ^ BIT(11),
                                BIT(0) ^ BIT(2),
                                BIT(0) ^ BIT(2) ^ BIT(5),
                                BIT(0) ^ BIT(1) ^ BIT(5) ^ BIT(8) ^ BIT(10),
                                BIT(1) ^ BIT(2) ^ BIT(4),
                                BIT(1) ^ BIT(2) ^ BIT(4) ^ BIT(7),
                                BIT(1) ^ BIT(2) ^ BIT(4) ^ BIT(7) ^ BIT(11)};
#undef BIT
  uint8_t input[12], output[10];
  for (unsigned bits = 0; bits < 4096; ++bits) {
    for (unsigned i = 0; i < 12; ++i)
      input[i] = (uint8_t)((bits >> i) & 1);
    TA_CHECK(cudaq_gf2_map_apply(&composed.map, input, 12, output, 10) ==
             CUDAQ_GF2_OK);
    for (unsigned i = 0; i < 10; ++i)
      TA_CHECK(output[i] == parity(bits & equations[i]));
  }
  failure_sites(sequence, 4, 0);
  failure_sites(&composed, 1, 1);

  // One open occurrence is an identity operation on the packed content.
  cudaq_gf2_owned_buffer open_owner = {0};
  cudaq_edem_view open;
  TA_CHECK(cudaq_edem_compose_chain(values + 1, 1, 0, &open_owner, &open) ==
           CUDAQ_EDEM_OK);
  TA_CHECK(open.map.size == values[1].map.size &&
           memcmp(open.map.data, values[1].map.data, open.map.size) == 0);
  cudaq_edem_dem_view dem = {composed};
  cudaq_gf2_owned_buffer dem_owner = {0};
  TA_CHECK(cudaq_edem_project_dem(&open, 0, &dem_owner, &dem) ==
           CUDAQ_EDEM_INVALID_CONTENT);
  TA_CHECK(!dem_owner.data && dem.edem.map.data == composed.map.data);
  TA_CHECK(cudaq_edem_project_dem(&composed, 0, &dem_owner, &dem) ==
           CUDAQ_EDEM_OK);
  TA_CHECK(dem_owner.data != composed_owner.data &&
           dem.edem.map.size == composed.map.size &&
           memcmp(dem.edem.map.data, composed.map.data, composed.map.size) ==
               0);
  cudaq_gf2_owned_buffer_release(&composed_owner);
  for (unsigned i = 0; i < 3; ++i)
    cudaq_gf2_owned_buffer_release(owners + i);
  cudaq_gf2_owned_buffer_release(&open_owner);
  // Projection retains success responses and owns every law and matrix after
  // all definitions and the composed source have been released.
  check_laws(&dem.edem, 4);
  TA_CHECK(cudaq_gf2_map_apply(&dem.edem.map, input, 12, output, 10) ==
           CUDAQ_GF2_OK);
  cudaq_gf2_owned_buffer_release(&dem_owner);
}

static void bounded(void) {
  cudaq_gf2_owned_buffer owners[3] = {{0}};
  cudaq_edem_view values[3];
  for (unsigned i = 0; i < 3; ++i)
    make_gadget(i, owners + i, values + i);
  _Alignas(8) unsigned char scratch_buffer[65536], result_buffer[65536];
  test_resource scratch, result;
  test_resource_bump(&scratch, scratch_buffer, sizeof(scratch_buffer));
  test_resource_bump(&result, result_buffer, sizeof(result_buffer));
  cudaq_gf2_memory memory = test_memory(&scratch, &result);
  cudaq_gf2_owned_buffer owner = {0}, dem_owner = {0};
  cudaq_edem_view composed;
  cudaq_edem_dem_view dem;
  TA_CHECK(cudaq_edem_compose_chain(values, 3, &memory, &owner, &composed) ==
           CUDAQ_EDEM_OK);
  test_resource_balanced(&scratch);
  TA_CHECK(owner.data >= (void *)result_buffer &&
           (unsigned char *)owner.data < result_buffer + sizeof(result_buffer));
  TA_CHECK(cudaq_edem_project_dem(&composed, &memory, &dem_owner, &dem) ==
           CUDAQ_EDEM_OK);
  check_laws(&dem.edem, 3);
  test_resource_balanced(&scratch);
  cudaq_gf2_owned_buffer_release(&dem_owner);
  cudaq_gf2_owned_buffer_release(&owner);
  test_resource_balanced(&result);
  test_resource_reset(&scratch);
  test_resource_reset(&result);
  // Explicit bounded resources must fail rather than fall back to the heap.
  scratch.arena.capacity = 0;
  composed = values[0];
  TA_CHECK(cudaq_edem_compose_chain(values, 3, &memory, &owner, &composed) ==
           CUDAQ_EDEM_ALLOCATION_FAILED);
  TA_CHECK(!owner.data && composed.map.data == values[0].map.data);
  test_resource_balanced(&scratch);
  test_resource_balanced(&result);
  for (unsigned i = 0; i < 3; ++i)
    cudaq_gf2_owned_buffer_release(owners + i);
}

static void arguments(cudaq_edem_view source) {
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_edem_view result = source;
  TA_CHECK(cudaq_edem_compose_chain(0, 0, 0, &owner, &result) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  TA_CHECK(cudaq_edem_compose_chain(0, 1, 0, &owner, &result) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  TA_CHECK(cudaq_edem_compose_chain(&source, 1, 0, 0, &result) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  TA_CHECK(cudaq_edem_compose_chain(&source, 1, 0, &owner, 0) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  uint32_t closed = 7;
  TA_CHECK(cudaq_edem_is_dem(0, &closed) == CUDAQ_EDEM_INVALID_ARGUMENT &&
           closed == 7);
  TA_CHECK(cudaq_edem_is_dem(&source, 0) == CUDAQ_EDEM_INVALID_ARGUMENT);
  TA_CHECK(!owner.data && result.map.data == source.map.data);
  cudaq_gf2_memory memory = {0};
  TA_CHECK(cudaq_edem_compose_chain(&source, 1, &memory, &owner, &result) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  cudaq_edem_dem_view dem = {source};
  TA_CHECK(cudaq_edem_project_dem(&source, &memory, &owner, &dem) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  owner = (cudaq_gf2_owned_buffer){(void *)source.map.data, source.map.size,
                                   test_release_sentinel, 0};
  cudaq_gf2_owned_buffer sentinel = owner;
  TA_CHECK(cudaq_edem_compose_chain(&source, 1, 0, &owner, &result) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  TA_CHECK(cudaq_edem_project_dem(&source, 0, &owner, &dem) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  TA_CHECK(test_same_owner(owner, sentinel) &&
           result.map.data == source.map.data &&
           dem.edem.map.data == source.map.data);
}

int main(void) {
  cudaq_edem_description empty = {0};
  cudaq_gf2_owned_buffer source_owner = {0};
  cudaq_edem_view source;
  TA_CHECK(cudaq_edem_construct(&empty, 0, &source_owner, &source) ==
           CUDAQ_EDEM_OK);
  failure_sites(&source, 1, 0);
  failure_sites(&source, 1, 1);
  arguments(source);
  boundaries();
  serial_chain();
  reordered_chain();
  interleaved_multiport_chain();
  bounded();
  cudaq_gf2_owned_buffer_release(&source_owner);
  return 0;
}
