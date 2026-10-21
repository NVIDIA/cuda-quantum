/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#include "cudaq/edem/composition.h"
#include "../../math/gf2/test/test_allocator.h"

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
  cudaq_edem_code_presentation codes[] = {{"left", 4, 0, 0},
                                          {"right", 5, 0, 0}};
  uint32_t code = 0;
  cudaq_edem_description description = {0};
  description.shape.code_presentations = 1;
  description.shape.output_code_blocks = 1;
  description.codes = codes;
  description.output_codes = &code;
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

  description.shape.output_code_blocks = 0;
  description.shape.input_code_blocks = 1;
  description.output_codes = 0;
  description.input_codes = &code;
  description.codes = codes + 1;
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
  // Matching names alone are insufficient: presentation metadata must agree.
  cudaq_gf2_owned_buffer_release(&mz_owner);
  codes[1] = (cudaq_edem_code_presentation){"left", 4, 1, 0};
  TA_CHECK(cudaq_edem_construct(&description, 0, &mz_owner, &mz) ==
           CUDAQ_EDEM_OK);
  sequence[1] = mz;
  TA_CHECK(cudaq_edem_compose_chain(sequence, 2, 0, &result_owner, &result) ==
           CUDAQ_EDEM_INVALID_CONTENT);
  TA_CHECK(!result_owner.data && result.map.data == prep.map.data);
  cudaq_gf2_owned_buffer_release(&mz_owner);
  description.codes = codes;
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
  const cudaq_edem_code_presentation code = {"test", 4, 1, 1};
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
  cudaq_edem_block blocks[16];
  uint32_t n = 0;
#define BLOCK(out, in, m)                                                      \
  blocks[n++] = (cudaq_edem_block) {                                           \
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
      {1, kind != 0, kind != 2, 1, 1, kind != 0, kind != 0, n},
      &code,
      &code_id,
      &code_id,
      &faults,
      &width,
      &width,
      &width,
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
  TA_CHECK(shape.fault_ports == 4 && shape.detector_ports == 4 &&
           shape.observable_ports == 3 && shape.success_ports == 3 &&
           !shape.input_code_blocks && !shape.output_code_blocks);
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
  bounded();
  cudaq_gf2_owned_buffer_release(&source_owner);
  return 0;
}
