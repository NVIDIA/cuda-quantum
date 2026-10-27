#include "../../math/gf2/test/test_allocator.h"
/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#include "cudaq/edem/core.h"

enum { FAULT_INDICATORS = 2048, SCRATCH_WORDS = 8192, RESULT_WORDS = 16384 };
typedef struct fixture {
  uint32_t ids[FAULT_INDICATORS], rows[FAULT_INDICATORS], detector_size;
  double probabilities[FAULT_INDICATORS];
  uint64_t offsets[FAULT_INDICATORS + 1];
  cudaq_edem_fault_group group;
  cudaq_edem_fault_distribution fault;
  cudaq_edem_matrix_block block;
  cudaq_edem_description description;
} fixture;
static void fixture_init(fixture *f) {
  memset(f, 0, sizeof(*f));
  for (uint32_t i = 0; i < FAULT_INDICATORS; ++i) {
    f->ids[i] = i;
    f->probabilities[i] = 0.001;
    f->offsets[i] = i;
  }
  f->offsets[FAULT_INDICATORS] = FAULT_INDICATORS;
  f->detector_size = 1;
  f->group = (cudaq_edem_fault_group){CUDAQ_EDEM_INDEPENDENT, FAULT_INDICATORS,
                                      f->ids, f->probabilities};
  f->fault = (cudaq_edem_fault_distribution){FAULT_INDICATORS, 1, &f->group};
  f->block = (cudaq_edem_matrix_block){
      {CUDAQ_EDEM_DETECTOR, 0},
      {CUDAQ_EDEM_FAULT, 0},
      {1, FAULT_INDICATORS, FAULT_INDICATORS, f->offsets, f->rows}};
  f->description.shape.fault_ports_count = 1;
  f->description.shape.detector_ports_count = 1;
  f->description.block_count = 1;
  f->description.fault_ports_distributions = &f->fault;
  f->description.detector_ports_sizes = &f->detector_size;
  f->description.blocks = &f->block;
}
static void check_apply(const cudaq_edem_view *view) {
  uint8_t input[FAULT_INDICATORS], output;
  for (unsigned pattern = 0; pattern < 3; ++pattern) {
    uint8_t parity = 0;
    for (uint32_t i = 0; i < FAULT_INDICATORS; ++i) {
      input[i] = pattern == 0 ? 0 : pattern == 1 ? i == 0 : i % 3 == 0;
      parity ^= input[i];
    }
    TA_CHECK(cudaq_gf2_map_apply(&view->map, input, sizeof(input), &output,
                                 1) == CUDAQ_GF2_OK);
    TA_CHECK(output == parity);
  }
}
enum operation { CONSTRUCT, VALIDATE, READ, CHECK };
static cudaq_edem_status invoke(enum operation op, fixture *f,
                                cudaq_edem_view valid,
                                const cudaq_gf2_memory *memory,
                                cudaq_gf2_owned_buffer *owner,
                                cudaq_edem_view *view) {
  switch (op) {
  case CONSTRUCT:
    return cudaq_edem_construct(&f->description, memory, owner, view);
  case VALIDATE:
    return cudaq_edem_validate(&f->description, memory);
  case READ:
    return cudaq_edem_read(valid.map.data, valid.map.size, memory, view);
  case CHECK:
    return cudaq_edem_check(&valid.map, memory, view);
  }
  abort();
}
static uint64_t failure_sites(enum operation op, fixture *f,
                              cudaq_edem_view valid) {
  test_resource resource;
  test_resource_init(&resource);
  cudaq_gf2_memory memory = test_memory(&resource, &resource);
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_edem_view view;
  TA_CHECK(invoke(op, f, valid, &memory, &owner, &view) == CUDAQ_EDEM_OK);
  const uint64_t sites = resource.attempts;
  cudaq_gf2_owned_buffer_release(&owner);
  test_resource_balanced(&resource);
  TA_CHECK(sites > 0);
  for (uint64_t fail = 1; fail <= sites; ++fail) {
    test_resource_init(&resource);
    resource.fail_on_attempt = fail;
    memory = test_memory(&resource, &resource);
    const cudaq_gf2_owned_buffer sentinel = {0};
    owner = sentinel;
    view = valid;
    TA_CHECK(invoke(op, f, valid, &memory, &owner, &view) ==
             CUDAQ_EDEM_ALLOCATION_FAILED);
    TA_CHECK(test_same_owner(owner, sentinel));
    TA_CHECK(view.map.data == valid.map.data &&
             view.map.size == valid.map.size);
    TA_CHECK(resource.attempts == fail);
    test_resource_balanced(&resource);
  }
  return sites;
}
static void test_heap_lifetimes(fixture *f) {
  test_resource scratch, result;
  test_resource_init(&scratch);
  test_resource_init(&result);
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_edem_view view;
  {
    cudaq_gf2_memory temporary = test_memory(&scratch, &result);
    cudaq_gf2_owned_buffer source = {0};
    TA_CHECK(cudaq_edem_construct(&f->description, &temporary, &source,
                                  &view) == CUDAQ_EDEM_OK);
    owner = source;
    source = (cudaq_gf2_owned_buffer){0};
    memset(&temporary, 0x5a, sizeof(temporary));
    TA_CHECK(!source.data);
  }
  test_resource_balanced(&scratch);
  TA_CHECK(owner.context == &result &&
           owner.deallocate == test_resource_deallocate);
  TA_CHECK(result.live_bytes == owner.size && result.allocations == 1);
  cudaq_gf2_memory scratch_only = {test_resource_allocator(&scratch), {0}};
  TA_CHECK(cudaq_edem_validate(&f->description, &scratch_only) ==
           CUDAQ_EDEM_OK);
  scratch.largest_request = 0;
  cudaq_edem_view borrowed, checked;
  TA_CHECK(cudaq_edem_read(owner.data, owner.size, &scratch_only, &borrowed) ==
           CUDAQ_EDEM_OK);
  TA_CHECK(borrowed.map.data == owner.data && borrowed.map.size == owner.size);
  TA_CHECK(cudaq_edem_check(&borrowed.map, &scratch_only, &checked) ==
           CUDAQ_EDEM_OK);
  TA_CHECK(checked.map.data == owner.data && checked.map.size == owner.size);
  TA_CHECK(scratch.largest_request >= FAULT_INDICATORS / 8);
  test_resource_balanced(&scratch);
  uint64_t attempts = scratch.attempts + result.attempts;
  check_apply(&checked);
  TA_CHECK(attempts == scratch.attempts + result.attempts);
  cudaq_gf2_owned_buffer_release(&owner);
  cudaq_gf2_owned_buffer_release(&owner);
  TA_CHECK(test_same_owner(owner, (cudaq_gf2_owned_buffer){0}));
  test_resource_balanced(&result);
}
static void test_shared_heap(fixture *f) {
  test_resource shared;
  test_resource_init(&shared);
  cudaq_gf2_memory memory = test_memory(&shared, &shared);
  cudaq_gf2_owned_buffer owners[2] = {{0}};
  cudaq_edem_view views[2];
  for (unsigned i = 0; i < 2; ++i)
    TA_CHECK(cudaq_edem_construct(&f->description, &memory, owners + i,
                                  views + i) == CUDAQ_EDEM_OK);
  cudaq_gf2_owned_buffer_release(owners + 1);
  check_apply(views);
  cudaq_gf2_owned_buffer_release(owners);
  test_resource_balanced(&shared);
}
static void test_bounded(fixture *f, uint64_t construct_sites) {
  uint64_t scratch_storage[SCRATCH_WORDS], result_storage[RESULT_WORDS];
  test_resource scratch, result;
  test_resource_bump(&scratch, scratch_storage, sizeof(scratch_storage));
  test_resource_bump(&result, result_storage, sizeof(result_storage));
  cudaq_gf2_memory memory = test_memory(&scratch, &result);
  cudaq_gf2_owned_buffer owners[2] = {{0}};
  cudaq_edem_view views[2];
  for (unsigned i = 0; i < 2; ++i) {
    TA_CHECK(cudaq_edem_construct(&f->description, &memory, owners + i,
                                  views + i) == CUDAQ_EDEM_OK);
    test_resource_reset(&scratch);
    for (unsigned j = 0; j <= i; ++j)
      check_apply(views + j);
  }
  cudaq_gf2_owned_buffer_release(owners + 1);
  check_apply(views);
  cudaq_gf2_owned_buffer_release(owners);
  test_resource_balanced(&result);
  TA_CHECK(result.arena.used > 0);

  test_resource_bump(&scratch, result_storage, sizeof(result_storage));
  scratch.fail_on_attempt = construct_sites;
  memory = test_memory(&scratch, &scratch);
  cudaq_edem_view unchanged = {{scratch_storage, 99}};
  TA_CHECK(cudaq_edem_construct(&f->description, &memory, owners, &unchanged) ==
           CUDAQ_EDEM_ALLOCATION_FAILED);
  test_resource_balanced(&scratch);
  TA_CHECK(scratch.arena.used > 0 && unchanged.map.data == scratch_storage &&
           unchanged.map.size == 99 && !owners[0].data);
  test_resource_reset(&scratch);
  scratch.fail_on_attempt = 0;
  TA_CHECK(cudaq_edem_construct(&f->description, &memory, owners, views) ==
           CUDAQ_EDEM_OK);
  check_apply(views);
  cudaq_gf2_owned_buffer_release(owners);
  test_resource_balanced(&scratch);
  test_resource_bump(&scratch, scratch_storage, 1);
  memory = test_memory(&scratch, &scratch);
  TA_CHECK(cudaq_edem_construct(&f->description, &memory, owners, views) ==
           CUDAQ_EDEM_ALLOCATION_FAILED);
  TA_CHECK(!owners[0].data && !scratch.allocations);
}
static void test_invalid_resources(fixture *f) {
  test_resource resource;
  test_resource_init(&resource);
  cudaq_gf2_memory memory = test_memory(&resource, &resource);
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_edem_view view = {{NULL, 0}};
  memory.scratch.allocate = NULL;
  TA_CHECK(cudaq_edem_construct(&f->description, &memory, &owner, &view) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  TA_CHECK(!resource.attempts && !owner.data && !view.map.data);
  memory = test_memory(&resource, &resource);
  memory.result.deallocate = NULL;
  TA_CHECK(cudaq_edem_construct(&f->description, &memory, &owner, &view) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  TA_CHECK(!resource.attempts);
  memory = test_memory(&resource, &resource);
  owner =
      (cudaq_gf2_owned_buffer){&resource, 77, test_release_sentinel, &resource};
  cudaq_gf2_owned_buffer saved = owner;
  TA_CHECK(cudaq_edem_construct(&f->description, &memory, &owner, &view) ==
           CUDAQ_EDEM_INVALID_ARGUMENT);
  TA_CHECK(!resource.attempts && test_same_owner(owner, saved) &&
           !view.map.data);
}
int main(void) {
  fixture f;
  fixture_init(&f);
  cudaq_gf2_owned_buffer reference = {0};
  cudaq_edem_view valid;
  TA_CHECK(cudaq_edem_construct(&f.description, NULL, &reference, &valid) ==
           CUDAQ_EDEM_OK);
  uint64_t sites = failure_sites(CONSTRUCT, &f, valid);
  failure_sites(VALIDATE, &f, valid);
  failure_sites(READ, &f, valid);
  failure_sites(CHECK, &f, valid);
  test_heap_lifetimes(&f);
  test_shared_heap(&f);
  test_bounded(&f, sites);
  test_invalid_resources(&f);
  cudaq_gf2_owned_buffer_release(&reference);
  return 0;
}
