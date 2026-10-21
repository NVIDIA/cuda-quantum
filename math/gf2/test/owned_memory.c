/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

/* Exercise allocation routing against the standalone GF2 library. */
#include "test_allocator.h"
#include "cudaq/gf2/map.h"

typedef struct fixture {
  cudaq_gf2_bundle_type type;
  uint32_t width, rows[3];
  uint64_t offsets[4];
  cudaq_gf2_bundle bundle;
  cudaq_gf2_map_block block;
  cudaq_gf2_map_description description;
} fixture;
static void fixture_init(fixture *f) {
  memset(f, 0, sizeof(*f));
  f->type =
      (cudaq_gf2_bundle_type){"test.interface", 14, CUDAQ_GF2_INTERFACE, 1};
  f->width = 3;
  for (uint32_t i = 0; i < 3; ++i) {
    f->rows[i] = i;
    f->offsets[i] = i;
  }
  f->offsets[3] = 3;
  f->bundle = (cudaq_gf2_bundle){0, 0, &f->width};
  f->block =
      (cudaq_gf2_map_block){{0, 0}, {0, 0}, {3, 3, 3, f->offsets, f->rows}};
  f->description = (cudaq_gf2_map_description){
      1, 1, 1, 0, 1, &f->type, &f->bundle, &f->bundle, &f->block, NULL};
}
static void check_apply(const cudaq_gf2_map_view *view) {
  for (unsigned bits = 0; bits < 8; ++bits) {
    uint8_t input[3], output[3];
    for (unsigned i = 0; i < 3; ++i)
      input[i] = (bits >> i) & 1;
    TA_CHECK(cudaq_gf2_map_apply(view, input, 3, output, 3) == CUDAQ_GF2_OK);
    TA_CHECK(!memcmp(input, output, sizeof(input)));
  }
}
enum operation { CONSTRUCT, VALIDATE, READ };
static cudaq_gf2_status invoke(enum operation op, fixture *f,
                               cudaq_gf2_map_view valid,
                               const cudaq_gf2_memory *memory,
                               cudaq_gf2_owned_buffer *owner,
                               cudaq_gf2_map_view *view) {
  if (op == CONSTRUCT)
    return cudaq_gf2_map_construct(&f->description, memory, owner, view);
  if (op == VALIDATE)
    return cudaq_gf2_map_validate(&f->description, memory);
  return cudaq_gf2_map_read(valid.data, valid.size, NULL, memory, view);
}
static uint64_t failure_sites(enum operation op, fixture *f,
                              cudaq_gf2_map_view valid) {
  test_resource resource;
  test_resource_init(&resource);
  cudaq_gf2_memory memory = test_memory(&resource, &resource);
  cudaq_gf2_owned_buffer result = {0};
  cudaq_gf2_map_view view;
  TA_CHECK(invoke(op, f, valid, &memory, &result, &view) == CUDAQ_GF2_OK);
  uint64_t sites = resource.attempts;
  cudaq_gf2_owned_buffer_release(&result);
  test_resource_balanced(&resource);
  TA_CHECK(sites > 0);
  for (uint64_t fail = 1; fail <= sites; ++fail) {
    test_resource_init(&resource);
    resource.fail_on_attempt = fail;
    memory = test_memory(&resource, &resource);
    const cudaq_gf2_owned_buffer sentinel = {0};
    result = sentinel;
    view = valid;
    TA_CHECK(invoke(op, f, valid, &memory, &result, &view) ==
             CUDAQ_GF2_ALLOCATION_FAILED);
    TA_CHECK(test_same_owner(result, sentinel));
    TA_CHECK(view.data == valid.data && view.size == valid.size);
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
  cudaq_gf2_map_view view;
  {
    cudaq_gf2_memory temporary = test_memory(&scratch, &result);
    cudaq_gf2_owned_buffer source = {0};
    TA_CHECK(cudaq_gf2_map_construct(&f->description, &temporary, &source,
                                     &view) == CUDAQ_GF2_OK);
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
  TA_CHECK(cudaq_gf2_map_validate(&f->description, &scratch_only) ==
           CUDAQ_GF2_OK);
  cudaq_gf2_map_view borrowed;
  TA_CHECK(cudaq_gf2_map_read(owner.data, owner.size, NULL, &scratch_only,
                              &borrowed) == CUDAQ_GF2_OK);
  TA_CHECK(borrowed.data == owner.data && borrowed.size == owner.size);
  test_resource_balanced(&scratch);
  const uint64_t before = scratch.attempts + result.attempts;
  check_apply(&borrowed);
  TA_CHECK(before == scratch.attempts + result.attempts);
  cudaq_gf2_owned_buffer_release(&owner);
  cudaq_gf2_owned_buffer_release(&owner);
  cudaq_gf2_owned_buffer_release(NULL);
  TA_CHECK(test_same_owner(owner, (cudaq_gf2_owned_buffer){0}));
  test_resource_balanced(&result);
}
static void test_bounded(fixture *f, uint64_t construct_sites) {
  uint64_t scratch_storage[1024], result_storage[1024];
  test_resource scratch, result;
  test_resource_bump(&scratch, scratch_storage, sizeof(scratch_storage));
  test_resource_bump(&result, result_storage, sizeof(result_storage));
  cudaq_gf2_memory memory = test_memory(&scratch, &result);
  cudaq_gf2_owned_buffer owners[2] = {{0}};
  cudaq_gf2_map_view views[2];
  for (unsigned i = 0; i < 2; ++i) {
    TA_CHECK(cudaq_gf2_map_construct(&f->description, &memory, owners + i,
                                     views + i) == CUDAQ_GF2_OK);
    test_resource_reset(&scratch);
    for (unsigned j = 0; j <= i; ++j)
      check_apply(views + j);
  }
  cudaq_gf2_owned_buffer_release(owners + 1);
  check_apply(views);
  cudaq_gf2_owned_buffer_release(owners);
  test_resource_balanced(&result);
  TA_CHECK(result.arena.used > 0);

  test_resource_bump(&scratch, scratch_storage, sizeof(scratch_storage));
  scratch.fail_on_attempt = construct_sites;
  memory = test_memory(&scratch, &scratch);
  cudaq_gf2_map_view unchanged = {result_storage, 99};
  TA_CHECK(cudaq_gf2_map_construct(&f->description, &memory, owners,
                                   &unchanged) == CUDAQ_GF2_ALLOCATION_FAILED);
  test_resource_balanced(&scratch);
  TA_CHECK(scratch.arena.used > 0 && unchanged.data == result_storage &&
           unchanged.size == 99 && !owners[0].data);
  test_resource_reset(&scratch);
  scratch.fail_on_attempt = 0;
  TA_CHECK(cudaq_gf2_map_construct(&f->description, &memory, owners, views) ==
           CUDAQ_GF2_OK);
  check_apply(views);
  cudaq_gf2_owned_buffer_release(owners);
  test_resource_balanced(&scratch);

  test_resource_bump(&scratch, scratch_storage, 1);
  memory = test_memory(&scratch, &scratch);
  TA_CHECK(cudaq_gf2_map_construct(&f->description, &memory, owners, views) ==
           CUDAQ_GF2_ALLOCATION_FAILED);
  TA_CHECK(!owners[0].data && !scratch.allocations);
}
static void test_invalid_resources(fixture *f) {
  test_resource resource;
  test_resource_init(&resource);
  cudaq_gf2_memory memory = test_memory(&resource, &resource);
  cudaq_gf2_owned_buffer owner = {0};
  cudaq_gf2_map_view view = {NULL, 0};
  memory.scratch.deallocate = NULL;
  TA_CHECK(cudaq_gf2_map_construct(&f->description, &memory, &owner, &view) ==
           CUDAQ_GF2_INVALID_ARGUMENT);
  TA_CHECK(!resource.attempts && !owner.data && !view.data);
  memory = test_memory(&resource, &resource);
  memory.result.allocate = NULL;
  TA_CHECK(cudaq_gf2_map_construct(&f->description, &memory, &owner, &view) ==
           CUDAQ_GF2_INVALID_ARGUMENT);
  TA_CHECK(!resource.attempts);
  memory = test_memory(&resource, &resource);
  owner =
      (cudaq_gf2_owned_buffer){&resource, 77, test_release_sentinel, &resource};
  cudaq_gf2_owned_buffer saved = owner;
  TA_CHECK(cudaq_gf2_map_construct(&f->description, &memory, &owner, &view) ==
           CUDAQ_GF2_INVALID_ARGUMENT);
  TA_CHECK(!resource.attempts && test_same_owner(owner, saved) && !view.data);
  cudaq_gf2_allocator defaults = cudaq_gf2_default_allocator();
  TA_CHECK(defaults.allocate && defaults.deallocate);
}
int main(void) {
  fixture f;
  fixture_init(&f);
  cudaq_gf2_owned_buffer reference = {0};
  cudaq_gf2_map_view view;
  TA_CHECK(cudaq_gf2_map_construct(&f.description, NULL, &reference, &view) ==
           CUDAQ_GF2_OK);
  uint64_t sites = failure_sites(CONSTRUCT, &f, view);
  failure_sites(VALIDATE, &f, view);
  failure_sites(READ, &f, view);
  test_heap_lifetimes(&f);
  test_bounded(&f, sites);
  test_invalid_resources(&f);
  cudaq_gf2_owned_buffer_release(&reference);
  return 0;
}
