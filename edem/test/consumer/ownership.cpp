/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#include "cudaq/edem/owned.hpp"
#include "cudaq/gf2/bump_allocator.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

using cudaq::edem::PackedEDEM;

static_assert(std::is_nothrow_move_constructible_v<PackedEDEM>);
static_assert(std::is_nothrow_move_assignable_v<PackedEDEM>);
static_assert(!std::is_copy_constructible_v<PackedEDEM>);
static_assert(!std::is_copy_assignable_v<PackedEDEM>);

template <class T, class = void>
struct has_rvalue_view : std::false_type {};
template <class T>
struct has_rvalue_view<T, std::void_t<decltype(std::declval<T &&>().view())>>
    : std::true_type {};
static_assert(!has_rvalue_view<PackedEDEM>::value);
static_assert(!has_rvalue_view<const PackedEDEM>::value);

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #condition);     \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

namespace {
struct Resource {
  struct Allocation {
    void *data = nullptr;
    uint64_t size = 0;
  };
  std::array<Allocation, 64> ledger{};
  uint64_t attempts = 0, allocations = 0, releases = 0, live = 0, fail_at = 0;

  static void *allocate(void *context, uint64_t size, uint32_t alignment) {
    auto &self = *static_cast<Resource *>(context);
    CHECK(alignment == CUDAQ_EDEM_PACKED_ALIGNMENT && size <= SIZE_MAX);
    if (++self.attempts == self.fail_at)
      return nullptr;
    for (auto &entry : self.ledger) {
      if (entry.data)
        continue;
      void *data = std::malloc(static_cast<size_t>(size));
      if (!data)
        return nullptr;
      entry = {data, size};
      ++self.allocations;
      ++self.live;
      return data;
    }
    return nullptr;
  }
  static void deallocate(void *context, void *data, uint64_t size,
                         uint32_t alignment) {
    auto &self = *static_cast<Resource *>(context);
    CHECK(alignment == CUDAQ_EDEM_PACKED_ALIGNMENT);
    for (auto &entry : self.ledger)
      if (entry.data == data) {
        CHECK(entry.size == size);
        std::free(data);
        entry = {};
        ++self.releases;
        --self.live;
        return;
      }
    CHECK(false);
  }
  cudaq_gf2_allocator provider() { return {this, allocate, deallocate}; }
};

const cudaq_edem_description &description() {
  static const uint32_t id[] = {0}, sizes[] = {1};
  static const uint64_t offsets[] = {0, 1};
  static const double probability[] = {0.125};
  static const cudaq_edem_fault_group group = {CUDAQ_EDEM_INDEPENDENT, 1, id,
                                               probability};
  static const cudaq_edem_fault_distribution fault = {1, 1, &group};
  static const cudaq_edem_matrix_block block = {
      {CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_FAULT, 0}, {1, 1, 1, offsets, id}};
  static const cudaq_edem_description value = {{0, 0, 1, 1, 0, 0},
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               &fault,
                                               sizes,
                                               nullptr,
                                               nullptr,
                                               1,
                                               &block};
  return value;
}

void apply(const cudaq_edem_view &view) {
  uint8_t input = 1, output = 0;
  CHECK(cudaq_gf2_map_apply(&view.map, &input, 1, &output, 1) == CUDAQ_GF2_OK);
  CHECK(output == 1);
}

bool same(cudaq_gf2_owned_buffer a, cudaq_gf2_owned_buffer b) {
  return a.data == b.data && a.size == b.size && a.deallocate == b.deallocate &&
         a.context == b.context;
}

void test_moves_and_handoff() {
  Resource scratch, result;
  {
    cudaq_gf2_memory memory{scratch.provider(), result.provider()};
    PackedEDEM first;
    CHECK(PackedEDEM::construct(description(), first, &memory) ==
          CUDAQ_EDEM_OK);
    CHECK(scratch.live == 0 && result.live == 1);
    const auto copied_view = first.view();
    const auto &source_slot = first.view();
    const void *address = first.data();
    memory = {}; // Destruction uses the captured callback, not this policy.

    PackedEDEM moved(std::move(first));
    CHECK(first.empty() && !source_slot.map.data && !source_slot.map.size);
    CHECK(moved.data() == address);
    PackedEDEM last;
    last = std::move(moved);
    CHECK(moved.empty() && last.data() == address);
    const auto &released_slot = last.view();
    auto record = last.release();
    CHECK(last.empty() && !released_slot.map.data && !released_slot.map.size);

    memory = {scratch.provider(), result.provider()};
    PackedEDEM adopted;
    CHECK(PackedEDEM::adopt(record, adopted, &memory) == CUDAQ_EDEM_OK);
    CHECK(same(record, {}));
    CHECK(adopted.data() == address && result.allocations == 1);
    apply(copied_view);
    apply(adopted.view());
    adopted.reset();
    CHECK(adopted.empty() && result.live == 0 && result.releases == 1);
  }
  CHECK(scratch.live == 0 && scratch.allocations == scratch.releases);
  CHECK(result.allocations == result.releases);
}

void test_clone_and_defaults() {
  PackedEDEM original, copy;
  CHECK(PackedEDEM::construct(description(), original) == CUDAQ_EDEM_OK);
  CHECK(original.clone(copy) == CUDAQ_EDEM_OK);
  CHECK(copy.data() != original.data() && copy.size() == original.size());
  CHECK(std::memcmp(copy.data(), original.data(), copy.size()) == 0);
  CHECK(copy.view().map.data == copy.data());
  CHECK(original.clone(original) == CUDAQ_EDEM_OK);
  CHECK(std::memcmp(copy.data(), original.data(), copy.size()) == 0);
  original.reset();
  apply(copy.view());
  PackedEDEM empty;
  const void *address = copy.data();
  CHECK(empty.clone(copy) == CUDAQ_EDEM_INVALID_ARGUMENT);
  CHECK(copy.data() == address);
  auto record = copy.release();
  CHECK(PackedEDEM::adopt(record, original) == CUDAQ_EDEM_OK);
  apply(original.view());
}

void test_failures_preserve_owners() {
  Resource scratch, result;
  {
    cudaq_gf2_memory memory{scratch.provider(), result.provider()};
    PackedEDEM source, destination;
    CHECK(PackedEDEM::construct(description(), source, &memory) ==
          CUDAQ_EDEM_OK);
    CHECK(PackedEDEM::construct(description(), destination, &memory) ==
          CUDAQ_EDEM_OK);
    const void *address = destination.data();
    result.fail_at = result.attempts + 1;
    CHECK(source.clone(destination, &memory) == CUDAQ_EDEM_ALLOCATION_FAILED);
    CHECK(destination.data() == address && result.live == 2);
    result.fail_at = 0;

    scratch.fail_at = scratch.attempts + 1;
    CHECK(source.clone(destination, &memory) == CUDAQ_EDEM_ALLOCATION_FAILED);
    CHECK(destination.data() == address && result.live == 2 &&
          scratch.live == 0);
    scratch.fail_at = scratch.attempts + 1;
    CHECK(PackedEDEM::construct(description(), destination, &memory) ==
          CUDAQ_EDEM_ALLOCATION_FAILED);
    CHECK(destination.data() == address && result.live == 2 &&
          scratch.live == 0);
    scratch.fail_at = 0;

    auto incomplete = memory;
    incomplete.result.deallocate = nullptr;
    CHECK(source.clone(destination, &incomplete) ==
          CUDAQ_EDEM_INVALID_ARGUMENT);
    CHECK(destination.data() == address && result.live == 2);
    auto invalid = description();
    invalid.blocks = nullptr;
    CHECK(PackedEDEM::construct(invalid, destination, &memory) ==
          CUDAQ_EDEM_INVALID_ARGUMENT);
    CHECK(destination.data() == address && result.live == 2);
    cudaq_gf2_owned_buffer empty{};
    CHECK(PackedEDEM::adopt(empty, destination, &memory) ==
          CUDAQ_EDEM_INVALID_ARGUMENT);
    CHECK(destination.data() == address && same(empty, {}));

    auto record = source.release();
    auto saved = record;
    auto *bytes = static_cast<unsigned char *>(const_cast<void *>(record.data));
    bytes[0] ^= 1;
    cudaq_edem_view rejected{};
    auto native_status =
        cudaq_edem_read(record.data, record.size, &memory, &rejected);
    CHECK(native_status != CUDAQ_EDEM_OK && !rejected.map.data);
    CHECK(PackedEDEM::adopt(record, destination, &memory) == native_status);
    CHECK(same(record, saved) && destination.data() == address &&
          result.live == 2);
    bytes[0] ^= 1;
    scratch.fail_at = scratch.attempts + 1;
    CHECK(PackedEDEM::adopt(record, destination, &memory) ==
          CUDAQ_EDEM_ALLOCATION_FAILED);
    CHECK(same(record, saved) && destination.data() == address &&
          result.live == 2);
    scratch.fail_at = 0;
    CHECK(PackedEDEM::adopt(record, destination, &memory) == CUDAQ_EDEM_OK);
    CHECK(same(record, {}) && destination.data() == saved.data &&
          result.live == 1);
    apply(destination.view());
  }
  CHECK(scratch.live == 0 && scratch.allocations == scratch.releases);
  CHECK(result.live == 0 && result.allocations == result.releases);
}

void test_bounded_scratch_reuse() {
  alignas(std::max_align_t) std::array<uint8_t, 16384> scratch{}, result{};
  cudaq_gf2_bump_allocator temporary{scratch.data(), scratch.size(), 0};
  cudaq_gf2_bump_allocator retained{result.data(), result.size(), 0};
  cudaq_gf2_memory memory{cudaq_gf2_bump_resource(&temporary),
                          cudaq_gf2_bump_resource(&retained)};
  PackedEDEM first, copy, adopted;
  CHECK(PackedEDEM::construct(description(), first, &memory) == CUDAQ_EDEM_OK);
  cudaq_gf2_bump_reset(&temporary);
  CHECK(first.clone(copy, &memory) == CUDAQ_EDEM_OK);
  CHECK(copy.data() != first.data());
  cudaq_gf2_bump_reset(&temporary);
  auto record = first.release();
  CHECK(PackedEDEM::adopt(record, adopted, &memory) == CUDAQ_EDEM_OK);
  cudaq_gf2_bump_reset(&temporary);
  apply(copy.view());
  apply(adopted.view());
}
} // namespace

int main() {
  test_moves_and_handoff();
  test_clone_and_defaults();
  test_failures_preserve_owners();
  test_bounded_scratch_reuse();
  return 0;
}
