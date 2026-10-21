/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#include "cudaq/edem/composition.hpp"
#include <array>
#include <cstdlib>
#include <type_traits>
#include <utility>

using cudaq::edem::PackedEDEM;

namespace {
template <class T, class = void>
struct HasRvalueView : std::false_type {};
template <class T>
struct HasRvalueView<T, std::void_t<decltype(std::declval<T &&>().view())>>
    : std::true_type {};
static_assert(!HasRvalueView<PackedEDEM>::value);
static_assert(!HasRvalueView<const PackedEDEM>::value);
static_assert(std::is_nothrow_move_constructible_v<PackedEDEM>);
static_assert(std::is_nothrow_move_assignable_v<PackedEDEM>);
static_assert(noexcept(
    cudaq::edem::compose_chain(nullptr, 0, std::declval<PackedEDEM &>())));
static_assert(noexcept(cudaq::edem::project_DEM(
    std::declval<const PackedEDEM &>(), std::declval<PackedEDEM &>())));

void check(bool condition) {
  if (!condition)
    std::abort();
}

struct Resource {
  uint64_t attempts = 0, fail_at = 0, live = 0;
  static void *allocate(void *context, uint64_t bytes, uint32_t alignment) {
    auto &self = *static_cast<Resource *>(context);
    if (++self.attempts == self.fail_at)
      return nullptr;
    auto heap = cudaq_gf2_default_allocator();
    void *data = heap.allocate(heap.context, bytes, alignment);
    if (data)
      ++self.live;
    return data;
  }
  static void deallocate(void *context, void *data, uint64_t bytes,
                         uint32_t alignment) {
    auto &self = *static_cast<Resource *>(context);
    check(self.live != 0);
    --self.live;
    auto heap = cudaq_gf2_default_allocator();
    heap.deallocate(heap.context, data, bytes, alignment);
  }
  cudaq_gf2_memory memory() {
    cudaq_gf2_allocator allocator{this, allocate, deallocate};
    return {allocator, allocator};
  }
};

template <class Operation>
void failure_sites(Operation operation) {
  Resource resource;
  auto memory = resource.memory();
  check(operation(&memory) == CUDAQ_EDEM_OK);
  check(!resource.live);
  const auto attempts = resource.attempts;
  check(attempts > 0);
  for (uint64_t fail = 1; fail <= attempts; ++fail) {
    resource = {0, fail, 0};
    check(operation(&memory) == CUDAQ_EDEM_ALLOCATION_FAILED);
    check(!resource.live);
  }
}
} // namespace

int main() {
  cudaq_edem_description empty{};
  PackedEDEM value, copy, composed;
  check(PackedEDEM::construct(empty, value) == CUDAQ_EDEM_OK);
  check(value.clone(copy) == CUDAQ_EDEM_OK);
  std::array occurrences{value.view(), copy.view()};
  check(cudaq::edem::compose_chain(occurrences.data(), occurrences.size(),
                                   composed) == CUDAQ_EDEM_OK);
  uint32_t closed = 0;
  check(cudaq_edem_is_dem(&composed.view(), &closed) == CUDAQ_EDEM_OK &&
        closed);
  auto address = composed.view().map.data;
  auto moved = std::move(composed);
  check(composed.empty() && address == moved.view().map.data);
  failure_sites([&](const cudaq_gf2_memory *memory) {
    PackedEDEM result;
    check(value.clone(result) == CUDAQ_EDEM_OK);
    const void *sentinel = result.data();
    auto status = cudaq::edem::compose_chain(
        occurrences.data(), occurrences.size(), result, memory);
    check(status == CUDAQ_EDEM_OK || result.data() == sentinel);
    return status;
  });
  failure_sites([&](const cudaq_gf2_memory *memory) {
    PackedEDEM result;
    check(value.clone(result) == CUDAQ_EDEM_OK);
    const void *sentinel = result.data();
    auto status = cudaq::edem::project_DEM(moved, result, memory);
    check(status == CUDAQ_EDEM_OK || result.data() == sentinel);
    return status;
  });
  PackedEDEM projected;
  check(cudaq::edem::project_DEM(moved, projected) == CUDAQ_EDEM_OK);
  check(projected.data() != moved.data());
  // Projection into the source owner is supported: replacement happens only
  // after the independent result has been validated and adopted.
  check(cudaq::edem::project_DEM(projected, projected) == CUDAQ_EDEM_OK);
  check(cudaq_gf2_map_apply(&projected.view().map, nullptr, 0, nullptr, 0) ==
        CUDAQ_GF2_OK);
  auto record = projected.release();
  PackedEDEM adopted;
  check(PackedEDEM::adopt(record, adopted) == CUDAQ_EDEM_OK && !record.data);
  check(cudaq::edem::compose_chain(nullptr, 0, adopted) ==
        CUDAQ_EDEM_INVALID_ARGUMENT);
  return 0;
}
