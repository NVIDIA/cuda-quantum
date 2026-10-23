/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#include "cudaq/edem/composition.hpp"
#include "cudaq/edem/noise.hpp"
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
static_assert(!std::is_copy_constructible_v<cudaq::edem::ColumnMapping>);
static_assert(std::is_nothrow_move_constructible_v<cudaq::edem::ColumnMapping>);
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
  failure_sites([&](const cudaq_gf2_memory *memory) {
    PackedEDEM result;
    return cudaq::edem::factorize_dem(projected, result, memory);
  });
  failure_sites([&](const cudaq_gf2_memory *memory) {
    PackedEDEM result;
    cudaq::edem::ColumnMapping mapping;
    return cudaq::edem::coalesce_dem_columns(projected, result, mapping,
                                             memory);
  });
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
  uint32_t ids[] = {0, 1}, rows[] = {0, 0}, width = 1;
  uint64_t offsets[] = {0, 1, 2};
  double probabilities[] = {.1, .2};
  cudaq_edem_fault_group group{CUDAQ_EDEM_INDEPENDENT, 2, ids, probabilities};
  cudaq_edem_fault_distribution fault{2, 1, &group};
  cudaq_edem_block block{{CUDAQ_EDEM_DETECTOR, 0},
                         {CUDAQ_EDEM_FAULT, 0},
                         {1, 2, 2, offsets, rows}};
  cudaq_edem_description description{};
  description.shape.fault_ports = description.shape.detector_ports = 1;
  description.shape.blocks = 1;
  description.faults = &fault;
  description.detector_sizes = &width;
  description.blocks = &block;
  check(PackedEDEM::construct(description, value) == CUDAQ_EDEM_OK);
  cudaq::edem::ColumnMapping mapping;
  check(cudaq::edem::factorize_dem(value, copy) == CUDAQ_EDEM_OK);
  check(cudaq::edem::coalesce_dem_columns(copy, adopted, mapping) ==
        CUDAQ_EDEM_OK);
  check(mapping.size() == 2 && mapping.data()[0] == 0 &&
        mapping.data()[1] == 0);
  auto mapping_address = mapping.data();
  auto moved_mapping = std::move(mapping);
  check(!mapping.size() && moved_mapping.data() == mapping_address);
  auto mapping_record = moved_mapping.release();
  check(!moved_mapping.size() && mapping_record.size == 2 * sizeof(uint64_t));
  cudaq_gf2_owned_buffer_release(&mapping_record);
  return 0;
}
