/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

// Native producer used only by build-tree tests. It is never installed.
#include "cudaq/edem/owned.hpp"
#include "cudaq/edem/python.h"
#include <nanobind/nanobind.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

namespace {
struct Tracker;
struct Resource {
  Tracker *tracker;
  bool result;
};
struct Allocation {
  void *data = nullptr;
  uint64_t bytes = 0;
  bool result = false;
};
struct Tracker {
  uint64_t attempts = 0, allocations = 0, deallocations = 0;
  uint64_t result_allocations = 0, result_deallocations = 0, live_results = 0;
  uint64_t invalid_frees = 0, capsule_destructions = 0, unconsumed_releases = 0;
  uint64_t native_moves = 0, fail_at = 0;
  Resource scratch{this, false}, result{this, true};
  std::array<Allocation, 64> ledger{};
};

void *allocate(void *context, uint64_t bytes, uint32_t alignment) {
  auto &resource = *static_cast<Resource *>(context);
  auto &tracker = *resource.tracker;
  ++tracker.attempts;
  if (tracker.attempts == tracker.fail_at || alignment > alignof(max_align_t) ||
      bytes > SIZE_MAX)
    return nullptr;
  for (auto &entry : tracker.ledger) {
    if (entry.data)
      continue;
    void *data = std::malloc(static_cast<size_t>(bytes));
    if (!data)
      return nullptr;
    entry = {data, bytes, resource.result};
    ++tracker.allocations;
    if (resource.result) {
      ++tracker.result_allocations;
      ++tracker.live_results;
    }
    return data;
  }
  return nullptr;
}
void deallocate(void *context, void *data, uint64_t bytes, uint32_t alignment) {
  auto &resource = *static_cast<Resource *>(context);
  auto &tracker = *resource.tracker;
  for (auto &entry : tracker.ledger) {
    if (entry.data != data || entry.bytes != bytes ||
        entry.result != resource.result ||
        alignment != CUDAQ_EDEM_PACKED_ALIGNMENT)
      continue;
    entry = {};
    ++tracker.deallocations;
    if (resource.result) {
      ++tracker.result_deallocations;
      --tracker.live_results;
    }
    std::memset(data, 0xdd, static_cast<size_t>(bytes));
    std::free(data);
    return;
  }
  ++tracker.invalid_frees;
}

class MovingOwner {
  cudaq::edem::PackedEDEM buffer;
  Tracker *tracker = nullptr;

public:
  MovingOwner() = default;
  MovingOwner(cudaq::edem::PackedEDEM &source, Tracker *counter)
      : buffer(std::move(source)), tracker(counter) {}
  MovingOwner(const MovingOwner &) = delete;
  MovingOwner &operator=(const MovingOwner &) = delete;
  MovingOwner(MovingOwner &&other) noexcept
      : buffer(std::move(other.buffer)), tracker(other.tracker) {
    if (tracker)
      ++tracker->native_moves;
  }
  MovingOwner &operator=(MovingOwner &&other) noexcept {
    if (this != &other) {
      buffer = std::move(other.buffer);
      tracker = other.tracker;
      if (tracker)
        ++tracker->native_moves;
    }
    return *this;
  }
  cudaq_gf2_owned_buffer detach() { return buffer.release(); }
};
MovingOwner throughReturn(MovingOwner value) { return value; }

cudaq_edem_status constructRound(const cudaq_gf2_memory *memory,
                                 cudaq::edem::PackedEDEM &owner) {
  const cudaq_edem_code_presentation codes[] = {
      {1, "repetition-3", 12, 2, 2, 1}};
  const uint32_t boundary[] = {0}, sizes[] = {2}, ids[] = {0, 1};
  const double probabilities[] = {0.01, 0.02};
  const cudaq_edem_fault_group group = {CUDAQ_EDEM_INDEPENDENT, 2, ids,
                                        probabilities};
  const cudaq_edem_fault_distribution fault = {2, 1, &group};
  const uint64_t identity_offsets[] = {0, 1, 2},
                 measurement_offsets[] = {0, 1, 1}, data_offsets[] = {0, 0, 1};
  const uint32_t identity_rows[] = {0, 1}, row[] = {0};
  const cudaq_gf2_csc_view identity = {2, 2, 2, identity_offsets,
                                       identity_rows};
  const cudaq_gf2_csc_view measurement = {2, 2, 1, measurement_offsets, row};
  const cudaq_gf2_csc_view data = {2, 2, 1, data_offsets, row};
  const cudaq_edem_block blocks[] = {
      {{CUDAQ_EDEM_LOGICAL, 0}, {CUDAQ_EDEM_LOGICAL, 0}, identity},
      {{CUDAQ_EDEM_SYNDROME, 0}, {CUDAQ_EDEM_FAULT, 0}, data},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_FAULT, 0}, measurement},
      {{CUDAQ_EDEM_VIRTUAL, 0}, {CUDAQ_EDEM_SYNDROME, 0}, identity},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_SYNDROME, 0}, identity},
      {{CUDAQ_EDEM_SYNDROME, 0}, {CUDAQ_EDEM_SYNDROME, 0}, identity},
      {{CUDAQ_EDEM_VIRTUAL, 0}, {CUDAQ_EDEM_FAULT, 0}, measurement},
      {{CUDAQ_EDEM_DETECTOR, 0}, {CUDAQ_EDEM_VIRTUAL, 0}, identity}};
  const cudaq_edem_description description = {{1, 1, 1, 1, 0, 0},
                                              {1, codes},
                                              boundary,
                                              boundary,
                                              &fault,
                                              sizes,
                                              nullptr,
                                              nullptr,
                                              8,
                                              blocks};
  return cudaq::edem::PackedEDEM::construct(description, owner, memory);
}

struct CapsuleOwner {
  cudaq_gf2_owned_buffer buffer{};
  nb::object context_keepalive;
  nb::object module_keepalive;
  Tracker *tracker = nullptr;
  bool corrupted = false;
  ~CapsuleOwner() {
    if (tracker) {
      ++tracker->capsule_destructions;
      if (buffer.data)
        ++tracker->unconsumed_releases;
    }
    cudaq_gf2_owned_buffer_release(&buffer);
  }
};
void destroyCapsule(PyObject *capsule) noexcept {
  auto *owner = static_cast<CapsuleOwner *>(PyCapsule_GetContext(capsule));
  delete owner;
}
CapsuleOwner *getOwner(nb::handle capsule) {
  if (!PyCapsule_CheckExact(capsule.ptr()))
    throw nb::type_error("expected native producer capsule");
  auto *owner =
      static_cast<CapsuleOwner *>(PyCapsule_GetContext(capsule.ptr()));
  if (!owner)
    throw nb::value_error("missing producer record");
  return owner;
}

nb::object makeCapsule(nb::object tracker_object, bool corrupt,
                       bool use_default, bool wrong_name) {
  auto *tracker = nb::cast<Tracker *>(tracker_object);
  auto owner = std::make_unique<CapsuleOwner>();
  owner->context_keepalive = std::move(tracker_object);
  owner->module_keepalive = nb::module_::import_("cudaq_edem._adoption_test");
  owner->tracker = tracker;
  cudaq::edem::PackedEDEM raw;
  {
    cudaq_gf2_memory temporary{{&tracker->scratch, allocate, deallocate},
                               {&tracker->result, allocate, deallocate}};
    auto status = constructRound(use_default ? nullptr : &temporary, raw);
    std::memset(&temporary, 0x5a, sizeof(temporary));
    if (status == CUDAQ_EDEM_ALLOCATION_FAILED)
      throw std::bad_alloc();
    if (status != CUDAQ_EDEM_OK)
      throw nb::value_error(cudaq_edem_status_string(status));
  }
  MovingOwner first(raw, tracker);
  std::array<MovingOwner, 2> container;
  container[1] = throughReturn(std::move(first));
  struct Wrapper {
    MovingOwner value;
  } wrapper{std::move(container[1])};
  owner->buffer = wrapper.value.detach();
  if (corrupt) {
    static_cast<unsigned char *>(const_cast<void *>(owner->buffer.data))[0] ^=
        1;
    owner->corrupted = true;
  }
  const char *name =
      wrong_name ? "cudaq_edem.unsupported" : CUDAQ_EDEM_OWNED_BUFFER_CAPSULE;
  nb::object capsule = nb::steal<nb::object>(
      PyCapsule_New(&owner->buffer, name, destroyCapsule));
  if (!capsule)
    throw nb::python_error();
  if (PyCapsule_SetContext(capsule.ptr(), owner.get()) != 0)
    throw nb::python_error();
  owner.release();
  return capsule;
}
} // namespace

NB_MODULE(_adoption_test, m) {
  nb::class_<Tracker>(m, "Tracker", nb::is_weak_referenceable())
      .def(nb::init<>())
      .def_ro("attempts", &Tracker::attempts)
      .def_ro("allocations", &Tracker::allocations)
      .def_ro("deallocations", &Tracker::deallocations)
      .def_ro("result_allocations", &Tracker::result_allocations)
      .def_ro("result_deallocations", &Tracker::result_deallocations)
      .def_ro("live_results", &Tracker::live_results)
      .def_ro("invalid_frees", &Tracker::invalid_frees)
      .def_ro("capsule_destructions", &Tracker::capsule_destructions)
      .def_ro("unconsumed_releases", &Tracker::unconsumed_releases)
      .def_ro("native_moves", &Tracker::native_moves)
      .def_rw("fail_at", &Tracker::fail_at);
  m.def("make_capsule", &makeCapsule, "tracker"_a, "corrupt"_a = false,
        "use_default"_a = false, "wrong_name"_a = false);
  m.def("payload_address", [](nb::handle capsule) {
    return reinterpret_cast<uintptr_t>(getOwner(capsule)->buffer.data);
  });
  m.def("source_empty", [](nb::handle capsule) {
    auto &value = getOwner(capsule)->buffer;
    return !value.data && !value.size && !value.deallocate && !value.context;
  });
  m.def("consumed", [](nb::handle capsule) {
    return PyCapsule_IsValid(capsule.ptr(), CUDAQ_EDEM_USED_BUFFER_CAPSULE) !=
           0;
  });
  m.def("repair", [](nb::handle capsule) {
    auto *owner = getOwner(capsule);
    if (!owner->corrupted || !owner->buffer.data)
      throw nb::value_error("capsule has no corrupt, unconsumed buffer");
    static_cast<unsigned char *>(const_cast<void *>(owner->buffer.data))[0] ^=
        1;
    owner->corrupted = false;
  });
}
