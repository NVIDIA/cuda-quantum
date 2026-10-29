/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

// Python.h (via nanobind) must precede every other header, or glibc's feature
// macros (_POSIX_C_SOURCE, _XOPEN_SOURCE) are redefined by pyconfig.h.
#include <nanobind/nanobind.h>

#include "cudaq/edem/composition.h"
#include "cudaq/edem/core.h"
#include "cudaq/edem/format.h"
#include "cudaq/edem/noise.h"
#include "cudaq/edem/python.h"
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;

// Owners are private to this extension. Only packed values leave the package.
namespace cudaq_edem_python {
inline namespace v0 {

static void check(cudaq_edem_status status) {
  if (status == CUDAQ_EDEM_ALLOCATION_FAILED)
    throw std::bad_alloc();
  if (status != CUDAQ_EDEM_OK)
    throw nb::value_error(cudaq_edem_status_string(status));
}

static void check_gf2(cudaq_gf2_status status) {
  if (status == CUDAQ_GF2_ALLOCATION_FAILED)
    throw std::bad_alloc();
  if (status != CUDAQ_GF2_OK)
    throw nb::value_error(status == CUDAQ_GF2_INSUFFICIENT_CAPACITY
                              ? "insufficient GF(2) capacity"
                              : "invalid GF(2) matrix or vector");
}

struct SparseMatrix {
  uint32_t rows = 0, columns = 0;
  std::vector<uint64_t> offsets{0};
  std::vector<uint32_t> indices;

  SparseMatrix() = default;
  SparseMatrix(uint32_t r, uint32_t c,
               const std::vector<std::pair<uint32_t, uint32_t>> &entries)
      : rows(r), columns(c), offsets(static_cast<size_t>(c) + 1),
        indices(entries.size()) {
    std::vector<cudaq_gf2_entry> input, scratch(entries.size());
    input.reserve(entries.size());
    for (auto [row, column] : entries)
      input.push_back({row, column});
    cudaq_gf2_csc_view result;
    check_gf2(cudaq_gf2_csc_construct(
        r, c, input.data(), input.size(), offsets.data(), offsets.size(),
        indices.data(), indices.size(), scratch.data(),
        scratch.size() * sizeof(cudaq_gf2_entry), &result));
    indices.resize(result.nonzeros);
  }

  explicit SparseMatrix(cudaq_gf2_csc_view matrix)
      : rows(matrix.rows), columns(matrix.columns),
        offsets(matrix.column_offsets,
                matrix.column_offsets + static_cast<uint64_t>(columns) + 1) {
    if (matrix.nonzeros)
      indices.assign(matrix.row_indices, matrix.row_indices + matrix.nonzeros);
  }

  cudaq_gf2_csc_view view() const {
    return {rows, columns, indices.size(), offsets.data(), indices.data()};
  }

  std::vector<std::pair<uint32_t, uint32_t>> entries() const {
    std::vector<std::pair<uint32_t, uint32_t>> result;
    result.reserve(indices.size());
    for (uint64_t c = 0; c < columns; ++c)
      for (uint64_t p = offsets[c]; p < offsets[c + 1]; ++p)
        result.emplace_back(indices[p], static_cast<uint32_t>(c));
    return result;
  }

  std::vector<uint8_t>
  apply(const std::vector<uint8_t> &input,
        const std::optional<std::vector<uint8_t>> &offset) const {
    if (offset && offset->size() != rows)
      throw nb::value_error("affine offset must have one bit per matrix row");
    auto matrix = view();
    std::vector<uint8_t> output(rows);
    check_gf2(cudaq_gf2_csc_apply(&matrix, input.data(), input.size(),
                                  offset ? offset->data() : nullptr,
                                  output.data(), output.size()));
    return output;
  }
};

enum class PortKind : uint32_t {
  FAULT = CUDAQ_EDEM_FAULT,
  VIRTUAL = CUDAQ_EDEM_VIRTUAL,
  SYNDROME = CUDAQ_EDEM_SYNDROME,
  LOGICAL = CUDAQ_EDEM_LOGICAL,
  DETECTOR = CUDAQ_EDEM_DETECTOR,
  OBSERVABLE = CUDAQ_EDEM_OBSERVABLE,
  SUCCESS = CUDAQ_EDEM_SUCCESS
};

enum class EndpointDirection : uint32_t {
  INPUT = CUDAQ_EDEM_INPUT,
  OUTPUT = CUDAQ_EDEM_OUTPUT
};

struct Endpoint {
  PortKind kind;
  uint32_t index;
  cudaq_edem_endpoint view() const {
    return {static_cast<uint32_t>(kind), index};
  }
};

struct CodePresentation {
  std::string name;
  uint32_t syndrome_count, virtual_outcome_count, logical_qubit_count;
  cudaq_edem_code_presentation view() const {
    return {name.data(), name.size(), syndrome_count, virtual_outcome_count,
            logical_qubit_count};
  }
};

enum class GroupKind : uint32_t {
  INDEPENDENT = CUDAQ_EDEM_INDEPENDENT,
  EXCLUSIVE = CUDAQ_EDEM_EXCLUSIVE
};

struct FaultGroup {
  GroupKind kind;
  std::vector<uint32_t> fault_ids;
  std::vector<double> probabilities;
  cudaq_edem_fault_group view() const {
    if (fault_ids.size() != probabilities.size() ||
        fault_ids.size() > UINT32_MAX)
      throw nb::value_error(
          "fault IDs and probabilities must have equal widths");
    return {static_cast<uint32_t>(kind),
            static_cast<uint32_t>(fault_ids.size()), fault_ids.data(),
            probabilities.data()};
  }
};

struct FaultDistribution {
  uint32_t size;
  std::vector<FaultGroup> groups;

  static FaultDistribution
  independent(const std::vector<double> &probabilities) {
    if (probabilities.size() > UINT32_MAX)
      throw nb::value_error("too many fault indicators");
    std::vector<uint32_t> ids(probabilities.size());
    std::iota(ids.begin(), ids.end(), 0);
    FaultDistribution result{static_cast<uint32_t>(ids.size()), {}};
    if (!ids.empty())
      result.groups.push_back(
          {GroupKind::INDEPENDENT, std::move(ids), probabilities});
    return result;
  }
};

struct MatrixBlock {
  Endpoint output, input;
  SparseMatrix matrix;
  cudaq_edem_matrix_block view() const {
    return {output.view(), input.view(), matrix.view()};
  }
};

class DEM;

class EDEM {
  friend class DEM;
  // Release owned bytes before dropping the capsule that keeps callback code
  // and its allocator context alive. Only successful construction/read
  // publishes checked; moving the owner preserves the payload address.
  nb::object dependency;
  cudaq_gf2_owned_buffer storage{};
  cudaq_edem_view checked{};

  cudaq_edem_view view() const { return checked; }

  static uint32_t count(size_t size) {
    if (size > UINT32_MAX)
      throw nb::value_error("too many EDEM ports");
    return static_cast<uint32_t>(size);
  }

public:
  EDEM() = default;
  EDEM(const EDEM &) = delete;
  EDEM &operator=(const EDEM &) = delete;
  EDEM(EDEM &&other) noexcept
      : dependency(std::move(other.dependency)),
        storage(std::exchange(other.storage, {})),
        checked(std::exchange(other.checked, {})) {}
  EDEM &operator=(EDEM &&other) noexcept {
    if (this != &other) {
      cudaq_gf2_owned_buffer_release(&storage);
      dependency = std::move(other.dependency);
      storage = std::exchange(other.storage, {});
      checked = std::exchange(other.checked, {});
    }
    return *this;
  }
  ~EDEM() { cudaq_gf2_owned_buffer_release(&storage); }

  const EDEM &model() const { return *this; }

  static EDEM compose_chain(nb::sequence occurrences) {
    // A sequence may construct fresh owners in __getitem__. Retain them until
    // native composition finishes instead of converting to borrowed pointers.
    nb::tuple owners(occurrences);
    auto size = count(owners.size());
    std::vector<cudaq_edem_view> views;
    views.reserve(size);
    for (nb::handle occurrence : owners) {
      if (!nb::isinstance<EDEM>(occurrence))
        throw nb::type_error("chain occurrences must be EDEM values");
      views.push_back(nb::cast<const EDEM &>(occurrence).view());
    }
    EDEM result;
    check(cudaq_edem_compose_chain(views.data(), size, nullptr, &result.storage,
                                   &result.checked));
    return result;
  }

  bool is_dem() const {
    auto value = view();
    uint32_t result;
    check(cudaq_edem_is_dem(&value, &result));
    return result != 0;
  }

  EDEM(const std::vector<CodePresentation> &codes,
       const std::vector<uint32_t> &inputs,
       const std::vector<uint32_t> &outputs,
       const std::vector<FaultDistribution> &faults,
       const std::vector<uint32_t> &detectors,
       const std::vector<uint32_t> &observables,
       const std::vector<uint32_t> &successes,
       const std::vector<MatrixBlock> &blocks) {
    std::vector<cudaq_edem_code_presentation> code_views;
    std::vector<std::vector<cudaq_edem_fault_group>> group_views(faults.size());
    std::vector<cudaq_edem_fault_distribution> fault_views;
    std::vector<cudaq_edem_matrix_block> block_views;
    for (const auto &code : codes)
      code_views.push_back(code.view());
    for (size_t p = 0; p < faults.size(); ++p) {
      for (const auto &group : faults[p].groups)
        group_views[p].push_back(group.view());
      fault_views.push_back({faults[p].size, count(group_views[p].size()),
                             group_views[p].data()});
    }
    for (const auto &block : blocks)
      block_views.push_back(block.view());
    cudaq_edem_description description = {
        {count(inputs.size()), count(outputs.size()), count(faults.size()),
         count(detectors.size()), count(observables.size()),
         count(successes.size())},
        {count(codes.size()), code_views.data()},
        inputs.data(),
        outputs.data(),
        fault_views.data(),
        detectors.data(),
        observables.data(),
        successes.data(),
        blocks.size(),
        block_views.data()};
    check(cudaq_edem_construct(&description, nullptr, &storage, &checked));
  }

  static EDEM from_bytes(nb::bytes bytes) {
    size_t size = bytes.size();
    if (!size)
      throw nb::value_error("empty packed EDEM");
    EDEM result;
    auto resource = cudaq_gf2_default_allocator();
    void *data =
        resource.allocate(resource.context, size, CUDAQ_EDEM_PACKED_ALIGNMENT);
    if (!data)
      throw std::bad_alloc();
    result.storage = {data, size, resource.deallocate, resource.context};
    std::memcpy(data, bytes.c_str(), size);
    check(cudaq_edem_read(data, size, nullptr, &result.checked));
    return result;
  }

  static nb::object adopt(nb::handle capsule) {
    if (!PyCapsule_CheckExact(capsule.ptr()))
      throw nb::type_error("expected an EDEM owned-buffer capsule");
    if (PyCapsule_IsValid(capsule.ptr(), CUDAQ_EDEM_USED_BUFFER_CAPSULE))
      throw nb::value_error(
          "EDEM owned-buffer capsule has already been consumed");
    if (!PyCapsule_IsValid(capsule.ptr(), CUDAQ_EDEM_OWNED_BUFFER_CAPSULE))
      throw nb::value_error("expected an EDEM owned-buffer capsule");
    auto *source = static_cast<cudaq_gf2_owned_buffer *>(
        PyCapsule_GetPointer(capsule.ptr(), CUDAQ_EDEM_OWNED_BUFFER_CAPSULE));
    if (!source->data || !source->size || !source->deallocate)
      throw nb::value_error("empty EDEM owned-buffer record");
    cudaq_edem_view validated;
    check(cudaq_edem_read(source->data, source->size, nullptr, &validated));

    // Allocate/initialize the Python receiver before irreversible consumption.
    // Its ready destructor can clean up if capsule renaming unexpectedly fails.
    nb::object receiver = nb::inst_alloc(nb::type<EDEM>());
    auto *value = new (nb::inst_ptr<EDEM>(receiver)) EDEM;
    nb::inst_mark_ready(receiver);
    value->dependency = nb::borrow<nb::object>(capsule);
    if (PyCapsule_SetName(capsule.ptr(), CUDAQ_EDEM_USED_BUFFER_CAPSULE) != 0)
      throw nb::python_error();
    // No allocation or Python call follows the ownership transfer.
    value->storage = std::exchange(*source, {});
    value->checked = validated;
    return receiver;
  }

  uintptr_t data_address() const {
    return reinterpret_cast<uintptr_t>(storage.data);
  }

  nb::bytes to_bytes() const {
    auto value = view();
    return nb::bytes(value.map.data, value.map.size);
  }

  cudaq_edem_shape shape() const {
    auto value = view();
    cudaq_edem_shape result;
    check(cudaq_edem_get_shape(&value, &result));
    return result;
  }

  uint32_t code_count() const {
    auto value = view();
    uint32_t result;
    check(cudaq_edem_get_code_count(&value, &result));
    return result;
  }

  std::vector<CodePresentation> codes() const {
    auto value = view();
    std::vector<CodePresentation> result;
    auto n = code_count();
    for (uint64_t i = 0; i < n; ++i) {
      cudaq_edem_code_presentation code;
      check(cudaq_edem_get_code(&value, static_cast<uint32_t>(i), &code));
      result.push_back({std::string(code.name, code.name_size),
                        code.syndrome_count, code.virtual_outcome_count,
                        code.logical_qubit_count});
    }
    return result;
  }

  std::vector<uint32_t> boundaries(EndpointDirection direction) const {
    auto value = view();
    auto side = static_cast<cudaq_edem_endpoint_direction>(direction);
    cudaq_gf2_map_shape map;
    check_gf2(cudaq_gf2_map_get_shape(&value.map, &map));
    uint32_t n = side == CUDAQ_EDEM_OUTPUT ? map.output_bundle_count
                                           : map.input_bundle_count;
    auto codes = code_count();
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < n; ++i) {
      cudaq_gf2_bundle bundle;
      check_gf2(cudaq_gf2_map_get_bundle(&value.map, side, i, &bundle));
      if (bundle.type < codes)
        result.push_back(bundle.type);
    }
    return result;
  }

  std::vector<FaultDistribution> faults() const {
    auto value = view();
    std::vector<FaultDistribution> result;
    auto codes = code_count();
    cudaq_gf2_map_shape map;
    check_gf2(cudaq_gf2_map_get_shape(&value.map, &map));
    for (uint64_t p = 0; p < map.input_bundle_count; ++p) {
      cudaq_gf2_bundle bundle;
      check_gf2(cudaq_gf2_map_get_bundle(&value.map, CUDAQ_EDEM_INPUT,
                                         static_cast<uint32_t>(p), &bundle));
      if (bundle.type != codes + 1)
        continue;
      uint32_t size, groups;
      check(cudaq_edem_get_fault(&value, static_cast<uint32_t>(p), &size,
                                 &groups));
      FaultDistribution distribution{size, {}};
      for (uint64_t i = 0; i < groups; ++i) {
        cudaq_edem_fault_group group;
        check(cudaq_edem_get_group(&value, static_cast<uint32_t>(p),
                                   static_cast<uint32_t>(i), &group));
        distribution.groups.push_back(
            {static_cast<GroupKind>(group.kind),
             std::vector<uint32_t>(group.fault_ids,
                                   group.fault_ids + group.size),
             std::vector<double>(group.probabilities,
                                 group.probabilities + group.size)});
      }
      result.push_back(std::move(distribution));
    }
    return result;
  }

  std::vector<MatrixBlock> blocks() const {
    auto value = view();
    std::vector<MatrixBlock> result;
    cudaq_gf2_map_shape map;
    check_gf2(cudaq_gf2_map_get_shape(&value.map, &map));
    for (uint64_t i = 0; i < map.block_count; ++i) {
      cudaq_gf2_map_matrix_block block;
      cudaq_edem_endpoint output, input;
      check_gf2(cudaq_gf2_map_get_matrix_block(&value.map, i, &block));
      check(cudaq_edem_describe_wire(&value, CUDAQ_EDEM_OUTPUT, block.output,
                                     &output));
      check(cudaq_edem_describe_wire(&value, CUDAQ_EDEM_INPUT, block.input,
                                     &input));
      result.push_back({{static_cast<PortKind>(output.kind), output.index},
                        {static_cast<PortKind>(input.kind), input.index},
                        SparseMatrix(block.matrix)});
    }
    return result;
  }

  std::pair<uint32_t, uint64_t> port(Endpoint endpoint,
                                     EndpointDirection direction) const {
    auto value = view();
    auto side = static_cast<cudaq_edem_endpoint_direction>(direction);
    uint32_t width;
    uint64_t offset;
    cudaq_gf2_wire wire;
    check(cudaq_edem_resolve_endpoint(&value, side, endpoint.view(), &wire));
    check_gf2(cudaq_gf2_map_get_wire(&value.map, side, wire, &width, &offset));
    return {width, offset};
  }

  std::vector<uint32_t> port_sizes(PortKind kind) const {
    auto s = shape();
    uint32_t size = kind == PortKind::DETECTOR ? s.detector_ports_count
                                               : (kind == PortKind::OBSERVABLE
                                                      ? s.observable_ports_count
                                                      : s.success_ports_count);
    std::vector<uint32_t> result(size);
    for (uint64_t i = 0; i < size; ++i)
      result[i] =
          port({kind, static_cast<uint32_t>(i)}, EndpointDirection::OUTPUT)
              .first;
    return result;
  }

  std::pair<uint64_t, uint64_t> vector_sizes() const {
    auto value = view();
    uint64_t inputs, outputs;
    check_gf2(cudaq_gf2_map_vector_sizes(&value.map, &inputs, &outputs));
    return {inputs, outputs};
  }

  std::vector<uint8_t> apply(const std::vector<uint8_t> &input) const {
    auto value = view();
    std::vector<uint8_t> output(vector_sizes().second);
    check_gf2(cudaq_gf2_map_apply(&value.map, input.data(), input.size(),
                                  output.data(), output.size()));
    return output;
  }
};

// A separate Python type makes structural closure a checked precondition for
// DEM consumers. Its packed representation remains an ordinary closed EDEM.
class DEM {
  EDEM value;

public:
  DEM() = default;
  DEM(const DEM &) = delete;
  DEM &operator=(const DEM &) = delete;
  DEM(DEM &&) noexcept = default;
  DEM &operator=(DEM &&) noexcept = default;

  static DEM project(const EDEM &source) {
    DEM result;
    auto input = source.view();
    cudaq_edem_dem_view checked;
    check(cudaq_edem_project_dem(&input, nullptr, &result.value.storage,
                                 &checked));
    result.value.checked = checked.edem;
    return result;
  }

  static DEM from_bytes(nb::bytes bytes) {
    DEM result;
    result.value = EDEM::from_bytes(bytes);
    if (!result.value.is_dem())
      throw nb::value_error("DEM must have no exposed code boundaries");
    return result;
  }

  static DEM factorize(const DEM &source) {
    DEM result;
    cudaq_edem_dem_view input{source.value.view()}, checked;
    check(cudaq_edem_into_independent_faults_dem(
        &input, nullptr, &result.value.storage, &checked));
    result.value.checked = checked.edem;
    return result;
  }

  static nb::tuple coalesce(const DEM &source) {
    DEM result;
    struct MappingOwner {
      cudaq_gf2_owned_buffer storage{};
      ~MappingOwner() { cudaq_gf2_owned_buffer_release(&storage); }
    } mapping;
    cudaq_edem_dem_view input{source.value.view()}, checked;
    check(cudaq_edem_coalesce_dem_columns(
        &input, nullptr, &result.value.storage, &checked, &mapping.storage));
    result.value.checked = checked.edem;
    nb::list columns;
    auto *indices = static_cast<const uint64_t *>(mapping.storage.data);
    for (uint64_t i = 0; i < mapping.storage.size / sizeof(uint64_t); ++i)
      columns.append(indices[i] == UINT64_MAX ? nb::none()
                                              : nb::cast(indices[i]));
    return nb::make_tuple(std::move(result), std::move(columns));
  }

  const EDEM &model() const { return value; }
};

template <class Value>
static void bind_value(nb::class_<Value> &type) {
  type.def_prop_ro("_data_address",
                   [](const Value &v) { return v.model().data_address(); })
      .def("to_bytes", [](const Value &v) { return v.model().to_bytes(); })
      .def_prop_ro("is_dem", [](const Value &v) { return v.model().is_dem(); })
      .def_prop_ro("codes", [](const Value &v) { return v.model().codes(); })
      .def_prop_ro("inputs",
                   [](const Value &v) {
                     return v.model().boundaries(EndpointDirection::INPUT);
                   })
      .def_prop_ro("outputs",
                   [](const Value &v) {
                     return v.model().boundaries(EndpointDirection::OUTPUT);
                   })
      .def_prop_ro("faults", [](const Value &v) { return v.model().faults(); })
      .def_prop_ro("detectors",
                   [](const Value &v) {
                     return v.model().port_sizes(PortKind::DETECTOR);
                   })
      .def_prop_ro("observables",
                   [](const Value &v) {
                     return v.model().port_sizes(PortKind::OBSERVABLE);
                   })
      .def_prop_ro("successes",
                   [](const Value &v) {
                     return v.model().port_sizes(PortKind::SUCCESS);
                   })
      .def_prop_ro("blocks", [](const Value &v) { return v.model().blocks(); })
      .def_prop_ro("vector_sizes",
                   [](const Value &v) { return v.model().vector_sizes(); })
      .def(
          "port",
          [](const Value &v, Endpoint endpoint, EndpointDirection direction) {
            return v.model().port(endpoint, direction);
          },
          "endpoint"_a, nb::kw_only(), "direction"_a = EndpointDirection::INPUT)
      .def(
          "apply",
          [](const Value &v, const std::vector<uint8_t> &input) {
            return v.model().apply(input);
          },
          "input"_a);
}

} // namespace v0
} // namespace cudaq_edem_python

NB_MODULE(_core, m) {
  using namespace cudaq_edem_python;
  m.doc() = "cudaq-edem native bindings. The API, ABI, and packed format are "
            "unstable during v0.x development. ABI_VERSION and FORMAT_VERSION "
            "identify concrete development revisions.";
  m.attr("ABI_VERSION") = CUDAQ_EDEM_ABI_VERSION;
  m.attr("FORMAT_VERSION") = CUDAQ_EDEM_FORMAT_VERSION;
  nb::enum_<PortKind>(m, "PortKind")
      .value("FAULT", PortKind::FAULT)
      .value("VIRTUAL", PortKind::VIRTUAL)
      .value("SYNDROME", PortKind::SYNDROME)
      .value("LOGICAL", PortKind::LOGICAL)
      .value("DETECTOR", PortKind::DETECTOR)
      .value("OBSERVABLE", PortKind::OBSERVABLE)
      .value("SUCCESS", PortKind::SUCCESS);
  nb::enum_<EndpointDirection>(m, "EndpointDirection")
      .value("INPUT", EndpointDirection::INPUT)
      .value("OUTPUT", EndpointDirection::OUTPUT);
  nb::enum_<GroupKind>(m, "GroupKind")
      .value("INDEPENDENT", GroupKind::INDEPENDENT)
      .value("EXCLUSIVE", GroupKind::EXCLUSIVE);
  nb::class_<Endpoint>(m, "Endpoint")
      .def(nb::init<PortKind, uint32_t>(), "kind"_a, "index"_a = 0)
      .def_ro("kind", &Endpoint::kind)
      .def_ro("index", &Endpoint::index);
  nb::class_<CodePresentation>(m, "CodePresentation")
      .def(nb::init<std::string, uint32_t, uint32_t, uint32_t>(), "name"_a,
           "syndrome_count"_a, "virtual_outcome_count"_a,
           "logical_qubit_count"_a)
      .def_ro("name", &CodePresentation::name)
      .def_ro("syndrome_count", &CodePresentation::syndrome_count)
      .def_ro("virtual_outcome_count", &CodePresentation::virtual_outcome_count)
      .def_ro("logical_qubit_count", &CodePresentation::logical_qubit_count);
  nb::class_<FaultGroup>(m, "FaultGroup")
      .def(nb::init<GroupKind, std::vector<uint32_t>, std::vector<double>>(),
           "kind"_a, "fault_ids"_a, "probabilities"_a)
      .def_ro("kind", &FaultGroup::kind)
      .def_ro("fault_ids", &FaultGroup::fault_ids)
      .def_ro("probabilities", &FaultGroup::probabilities);
  nb::class_<FaultDistribution>(m, "FaultDistribution")
      .def(nb::init<uint32_t, std::vector<FaultGroup>>(), "size"_a, "groups"_a)
      .def_static("independent", &FaultDistribution::independent,
                  "probabilities"_a)
      .def_ro("size", &FaultDistribution::size)
      .def_ro("groups", &FaultDistribution::groups);
  nb::class_<SparseMatrix>(m, "SparseMatrix")
      .def(nb::init<uint32_t, uint32_t,
                    const std::vector<std::pair<uint32_t, uint32_t>> &>(),
           "rows"_a, "columns"_a,
           "entries"_a = std::vector<std::pair<uint32_t, uint32_t>>{})
      .def_prop_ro("shape",
                   [](const SparseMatrix &v) {
                     return std::make_pair(v.rows, v.columns);
                   })
      .def_prop_ro("entries", &SparseMatrix::entries)
      .def("apply", &SparseMatrix::apply, "input"_a, "offset"_a = nb::none());
  nb::class_<MatrixBlock>(m, "MatrixBlock")
      .def(nb::init<Endpoint, Endpoint, SparseMatrix>(), "output"_a, "input"_a,
           "matrix"_a)
      .def_ro("output", &MatrixBlock::output)
      .def_ro("input", &MatrixBlock::input)
      .def_ro("matrix", &MatrixBlock::matrix);
  auto edem = nb::class_<EDEM>(m, "EDEM");
  edem.def(
          nb::init<const std::vector<CodePresentation> &,
                   const std::vector<uint32_t> &, const std::vector<uint32_t> &,
                   const std::vector<FaultDistribution> &,
                   const std::vector<uint32_t> &, const std::vector<uint32_t> &,
                   const std::vector<uint32_t> &,
                   const std::vector<MatrixBlock> &>(),
          nb::kw_only(), "codes"_a = std::vector<CodePresentation>{},
          "inputs"_a = std::vector<uint32_t>{},
          "outputs"_a = std::vector<uint32_t>{},
          "faults"_a = std::vector<FaultDistribution>{},
          "detectors"_a = std::vector<uint32_t>{},
          "observables"_a = std::vector<uint32_t>{},
          "successes"_a = std::vector<uint32_t>{},
          "blocks"_a = std::vector<MatrixBlock>{})
      .def_static("from_bytes", &EDEM::from_bytes, "data"_a)
      .def_static("adopt", &EDEM::adopt, "capsule"_a,
                  "Validate and adopt an owned-buffer capsule without copying "
                  "its payload. Successful adoption consumes it once; invalid "
                  "content leaves the source usable.");
  bind_value(edem);
  auto dem = nb::class_<DEM>(m, "DEM");
  dem.def_static("from_bytes", &DEM::from_bytes, "data"_a,
                 "Read a packed EDEM and require structural closure.");
  bind_value(dem);
  m.def("compose_chain", &EDEM::compose_chain, "occurrences"_a,
        "Compose a nonempty serial chain, retaining each occurrence's fault "
        "indicators and exact laws. The exposed endpoint boundaries remain.");
  m.def("project_DEM", &DEM::project, "value"_a,
        "Require closed code boundaries and return an independently owned "
        "DEM with original fault columns and exact laws. Success outputs are "
        "retained; no noise approximation or column merging is performed.");
  m.def(
      "factorize_dem", &DEM::factorize, "value"_a,
      "Exactly factor supported uniform Pauli laws into independent "
      "mechanisms. "
      "Response columns and ports stay unchanged; original event laws "
      "remain in the source DEM. Unsupported distributions raise ValueError.");
  m.def("coalesce_dem_columns", &DEM::coalesce, "value"_a,
        "Return (DEM, source_to_result) for independent mechanisms with equal "
        "complete responses. Global D/O/success rows retain their numbering; "
        "ports consolidate. Mapping entries are result column IDs or None "
        "for discarded mechanisms. Factorize exclusive laws explicitly first.");
}
