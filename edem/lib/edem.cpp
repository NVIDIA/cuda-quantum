/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#pragma GCC visibility push(default)
#include "cudaq/edem/core.h"
#pragma GCC visibility pop
#include "allocation.h"
#include "storage.h"
#include "cudaq/edem/format.h"
#include <float.h>
#include <math.h>
using namespace cudaq_gf2_private;
namespace {
static_assert(sizeof(cudaq_edem_packed_code) == 16);
static_assert(sizeof(cudaq_edem_packed_fault) == 24);
static_assert(sizeof(cudaq_edem_packed_group) == 24);
static_assert(sizeof(double) == 8 && FLT_RADIX == 2 && DBL_MANT_DIG == 53 &&
              DBL_MAX_EXP == 1024);
constexpr char code_prefix[] = "edem.code/";
constexpr const char *free_names[] = {"edem.detector", "edem.fault",
                                      "edem.observable", "edem.success"};
constexpr uint32_t free_kinds[] = {CUDAQ_EDEM_DETECTOR, CUDAQ_EDEM_FAULT,
                                   CUDAQ_EDEM_OBSERVABLE, CUDAQ_EDEM_SUCCESS};
constexpr uint32_t required_sections[] = {CUDAQ_EDEM_SECTION_PRESENTATIONS,
                                          CUDAQ_EDEM_SECTION_DISTRIBUTIONS};
cudaq_edem_status status(cudaq_gf2_status s) {
  return static_cast<cudaq_edem_status>(s);
}
struct plan {
  uint64_t names = 0, fault_bytes = 0, total = 0;
  uint32_t inputs = 0, outputs = 0, max_groups = 0, max_faults = 0;
  uint64_t wires = 0;
};
cudaq_edem_status describe(const cudaq_edem_description *d, plan &p) {
  if (!d)
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  const auto &s = d->shape;
  if (s.code_presentations > UINT32_MAX - 4 || s.blocks > UINT32_MAX ||
      uint64_t{s.fault_ports} + s.input_code_blocks > UINT32_MAX ||
      uint64_t{s.detector_ports} + s.observable_ports + s.success_ports +
              s.output_code_blocks >
          UINT32_MAX ||
      !array_ok(d->codes, s.code_presentations, sizeof(*d->codes)) ||
      !array_ok(d->input_codes, s.input_code_blocks, 4) ||
      !array_ok(d->output_codes, s.output_code_blocks, 4) ||
      !array_ok(d->faults, s.fault_ports, sizeof(*d->faults)) ||
      !array_ok(d->detector_sizes, s.detector_ports, 4) ||
      !array_ok(d->observable_sizes, s.observable_ports, 4) ||
      !array_ok(d->success_sizes, s.success_ports, 4) ||
      !array_ok(d->blocks, s.blocks, sizeof(*d->blocks)))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  p.inputs = s.fault_ports + s.input_code_blocks;
  p.outputs = s.detector_ports + s.observable_ports + s.success_ports +
              s.output_code_blocks;
  p.wires = uint64_t{s.fault_ports} + s.detector_ports + s.observable_ports +
            s.success_ports +
            3 * (uint64_t{s.input_code_blocks} + s.output_code_blocks);
  uint64_t ignored;
  for (uint32_t i = 0; i < s.code_presentations; ++i) {
    auto c = d->codes[i];
    if (!array_ok(c.name, c.name_size, 1) || c.name_size > SIZE_MAX - 10 ||
        !reserve(p.names, c.name_size + 10, 1, ignored, 1))
      return CUDAQ_EDEM_INVALID_ARGUMENT;
  }
  if (!reserve(p.fault_bytes, s.fault_ports, sizeof(cudaq_edem_packed_fault),
               ignored))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  for (uint32_t i = 0; i < s.fault_ports; ++i) {
    auto f = d->faults[i];
    if (!array_ok(f.groups, f.group_count, sizeof(*f.groups)))
      return CUDAQ_EDEM_INVALID_ARGUMENT;
    if (f.group_count > f.size || (f.size && !f.group_count))
      return CUDAQ_EDEM_INVALID_CONTENT;
    p.max_groups = std::max(p.max_groups, f.group_count);
    p.max_faults = std::max(p.max_faults, f.size);
    if (!reserve(p.fault_bytes, f.group_count, sizeof(cudaq_edem_packed_group),
                 ignored))
      return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
    for (uint32_t j = 0; j < f.group_count; ++j)
      if (!reserve(p.fault_bytes, f.groups[j].size, 4, ignored, 4) ||
          !reserve(p.fault_bytes, f.groups[j].size, 8, ignored))
        return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  }
  uint64_t &n = p.total;
  if (!reserve(n, s.code_presentations, 4, ignored) ||
      !reserve(n, s.code_presentations, 4, ignored) ||
      !reserve(n, s.code_presentations + 4, sizeof(cudaq_gf2_bundle_type),
               ignored) ||
      !reserve(n, p.inputs, sizeof(cudaq_gf2_bundle), ignored) ||
      !reserve(n, p.outputs, sizeof(cudaq_gf2_bundle), ignored) ||
      !reserve(n, p.wires, 4, ignored) ||
      !reserve(n, s.blocks, sizeof(cudaq_gf2_map_block), ignored) ||
      !reserve(n, p.names, 1, ignored) ||
      !reserve(n, s.code_presentations, sizeof(cudaq_edem_packed_code),
               ignored) ||
      !reserve(n, p.fault_bytes, 1, ignored) ||
      !reserve(n, p.max_groups, 4, ignored) ||
      !reserve(n, (uint64_t{p.max_faults} + 7) / 8, 1, ignored))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  return CUDAQ_EDEM_OK;
}
template <class Group>
cudaq_edem_status check_distribution(uint32_t size, uint32_t count, Group group,
                                     std::span<uint8_t> seen, bool packed) {
  if (count > size || (size && !count))
    return CUDAQ_EDEM_INVALID_CONTENT;
  if (size)
    memset(seen.data(), 0, (uint64_t{size} + 7) / 8);
  uint64_t total = 0;
  uint32_t independent = 0;
  for (uint32_t i = 0; i < count; ++i) {
    auto g = group(i);
    if (!array_ok(g.fault_ids, g.size, 4) ||
        !array_ok(g.probabilities, g.size, 8))
      return CUDAQ_EDEM_INVALID_ARGUMENT;
    if (!g.size || g.kind > CUDAQ_EDEM_EXCLUSIVE ||
        (g.kind == CUDAQ_EDEM_EXCLUSIVE && g.size < 2) ||
        (g.kind == CUDAQ_EDEM_INDEPENDENT && independent++))
      return CUDAQ_EDEM_INVALID_CONTENT;
    total += g.size;
    double sum = 0;
    for (uint64_t j = 0; j < g.size; ++j) {
      uint32_t bit = g.fault_ids[j];
      double probability = g.probabilities[j];
      if (bit >= size || (j && g.fault_ids[j - 1] >= bit) ||
          !isfinite(probability) || probability < 0 || probability > 1 ||
          (packed && probability == 0 && signbit(probability)))
        return CUDAQ_EDEM_INVALID_CONTENT;
      uint8_t mask = static_cast<uint8_t>(1u << (bit % 8));
      if (seen[bit / 8] & mask)
        return CUDAQ_EDEM_INVALID_CONTENT;
      seen[bit / 8] |= mask;
      sum += probability;
    }
    if (g.kind == CUDAQ_EDEM_EXCLUSIVE && sum > 1)
      return CUDAQ_EDEM_INVALID_CONTENT;
    if (packed && i) {
      auto previous = group(i - 1);
      if (previous.kind > g.kind ||
          (previous.kind == g.kind && previous.fault_ids[0] >= g.fault_ids[0]))
        return CUDAQ_EDEM_INVALID_CONTENT;
    }
  }
  return total == size ? CUDAQ_EDEM_OK : CUDAQ_EDEM_INVALID_CONTENT;
}
bool semantic_wire(const cudaq_edem_shape &s, uint32_t side,
                   cudaq_edem_endpoint e, cudaq_gf2_wire &wire) {
  if (e.kind >= CUDAQ_EDEM_VIRTUAL && e.kind <= CUDAQ_EDEM_LOGICAL) {
    if (e.index >= (side ? s.output_code_blocks : s.input_code_blocks))
      return false;
    uint32_t start =
        side ? s.detector_ports + s.observable_ports + s.success_ports
             : s.fault_ports;
    wire = {start + e.index, e.kind - CUDAQ_EDEM_VIRTUAL};
    return true;
  }
  if (!side && e.kind == CUDAQ_EDEM_FAULT && e.index < s.fault_ports) {
    wire = {e.index, 0};
    return true;
  }
  if (side) {
    if (e.kind == CUDAQ_EDEM_DETECTOR && e.index < s.detector_ports) {
      wire = {e.index, 0};
      return true;
    }
    if (e.kind == CUDAQ_EDEM_OBSERVABLE && e.index < s.observable_ports) {
      wire = {s.detector_ports + e.index, 0};
      return true;
    }
    if (e.kind == CUDAQ_EDEM_SUCCESS && e.index < s.success_ports) {
      wire = {s.detector_ports + s.observable_ports + e.index, 0};
      return true;
    }
  }
  return false;
}
template <class Operation>
cudaq_edem_status lower(const cudaq_edem_description *d,
                        const cudaq_gf2_allocator &resource, Operation op) {
  if (!valid_allocator(resource))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  plan p;
  auto st = describe(d, p);
  if (st != CUDAQ_EDEM_OK)
    return st;
  if (!little_endian())
    return CUDAQ_EDEM_UNSUPPORTED_VERSION;
  allocation scratch(resource);
  auto allocated = scratch.reserve(p.total, CUDAQ_EDEM_PACKED_ALIGNMENT);
  if (allocated != CUDAQ_GF2_OK)
    return status(allocated);
  const auto &s = d->shape;
  arena a{static_cast<std::byte *>(scratch.data()), p.total};
  auto order = a.take<uint32_t>(s.code_presentations),
       ids = a.take<uint32_t>(s.code_presentations);
  auto types = a.take<cudaq_gf2_bundle_type>(s.code_presentations + 4);
  auto inputs = a.take<cudaq_gf2_bundle>(p.inputs),
       outputs = a.take<cudaq_gf2_bundle>(p.outputs);
  auto widths = a.take<uint32_t>(p.wires);
  auto blocks = a.take<cudaq_gf2_map_block>(s.blocks);
  auto names = a.take<char>(p.names);
  auto codes = a.take<cudaq_edem_packed_code>(s.code_presentations);
  auto fault_data = a.take<uint8_t>(p.fault_bytes);
  auto group_order = a.take<uint32_t>(p.max_groups);
  auto seen = a.take<uint8_t>((uint64_t{p.max_faults} + 7) / 8);
  for (uint32_t i = 0; i < s.code_presentations; ++i)
    if (!utf8_name(d->codes[i].name, d->codes[i].name_size) ||
        d->codes[i].logical_qubits > UINT32_MAX / 2)
      return CUDAQ_EDEM_INVALID_CONTENT;
  sort_indices(order, [&](uint32_t x, uint32_t y) {
    auto c = d->codes[x], b = d->codes[y];
    return name_compare(c.name, c.name_size, b.name, b.name_size) < 0;
  });
  uint64_t cursor = 0;
  for (uint32_t i = 0; i < s.code_presentations; ++i) {
    auto c = d->codes[order[i]];
    if (i) {
      auto previous = d->codes[order[i - 1]];
      if (!name_compare(previous.name, previous.name_size, c.name, c.name_size))
        return CUDAQ_EDEM_INVALID_CONTENT;
    }
    char *name = names.data() + cursor;
    memcpy(name, code_prefix, 10);
    memcpy(name + 10, c.name, c.name_size);
    cursor += 10 + c.name_size;
    types[i] = {name, 10 + c.name_size, CUDAQ_GF2_INTERFACE, 3};
    codes[i] = {i, c.generators, c.logical_qubits, 0};
    ids[order[i]] = i;
  }
  for (uint32_t i = 0; i < 4; ++i)
    types[s.code_presentations + i] = {free_names[i], strlen(free_names[i]),
                                       CUDAQ_GF2_FREE, 1};
  uint64_t wi = 0;
  for (uint32_t side = 0; side < 2; ++side) {
    auto bundles = side ? outputs : inputs;
    uint32_t free_count =
        side ? s.detector_ports + s.observable_ports + s.success_ports
             : s.fault_ports;
    for (uint32_t i = 0; i < free_count; ++i) {
      uint32_t type, width;
      if (!side) {
        type = 1;
        width = d->faults[i].size;
      } else if (i < s.detector_ports) {
        type = 0;
        width = d->detector_sizes[i];
      } else if (i < s.detector_ports + s.observable_ports) {
        type = 2;
        width = d->observable_sizes[i - s.detector_ports];
      } else {
        type = 3;
        width = d->success_sizes[i - s.detector_ports - s.observable_ports];
      }
      widths[wi] = width;
      bundles[i] = {s.code_presentations + type, 0, widths.data() + wi++};
    }
    uint32_t count = side ? s.output_code_blocks : s.input_code_blocks;
    const auto *code_ids = side ? d->output_codes : d->input_codes;
    for (uint32_t i = 0; i < count; ++i) {
      if (code_ids[i] >= s.code_presentations)
        return CUDAQ_EDEM_INVALID_CONTENT;
      uint32_t id = ids[code_ids[i]];
      auto c = codes[id];
      widths[wi] = c.generators;
      widths[wi + 1] = c.generators;
      widths[wi + 2] = 2 * c.logical_qubits;
      bundles[free_count + i] = {id, 0, widths.data() + wi};
      wi += 3;
    }
  }
  for (uint32_t i = 0; i < s.blocks; ++i) {
    auto b = d->blocks[i];
    if (!semantic_wire(s, 1, b.output, blocks[i].output) ||
        !semantic_wire(s, 0, b.input, blocks[i].input))
      return CUDAQ_EDEM_INVALID_CONTENT;
    if (b.matrix.nonzeros && b.input.kind == CUDAQ_EDEM_VIRTUAL &&
        (b.output.kind == CUDAQ_EDEM_OBSERVABLE ||
         b.output.kind == CUDAQ_EDEM_SYNDROME ||
         b.output.kind == CUDAQ_EDEM_LOGICAL))
      return CUDAQ_EDEM_INVALID_CONTENT;
    blocks[i].matrix = b.matrix;
  }
  auto *faults = reinterpret_cast<cudaq_edem_packed_fault *>(fault_data.data());
  cursor = uint64_t{s.fault_ports} * sizeof(*faults);
  for (uint32_t i = 0; i < s.fault_ports; ++i) {
    auto f = d->faults[i];
    st = check_distribution(
        f.size, f.group_count, [&](uint32_t j) { return f.groups[j]; }, seen,
        false);
    if (st != CUDAQ_EDEM_OK)
      return st;
    uint64_t groups;
    reserve(cursor, f.group_count, sizeof(cudaq_edem_packed_group), groups);
    faults[i] = {i, f.size, f.group_count, 0, groups};
    auto ordered = group_order.first(f.group_count);
    sort_indices(ordered, [&](uint32_t x, uint32_t y) {
      auto g = f.groups[x], b = f.groups[y];
      return g.kind < b.kind ||
             (g.kind == b.kind && g.fault_ids[0] < b.fault_ids[0]);
    });
    for (uint32_t j = 0; j < f.group_count; ++j) {
      auto g = f.groups[ordered[j]];
      uint64_t fault_ids, probabilities;
      reserve(cursor, g.size, 4, fault_ids, 4);
      reserve(cursor, g.size, 8, probabilities);
      reinterpret_cast<cudaq_edem_packed_group *>(
          fault_data.data() + groups)[j] = {g.kind, g.size, fault_ids,
                                            probabilities};
      memcpy(fault_data.data() + fault_ids, g.fault_ids, uint64_t{g.size} * 4);
      auto *values =
          reinterpret_cast<double *>(fault_data.data() + probabilities);
      for (uint32_t k = 0; k < g.size; ++k)
        values[k] = g.probabilities[k] == 0 ? 0 : g.probabilities[k];
    }
  }
  cudaq_gf2_section sections[] = {
      {CUDAQ_EDEM_SECTION_PRESENTATIONS, CUDAQ_GF2_SECTION_REQUIRED,
       s.code_presentations, codes.size_bytes(), codes.data()},
      {CUDAQ_EDEM_SECTION_DISTRIBUTIONS, CUDAQ_GF2_SECTION_REQUIRED,
       s.fault_ports, fault_data.size(), fault_data.data()}};
  cudaq_gf2_map_description map = {s.code_presentations + 4,
                                   p.inputs,
                                   p.outputs,
                                   2,
                                   s.blocks,
                                   types.data(),
                                   inputs.data(),
                                   outputs.data(),
                                   blocks.data(),
                                   sections};
  return status(op(map));
}
struct sections {
  cudaq_gf2_section codes{}, faults{};
  const cudaq_edem_packed_code *code_data = nullptr;
  const cudaq_edem_packed_fault *fault_data = nullptr;
  cudaq_edem_fault_group group(uint32_t f, uint32_t i) const {
    auto *p = static_cast<const uint8_t *>(faults.data);
    const auto &g = reinterpret_cast<const cudaq_edem_packed_group *>(
        p + fault_data[f].groups)[i];
    return {g.kind, g.size, reinterpret_cast<const uint32_t *>(p + g.fault_ids),
            reinterpret_cast<const double *>(p + g.probabilities)};
  }
};
cudaq_edem_status inspect_sections(sections &s, uint64_t &scratch) {
  if (s.codes.count > UINT32_MAX || s.faults.count > UINT32_MAX ||
      s.codes.bytes != s.codes.count * sizeof(cudaq_edem_packed_code))
    return CUDAQ_EDEM_INVALID_CONTENT;
  s.code_data = static_cast<const cudaq_edem_packed_code *>(s.codes.data);
  s.fault_data = static_cast<const cudaq_edem_packed_fault *>(s.faults.data);
  reader r{static_cast<const uint8_t *>(s.faults.data), s.faults.bytes};
  if (!r.take(0, s.faults.count, sizeof(cudaq_edem_packed_fault)))
    return CUDAQ_EDEM_INVALID_CONTENT;
  scratch = 0;
  for (uint32_t i = 0; i < s.faults.count; ++i) {
    auto f = s.fault_data[i];
    if (f.reserved || (i && s.fault_data[i - 1].bundle >= f.bundle) ||
        f.group_count > f.size || (f.size && !f.group_count) ||
        !r.take(f.groups, f.group_count, sizeof(cudaq_edem_packed_group)))
      return CUDAQ_EDEM_INVALID_CONTENT;
    scratch = std::max(scratch, (uint64_t{f.size} + 7) / 8);
    auto *groups =
        reinterpret_cast<const cudaq_edem_packed_group *>(r.data + f.groups);
    for (uint32_t j = 0; j < f.group_count; ++j)
      if (!r.take(groups[j].fault_ids, groups[j].size, 4, 4) ||
          !r.take(groups[j].probabilities, groups[j].size, 8))
        return CUDAQ_EDEM_INVALID_CONTENT;
  }
  return r.end() ? CUDAQ_EDEM_OK : CUDAQ_EDEM_INVALID_CONTENT;
}
cudaq_edem_status inspect(const cudaq_gf2_map_view *view, sections &s,
                          uint64_t &scratch) {
  if (cudaq_gf2_map_get_section(view, CUDAQ_EDEM_SECTION_PRESENTATIONS,
                                &s.codes) != CUDAQ_GF2_OK ||
      cudaq_gf2_map_get_section(view, CUDAQ_EDEM_SECTION_DISTRIBUTIONS,
                                &s.faults) != CUDAQ_GF2_OK)
    return CUDAQ_EDEM_INVALID_CONTENT;
  return inspect_sections(s, scratch);
}
bool same_name(cudaq_gf2_bundle_type t, const char *name) {
  return t.name_size == strlen(name) && !memcmp(t.name, name, t.name_size);
}
bool find_fault(const cudaq_gf2_map_view *view, uint32_t bundle, sections &s,
                uint32_t &index) {
  if (cudaq_gf2_map_get_section(view, CUDAQ_EDEM_SECTION_DISTRIBUTIONS,
                                &s.faults) != CUDAQ_GF2_OK)
    return false;
  s.fault_data = static_cast<const cudaq_edem_packed_fault *>(s.faults.data);
  uint32_t lo = 0, hi = static_cast<uint32_t>(s.faults.count);
  while (lo < hi) {
    uint32_t mid = lo + (hi - lo) / 2;
    if (s.fault_data[mid].bundle < bundle)
      lo = mid + 1;
    else
      hi = mid;
  }
  index = lo;
  return lo < s.faults.count && s.fault_data[lo].bundle == bundle;
}
} // namespace
uint32_t cudaq_edem_abi_version(void) { return CUDAQ_EDEM_ABI_VERSION; }
const char *cudaq_edem_status_string(cudaq_edem_status s) {
  return cudaq_gf2_status_string(static_cast<cudaq_gf2_status>(s));
}
cudaq_gf2_schema cudaq_edem_schema(void) { return {required_sections, 2, 0}; }
cudaq_edem_status cudaq_edem_validate(const cudaq_edem_description *d,
                                      const cudaq_gf2_memory *memory) {
  cudaq_gf2_memory resources{};
  resources.scratch = scratch_allocator(memory);
  return lower(d, resources.scratch, [&](const auto &map) {
    return cudaq_gf2_map_validate(&map, &resources);
  });
}
cudaq_edem_status cudaq_edem_construct(const cudaq_edem_description *d,
                                       const cudaq_gf2_memory *memory,
                                       cudaq_gf2_owned_buffer *owner,
                                       cudaq_edem_view *out) {
  if (!owner || !out || !empty_owner(*owner))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  const cudaq_gf2_memory resources{scratch_allocator(memory),
                                   result_allocator(memory)};
  if (!valid_allocator(resources.result))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  cudaq_gf2_owned_buffer candidate_owner{};
  cudaq_gf2_map_view candidate;
  auto st = lower(d, resources.scratch, [&](const auto &map) {
    return cudaq_gf2_map_construct(&map, &resources, &candidate_owner,
                                   &candidate);
  });
  if (st == CUDAQ_EDEM_OK) {
    *owner = candidate_owner;
    *out = {candidate};
  }
  return st;
}
cudaq_edem_status cudaq_edem_read(const void *buffer, uint64_t bytes,
                                  const cudaq_gf2_memory *memory,
                                  cudaq_edem_view *out) {
  if (!out)
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  cudaq_gf2_memory resources{};
  resources.scratch = scratch_allocator(memory);
  auto schema = cudaq_edem_schema();
  cudaq_gf2_map_view map;
  auto read = cudaq_gf2_map_read(buffer, bytes, &schema, &resources, &map);
  if (read != CUDAQ_GF2_OK)
    return status(read);
  return cudaq_edem_check(&map, &resources, out);
}
cudaq_edem_status cudaq_edem_check(const cudaq_gf2_map_view *view,
                                   const cudaq_gf2_memory *memory,
                                   cudaq_edem_view *out) {
  const auto resource = scratch_allocator(memory);
  if (!out || !valid_allocator(resource))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  sections s;
  uint64_t required;
  auto st = inspect(view, s, required);
  if (st != CUDAQ_EDEM_OK)
    return st;
  allocation scratch(resource);
  auto allocated = scratch.reserve(required, CUDAQ_EDEM_PACKED_ALIGNMENT);
  if (allocated != CUDAQ_GF2_OK)
    return status(allocated);
  cudaq_gf2_map_shape shape;
  if (cudaq_gf2_map_get_shape(view, &shape) != CUDAQ_GF2_OK ||
      s.codes.count > UINT32_MAX - 4 || shape.type_count != s.codes.count + 4)
    return CUDAQ_EDEM_INVALID_CONTENT;
  for (uint32_t i = 0; i < shape.type_count; ++i) {
    cudaq_gf2_bundle_type t;
    cudaq_gf2_map_get_type(view, i, &t);
    if (i < s.codes.count) {
      auto c = s.code_data[i];
      if (c.type != i || c.reserved || c.logical_qubits > UINT32_MAX / 2 ||
          t.role != CUDAQ_GF2_INTERFACE || t.wire_count != 3 ||
          t.name_size <= 10 || memcmp(t.name, code_prefix, 10))
        return CUDAQ_EDEM_INVALID_CONTENT;
    } else if (t.role != CUDAQ_GF2_FREE || t.wire_count != 1 ||
               !same_name(t, free_names[i - s.codes.count]))
      return CUDAQ_EDEM_INVALID_CONTENT;
  }
  uint32_t fault = 0;
  for (uint32_t side = 0; side < 2; ++side) {
    uint32_t count =
        side ? shape.output_bundle_count : shape.input_bundle_count;
    for (uint32_t i = 0; i < count; ++i) {
      cudaq_gf2_bundle b;
      cudaq_gf2_map_get_bundle(view, side, i, &b);
      if (b.type < s.codes.count) {
        auto c = s.code_data[b.type];
        if (b.widths[0] != c.generators || b.widths[1] != c.generators ||
            b.widths[2] != 2 * c.logical_qubits)
          return CUDAQ_EDEM_INVALID_CONTENT;
      } else {
        uint32_t kind = free_kinds[b.type - s.codes.count];
        if ((!side && kind != CUDAQ_EDEM_FAULT) ||
            (side && kind == CUDAQ_EDEM_FAULT))
          return CUDAQ_EDEM_INVALID_CONTENT;
        if (!side) {
          if (fault >= s.faults.count || s.fault_data[fault].bundle != i ||
              s.fault_data[fault].size != b.widths[0])
            return CUDAQ_EDEM_INVALID_CONTENT;
          auto f = s.fault_data[fault];
          st = check_distribution(
              f.size, f.group_count,
              [&](uint32_t j) { return s.group(fault, j); },
              std::span(static_cast<uint8_t *>(scratch.data()),
                        static_cast<size_t>(required)),
              true);
          if (st != CUDAQ_EDEM_OK)
            return st;
          ++fault;
        }
      }
    }
  }
  if (fault != s.faults.count)
    return CUDAQ_EDEM_INVALID_CONTENT;
  for (uint64_t i = 0; i < shape.block_count; ++i) {
    cudaq_gf2_map_block b;
    cudaq_gf2_map_get_block(view, i, &b);
    cudaq_gf2_bundle in, out;
    cudaq_gf2_map_get_bundle(view, 0, b.input.bundle, &in);
    cudaq_gf2_map_get_bundle(view, 1, b.output.bundle, &out);
    if (in.type < s.codes.count && b.input.wire == 0 &&
        ((out.type < s.codes.count && b.output.wire != 0) ||
         out.type == s.codes.count + 2))
      return CUDAQ_EDEM_INVALID_CONTENT;
  }
  scratch.reset();
  *out = {*view};
  return CUDAQ_EDEM_OK;
}
cudaq_edem_status cudaq_edem_get_code(const cudaq_edem_view *value, uint32_t i,
                                      cudaq_edem_code_presentation *out) {
  const auto *view = value ? &value->map : nullptr;
  cudaq_gf2_section section;
  if (!out ||
      cudaq_gf2_map_get_section(view, CUDAQ_EDEM_SECTION_PRESENTATIONS,
                                &section) != CUDAQ_GF2_OK ||
      i >= section.count)
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  auto c = static_cast<const cudaq_edem_packed_code *>(section.data)[i];
  cudaq_gf2_bundle_type t;
  if (cudaq_gf2_map_get_type(view, c.type, &t) != CUDAQ_GF2_OK)
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  *out = {t.name + 10, t.name_size - 10, c.generators, c.logical_qubits};
  return CUDAQ_EDEM_OK;
}
cudaq_edem_status cudaq_edem_describe_wire(const cudaq_edem_view *value,
                                           uint32_t side, cudaq_gf2_wire wire,
                                           cudaq_edem_endpoint *out) {
  const auto *view = value ? &value->map : nullptr;
  cudaq_gf2_map_shape shape;
  cudaq_gf2_section section;
  if (!out || side > 1 ||
      cudaq_gf2_map_get_shape(view, &shape) != CUDAQ_GF2_OK ||
      cudaq_gf2_map_get_section(view, CUDAQ_EDEM_SECTION_PRESENTATIONS,
                                &section) != CUDAQ_GF2_OK ||
      wire.bundle >=
          (side ? shape.output_bundle_count : shape.input_bundle_count))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  cudaq_gf2_bundle bundle;
  cudaq_gf2_map_get_bundle(view, side, wire.bundle, &bundle);
  bool code = bundle.type < section.count;
  if (wire.wire >= (code ? 3u : 1u))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  uint32_t index = 0;
  for (uint32_t i = 0; i < wire.bundle; ++i) {
    cudaq_gf2_bundle b;
    cudaq_gf2_map_get_bundle(view, side, i, &b);
    if (code ? b.type < section.count : b.type == bundle.type)
      ++index;
  }
  *out = {code ? CUDAQ_EDEM_VIRTUAL + wire.wire
               : free_kinds[bundle.type - section.count],
          index};
  return CUDAQ_EDEM_OK;
}
cudaq_edem_status cudaq_edem_resolve_endpoint(const cudaq_edem_view *value,
                                              uint32_t side,
                                              cudaq_edem_endpoint endpoint,
                                              cudaq_gf2_wire *out) {
  const auto *view = value ? &value->map : nullptr;
  cudaq_gf2_map_shape shape;
  cudaq_gf2_section section;
  if (!out || side > 1 ||
      cudaq_gf2_map_get_shape(view, &shape) != CUDAQ_GF2_OK ||
      cudaq_gf2_map_get_section(view, CUDAQ_EDEM_SECTION_PRESENTATIONS,
                                &section) != CUDAQ_GF2_OK)
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  bool code = endpoint.kind >= CUDAQ_EDEM_VIRTUAL &&
              endpoint.kind <= CUDAQ_EDEM_LOGICAL;
  uint32_t index = 0,
           count = side ? shape.output_bundle_count : shape.input_bundle_count;
  for (uint32_t i = 0; i < count; ++i) {
    cudaq_gf2_bundle b;
    cudaq_gf2_map_get_bundle(view, side, i, &b);
    bool matches = code ? b.type < section.count
                        : (b.type >= section.count &&
                           free_kinds[b.type - section.count] == endpoint.kind);
    if (matches && index++ == endpoint.index) {
      *out = {i, code ? endpoint.kind - CUDAQ_EDEM_VIRTUAL : 0};
      return CUDAQ_EDEM_OK;
    }
  }
  return CUDAQ_EDEM_INVALID_ARGUMENT;
}
cudaq_edem_status cudaq_edem_get_fault(const cudaq_edem_view *value,
                                       uint32_t bundle, uint32_t *size,
                                       uint32_t *groups) {
  const auto *view = value ? &value->map : nullptr;
  sections s;
  uint32_t i;
  if (!size || !groups || !find_fault(view, bundle, s, i))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  *size = s.fault_data[i].size;
  *groups = s.fault_data[i].group_count;
  return CUDAQ_EDEM_OK;
}
cudaq_edem_status cudaq_edem_get_group(const cudaq_edem_view *value,
                                       uint32_t bundle, uint32_t group,
                                       cudaq_edem_fault_group *out) {
  const auto *view = value ? &value->map : nullptr;
  sections s;
  uint32_t i;
  if (!out || !find_fault(view, bundle, s, i) ||
      group >= s.fault_data[i].group_count)
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  *out = s.group(i, group);
  return CUDAQ_EDEM_OK;
}
