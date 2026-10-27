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
static_assert(sizeof(cudaq_edem_packed_code) == 24);
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
// Private checked code table bytes: a header, entries in input order, then
// names. Ranks give each entry's position in name order.
struct table_header {
  uint32_t code_count, reserved;
};
struct table_entry {
  uint64_t id, name_offset, name_size;
  uint32_t syndrome_count, virtual_outcome_count, logical_qubit_count, rank;
};
static_assert(sizeof(table_header) == 8 && sizeof(table_entry) == 40);
uint32_t table_count(const cudaq_edem_checked_code_table &t) {
  return static_cast<const table_header *>(t.data)->code_count;
}
const table_entry &table_at(const cudaq_edem_checked_code_table &t,
                            uint32_t i) {
  return reinterpret_cast<const table_entry *>(
      static_cast<const std::byte *>(t.data) + sizeof(table_header))[i];
}
const char *table_name(const cudaq_edem_checked_code_table &t,
                       const table_entry &e) {
  return static_cast<const char *>(t.data) + e.name_offset;
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
  const auto *table = d->shared_code_table;
  const uint64_t code_blocks =
      uint64_t{s.input_code_blocks_count} + s.output_code_blocks_count;
  if ((code_blocks && (!table || !table->data)) ||
      code_blocks > UINT32_MAX - 4 || d->block_count > UINT32_MAX ||
      uint64_t{s.fault_ports_count} + s.input_code_blocks_count > UINT32_MAX ||
      uint64_t{s.detector_ports_count} + s.observable_ports_count +
              s.success_ports_count + s.output_code_blocks_count >
          UINT32_MAX ||
      !array_ok(d->input_code_indices, s.input_code_blocks_count, 4) ||
      !array_ok(d->output_code_indices, s.output_code_blocks_count, 4) ||
      !array_ok(d->fault_ports_distributions, s.fault_ports_count,
                sizeof(*d->fault_ports_distributions)) ||
      !array_ok(d->detector_ports_sizes, s.detector_ports_count, 4) ||
      !array_ok(d->observable_ports_sizes, s.observable_ports_count, 4) ||
      !array_ok(d->success_ports_sizes, s.success_ports_count, 4) ||
      !array_ok(d->blocks, d->block_count, sizeof(*d->blocks)))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  p.inputs = s.fault_ports_count + s.input_code_blocks_count;
  p.outputs = s.detector_ports_count + s.observable_ports_count +
              s.success_ports_count + s.output_code_blocks_count;
  p.wires =
      uint64_t{s.fault_ports_count} + s.detector_ports_count +
      s.observable_ports_count + s.success_ports_count +
      3 * (uint64_t{s.input_code_blocks_count} + s.output_code_blocks_count);
  uint64_t ignored;
  // Names are reserved per code block, an upper bound on the codes used.
  for (uint32_t side = 0; side < 2; ++side) {
    uint32_t count =
        side ? s.output_code_blocks_count : s.input_code_blocks_count;
    const auto *indices = side ? d->output_code_indices : d->input_code_indices;
    for (uint32_t i = 0; i < count; ++i) {
      if (indices[i] >= table_count(*table))
        return CUDAQ_EDEM_INVALID_CONTENT;
      if (!reserve(p.names, table_at(*table, indices[i]).name_size + 10, 1,
                   ignored, 1))
        return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
    }
  }
  if (!reserve(p.fault_bytes, s.fault_ports_count,
               sizeof(cudaq_edem_packed_fault), ignored))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  for (uint32_t i = 0; i < s.fault_ports_count; ++i) {
    auto f = d->fault_ports_distributions[i];
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
  if (!reserve(n, code_blocks, 4, ignored) ||
      !reserve(n, code_blocks + 4, sizeof(cudaq_gf2_bundle_type), ignored) ||
      !reserve(n, p.inputs, sizeof(cudaq_gf2_bundle), ignored) ||
      !reserve(n, p.outputs, sizeof(cudaq_gf2_bundle), ignored) ||
      !reserve(n, p.wires, 4, ignored) ||
      !reserve(n, d->block_count, sizeof(cudaq_gf2_map_block), ignored) ||
      !reserve(n, p.names, 1, ignored) ||
      !reserve(n, code_blocks, sizeof(cudaq_edem_packed_code), ignored) ||
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
    double sum = 0, correction = 0;
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
      // Kahan summation keeps rounding in normalized groups from accumulating.
      double adjusted = probability - correction;
      double next = sum + adjusted;
      correction = (next - sum) - adjusted;
      sum = next;
    }
    if (g.kind == CUDAQ_EDEM_EXCLUSIVE &&
        sum > 1 + CUDAQ_EDEM_EXCLUSIVE_TOTAL_TOLERANCE)
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
    if (e.index >=
        (side ? s.output_code_blocks_count : s.input_code_blocks_count))
      return false;
    uint32_t start = side ? s.detector_ports_count + s.observable_ports_count +
                                s.success_ports_count
                          : s.fault_ports_count;
    wire = {start + e.index, e.kind - CUDAQ_EDEM_VIRTUAL};
    return true;
  }
  if (!side && e.kind == CUDAQ_EDEM_FAULT && e.index < s.fault_ports_count) {
    wire = {e.index, 0};
    return true;
  }
  if (side) {
    if (e.kind == CUDAQ_EDEM_DETECTOR && e.index < s.detector_ports_count) {
      wire = {e.index, 0};
      return true;
    }
    if (e.kind == CUDAQ_EDEM_OBSERVABLE && e.index < s.observable_ports_count) {
      wire = {s.detector_ports_count + e.index, 0};
      return true;
    }
    if (e.kind == CUDAQ_EDEM_SUCCESS && e.index < s.success_ports_count) {
      wire = {s.detector_ports_count + s.observable_ports_count + e.index, 0};
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
  const auto *table = d->shared_code_table;
  const uint32_t code_blocks =
      s.input_code_blocks_count + s.output_code_blocks_count;
  arena a{static_cast<std::byte *>(scratch.data()), p.total};
  auto used = a.take<uint32_t>(code_blocks);
  auto types = a.take<cudaq_gf2_bundle_type>(code_blocks + 4);
  auto inputs = a.take<cudaq_gf2_bundle>(p.inputs),
       outputs = a.take<cudaq_gf2_bundle>(p.outputs);
  auto widths = a.take<uint32_t>(p.wires);
  auto blocks = a.take<cudaq_gf2_map_block>(d->block_count);
  auto names = a.take<char>(p.names);
  auto codes = a.take<cudaq_edem_packed_code>(code_blocks);
  auto fault_data = a.take<uint8_t>(p.fault_bytes);
  auto group_order = a.take<uint32_t>(p.max_groups);
  auto seen = a.take<uint8_t>((uint64_t{p.max_faults} + 7) / 8);
  // The table was checked when built; keep the codes used, in name order.
  std::copy_n(d->input_code_indices, s.input_code_blocks_count, used.begin());
  std::copy_n(d->output_code_indices, s.output_code_blocks_count,
              used.begin() + s.input_code_blocks_count);
  auto by_rank = [&](uint32_t x, uint32_t y) {
    return table_at(*table, x).rank < table_at(*table, y).rank;
  };
  std::sort(used.begin(), used.end(), by_rank);
  const uint32_t code_count = static_cast<uint32_t>(
      std::unique(used.begin(), used.end()) - used.begin());
  uint64_t cursor = 0;
  for (uint32_t i = 0; i < code_count; ++i) {
    const auto &c = table_at(*table, used[i]);
    char *name = names.data() + cursor;
    memcpy(name, code_prefix, 10);
    memcpy(name + 10, table_name(*table, c), c.name_size);
    cursor += 10 + c.name_size;
    types[i] = {name, 10 + c.name_size, CUDAQ_GF2_INTERFACE, 3};
    codes[i] = {c.id, i, c.syndrome_count, c.virtual_outcome_count,
                c.logical_qubit_count};
  }
  auto local = [&](uint32_t index) {
    return static_cast<uint32_t>(std::lower_bound(used.begin(),
                                                  used.begin() + code_count,
                                                  index, by_rank) -
                                 used.begin());
  };
  for (uint32_t i = 0; i < 4; ++i)
    types[code_count + i] = {free_names[i], strlen(free_names[i]),
                             CUDAQ_GF2_FREE, 1};
  uint64_t wi = 0;
  for (uint32_t side = 0; side < 2; ++side) {
    auto bundles = side ? outputs : inputs;
    uint32_t free_count = side ? s.detector_ports_count +
                                     s.observable_ports_count +
                                     s.success_ports_count
                               : s.fault_ports_count;
    for (uint32_t i = 0; i < free_count; ++i) {
      uint32_t type, width;
      if (!side) {
        type = 1;
        width = d->fault_ports_distributions[i].size;
      } else if (i < s.detector_ports_count) {
        type = 0;
        width = d->detector_ports_sizes[i];
      } else if (i < s.detector_ports_count + s.observable_ports_count) {
        type = 2;
        width = d->observable_ports_sizes[i - s.detector_ports_count];
      } else {
        type = 3;
        width = d->success_ports_sizes[i - s.detector_ports_count -
                                       s.observable_ports_count];
      }
      widths[wi] = width;
      bundles[i] = {code_count + type, 0, widths.data() + wi++};
    }
    uint32_t count =
        side ? s.output_code_blocks_count : s.input_code_blocks_count;
    const auto *code_indices =
        side ? d->output_code_indices : d->input_code_indices;
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t id = local(code_indices[i]);
      auto c = codes[id];
      widths[wi] = c.virtual_outcome_count;
      widths[wi + 1] = c.syndrome_count;
      widths[wi + 2] = 2 * c.logical_qubit_count;
      bundles[free_count + i] = {id, 0, widths.data() + wi};
      wi += 3;
    }
  }
  for (uint32_t i = 0; i < d->block_count; ++i) {
    auto b = d->blocks[i];
    if (!semantic_wire(s, 1, b.output, blocks[i].output) ||
        !semantic_wire(s, 0, b.input, blocks[i].input))
      return CUDAQ_EDEM_INVALID_CONTENT;
    if (b.matrix.nonzeros && b.input.kind == CUDAQ_EDEM_VIRTUAL &&
        (b.output.kind == CUDAQ_EDEM_SYNDROME ||
         b.output.kind == CUDAQ_EDEM_LOGICAL))
      return CUDAQ_EDEM_INVALID_CONTENT;
    blocks[i].matrix = b.matrix;
  }
  auto *faults = reinterpret_cast<cudaq_edem_packed_fault *>(fault_data.data());
  cursor = uint64_t{s.fault_ports_count} * sizeof(*faults);
  for (uint32_t i = 0; i < s.fault_ports_count; ++i) {
    auto f = d->fault_ports_distributions[i];
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
      {CUDAQ_EDEM_SECTION_PRESENTATIONS, CUDAQ_GF2_SECTION_REQUIRED, code_count,
       code_count * sizeof(cudaq_edem_packed_code), codes.data()},
      {CUDAQ_EDEM_SECTION_DISTRIBUTIONS, CUDAQ_GF2_SECTION_REQUIRED,
       s.fault_ports_count, fault_data.size(), fault_data.data()}};
  cudaq_gf2_map_description map = {
      code_count + 4, p.inputs,     p.outputs,     2,
      d->block_count, types.data(), inputs.data(), outputs.data(),
      blocks.data(),  sections};
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
  if (s == CUDAQ_EDEM_UNSUPPORTED_DISTRIBUTION)
    return "unsupported fault distribution";
  return cudaq_gf2_status_string(static_cast<cudaq_gf2_status>(s));
}
cudaq_gf2_schema cudaq_edem_schema(void) { return {required_sections, 2, 0}; }
cudaq_edem_status cudaq_edem_code_table_construct(
    const cudaq_edem_code_table *table, const cudaq_gf2_memory *memory,
    cudaq_gf2_owned_buffer *owner, cudaq_edem_checked_code_table *out) {
  if (!table || !owner || !out || !empty_owner(*owner) ||
      !array_ok(table->codes, table->code_count, sizeof(*table->codes)))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  const auto scratch_resource = scratch_allocator(memory),
             result_resource = result_allocator(memory);
  if (!valid_allocator(scratch_resource) || !valid_allocator(result_resource))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  const uint32_t count = table->code_count;
  const auto *codes = table->codes;
  uint64_t bytes = sizeof(table_header), entries, ignored;
  if (!reserve(bytes, count, sizeof(table_entry), entries))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  for (uint32_t i = 0; i < count; ++i) {
    if (!array_ok(codes[i].name, codes[i].name_size, 1))
      return CUDAQ_EDEM_INVALID_ARGUMENT;
    if (!codes[i].name_size || codes[i].name_size > SIZE_MAX - 10 ||
        !utf8_name(codes[i].name, codes[i].name_size) ||
        codes[i].logical_qubit_count > UINT32_MAX / 2)
      return CUDAQ_EDEM_INVALID_CONTENT;
    if (!reserve(bytes, codes[i].name_size, 1, ignored, 1))
      return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  }
  uint64_t scratch_bytes = 0;
  if (!reserve(scratch_bytes, count, 4, ignored) ||
      !reserve(scratch_bytes, count, 4, ignored))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  allocation scratch(scratch_resource);
  auto allocated = scratch.reserve(scratch_bytes, CUDAQ_EDEM_PACKED_ALIGNMENT);
  if (allocated != CUDAQ_GF2_OK)
    return status(allocated);
  arena a{static_cast<std::byte *>(scratch.data()), scratch_bytes};
  auto order = a.take<uint32_t>(count), rank = a.take<uint32_t>(count);
  sort_indices(
      order, [&](uint32_t x, uint32_t y) { return codes[x].id < codes[y].id; });
  for (uint32_t i = 1; i < count; ++i)
    if (codes[order[i - 1]].id == codes[order[i]].id)
      return CUDAQ_EDEM_INVALID_CONTENT;
  sort_indices(order, [&](uint32_t x, uint32_t y) {
    return name_compare(codes[x].name, codes[x].name_size, codes[y].name,
                        codes[y].name_size) < 0;
  });
  for (uint32_t i = 0; i < count; ++i) {
    if (i &&
        !name_compare(codes[order[i - 1]].name, codes[order[i - 1]].name_size,
                      codes[order[i]].name, codes[order[i]].name_size))
      return CUDAQ_EDEM_INVALID_CONTENT;
    rank[order[i]] = i;
  }
  allocation storage(result_resource);
  allocated = storage.reserve(bytes, CUDAQ_EDEM_PACKED_ALIGNMENT);
  if (allocated != CUDAQ_GF2_OK)
    return status(allocated);
  auto *data = static_cast<std::byte *>(storage.data());
  memset(data, 0, static_cast<size_t>(bytes));
  *reinterpret_cast<table_header *>(data) = {count, 0};
  auto *entry = reinterpret_cast<table_entry *>(data + entries);
  uint64_t cursor = entries + uint64_t{count} * sizeof(table_entry);
  for (uint32_t i = 0; i < count; ++i) {
    const auto &c = codes[i];
    entry[i] = {c.id,
                cursor,
                c.name_size,
                c.syndrome_count,
                c.virtual_outcome_count,
                c.logical_qubit_count,
                rank[i]};
    memcpy(data + cursor, c.name, static_cast<size_t>(c.name_size));
    cursor += c.name_size;
  }
  *owner = storage.relinquish();
  *out = {owner->data, bytes};
  return CUDAQ_EDEM_OK;
}
cudaq_edem_status
cudaq_edem_code_table_get_count(const cudaq_edem_checked_code_table *table,
                                uint32_t *count) {
  if (!table || !table->data || !count)
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  *count = table_count(*table);
  return CUDAQ_EDEM_OK;
}
cudaq_edem_status
cudaq_edem_code_table_get_code(const cudaq_edem_checked_code_table *table,
                               uint32_t index,
                               cudaq_edem_code_presentation *out) {
  if (!table || !table->data || !out || index >= table_count(*table))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  const auto &e = table_at(*table, index);
  *out = {e.id,
          table_name(*table, e),
          e.name_size,
          e.syndrome_count,
          e.virtual_outcome_count,
          e.logical_qubit_count};
  return CUDAQ_EDEM_OK;
}
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
  // Scratch holds a distribution bitset, then the sorted code ids, which are
  // later reused as one used flag per code.
  uint64_t bytes = required, id_offset;
  if (!reserve(bytes, s.codes.count, sizeof(uint64_t), id_offset))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  allocation scratch(resource);
  auto allocated = scratch.reserve(bytes, CUDAQ_EDEM_PACKED_ALIGNMENT);
  if (allocated != CUDAQ_GF2_OK)
    return status(allocated);
  auto *sorted_ids = reinterpret_cast<uint64_t *>(
      static_cast<std::byte *>(scratch.data()) + id_offset);
  for (uint64_t i = 0; i < s.codes.count; ++i)
    sorted_ids[i] = s.code_data[i].id;
  std::sort(sorted_ids, sorted_ids + s.codes.count);
  if (std::adjacent_find(sorted_ids, sorted_ids + s.codes.count) !=
      sorted_ids + s.codes.count)
    return CUDAQ_EDEM_INVALID_CONTENT;
  auto *used = reinterpret_cast<uint8_t *>(sorted_ids);
  std::fill_n(used, s.codes.count, uint8_t{0});
  cudaq_gf2_map_shape shape;
  if (cudaq_gf2_map_get_shape(view, &shape) != CUDAQ_GF2_OK ||
      s.codes.count > UINT32_MAX - 4 || shape.type_count != s.codes.count + 4)
    return CUDAQ_EDEM_INVALID_CONTENT;
  for (uint32_t i = 0; i < shape.type_count; ++i) {
    cudaq_gf2_bundle_type t;
    cudaq_gf2_map_get_type(view, i, &t);
    if (i < s.codes.count) {
      auto c = s.code_data[i];
      if (c.type != i || c.logical_qubit_count > UINT32_MAX / 2 ||
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
        used[b.type] = 1;
        auto c = s.code_data[b.type];
        if (b.widths[0] != c.virtual_outcome_count ||
            b.widths[1] != c.syndrome_count ||
            b.widths[2] != 2 * c.logical_qubit_count)
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
  // Only presentations used by code bundles are retained.
  if (fault != s.faults.count ||
      std::find(used, used + s.codes.count, uint8_t{0}) != used + s.codes.count)
    return CUDAQ_EDEM_INVALID_CONTENT;
  for (uint64_t i = 0; i < shape.block_count; ++i) {
    cudaq_gf2_map_block b;
    cudaq_gf2_map_get_block(view, i, &b);
    cudaq_gf2_bundle in, out;
    cudaq_gf2_map_get_bundle(view, 0, b.input.bundle, &in);
    cudaq_gf2_map_get_bundle(view, 1, b.output.bundle, &out);
    if (in.type < s.codes.count && b.input.wire == 0 &&
        out.type < s.codes.count && b.output.wire != 0)
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
  *out = {c.id,
          t.name + 10,
          t.name_size - 10,
          c.syndrome_count,
          c.virtual_outcome_count,
          c.logical_qubit_count};
  return CUDAQ_EDEM_OK;
}
cudaq_edem_status cudaq_edem_get_code_count(const cudaq_edem_view *value,
                                            uint32_t *count) {
  const auto *view = value ? &value->map : nullptr;
  cudaq_gf2_section section;
  if (!count ||
      cudaq_gf2_map_get_section(view, CUDAQ_EDEM_SECTION_PRESENTATIONS,
                                &section) != CUDAQ_GF2_OK)
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  *count = static_cast<uint32_t>(section.count);
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
