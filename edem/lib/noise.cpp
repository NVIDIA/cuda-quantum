/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#pragma GCC visibility push(default)
#include "cudaq/edem/noise.h"
#pragma GCC visibility pop
#include "allocation.h"
#include "storage.h"
#include <cmath>

using namespace cudaq_gf2_private;

#define EDEM_TRY(expression)                                                   \
  do {                                                                         \
    auto status_ = static_cast<cudaq_edem_status>(expression);                 \
    if (status_ != CUDAQ_EDEM_OK)                                              \
      return status_;                                                          \
  } while (false)

namespace {
template <class T>
bool space(uint64_t &bytes, uint64_t count) {
  uint64_t offset;
  return reserve(bytes, count, sizeof(T), offset);
}

bool add(uint64_t &sum, uint64_t n) {
  if (n > SIZE_MAX - sum)
    return false;
  sum += n;
  return true;
}

struct output_port {
  cudaq_edem_endpoint endpoint;
  uint64_t offset;
  uint32_t size;
};

// Sparse complete response columns in semantic D/O/Success row order. All
// temporary containers live in one allocation from the caller's scratch.
struct responses {
  allocation storage;
  cudaq_edem_shape shape{};
  uint64_t block_count = 0, columns = 0, nonzeros = 0;
  std::span<cudaq_edem_block> blocks;
  std::span<output_port> outputs;
  std::span<uint32_t> sizes, ids;
  std::span<cudaq_edem_fault_distribution> faults;
  std::span<cudaq_edem_fault_group> groups;
  std::span<uint64_t> fault_offsets, offsets, rows;
  std::span<double> probabilities;

  explicit responses(const cudaq_gf2_memory *memory)
      : storage(scratch_allocator(memory)) {}

  std::span<const uint64_t> column(uint64_t id) const {
    // columns is also the sentinel for the identity response.
    return id == columns
               ? std::span<const uint64_t>{}
               : rows.subspan(offsets[id], offsets[id + 1] - offsets[id]);
  }

  uint64_t output_index(cudaq_edem_endpoint endpoint) const {
    return endpoint.index + (endpoint.kind == CUDAQ_EDEM_DETECTOR ? 0
                             : endpoint.kind == CUDAQ_EDEM_OBSERVABLE
                                 ? uint64_t(shape.detector_ports_count)
                                 : uint64_t(shape.detector_ports_count) +
                                       shape.observable_ports_count);
  }

  cudaq_edem_status load(const cudaq_edem_dem_view *value) {
    if (!value)
      return CUDAQ_EDEM_INVALID_ARGUMENT;
    EDEM_TRY(cudaq_edem_get_shape(&value->edem, &shape));
    if (shape.input_code_blocks_count || shape.output_code_blocks_count)
      return CUDAQ_EDEM_INVALID_CONTENT;
    cudaq_gf2_map_shape map;
    EDEM_TRY(cudaq_gf2_map_get_shape(&value->edem.map, &map));
    block_count = map.block_count;
    uint64_t row_count;
    EDEM_TRY(
        cudaq_gf2_map_vector_sizes(&value->edem.map, &columns, &row_count));
    if (columns == SIZE_MAX)
      return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
    for (uint64_t b = 0; b < block_count; ++b) {
      cudaq_gf2_map_block block;
      EDEM_TRY(cudaq_gf2_map_get_block(&value->edem.map, b, &block));
      if (!add(nonzeros, block.matrix.nonzeros))
        return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
    }
    const uint64_t output_count = uint64_t(shape.detector_ports_count) +
                                  shape.observable_ports_count +
                                  shape.success_ports_count;
    uint64_t bytes = 0;
    if (!space<cudaq_edem_block>(bytes, block_count) ||
        !space<output_port>(bytes, output_count) ||
        !space<uint32_t>(bytes, output_count) ||
        !space<cudaq_edem_fault_distribution>(bytes, shape.fault_ports_count) ||
        !space<cudaq_edem_fault_group>(bytes, shape.fault_ports_count) ||
        !space<uint64_t>(bytes, uint64_t(shape.fault_ports_count) + 1) ||
        !space<uint64_t>(bytes, columns + 1) ||
        !space<uint64_t>(bytes, columns) || !space<uint64_t>(bytes, nonzeros) ||
        !space<uint32_t>(bytes, columns) || !space<double>(bytes, columns))
      return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
    EDEM_TRY(storage.reserve(bytes, CUDAQ_EDEM_PACKED_ALIGNMENT));
    arena a{static_cast<std::byte *>(storage.data()), bytes};
    blocks = a.take<cudaq_edem_block>(block_count);
    outputs = a.take<output_port>(output_count);
    sizes = a.take<uint32_t>(output_count);
    faults = a.take<cudaq_edem_fault_distribution>(shape.fault_ports_count);
    groups = a.take<cudaq_edem_fault_group>(shape.fault_ports_count);
    fault_offsets = a.take<uint64_t>(uint64_t(shape.fault_ports_count) + 1);
    offsets = a.take<uint64_t>(columns + 1);
    auto cursor = a.take<uint64_t>(columns);
    rows = a.take<uint64_t>(nonzeros);
    ids = a.take<uint32_t>(columns);
    probabilities = a.take<double>(columns);
    for (uint32_t p = 0; p < shape.fault_ports_count; ++p) {
      uint32_t size, ignored;
      EDEM_TRY(cudaq_edem_get_fault(&value->edem, p, &size, &ignored));
      const uint64_t start = fault_offsets[p];
      fault_offsets[p + 1] = start + size;
      faults[p] = {size, size ? 1u : 0u, size ? &groups[p] : nullptr};
      if (size) {
        groups[p] = {CUDAQ_EDEM_INDEPENDENT, size, ids.data() + start,
                     probabilities.data() + start};
        for (uint32_t i = 0; i < size; ++i)
          ids[start + i] = i;
      }
    }
    for (uint32_t p = 0; p < output_count; ++p) {
      cudaq_edem_endpoint endpoint;
      uint32_t width;
      uint64_t ignored;
      EDEM_TRY(cudaq_edem_describe_wire(&value->edem, true, {p, 0}, &endpoint));
      EDEM_TRY(cudaq_gf2_map_get_wire(&value->edem.map, true, {p, 0}, &width,
                                      &ignored));
      const uint64_t index = output_index(endpoint);
      outputs[index] = {endpoint, 0, width};
      sizes[index] = width;
    }
    uint64_t start = 0;
    for (auto &output : outputs) {
      output.offset = start;
      start += output.size;
    }
    for (uint64_t b = 0; b < block_count; ++b) {
      cudaq_gf2_map_block block;
      EDEM_TRY(cudaq_gf2_map_get_block(&value->edem.map, b, &block));
      EDEM_TRY(cudaq_edem_describe_wire(&value->edem, true, block.output,
                                        &blocks[b].output));
      EDEM_TRY(cudaq_edem_describe_wire(&value->edem, false, block.input,
                                        &blocks[b].input));
      blocks[b].matrix = block.matrix;
      const uint64_t column_start = fault_offsets[blocks[b].input.index];
      for (uint64_t c = 0; c < block.matrix.columns; ++c)
        offsets[column_start + c + 1] +=
            block.matrix.column_offsets[c + 1] - block.matrix.column_offsets[c];
    }
    for (uint64_t c = 0; c < columns; ++c) {
      offsets[c + 1] += offsets[c];
      cursor[c] = offsets[c];
    }
    for (const auto &block : blocks) {
      const uint64_t column_start = fault_offsets[block.input.index];
      const uint64_t row_start = outputs[output_index(block.output)].offset;
      for (uint64_t c = 0; c < block.matrix.columns; ++c)
        for (uint64_t i = block.matrix.column_offsets[c];
             i < block.matrix.column_offsets[c + 1]; ++i)
          rows[cursor[column_start + c]++] =
              row_start + block.matrix.row_indices[i];
    }
    for (uint64_t c = 0; c < columns; ++c) {
      auto entries = rows.subspan(offsets[c], offsets[c + 1] - offsets[c]);
      if (entries.size() > 1)
        std::sort(entries.begin(), entries.end(),
                  [](uint64_t x, uint64_t y) { return x < y; });
    }
    return CUDAQ_EDEM_OK;
  }

  cudaq_edem_description description() const {
    auto data_at = [&](uint64_t i) {
      return sizes.empty() ? nullptr : sizes.data() + i;
    };
    return {shape,
            nullptr,
            nullptr,
            nullptr,
            faults.data(),
            sizes.data(),
            data_at(shape.detector_ports_count),
            data_at(uint64_t(shape.detector_ports_count) +
                    shape.observable_ports_count),
            block_count,
            blocks.data()};
  }
};

bool equal(std::span<const uint64_t> a, std::span<const uint64_t> b) {
  return a.size() == b.size() &&
         (a.empty() || std::equal(a.begin(), a.end(), b.begin()));
}

bool xor_equals(std::span<const uint64_t> a, std::span<const uint64_t> b,
                std::span<const uint64_t> c) {
  size_t i = 0, j = 0, k = 0;
  while (i < a.size() || j < b.size()) {
    if (i < a.size() && j < b.size() && a[i] == b[j]) {
      ++i;
      ++j;
      continue;
    }
    uint64_t row =
        j == b.size() || (i < a.size() && a[i] < b[j]) ? a[i++] : b[j++];
    if (k == c.size() || c[k++] != row)
      return false;
  }
  return k == c.size();
}

// At most sixteen responses. Verify a uniform linear image, including identity
// and kernel multiplicities. Generate its span with sparse XOR comparisons;
// there is no dense response bitset and no scan across unrelated fault groups.
bool pauli_image(const responses &data, uint64_t start,
                 const cudaq_edem_fault_group &group) {
  uint64_t unique[16] = {data.columns};
  uint32_t multiplicity[16] = {1}, count = 1;
  for (uint32_t i = 0; i < group.size; ++i) {
    uint64_t id = start + group.fault_ids[i];
    uint32_t u = 0;
    while (u < count && !equal(data.column(id), data.column(unique[u])))
      ++u;
    if (u == count)
      unique[count++] = id;
    ++multiplicity[u];
  }
  if (count & (count - 1))
    return false;
  for (uint32_t i = 1; i < count; ++i)
    if (multiplicity[i] != multiplicity[0])
      return false;
  uint32_t span[16] = {0}, span_size = 1;
  for (uint32_t u = 1; u < count; ++u) {
    if (std::find(span, span + span_size, u) != span + span_size)
      continue;
    uint32_t old_size = span_size;
    for (uint32_t s = 0; s < old_size; ++s) {
      uint32_t v = 0;
      while (v < count &&
             !xor_equals(data.column(unique[span[s]]), data.column(unique[u]),
                         data.column(unique[v])))
        ++v;
      if (v == count || span_size == count ||
          std::find(span, span + span_size, v) != span + span_size)
        return false;
      span[span_size++] = v;
    }
  }
  return span_size == count;
}

void compensated_add(double x, double &sum, double &correction) {
  double adjusted = x - correction;
  double next = sum + adjusted;
  correction = (next - sum) - adjusted;
  sum = next;
}

cudaq_edem_status probabilities(const cudaq_edem_dem_view &value,
                                responses &data, bool factorize) {
  for (uint32_t p = 0; p < data.shape.fault_ports_count; ++p) {
    uint32_t width, count;
    EDEM_TRY(cudaq_edem_get_fault(&value.edem, p, &width, &count));
    uint64_t start = data.fault_offsets[p];
    for (uint32_t g = 0; g < count; ++g) {
      cudaq_edem_fault_group group;
      EDEM_TRY(cudaq_edem_get_group(&value.edem, p, g, &group));
      if (group.kind == CUDAQ_EDEM_INDEPENDENT) {
        for (uint32_t i = 0; i < group.size; ++i)
          data.probabilities[start + group.fault_ids[i]] =
              group.probabilities[i];
        continue;
      }
      if (!factorize || (group.size != 3 && group.size != 15))
        return CUDAQ_EDEM_UNSUPPORTED_DISTRIBUTION;
      double total = 0, correction = 0;
      for (uint32_t i = 0; i < group.size; ++i) {
        if (group.probabilities[i] != group.probabilities[0])
          return CUDAQ_EDEM_UNSUPPORTED_DISTRIBUTION;
        compensated_add(group.probabilities[i], total, correction);
      }
      const double limit = double(group.size) / (group.size + 1);
      if (total > limit || !pauli_image(data, start, group))
        return CUDAQ_EDEM_UNSUPPORTED_DISTRIBUTION;
      double q = total == 0 ? 0
                 : total == limit
                     ? 0.5
                     : -0.5 * std::expm1(std::log1p(-total / limit) /
                                         ((group.size + 1) / 2));
      for (uint32_t i = 0; i < group.size; ++i)
        data.probabilities[start + group.fault_ids[i]] = q;
    }
  }
  return CUDAQ_EDEM_OK;
}

bool valid_outputs(const cudaq_gf2_memory *memory,
                   const cudaq_gf2_owned_buffer *owner,
                   const cudaq_edem_dem_view *result) {
  return owner && result && empty_owner(*owner) &&
         valid_allocator(scratch_allocator(memory)) &&
         valid_allocator(result_allocator(memory));
}
} // namespace

cudaq_edem_status cudaq_edem_factorize_dem(const cudaq_edem_dem_view *value,
                                           const cudaq_gf2_memory *memory,
                                           cudaq_gf2_owned_buffer *owner,
                                           cudaq_edem_dem_view *result) {
  if (!valid_outputs(memory, owner, result))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  responses data(memory);
  EDEM_TRY(data.load(value));
  EDEM_TRY(probabilities(*value, data, true));
  auto description = data.description();
  cudaq_edem_view view;
  EDEM_TRY(cudaq_edem_construct(&description, memory, owner, &view));
  result->edem = view;
  return CUDAQ_EDEM_OK;
}

cudaq_edem_status cudaq_edem_coalesce_dem_columns(
    const cudaq_edem_dem_view *value, const cudaq_gf2_memory *memory,
    cudaq_gf2_owned_buffer *owner, cudaq_edem_dem_view *result,
    cudaq_gf2_owned_buffer *column_map) {
  if (!valid_outputs(memory, owner, result) || !column_map ||
      column_map == owner || !empty_owner(*column_map))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  responses data(memory);
  EDEM_TRY(data.load(value));
  EDEM_TRY(probabilities(*value, data, false));
  uint64_t bytes = 0;
  if (!space<uint64_t>(bytes, data.columns) ||
      !space<uint64_t>(bytes, data.columns) ||
      !space<double>(bytes, data.columns))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  allocation workspace(scratch_allocator(memory));
  EDEM_TRY(workspace.reserve(bytes, CUDAQ_EDEM_PACKED_ALIGNMENT));
  arena a{static_cast<std::byte *>(workspace.data()), bytes};
  auto order = a.take<uint64_t>(data.columns);
  auto representatives = a.take<uint64_t>(data.columns);
  auto merged = a.take<double>(data.columns);
  for (uint64_t i = 0; i < data.columns; ++i)
    order[i] = i;
  if (order.size() > 1)
    std::sort(order.begin(), order.end(), [&](uint64_t x, uint64_t y) {
      auto left = data.column(x), right = data.column(y);
      if (equal(left, right)) {
        if (data.probabilities[x] != data.probabilities[y])
          return data.probabilities[x] < data.probabilities[y];
        return x < y;
      }
      if (left.empty() || right.empty())
        return left.empty();
      return std::lexicographical_compare(left.begin(), left.end(),
                                          right.begin(), right.end());
    });
  if (data.columns > SIZE_MAX / sizeof(uint64_t))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  allocation mapping(result_allocator(memory));
  EDEM_TRY(mapping.reserve(data.columns * sizeof(uint64_t),
                           CUDAQ_EDEM_PACKED_ALIGNMENT));
  auto *indices = static_cast<uint64_t *>(mapping.data());
  uint64_t count = 0;
  for (uint64_t begin = 0; begin < data.columns;) {
    uint64_t end = begin + 1;
    while (end < data.columns &&
           equal(data.column(order[begin]), data.column(order[end])))
      ++end;
    double log_parity = 0, correction = 0;
    bool negative = false, half = false;
    for (uint64_t i = begin; i < end; ++i) {
      double p = data.probabilities[order[i]];
      half |= p == 0.5;
      negative ^= p > 0.5;
      if (p != 0.5)
        compensated_add(std::log1p(-2 * (p > 0.5 ? 1 - p : p)), log_parity,
                        correction);
    }
    double p = half       ? 0.5
               : negative ? 1 + 0.5 * std::expm1(log_parity)
                          : -0.5 * std::expm1(log_parity);
    bool keep = p != 0 && !data.column(order[begin]).empty();
    for (uint64_t i = begin; i < end; ++i)
      indices[order[i]] = keep ? count : UINT64_MAX;
    if (keep) {
      representatives[count] = order[begin];
      merged[count++] = p;
    }
    begin = end;
  }
  if (count > UINT32_MAX)
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  uint32_t widths[3] = {}, port_counts[3] = {};
  uint64_t row_starts[4] = {};
  for (const auto &port : data.outputs) {
    uint32_t kind = port.endpoint.kind - CUDAQ_EDEM_DETECTOR;
    if (port.size > UINT32_MAX - widths[kind])
      return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
    widths[kind] += port.size;
    port_counts[kind] = 1;
  }
  for (uint32_t kind = 0; kind < 3; ++kind)
    row_starts[kind + 1] = row_starts[kind] + widths[kind];
  bytes = 0;
  if (!space<uint64_t>(bytes, 3 * (count + 1)) ||
      !space<uint32_t>(bytes, data.nonzeros))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  allocation matrices(scratch_allocator(memory));
  EDEM_TRY(matrices.reserve(bytes, CUDAQ_EDEM_PACKED_ALIGNMENT));
  arena b{static_cast<std::byte *>(matrices.data()), bytes};
  auto offsets = b.take<uint64_t>(3 * (count + 1));
  auto rows = b.take<uint32_t>(data.nonzeros);
  cudaq_edem_block blocks[3] = {};
  uint64_t block_count = 0, cursor = 0;
  for (uint32_t kind = 0; kind < 3; ++kind) {
    uint64_t start = cursor;
    auto *column_offsets = offsets.data() + kind * (count + 1);
    for (uint64_t c = 0; c < count; ++c) {
      for (uint64_t row : data.column(representatives[c]))
        if (row >= row_starts[kind] && row < row_starts[kind + 1])
          rows[cursor++] = static_cast<uint32_t>(row - row_starts[kind]);
      column_offsets[c + 1] = cursor - start;
    }
    if (cursor != start)
      blocks[block_count++] = {{CUDAQ_EDEM_DETECTOR + kind, 0},
                               {CUDAQ_EDEM_FAULT, 0},
                               {widths[kind], static_cast<uint32_t>(count),
                                cursor - start, column_offsets,
                                rows.data() + start}};
  }
  for (uint64_t i = 0; i < count; ++i)
    data.ids[i] = static_cast<uint32_t>(i);
  cudaq_edem_fault_group group{CUDAQ_EDEM_INDEPENDENT,
                               static_cast<uint32_t>(count), data.ids.data(),
                               merged.data()};
  cudaq_edem_fault_distribution fault{static_cast<uint32_t>(count),
                                      count ? 1u : 0u, &group};
  cudaq_edem_description description{
      {0, 0, count ? 1u : 0u, port_counts[0], port_counts[1], port_counts[2]},
      nullptr,
      nullptr,
      nullptr,
      count ? &fault : nullptr,
      &widths[0],
      &widths[1],
      &widths[2],
      block_count,
      blocks};
  cudaq_edem_view view;
  EDEM_TRY(cudaq_edem_construct(&description, memory, owner, &view));
  result->edem = view;
  *column_map = data.columns ? mapping.relinquish() : cudaq_gf2_owned_buffer{};
  return CUDAQ_EDEM_OK;
}
