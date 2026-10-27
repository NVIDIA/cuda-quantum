/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */
#pragma GCC visibility push(default)
#include "cudaq/edem/composition.h"
#pragma GCC visibility pop
#include "allocation.h"
#include "storage.h"
#include "cudaq/edem/format.h"

using namespace cudaq_gf2_private;

#define EDEM_TRY(expression)                                                   \
  do {                                                                         \
    auto status_ = static_cast<cudaq_edem_status>(expression);                 \
    if (status_ != CUDAQ_EDEM_OK)                                              \
      return status_;                                                          \
  } while (false)

namespace {
struct occurrence_info {
  cudaq_edem_shape shape{};
  uint64_t inputs = 0, outputs = 0, faults = 0, free_outputs = 0;
  uint64_t groups = 0, transfer_begin = 0, transfer_count = 0;
};

struct boundary_transfer {
  uint64_t input_offset = 0, output_offset = 0;
  uint32_t width = 0;
};

struct input_wire {
  cudaq_edem_endpoint endpoint{};
  uint32_t occurrence = 0, width = 0;
  uint64_t local_offset = 0;
};

struct output_wire {
  cudaq_edem_endpoint endpoint{};
  uint32_t occurrence = 0, width = 0;
  uint64_t local_offset = 0, response_offset = 0;
};

cudaq_edem_status location(const cudaq_edem_view &value, bool output,
                           cudaq_edem_endpoint endpoint, uint32_t &width,
                           uint64_t &offset) {
  cudaq_gf2_wire wire;
  EDEM_TRY(cudaq_edem_resolve_endpoint(&value, output, endpoint, &wire));
  EDEM_TRY(cudaq_gf2_map_get_wire(&value.map, output, wire, &width, &offset));
  return CUDAQ_EDEM_OK;
}

cudaq_edem_status describe(const cudaq_edem_view &value,
                           occurrence_info &info) {
  EDEM_TRY(cudaq_edem_get_shape(&value, &info.shape));
  EDEM_TRY(cudaq_gf2_map_vector_sizes(&value.map, &info.inputs, &info.outputs));
  for (uint32_t side = 0; side < 2; ++side) {
    uint32_t count = side ? info.shape.detector_ports_count +
                                info.shape.observable_ports_count +
                                info.shape.success_ports_count +
                                info.shape.output_code_blocks_count
                          : info.shape.fault_ports_count +
                                info.shape.input_code_blocks_count;
    for (uint32_t i = 0; i < count; ++i) {
      cudaq_gf2_bundle bundle;
      EDEM_TRY(cudaq_gf2_map_get_bundle(&value.map, side, i, &bundle));
      if (bundle.type < info.shape.code_presentations)
        continue;
      if (side) {
        info.free_outputs += bundle.widths[0];
      } else {
        uint32_t width, groups;
        EDEM_TRY(cudaq_edem_get_fault(&value, i, &width, &groups));
        info.faults += width;
        info.groups += groups;
      }
    }
  }
  return CUDAQ_EDEM_OK;
}

bool add(uint64_t &sum, uint64_t value) {
  if (value > SIZE_MAX - sum)
    return false;
  sum += value;
  return true;
}

template <class T>
bool space(uint64_t &bytes, uint64_t count) {
  uint64_t ignored;
  return reserve(bytes, count, sizeof(T), ignored);
}

bool same_code(const cudaq_edem_code_presentation &a,
               const cudaq_edem_code_presentation &b) {
  return a.syndrome_count == b.syndrome_count &&
         a.virtual_outcome_count == b.virtual_outcome_count &&
         a.logical_qubit_count == b.logical_qubit_count &&
         name_compare(a.name, a.name_size, b.name, b.name_size) == 0;
}

cudaq_edem_status boundary_code(const cudaq_edem_view &value, bool output,
                                uint32_t index,
                                cudaq_edem_code_presentation &code) {
  cudaq_gf2_wire wire;
  EDEM_TRY(cudaq_edem_resolve_endpoint(&value, output,
                                       {CUDAQ_EDEM_VIRTUAL, index}, &wire));
  cudaq_gf2_bundle bundle;
  EDEM_TRY(cudaq_gf2_map_get_bundle(&value.map, output, wire.bundle, &bundle));
  return cudaq_edem_get_code(&value, bundle.type, &code);
}

cudaq_edem_status evaluate_column(const cudaq_edem_view *values,
                                  std::span<const occurrence_info> info,
                                  const input_wire &source, uint32_t column,
                                  std::span<const output_wire> outputs,
                                  std::span<const boundary_transfer> transfers,
                                  std::span<uint8_t> input,
                                  std::span<uint8_t> output,
                                  std::span<uint8_t> response) {
  std::fill(response.begin(), response.end(), uint8_t{0});
  for (uint64_t t = source.occurrence; t < info.size(); ++t) {
    const auto &current = info[t];
    std::fill_n(input.begin(), current.inputs, uint8_t{0});
    if (t > source.occurrence) {
      for (const auto &transfer :
           transfers.subspan(current.transfer_begin, current.transfer_count))
        if (transfer.width)
          std::memcpy(input.data() + transfer.input_offset,
                      output.data() + transfer.output_offset, transfer.width);
    }
    if (t == source.occurrence)
      input[source.local_offset + column] = 1;
    EDEM_TRY(cudaq_gf2_map_apply(&values[t].map, input.data(), current.inputs,
                                 output.data(), current.outputs));
    for (const auto &wire : outputs)
      if (wire.occurrence == t && wire.width)
        std::memcpy(response.data() + wire.response_offset,
                    output.data() + wire.local_offset, wire.width);
  }
  return CUDAQ_EDEM_OK;
}
} // namespace

cudaq_edem_status cudaq_edem_get_shape(const cudaq_edem_view *value,
                                       cudaq_edem_shape *result) {
  if (!value || !result)
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  cudaq_gf2_map_shape map;
  cudaq_gf2_section codes;
  EDEM_TRY(cudaq_gf2_map_get_shape(&value->map, &map));
  EDEM_TRY(cudaq_gf2_map_get_section(&value->map,
                                     CUDAQ_EDEM_SECTION_PRESENTATIONS, &codes));
  cudaq_edem_shape shape{};
  shape.code_presentations = static_cast<uint32_t>(codes.count);
  shape.block_count = map.block_count;
  for (uint32_t side = 0; side < 2; ++side) {
    uint32_t count = side ? map.output_bundle_count : map.input_bundle_count;
    for (uint32_t i = 0; i < count; ++i) {
      cudaq_gf2_bundle bundle;
      EDEM_TRY(cudaq_gf2_map_get_bundle(&value->map, side, i, &bundle));
      if (bundle.type < codes.count) {
        if (side)
          ++shape.output_code_blocks_count;
        else
          ++shape.input_code_blocks_count;
      } else {
        switch (bundle.type - codes.count) {
        case 0:
          ++shape.detector_ports_count;
          break;
        case 1:
          ++shape.fault_ports_count;
          break;
        case 2:
          ++shape.observable_ports_count;
          break;
        case 3:
          ++shape.success_ports_count;
          break;
        }
      }
    }
  }
  *result = shape;
  return CUDAQ_EDEM_OK;
}

cudaq_edem_status cudaq_edem_is_dem(const cudaq_edem_view *value,
                                    uint32_t *result) {
  if (!result)
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  cudaq_edem_shape shape;
  EDEM_TRY(cudaq_edem_get_shape(value, &shape));
  *result = !shape.input_code_blocks_count && !shape.output_code_blocks_count;
  return CUDAQ_EDEM_OK;
}

cudaq_edem_status cudaq_edem_project_dem(const cudaq_edem_view *value,
                                         const cudaq_gf2_memory *memory,
                                         cudaq_gf2_owned_buffer *owner,
                                         cudaq_edem_dem_view *result) {
  if (!owner || !result || !empty_owner(*owner) ||
      !valid_allocator(scratch_allocator(memory)) ||
      !valid_allocator(result_allocator(memory)))
    return CUDAQ_EDEM_INVALID_ARGUMENT;
  uint32_t closed;
  EDEM_TRY(cudaq_edem_is_dem(value, &closed));
  if (!closed)
    return CUDAQ_EDEM_INVALID_CONTENT;
  allocation storage(result_allocator(memory));
  EDEM_TRY(storage.reserve(value->map.size, CUDAQ_EDEM_PACKED_ALIGNMENT));
  std::memcpy(storage.data(), value->map.data, value->map.size);
  cudaq_edem_dem_view view{{{storage.data(), value->map.size}}};
  *owner = storage.relinquish();
  *result = view;
  return CUDAQ_EDEM_OK;
}

cudaq_edem_status cudaq_edem_compose_chain(const cudaq_edem_view *values,
                                           uint32_t count,
                                           const cudaq_gf2_memory *memory,
                                           cudaq_gf2_owned_buffer *owner,
                                           cudaq_edem_view *result) {
  if (!count || !array_ok(values, count, sizeof(*values)) || !owner ||
      !result || !empty_owner(*owner) ||
      !valid_allocator(scratch_allocator(memory)) ||
      !valid_allocator(result_allocator(memory)))
    return CUDAQ_EDEM_INVALID_ARGUMENT;

  uint64_t code_capacity = 0, faults = 0, groups = 0, detectors = 0,
           observables = 0, successes = 0, columns = 0, response_size = 0,
           max_input = 0, max_output = 0, transfer_count = 0;
  occurrence_info first{}, last{};
  for (uint32_t t = 0; t < count; ++t) {
    occurrence_info current;
    EDEM_TRY(describe(values[t], current));
    if (!t)
      first = current;
    last = current;
    if (!add(code_capacity, current.shape.code_presentations) ||
        !add(faults, current.shape.fault_ports_count) ||
        !add(detectors, current.shape.detector_ports_count) ||
        !add(observables, current.shape.observable_ports_count) ||
        !add(successes, current.shape.success_ports_count) ||
        !add(columns, current.faults) ||
        !add(response_size, current.free_outputs) ||
        !add(groups, current.groups) ||
        (t &&
         !add(transfer_count, 3ull * current.shape.input_code_blocks_count)))
      return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
    max_input = std::max(max_input, current.inputs);
    max_output = std::max(max_output, current.outputs);
  }
  if (!add(columns, first.inputs - first.faults) ||
      !add(response_size, last.outputs - last.free_outputs) ||
      code_capacity > UINT32_MAX - 4 || faults > UINT32_MAX ||
      detectors > UINT32_MAX || observables > UINT32_MAX ||
      successes > UINT32_MAX ||
      detectors + observables + successes > UINT32_MAX)
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  uint64_t input_count = faults + 3ull * first.shape.input_code_blocks_count;
  uint64_t output_count = detectors + observables + successes +
                          3ull * last.shape.output_code_blocks_count;
  if (input_count > UINT32_MAX || output_count > UINT32_MAX ||
      (input_count && output_count > UINT32_MAX / input_count) ||
      columns > SIZE_MAX - input_count ||
      (columns + input_count &&
       output_count > SIZE_MAX / (columns + input_count)))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  // This serial reference path reserves one descriptor per endpoint pair and
  // evaluates each basis column twice: once to count nonzeros, then to fill
  // them. Storage therefore includes Cartesian zero-block bookkeeping.
  uint64_t block_count = input_count * output_count;
  uint64_t offset_count = output_count * (columns + input_count);
  uint64_t bytes = 0;
  if (!space<occurrence_info>(bytes, count) ||
      !space<boundary_transfer>(bytes, transfer_count) ||
      !space<cudaq_edem_code_presentation>(bytes, code_capacity) ||
      !space<uint32_t>(bytes, first.shape.input_code_blocks_count) ||
      !space<uint32_t>(bytes, last.shape.output_code_blocks_count) ||
      !space<cudaq_edem_fault_distribution>(bytes, faults) ||
      !space<cudaq_edem_fault_group>(bytes, groups) ||
      !space<uint32_t>(bytes, detectors) ||
      !space<uint32_t>(bytes, observables) ||
      !space<uint32_t>(bytes, successes) ||
      !space<input_wire>(bytes, input_count) ||
      !space<output_wire>(bytes, output_count) ||
      !space<cudaq_edem_block>(bytes, block_count) ||
      !space<uint64_t>(bytes, offset_count) ||
      !space<uint8_t>(bytes, max_input) || !space<uint8_t>(bytes, max_output) ||
      !space<uint8_t>(bytes, response_size))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  allocation workspace(scratch_allocator(memory));
  EDEM_TRY(workspace.reserve(bytes, CUDAQ_EDEM_PACKED_ALIGNMENT));
  arena a{static_cast<std::byte *>(workspace.data()), bytes};
  auto info = a.take<occurrence_info>(count);
  auto transfers = a.take<boundary_transfer>(transfer_count);
  auto codes = a.take<cudaq_edem_code_presentation>(code_capacity);
  auto input_codes = a.take<uint32_t>(first.shape.input_code_blocks_count);
  auto output_codes = a.take<uint32_t>(last.shape.output_code_blocks_count);
  auto distributions = a.take<cudaq_edem_fault_distribution>(faults);
  auto laws = a.take<cudaq_edem_fault_group>(groups);
  auto detector_ports_sizes = a.take<uint32_t>(detectors);
  auto observable_ports_sizes = a.take<uint32_t>(observables);
  auto success_ports_sizes = a.take<uint32_t>(successes);
  auto inputs = a.take<input_wire>(input_count);
  auto outputs = a.take<output_wire>(output_count);
  auto blocks = a.take<cudaq_edem_block>(block_count);
  auto offsets = a.take<uint64_t>(offset_count);
  auto input = a.take<uint8_t>(max_input);
  auto output = a.take<uint8_t>(max_output);
  auto response = a.take<uint8_t>(response_size);

  uint32_t code_count = 0;
  uint64_t fault_index = 0, group_index = 0, transfer_index = 0;
  for (uint32_t t = 0; t < count; ++t) {
    EDEM_TRY(describe(values[t], info[t]));
    if (t) {
      if (info[t - 1].shape.output_code_blocks_count !=
          info[t].shape.input_code_blocks_count)
        return CUDAQ_EDEM_INVALID_CONTENT;
      info[t].transfer_begin = transfer_index;
      for (uint32_t p = 0; p < info[t].shape.input_code_blocks_count; ++p) {
        cudaq_edem_code_presentation left, right;
        EDEM_TRY(boundary_code(values[t - 1], true, p, left));
        EDEM_TRY(boundary_code(values[t], false, p, right));
        if (!same_code(left, right))
          return CUDAQ_EDEM_INVALID_CONTENT;
        for (uint32_t kind = CUDAQ_EDEM_VIRTUAL; kind <= CUDAQ_EDEM_LOGICAL;
             ++kind) {
          auto &transfer = transfers[transfer_index++];
          uint32_t output_width;
          EDEM_TRY(location(values[t - 1], true, {kind, p}, output_width,
                            transfer.output_offset));
          EDEM_TRY(location(values[t], false, {kind, p}, transfer.width,
                            transfer.input_offset));
          if (output_width != transfer.width)
            return CUDAQ_EDEM_INVALID_CONTENT;
        }
      }
      info[t].transfer_count = transfer_index - info[t].transfer_begin;
    }
    for (uint32_t p = 0; p < info[t].shape.code_presentations; ++p) {
      cudaq_edem_code_presentation code;
      EDEM_TRY(cudaq_edem_get_code(&values[t], p, &code));
      uint32_t i = 0;
      while (i < code_count && name_compare(codes[i].name, codes[i].name_size,
                                            code.name, code.name_size))
        ++i;
      if (i == code_count)
        codes[code_count++] = code;
      else if (!same_code(codes[i], code))
        return CUDAQ_EDEM_INVALID_CONTENT;
    }
    const uint32_t input_bundles =
        info[t].shape.fault_ports_count + info[t].shape.input_code_blocks_count;
    for (uint32_t p = 0; p < input_bundles; ++p) {
      cudaq_gf2_bundle bundle;
      EDEM_TRY(cudaq_gf2_map_get_bundle(&values[t].map, false, p, &bundle));
      if (bundle.type < info[t].shape.code_presentations)
        continue;
      uint32_t width, law_count;
      EDEM_TRY(cudaq_edem_get_fault(&values[t], p, &width, &law_count));
      distributions[fault_index] = {
          width, law_count, law_count ? laws.data() + group_index : nullptr};
      for (uint32_t g = 0; g < law_count; ++g)
        EDEM_TRY(cudaq_edem_get_group(&values[t], p, g, &laws[group_index++]));
      auto &wire = inputs[fault_index];
      wire.endpoint = {CUDAQ_EDEM_FAULT, static_cast<uint32_t>(fault_index)};
      wire.occurrence = t;
      EDEM_TRY(cudaq_gf2_map_get_wire(&values[t].map, false, {p, 0},
                                      &wire.width, &wire.local_offset));
      ++fault_index;
    }
  }
  for (uint32_t side = 0; side < 2; ++side) {
    auto destination = side ? output_codes : input_codes;
    uint32_t t = side ? count - 1 : 0;
    for (uint32_t p = 0; p < destination.size(); ++p) {
      cudaq_edem_code_presentation code;
      EDEM_TRY(boundary_code(values[t], side, p, code));
      for (uint32_t i = 0; i < code_count; ++i)
        if (same_code(codes[i], code)) {
          destination[p] = i;
          break;
        }
    }
  }
  uint64_t in_index = faults;
  for (uint32_t p = 0; p < first.shape.input_code_blocks_count; ++p)
    for (uint32_t kind = CUDAQ_EDEM_VIRTUAL; kind <= CUDAQ_EDEM_LOGICAL;
         ++kind) {
      auto &wire = inputs[in_index++];
      wire.endpoint = {kind, p};
      EDEM_TRY(location(values[0], false, wire.endpoint, wire.width,
                        wire.local_offset));
    }
  uint64_t out_index = 0, response_offset = 0;
  for (uint32_t kind = CUDAQ_EDEM_DETECTOR; kind <= CUDAQ_EDEM_SUCCESS;
       ++kind) {
    auto sizes = kind == CUDAQ_EDEM_DETECTOR     ? detector_ports_sizes
                 : kind == CUDAQ_EDEM_OBSERVABLE ? observable_ports_sizes
                                                 : success_ports_sizes;
    uint32_t partition = 0;
    for (uint32_t t = 0; t < count; ++t) {
      uint32_t n =
          kind == CUDAQ_EDEM_DETECTOR     ? info[t].shape.detector_ports_count
          : kind == CUDAQ_EDEM_OBSERVABLE ? info[t].shape.observable_ports_count
                                          : info[t].shape.success_ports_count;
      for (uint32_t p = 0; p < n; ++p) {
        auto &wire = outputs[out_index++];
        wire.endpoint = {kind, partition};
        wire.occurrence = t;
        wire.response_offset = response_offset;
        EDEM_TRY(location(values[t], true, {kind, p}, wire.width,
                          wire.local_offset));
        sizes[partition++] = wire.width;
        response_offset += wire.width;
      }
    }
  }
  for (uint32_t p = 0; p < last.shape.output_code_blocks_count; ++p)
    for (uint32_t kind = CUDAQ_EDEM_VIRTUAL; kind <= CUDAQ_EDEM_LOGICAL;
         ++kind) {
      auto &wire = outputs[out_index++];
      wire.endpoint = {kind, p};
      wire.occurrence = count - 1;
      wire.response_offset = response_offset;
      EDEM_TRY(location(values[count - 1], true, wire.endpoint, wire.width,
                        wire.local_offset));
      response_offset += wire.width;
    }

  uint64_t offset_index = 0;
  for (uint64_t o = 0; o < output_count; ++o)
    for (uint64_t i = 0; i < input_count; ++i) {
      blocks[o * input_count + i] = {outputs[o].endpoint,
                                     inputs[i].endpoint,
                                     {outputs[o].width, inputs[i].width, 0,
                                      offsets.data() + offset_index, nullptr}};
      offset_index += uint64_t{inputs[i].width} + 1;
    }
  uint64_t nonzeros = 0;
  for (uint64_t i = 0; i < input_count; ++i)
    for (uint32_t column = 0; column < inputs[i].width; ++column) {
      EDEM_TRY(evaluate_column(values, info, inputs[i], column, outputs,
                               transfers, input, output, response));
      for (uint64_t o = 0; o < output_count; ++o) {
        auto &matrix = blocks[o * input_count + i].matrix;
        uint64_t hits = 0;
        for (uint32_t row = 0; row < outputs[o].width; ++row)
          hits += response[outputs[o].response_offset + row] != 0;
        if (!add(nonzeros, hits))
          return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
        matrix.nonzeros += hits;
        const_cast<uint64_t *>(matrix.column_offsets)[column + 1] =
            matrix.nonzeros;
      }
    }
  uint64_t row_bytes = 0;
  if (!space<uint32_t>(row_bytes, nonzeros))
    return CUDAQ_EDEM_INSUFFICIENT_CAPACITY;
  allocation row_storage(scratch_allocator(memory));
  EDEM_TRY(row_storage.reserve(row_bytes, CUDAQ_EDEM_PACKED_ALIGNMENT));
  arena rows_arena{static_cast<std::byte *>(row_storage.data()), row_bytes};
  auto rows = rows_arena.take<uint32_t>(nonzeros);
  uint64_t row_index = 0;
  for (auto &block : blocks) {
    block.matrix.row_indices =
        block.matrix.nonzeros ? rows.data() + row_index : nullptr;
    row_index += block.matrix.nonzeros;
  }
  for (uint64_t i = 0; i < input_count; ++i)
    for (uint32_t column = 0; column < inputs[i].width; ++column) {
      EDEM_TRY(evaluate_column(values, info, inputs[i], column, outputs,
                               transfers, input, output, response));
      for (uint64_t o = 0; o < output_count; ++o) {
        auto &matrix = blocks[o * input_count + i].matrix;
        auto *indices = const_cast<uint32_t *>(matrix.row_indices);
        uint64_t position = matrix.column_offsets[column];
        for (uint32_t row = 0; row < outputs[o].width; ++row)
          if (response[outputs[o].response_offset + row])
            indices[position++] = row;
      }
    }
  cudaq_edem_description description{
      {code_count, first.shape.input_code_blocks_count,
       last.shape.output_code_blocks_count, static_cast<uint32_t>(faults),
       static_cast<uint32_t>(detectors), static_cast<uint32_t>(observables),
       static_cast<uint32_t>(successes), block_count},
      codes.data(),
      input_codes.data(),
      output_codes.data(),
      distributions.data(),
      detector_ports_sizes.data(),
      observable_ports_sizes.data(),
      success_ports_sizes.data(),
      blocks.data()};
  return cudaq_edem_construct(&description, memory, owner, result);
}
