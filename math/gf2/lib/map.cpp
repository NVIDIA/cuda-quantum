/*
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 */

#pragma GCC visibility push(default)
#include "cudaq/gf2/map.h"
#pragma GCC visibility pop
#include "allocation.h"
#include "csc_xor.h"
#include "storage.h"

using namespace cudaq_gf2_private;
namespace {
static_assert(sizeof(cudaq_gf2_packed_header) == 64);
static_assert(sizeof(cudaq_gf2_packed_section) == 32);
static_assert(sizeof(cudaq_gf2_packed_type) == 24);
static_assert(sizeof(cudaq_gf2_packed_bundle) == 24);
static_assert(sizeof(cudaq_gf2_packed_matrix_block) == 24);
static_assert(sizeof(cudaq_gf2_packed_matrix) == 32);
constexpr uint8_t magic[8] = {'C', 'Q', 'G', 'F', '2', 0, 0, 0};
struct source {
  cudaq_gf2_map_description d{};
  const uint8_t *data = nullptr;
  const cudaq_gf2_packed_section *directory = nullptr;
  const uint8_t *payload(unsigned i) const {
    return data + directory[i].offset;
  }
  cudaq_gf2_bundle_type type(uint32_t i) const {
    if (!data)
      return d.types[i];
    const auto &t =
        reinterpret_cast<const cudaq_gf2_packed_type *>(payload(0))[i];
    return {reinterpret_cast<const char *>(payload(0) + t.name), t.name_size,
            t.role, t.wire_count};
  }
  cudaq_gf2_bundle bundle(uint32_t side, uint32_t i) const {
    if (!data)
      return (side ? d.outputs : d.inputs)[i];
    auto *p = payload(side + 1);
    const auto &b = reinterpret_cast<const cudaq_gf2_packed_bundle *>(p)[i];
    return {b.type, b.reserved,
            reinterpret_cast<const uint32_t *>(p + b.widths)};
  }
  const uint64_t *offsets(uint32_t side, uint32_t i) const {
    auto *p = payload(side + 1);
    return reinterpret_cast<const uint64_t *>(
        p + reinterpret_cast<const cudaq_gf2_packed_bundle *>(p)[i].offsets);
  }
  cudaq_gf2_csc_view matrix(uint32_t i) const {
    auto *p = payload(4);
    const auto &m = reinterpret_cast<const cudaq_gf2_packed_matrix *>(p)[i];
    return {m.rows, m.columns, m.nonzeros,
            reinterpret_cast<const uint64_t *>(p + m.column_offsets),
            reinterpret_cast<const uint32_t *>(p + m.row_indices)};
  }
  cudaq_gf2_map_matrix_block block(uint32_t i) const {
    if (!data)
      return d.blocks[i];
    const auto &b =
        reinterpret_cast<const cudaq_gf2_packed_matrix_block *>(payload(3))[i];
    return {{b.output_bundle, b.output_wire},
            {b.input_bundle, b.input_wire},
            matrix(b.matrix)};
  }
  cudaq_gf2_section extension(uint32_t i) const {
    if (!data)
      return d.sections[i];
    const auto &e = directory[5 + i];
    return {e.id, e.flags, e.count, e.bytes, data + e.offset};
  }
};
struct matrix_record {
  uint64_t hash;
  uint32_t index, ordinal;
};
struct workspace {
  std::span<uint32_t> types, type_ids, blocks, matrix_ids, owners, extensions;
  std::span<matrix_record> matrices;
  uint32_t unique = 0;
};
struct workspace_requirements {
  uint64_t bytes;
  uint32_t alignment, reserved;
};
bool block_less(cudaq_gf2_map_matrix_block a, cudaq_gf2_map_matrix_block b) {
  if (a.output.bundle != b.output.bundle)
    return a.output.bundle < b.output.bundle;
  if (a.output.wire != b.output.wire)
    return a.output.wire < b.output.wire;
  if (a.input.bundle != b.input.bundle)
    return a.input.bundle < b.input.bundle;
  return a.input.wire < b.input.wire;
}
int type_compare(cudaq_gf2_bundle_type a, cudaq_gf2_bundle_type b) {
  return name_compare(a.name, a.name_size, b.name, b.name_size);
}
cudaq_gf2_status requirements(const source &s, workspace_requirements &req) {
  const auto &d = s.d;
  if (d.block_count > UINT32_MAX || d.section_count > UINT32_MAX - 5)
    return CUDAQ_GF2_INVALID_ARGUMENT;
  uint64_t bytes = 0, ignored;
  if (!reserve(bytes, d.type_count, 4, ignored) ||
      !reserve(bytes, d.type_count, 4, ignored) ||
      !reserve(bytes, d.block_count, 4, ignored) ||
      !reserve(bytes, d.block_count, 4, ignored) ||
      !reserve(bytes, d.block_count, 4, ignored) ||
      !reserve(bytes, d.block_count, sizeof(matrix_record), ignored) ||
      !reserve(bytes, d.section_count, 4, ignored))
    return CUDAQ_GF2_INSUFFICIENT_CAPACITY;
  req = {bytes, 8, 0};
  return CUDAQ_GF2_OK;
}
cudaq_gf2_status make_workspace(const source &s, void *buffer, uint64_t bytes,
                                workspace &w) {
  workspace_requirements req;
  auto status = requirements(s, req);
  if (status != CUDAQ_GF2_OK)
    return status;
  if (bytes < req.bytes || (req.bytes && !buffer))
    return CUDAQ_GF2_INSUFFICIENT_CAPACITY;
  if ((bytes && !buffer) || (buffer && reinterpret_cast<uintptr_t>(buffer) % 8))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  arena a{static_cast<std::byte *>(buffer), bytes};
  w.types = a.take<uint32_t>(s.d.type_count);
  w.type_ids = a.take<uint32_t>(s.d.type_count);
  w.blocks = a.take<uint32_t>(s.d.block_count);
  w.matrix_ids = a.take<uint32_t>(s.d.block_count);
  w.owners = a.take<uint32_t>(s.d.block_count);
  w.matrices = a.take<matrix_record>(s.d.block_count);
  w.extensions = a.take<uint32_t>(s.d.section_count);
  return CUDAQ_GF2_OK;
}
bool wire_width(const source &s, uint32_t side, cudaq_gf2_wire wire,
                uint32_t &width) {
  if (wire.bundle >= (side ? s.d.output_bundle_count : s.d.input_bundle_count))
    return false;
  auto b = s.bundle(side, wire.bundle);
  if (b.type >= s.d.type_count || wire.wire >= s.type(b.type).wire_count)
    return false;
  width = b.widths[wire.wire];
  return true;
}
cudaq_gf2_status validate(const source &s, workspace &w) {
  if (!s.data &&
      (!array_ok(s.d.types, s.d.type_count, sizeof(*s.d.types)) ||
       !array_ok(s.d.inputs, s.d.input_bundle_count, sizeof(*s.d.inputs)) ||
       !array_ok(s.d.outputs, s.d.output_bundle_count, sizeof(*s.d.outputs)) ||
       !array_ok(s.d.blocks, s.d.block_count, sizeof(*s.d.blocks)) ||
       !array_ok(s.d.sections, s.d.section_count, sizeof(*s.d.sections))))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  for (uint32_t i = 0; i < s.d.type_count; ++i) {
    auto t = s.type(i);
    if (!array_ok(t.name, t.name_size, 1))
      return CUDAQ_GF2_INVALID_ARGUMENT;
    if (!utf8_name(t.name, t.name_size) || t.role > CUDAQ_GF2_INTERFACE)
      return CUDAQ_GF2_INVALID_CONTENT;
    if (s.data && i && type_compare(s.type(i - 1), t) >= 0)
      return CUDAQ_GF2_INVALID_CONTENT;
  }
  sort_indices(w.types, [&](uint32_t a, uint32_t b) {
    return type_compare(s.type(a), s.type(b)) < 0;
  });
  for (uint32_t i = 0; i < s.d.type_count; ++i) {
    if (i && type_compare(s.type(w.types[i - 1]), s.type(w.types[i])) == 0)
      return CUDAQ_GF2_INVALID_CONTENT;
    w.type_ids[w.types[i]] = i;
  }
  for (uint32_t side = 0; side < 2; ++side) {
    uint64_t offset = 0;
    uint32_t count = side ? s.d.output_bundle_count : s.d.input_bundle_count;
    for (uint32_t i = 0; i < count; ++i) {
      auto b = s.bundle(side, i);
      if (b.type >= s.d.type_count || b.reserved)
        return CUDAQ_GF2_INVALID_CONTENT;
      uint32_t n = s.type(b.type).wire_count;
      if (!array_ok(b.widths, n, 4))
        return CUDAQ_GF2_INVALID_ARGUMENT;
      for (uint64_t j = 0; j < n; ++j) {
        if (s.data && s.offsets(side, i)[j] != offset)
          return CUDAQ_GF2_INVALID_CONTENT;
        if (b.widths[j] > SIZE_MAX - offset)
          return CUDAQ_GF2_INVALID_CONTENT;
        offset += b.widths[j];
      }
      if (s.data && s.offsets(side, i)[n] != offset)
        return CUDAQ_GF2_INVALID_CONTENT;
    }
  }
  for (uint32_t i = 0; i < s.d.block_count; ++i) {
    auto b = s.block(i);
    uint32_t rows, cols;
    if (!wire_width(s, 1, b.output, rows) || !wire_width(s, 0, b.input, cols) ||
        b.matrix.rows != rows || b.matrix.columns != cols ||
        cudaq_gf2_csc_validate(&b.matrix) != CUDAQ_GF2_OK ||
        (s.data &&
         (!b.matrix.nonzeros || (i && !block_less(s.block(i - 1), b)))))
      return CUDAQ_GF2_INVALID_CONTENT;
  }
  sort_indices(w.blocks, [&](uint32_t a, uint32_t b) {
    return block_less(s.block(a), s.block(b));
  });
  for (size_t i = 1; i < w.blocks.size(); ++i)
    if (!block_less(s.block(w.blocks[i - 1]), s.block(w.blocks[i])))
      return CUDAQ_GF2_INVALID_CONTENT;
  for (uint32_t i = 0; i < s.d.section_count; ++i) {
    auto e = s.extension(i);
    if (!array_ok(e.data, e.bytes, 1))
      return CUDAQ_GF2_INVALID_ARGUMENT;
    if (e.id < CUDAQ_GF2_EXTENSION_SECTION_MIN ||
        e.flags != (e.id < CUDAQ_GF2_OPTIONAL_SECTION_MIN
                        ? CUDAQ_GF2_SECTION_REQUIRED
                        : CUDAQ_GF2_SECTION_OPTIONAL))
      return CUDAQ_GF2_INVALID_CONTENT;
  }
  sort_indices(w.extensions, [&](uint32_t a, uint32_t b) {
    return s.extension(a).id < s.extension(b).id;
  });
  for (size_t i = 1; i < w.extensions.size(); ++i)
    if (s.extension(w.extensions[i - 1]).id == s.extension(w.extensions[i]).id)
      return CUDAQ_GF2_INVALID_CONTENT;
  return CUDAQ_GF2_OK;
}
uint64_t matrix_hash(cudaq_gf2_csc_view m) {
  uint64_t hash = 14695981039346656037ull;
  auto add = [&](const void *data, uint64_t bytes) {
    auto *p = static_cast<const uint8_t *>(data);
    for (uint64_t i = 0; i < bytes; ++i)
      hash = (hash ^ p[i]) * 1099511628211ull;
  };
  add(&m.rows, 4);
  add(&m.columns, 4);
  add(m.column_offsets, (uint64_t{m.columns} + 1) * 8);
  add(m.row_indices, m.nonzeros * 4);
  return hash;
}
int matrix_compare(cudaq_gf2_csc_view a, cudaq_gf2_csc_view b) {
  if (a.rows != b.rows)
    return a.rows < b.rows ? -1 : 1;
  if (a.columns != b.columns)
    return a.columns < b.columns ? -1 : 1;
  if (a.nonzeros != b.nonzeros)
    return a.nonzeros < b.nonzeros ? -1 : 1;
  int cmp =
      memcmp(a.column_offsets, b.column_offsets, (uint64_t{a.columns} + 1) * 8);
  return cmp || !a.nonzeros
             ? cmp
             : memcmp(a.row_indices, b.row_indices, a.nonzeros * 4);
}
void prepare_matrices(const source &s, workspace &w) {
  uint64_t count = 0;
  for (uint32_t i = 0; i < s.d.block_count; ++i)
    if (s.block(i).matrix.nonzeros)
      w.matrices[count++] = {matrix_hash(s.block(i).matrix), i, 0};
  if (count > 1)
    std::sort(w.matrices.begin(), w.matrices.begin() + count,
              [&](auto a, auto b) {
                return a.hash < b.hash ||
                       (a.hash == b.hash &&
                        matrix_compare(s.block(a.index).matrix,
                                       s.block(b.index).matrix) < 0);
              });
  for (uint64_t begin = 0; begin < count;) {
    uint64_t end = begin + 1;
    uint32_t owner = w.matrices[begin].index;
    while (end < count && w.matrices[begin].hash == w.matrices[end].hash &&
           matrix_compare(s.block(w.matrices[begin].index).matrix,
                          s.block(w.matrices[end].index).matrix) == 0) {
      if (block_less(s.block(w.matrices[end].index), s.block(owner)))
        owner = w.matrices[end].index;
      ++end;
    }
    for (uint64_t i = begin; i < end; ++i)
      w.matrix_ids[w.matrices[i].index] = owner;
    begin = end;
  }
  for (uint32_t i : w.blocks)
    if (s.block(i).matrix.nonzeros && w.matrix_ids[i] == i) {
      w.owners[w.unique] = i;
      w.matrices[i].ordinal = w.unique++;
    }
  for (uint32_t i = 0; i < s.d.block_count; ++i)
    if (s.block(i).matrix.nonzeros)
      w.matrix_ids[i] = w.matrices[w.matrix_ids[i]].ordinal;
}
struct layout {
  cudaq_gf2_packed_header h{};
  cudaq_gf2_packed_section sections[5]{};
};
bool measure(const source &s, const workspace &w, layout &out) {
  auto &h = out.h;
  memcpy(h.magic, magic, 8);
  h.version = CUDAQ_GF2_FORMAT_VERSION;
  h.kind = CUDAQ_GF2_KIND_LINEAR_MAP;
  h.alignment = CUDAQ_GF2_PACKED_ALIGNMENT;
  h.index_bits = 32;
  h.directory_count = 5 + uint64_t{s.d.section_count};
  h.directory_bytes = h.directory_count * 32;
  uint64_t block_count = 0;
  for (uint32_t i = 0; i < s.d.block_count; ++i)
    block_count += s.block(i).matrix.nonzeros != 0;
  uint64_t counts[5] = {s.d.type_count, s.d.input_bundle_count,
                        s.d.output_bundle_count, block_count, w.unique};
  uint64_t sizes[5] = {24, 24, 24, 24, 32};
  uint64_t cursor = sizeof(h), ignored;
  for (uint32_t part = 0; part < 5; ++part) {
    uint64_t bytes = 0;
    if (!reserve(bytes, counts[part], sizes[part], ignored))
      return false;
    if (part == 0)
      for (uint32_t i = 0; i < s.d.type_count; ++i) {
        if (!reserve(bytes, s.type(w.types[i]).name_size, 1, ignored, 1))
          return false;
      }
    if (part == 1 || part == 2)
      for (uint32_t i = 0; i < counts[part]; ++i) {
        auto b = s.bundle(part - 1, i);
        uint32_t n = s.type(b.type).wire_count;
        if (!reserve(bytes, n, 4, ignored, 4) ||
            !reserve(bytes, uint64_t{n} + 1, 8, ignored))
          return false;
      }
    if (part == 4)
      for (uint32_t i = 0; i < w.unique; ++i) {
        auto m = s.block(w.owners[i]).matrix;
        if (!reserve(bytes, uint64_t{m.columns} + 1, 8, ignored) ||
            !reserve(bytes, m.nonzeros, 4, ignored, 4))
          return false;
      }
    out.sections[part] = {part + 1, CUDAQ_GF2_SECTION_REQUIRED, counts[part],
                          bytes, 0};
    if (!reserve(cursor, bytes, 1, out.sections[part].offset))
      return false;
  }
  for (uint32_t i : w.extensions)
    if (!reserve(cursor, s.extension(i).bytes, 1, ignored))
      return false;
  if (!reserve(cursor, h.directory_count, 32, h.directory_offset))
    return false;
  h.size = cursor;
  return true;
}
void write(const source &s, const workspace &w, const layout &l,
           uint8_t *data) {
  memset(data, 0, l.h.size);
  memcpy(data, &l.h, sizeof(l.h));
  auto *dir =
      reinterpret_cast<cudaq_gf2_packed_section *>(data + l.h.directory_offset);
  memcpy(dir, l.sections, sizeof(l.sections));
  uint64_t cursor = s.d.type_count * uint64_t{24}, offset;
  auto *types = reinterpret_cast<cudaq_gf2_packed_type *>(data + dir[0].offset);
  for (uint32_t i = 0; i < s.d.type_count; ++i) {
    auto t = s.type(w.types[i]);
    reserve(cursor, t.name_size, 1, offset, 1);
    types[i] = {offset, t.name_size, t.role, t.wire_count};
    memcpy(data + dir[0].offset + offset, t.name, t.name_size);
  }
  for (uint32_t side = 0; side < 2; ++side) {
    auto &e = dir[side + 1];
    auto *p = data + e.offset;
    auto *bundles = reinterpret_cast<cudaq_gf2_packed_bundle *>(p);
    cursor = e.count * 24;
    uint64_t vector_offset = 0;
    for (uint32_t i = 0; i < e.count; ++i) {
      auto b = s.bundle(side, i);
      uint32_t n = s.type(b.type).wire_count;
      uint64_t widths, offsets;
      reserve(cursor, n, 4, widths, 4);
      reserve(cursor, uint64_t{n} + 1, 8, offsets);
      bundles[i] = {w.type_ids[b.type], 0, widths, offsets};
      if (n)
        memcpy(p + widths, b.widths, uint64_t{n} * 4);
      auto *positions = reinterpret_cast<uint64_t *>(p + offsets);
      for (uint64_t j = 0; j < n; ++j) {
        positions[j] = vector_offset;
        vector_offset += b.widths[j];
      }
      positions[n] = vector_offset;
    }
  }
  auto *blocks =
      reinterpret_cast<cudaq_gf2_packed_matrix_block *>(data + dir[3].offset);
  uint32_t written = 0;
  for (uint32_t i : w.blocks) {
    auto b = s.block(i);
    if (!b.matrix.nonzeros)
      continue;
    blocks[written++] = {b.output.bundle, b.output.wire,   b.input.bundle,
                         b.input.wire,    w.matrix_ids[i], 0};
  }
  auto *p = data + dir[4].offset;
  auto *matrices = reinterpret_cast<cudaq_gf2_packed_matrix *>(p);
  cursor = uint64_t{w.unique} * 32;
  for (uint32_t i = 0; i < w.unique; ++i) {
    auto m = s.block(w.owners[i]).matrix;
    uint64_t offsets, indices;
    reserve(cursor, uint64_t{m.columns} + 1, 8, offsets);
    reserve(cursor, m.nonzeros, 4, indices, 4);
    matrices[i] = {m.rows, m.columns, m.nonzeros, offsets, indices};
    memcpy(p + offsets, m.column_offsets, (uint64_t{m.columns} + 1) * 8);
    memcpy(p + indices, m.row_indices, m.nonzeros * 4);
  }
  cursor = sizeof(l.h);
  for (const auto &e : l.sections) {
    uint64_t ignored;
    reserve(cursor, e.bytes, 1, ignored);
  }
  uint32_t i = 5;
  for (uint32_t index : w.extensions) {
    auto e = s.extension(index);
    reserve(cursor, e.bytes, 1, offset);
    dir[i++] = {e.id, e.flags, e.count, e.bytes, offset};
    if (e.bytes)
      memcpy(data + offset, e.data, e.bytes);
  }
}
/* Validate byte ranges before any count-indexed or nested access. */
cudaq_gf2_status parse(const void *buffer, uint64_t bytes,
                       const cudaq_gf2_schema *schema, source &s) {
  if (!buffer || reinterpret_cast<uintptr_t>(buffer) % 8 ||
      bytes < sizeof(cudaq_gf2_packed_header) || bytes > SIZE_MAX)
    return CUDAQ_GF2_INVALID_ARGUMENT;
  if (!little_endian())
    return CUDAQ_GF2_UNSUPPORTED_VERSION;
  if (schema &&
      (schema->reserved ||
       !array_ok(schema->required_sections, schema->required_section_count, 4)))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  auto *data = static_cast<const uint8_t *>(buffer);
  const auto &h = *reinterpret_cast<const cudaq_gf2_packed_header *>(data);
  if (memcmp(h.magic, magic, 8) || h.version != CUDAQ_GF2_FORMAT_VERSION ||
      h.kind != CUDAQ_GF2_KIND_LINEAR_MAP)
    return CUDAQ_GF2_UNSUPPORTED_VERSION;
  if (h.size != bytes || h.alignment != 8 || h.index_bits != 32 || h.reserved ||
      h.directory_count < 5 || h.directory_count > UINT32_MAX ||
      h.directory_offset % 8 || h.directory_offset < sizeof(h) ||
      h.directory_offset > bytes ||
      h.directory_bytes != h.directory_count * 32 ||
      h.directory_bytes != bytes - h.directory_offset)
    return CUDAQ_GF2_INVALID_CONTENT;
  auto *dir = reinterpret_cast<const cudaq_gf2_packed_section *>(
      data + h.directory_offset);
  reader outer{data, h.directory_offset, sizeof(h)};
  for (uint32_t i = 0; i < h.directory_count; ++i) {
    const auto &e = dir[i];
    if ((i && dir[i - 1].id >= e.id) ||
        e.flags != (e.id < CUDAQ_GF2_OPTIONAL_SECTION_MIN
                        ? CUDAQ_GF2_SECTION_REQUIRED
                        : CUDAQ_GF2_SECTION_OPTIONAL) ||
        (i < 5 && e.id != i + 1) || !outer.take(e.offset, e.bytes, 1))
      return CUDAQ_GF2_INVALID_CONTENT;
    if (i >= 5 && e.id < CUDAQ_GF2_EXTENSION_SECTION_MIN)
      return CUDAQ_GF2_UNSUPPORTED_VERSION;
    if (i >= 5 && e.flags == CUDAQ_GF2_SECTION_REQUIRED) {
      bool known = false;
      if (schema)
        for (uint32_t j = 0; j < schema->required_section_count; ++j)
          known |= schema->required_sections[j] == e.id;
      if (!known)
        return CUDAQ_GF2_UNSUPPORTED_VERSION;
    }
  }
  uint64_t expected, aligned = outer.cursor;
  if (!reserve(aligned, 1, 1, expected) || expected != h.directory_offset ||
      !zeros(data, outer.cursor, h.directory_offset))
    return CUDAQ_GF2_INVALID_CONTENT;
  for (uint32_t part = 0; part < 5; ++part)
    if (dir[part].count > UINT32_MAX)
      return CUDAQ_GF2_INVALID_CONTENT;
  if (dir[4].count > dir[3].count)
    return CUDAQ_GF2_INVALID_CONTENT;
  s.data = data;
  s.directory = dir;
  s.d.type_count = static_cast<uint32_t>(dir[0].count);
  s.d.input_bundle_count = static_cast<uint32_t>(dir[1].count);
  s.d.output_bundle_count = static_cast<uint32_t>(dir[2].count);
  s.d.block_count = dir[3].count;
  s.d.section_count = static_cast<uint32_t>(h.directory_count - 5);
  uint64_t widths[5] = {24, 24, 24, 24, 32};
  for (uint32_t part = 0; part < 5; ++part) {
    auto *p = s.payload(part);
    const auto &e = dir[part];
    reader r{p, e.bytes};
    if (!r.take(0, e.count, widths[part]))
      return CUDAQ_GF2_INVALID_CONTENT;
    for (uint32_t i = 0; i < e.count; ++i) {
      if (part == 0) {
        const auto &t = reinterpret_cast<const cudaq_gf2_packed_type *>(p)[i];
        if (!r.take(t.name, t.name_size, 1, 1))
          return CUDAQ_GF2_INVALID_CONTENT;
      } else if (part == 1 || part == 2) {
        const auto &b = reinterpret_cast<const cudaq_gf2_packed_bundle *>(p)[i];
        if (b.type >= s.d.type_count || b.reserved)
          return CUDAQ_GF2_INVALID_CONTENT;
        uint32_t n = s.type(b.type).wire_count;
        if (!r.take(b.widths, n, 4, 4) ||
            !r.take(b.offsets, uint64_t{n} + 1, 8))
          return CUDAQ_GF2_INVALID_CONTENT;
      } else if (part == 3) {
        const auto &b =
            reinterpret_cast<const cudaq_gf2_packed_matrix_block *>(p)[i];
        if (b.matrix >= dir[4].count || b.reserved)
          return CUDAQ_GF2_INVALID_CONTENT;
      } else {
        const auto &m = reinterpret_cast<const cudaq_gf2_packed_matrix *>(p)[i];
        if (!r.take(m.column_offsets, uint64_t{m.columns} + 1, 8) ||
            !r.take(m.row_indices, m.nonzeros, 4, 4))
          return CUDAQ_GF2_INVALID_CONTENT;
      }
    }
    if (!r.end())
      return CUDAQ_GF2_INVALID_CONTENT;
  }
  return CUDAQ_GF2_OK;
}
bool view_source(const cudaq_gf2_map_view *view, source &s) {
  if (!view || !view->data || reinterpret_cast<uintptr_t>(view->data) % 8 ||
      view->size < sizeof(cudaq_gf2_packed_header))
    return false;
  auto *p = static_cast<const uint8_t *>(view->data);
  const auto &h = *reinterpret_cast<const cudaq_gf2_packed_header *>(p);
  if (h.size != view->size)
    return false;
  s.data = p;
  s.directory = reinterpret_cast<const cudaq_gf2_packed_section *>(
      p + h.directory_offset);
  s.d.type_count = static_cast<uint32_t>(s.directory[0].count);
  s.d.input_bundle_count = static_cast<uint32_t>(s.directory[1].count);
  s.d.output_bundle_count = static_cast<uint32_t>(s.directory[2].count);
  s.d.block_count = s.directory[3].count;
  s.d.section_count = static_cast<uint32_t>(h.directory_count - 5);
  return true;
}
uint64_t vector_size(const source &s, uint32_t side) {
  uint32_t n = side ? s.d.output_bundle_count : s.d.input_bundle_count;
  if (!n)
    return 0;
  auto b = s.bundle(side, n - 1);
  return s.offsets(side, n - 1)[s.type(b.type).wire_count];
}
template <class F>
cudaq_gf2_status with_description(const cudaq_gf2_map_description *d,
                                  const cudaq_gf2_memory *memory, F f) {
  const auto resource = scratch_allocator(memory);
  if (!d || !valid_allocator(resource))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  if (!little_endian())
    return CUDAQ_GF2_UNSUPPORTED_VERSION;
  source s;
  s.d = *d;
  workspace_requirements req;
  auto status = requirements(s, req);
  if (status != CUDAQ_GF2_OK)
    return status;
  allocation scratch(resource);
  status = scratch.reserve(req.bytes, req.alignment);
  if (status != CUDAQ_GF2_OK)
    return status;
  workspace w;
  status = make_workspace(s, scratch.data(), req.bytes, w);
  if (status != CUDAQ_GF2_OK)
    return status;
  status = validate(s, w);
  if (status != CUDAQ_GF2_OK)
    return status;
  return f(s, w);
}
} // namespace
cudaq_gf2_status cudaq_gf2_map_validate(const cudaq_gf2_map_description *d,
                                        const cudaq_gf2_memory *memory) {
  return with_description(
      d, memory, [](const source &, workspace &) { return CUDAQ_GF2_OK; });
}
cudaq_gf2_status cudaq_gf2_map_construct(const cudaq_gf2_map_description *d,
                                         const cudaq_gf2_memory *memory,
                                         cudaq_gf2_owned_buffer *owner,
                                         cudaq_gf2_map_view *out) {
  const auto resource = result_allocator(memory);
  if (!owner || !out || !empty_owner(*owner) || !valid_allocator(resource))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  cudaq_gf2_owned_buffer candidate_owner{};
  cudaq_gf2_map_view candidate;
  const auto status = with_description(
      d, memory, [&](const source &s, workspace &w) -> cudaq_gf2_status {
        prepare_matrices(s, w);
        layout l;
        if (!measure(s, w, l))
          return CUDAQ_GF2_INSUFFICIENT_CAPACITY;
        allocation output(resource);
        auto allocated = output.reserve(l.h.size, CUDAQ_GF2_PACKED_ALIGNMENT);
        if (allocated != CUDAQ_GF2_OK)
          return allocated;
        write(s, w, l, static_cast<uint8_t *>(output.data()));
        candidate = {output.data(), l.h.size};
        candidate_owner = output.relinquish();
        return CUDAQ_GF2_OK;
      });
  if (status == CUDAQ_GF2_OK) {
    *owner = candidate_owner;
    *out = candidate;
  }
  return status;
}
cudaq_gf2_status cudaq_gf2_map_read(const void *buffer, uint64_t bytes,
                                    const cudaq_gf2_schema *schema,
                                    const cudaq_gf2_memory *memory,
                                    cudaq_gf2_map_view *out) {
  const auto resource = scratch_allocator(memory);
  if (!out || !valid_allocator(resource))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  source s;
  auto status = parse(buffer, bytes, schema, s);
  if (status != CUDAQ_GF2_OK)
    return status;
  workspace_requirements req;
  status = requirements(s, req);
  if (status != CUDAQ_GF2_OK)
    return status;
  allocation scratch(resource);
  status = scratch.reserve(req.bytes, req.alignment);
  if (status != CUDAQ_GF2_OK)
    return status;
  workspace w;
  status = make_workspace(s, scratch.data(), req.bytes, w);
  if (status != CUDAQ_GF2_OK)
    return status;
  status = validate(s, w);
  if (status != CUDAQ_GF2_OK)
    return status;
  prepare_matrices(s, w);
  if (w.unique != s.directory[4].count)
    return CUDAQ_GF2_INVALID_CONTENT;
  auto *blocks =
      reinterpret_cast<const cudaq_gf2_packed_matrix_block *>(s.payload(3));
  for (uint32_t i = 0; i < s.d.block_count; ++i)
    if (blocks[i].matrix != w.matrix_ids[i])
      return CUDAQ_GF2_INVALID_CONTENT;
  scratch.reset();
  *out = {buffer, bytes};
  return CUDAQ_GF2_OK;
}
cudaq_gf2_status cudaq_gf2_map_get_shape(const cudaq_gf2_map_view *view,
                                         cudaq_gf2_map_shape *out) {
  source s;
  if (!out || !view_source(view, s))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  *out = {s.d.type_count,    s.d.input_bundle_count, s.d.output_bundle_count,
          s.d.section_count, s.d.block_count,        s.directory[4].count};
  return CUDAQ_GF2_OK;
}
cudaq_gf2_status cudaq_gf2_map_get_type(const cudaq_gf2_map_view *view,
                                        uint32_t i,
                                        cudaq_gf2_bundle_type *out) {
  source s;
  if (!out || !view_source(view, s) || i >= s.d.type_count)
    return CUDAQ_GF2_INVALID_ARGUMENT;
  *out = s.type(i);
  return CUDAQ_GF2_OK;
}
cudaq_gf2_status cudaq_gf2_map_get_bundle(const cudaq_gf2_map_view *view,
                                          cudaq_gf2_wire_direction direction,
                                          uint32_t i, cudaq_gf2_bundle *out) {
  source s;
  if (!out || direction > CUDAQ_GF2_OUTPUT || !view_source(view, s) ||
      i >= (direction == CUDAQ_GF2_OUTPUT ? s.d.output_bundle_count
                                          : s.d.input_bundle_count))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  *out = s.bundle(direction, i);
  return CUDAQ_GF2_OK;
}
cudaq_gf2_status cudaq_gf2_map_get_wire(const cudaq_gf2_map_view *view,
                                        cudaq_gf2_wire_direction direction,
                                        cudaq_gf2_wire wire, uint32_t *width,
                                        uint64_t *offset) {
  source s;
  uint32_t candidate;
  if (!width || !offset || direction > CUDAQ_GF2_OUTPUT ||
      !view_source(view, s) || !wire_width(s, direction, wire, candidate))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  *width = candidate;
  *offset = s.offsets(direction, wire.bundle)[wire.wire];
  return CUDAQ_GF2_OK;
}
cudaq_gf2_status
cudaq_gf2_map_get_matrix_block(const cudaq_gf2_map_view *view, uint64_t i,
                               cudaq_gf2_map_matrix_block *out) {
  source s;
  if (!out || !view_source(view, s) || i >= s.d.block_count)
    return CUDAQ_GF2_INVALID_ARGUMENT;
  *out = s.block(static_cast<uint32_t>(i));
  return CUDAQ_GF2_OK;
}
cudaq_gf2_status cudaq_gf2_map_get_section(const cudaq_gf2_map_view *view,
                                           cudaq_gf2_section_id id,
                                           cudaq_gf2_section *out) {
  source s;
  if (!out || !view_source(view, s))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  uint64_t lo = 0, hi = 5 + uint64_t{s.d.section_count};
  while (lo < hi) {
    uint64_t mid = lo + (hi - lo) / 2;
    if (s.directory[mid].id < id)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo >= 5 + uint64_t{s.d.section_count} || s.directory[lo].id != id)
    return CUDAQ_GF2_INVALID_ARGUMENT;
  auto e = s.directory[lo];
  *out = {e.id, e.flags, e.count, e.bytes, s.data + e.offset};
  return CUDAQ_GF2_OK;
}
cudaq_gf2_status cudaq_gf2_map_vector_sizes(const cudaq_gf2_map_view *view,
                                            uint64_t *inputs,
                                            uint64_t *outputs) {
  source s;
  if (!inputs || !outputs || !view_source(view, s))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  *inputs = vector_size(s, 0);
  *outputs = vector_size(s, 1);
  return CUDAQ_GF2_OK;
}
cudaq_gf2_status cudaq_gf2_map_apply(const cudaq_gf2_map_view *view,
                                     const uint8_t *input, uint64_t inputs,
                                     uint8_t *output, uint64_t outputs) {
  source s;
  if (!view_source(view, s) || inputs != vector_size(s, 0) ||
      outputs != vector_size(s, 1) || (inputs && !input) ||
      (outputs && !output) || (inputs && input == output))
    return CUDAQ_GF2_INVALID_ARGUMENT;
  for (uint64_t i = 0; i < inputs; ++i)
    if (input[i] > 1)
      return CUDAQ_GF2_INVALID_ARGUMENT;
  if (outputs)
    memset(output, 0, outputs);
  for (uint32_t i = 0; i < s.d.block_count; ++i) {
    auto b = s.block(i);
    uint64_t from = s.offsets(0, b.input.bundle)[b.input.wire],
             to = s.offsets(1, b.output.bundle)[b.output.wire];
    cudaq_gf2_csc_xor_kernel(b.matrix.columns, b.matrix.column_offsets,
                             b.matrix.row_indices, input + from, output + to);
  }
  return CUDAQ_GF2_OK;
}
