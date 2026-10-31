/******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 *****************************************************************************/

#pragma once

#include "cudaq/gf2/format.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <span>
#include <type_traits>
namespace cudaq_gf2_private {
// Scratch arrays are laid out by arena::take, each at this alignment.
constexpr uint64_t scratch_alignment = 8;
// Packed arrays use natural alignment: a scalar's size (one for strings, four
// for 32-bit and eight for 64-bit values) and the format alignment for records.
template <class T>
constexpr uint64_t packed_alignment =
    std::is_scalar_v<T> ? sizeof(T) : CUDAQ_GF2_PACKED_ALIGNMENT;
inline bool array_ok(const void *p, uint64_t n, uint64_t width) {
  return n <= SIZE_MAX / width && (!n || p);
}
template <class T>
bool array_ok(const T *p, uint64_t n) {
  return array_ok(p, n, sizeof(T));
}
inline bool reserve(uint64_t &cursor, uint64_t count, uint64_t width,
                    uint64_t &offset, uint64_t alignment) {
  offset = 0;
  if (!count)
    return true;
  if (cursor > SIZE_MAX - (alignment - 1) || count > SIZE_MAX / width)
    return false;
  uint64_t aligned = (cursor + alignment - 1) & ~(alignment - 1);
  if (count * width > SIZE_MAX - aligned)
    return false;
  offset = aligned;
  cursor = aligned + count * width;
  return true;
}
template <class T>
bool reserve_scratch(uint64_t &cursor, uint64_t count, uint64_t &offset) {
  return reserve(cursor, count, sizeof(T), offset, scratch_alignment);
}
template <class T>
bool reserve_scratch(uint64_t &cursor, uint64_t count) {
  uint64_t ignored;
  return reserve_scratch<T>(cursor, count, ignored);
}
template <class T>
bool reserve_packed(uint64_t &cursor, uint64_t count, uint64_t &offset) {
  return reserve(cursor, count, sizeof(T), offset, packed_alignment<T>);
}
template <class T>
bool reserve_packed(uint64_t &cursor, uint64_t count) {
  uint64_t ignored;
  return reserve_packed<T>(cursor, count, ignored);
}
// Section payloads are byte ranges at the format alignment.
inline bool reserve_payload(uint64_t &cursor, uint64_t bytes,
                            uint64_t &offset) {
  return reserve(cursor, bytes, 1, offset, CUDAQ_GF2_PACKED_ALIGNMENT);
}
struct arena {
  std::byte *data;
  uint64_t capacity, used = 0;
  template <class T>
  std::span<T> take(uint64_t count) {
    uint64_t offset, next = used;
    if (!count || !reserve_scratch<T>(next, count, offset) || next > capacity)
      return {};
    auto *p = reinterpret_cast<T *>(data + offset);
    for (uint64_t i = 0; i < count; ++i)
      ::new (static_cast<void *>(p + i)) T{};
    used = next;
    return {p, static_cast<size_t>(count)};
  }
};
inline int name_compare(const char *a, uint64_t na, const char *b,
                        uint64_t nb) {
  int cmp = std::memcmp(a, b, static_cast<size_t>(std::min(na, nb)));
  return cmp ? cmp : (na > nb) - (na < nb);
}
inline bool utf8_name(const char *name, uint64_t size) {
  for (uint64_t i = 0; i < size;) {
    uint32_t c = static_cast<uint8_t>(name[i++]), value, following, minimum;
    if (c > 0 && c < 0x80)
      continue;
    if (c >= 0xc2 && c <= 0xdf) {
      value = c & 0x1f;
      following = 1;
      minimum = 0x80;
    } else if (c >= 0xe0 && c <= 0xef) {
      value = c & 0xf;
      following = 2;
      minimum = 0x800;
    } else if (c >= 0xf0 && c <= 0xf4) {
      value = c & 7;
      following = 3;
      minimum = 0x10000;
    } else
      return false;
    if (following > size - i)
      return false;
    while (following--) {
      c = static_cast<uint8_t>(name[i++]);
      if ((c & 0xc0) != 0x80)
        return false;
      value = (value << 6) | (c & 0x3f);
    }
    if (value < minimum || value > 0x10ffff ||
        (value >= 0xd800 && value <= 0xdfff))
      return false;
  }
  return size != 0;
}
inline bool little_endian() {
  const uint32_t one = 1;
  return *reinterpret_cast<const uint8_t *>(&one) == 1;
}
inline bool zeros(const uint8_t *data, uint64_t begin, uint64_t end) {
  while (begin < end)
    if (data[begin++])
      return false;
  return true;
}
struct reader {
  const uint8_t *data;
  uint64_t bytes, cursor = 0;
  bool take(uint64_t offset, uint64_t count, uint64_t width,
            uint64_t alignment) {
    uint64_t expected, next = cursor;
    if (!reserve(next, count, width, expected, alignment) ||
        expected != offset || next > bytes ||
        (count && !zeros(data, cursor, offset)))
      return false;
    cursor = next;
    return true;
  }
  template <class T>
  bool take(uint64_t offset, uint64_t count) {
    return take(offset, count, sizeof(T), packed_alignment<T>);
  }
  bool take_payload(uint64_t offset, uint64_t bytes) {
    return take(offset, bytes, 1, CUDAQ_GF2_PACKED_ALIGNMENT);
  }
  bool end() const { return cursor == bytes; }
};
template <class Less>
void sort_indices(std::span<uint32_t> indices, Less less) {
  for (size_t i = 0; i < indices.size(); ++i)
    indices[i] = static_cast<uint32_t>(i);
  if (indices.size() > 1)
    std::sort(indices.begin(), indices.end(), less);
}
} // namespace cudaq_gf2_private
