/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QIRTypes.h"
#include "common/FmtCore.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <memory>
#include <sstream>

extern "C" {
extern bool initialized;
extern bool verbose;
}

int8_t *Array::operator[](std::size_t index) {
  if (static_cast<uint64_t>(index * element_size_bytes) >= storage.size())
    throw std::runtime_error(
        fmt::format("Provided index [{}] >= array size [{}]", index,
                    storage.size() / element_size_bytes));
  return &storage.at(index * element_size_bytes);
}
// Ctors
// Default items are pointers.
Array::Array(std::size_t _nitems, int _item_size)
    : element_size_bytes(_item_size),
      // Initialized to zero
      storage(_nitems * _item_size, 0) {
  assert(element_size_bytes > 0);
};

Array::Array(const Array &other)
    : element_size_bytes(other.element_size_bytes), storage(other.storage) {}

void Array::append(const Array &other) {
  if (other.element_size_bytes != element_size_bytes) {
    throw std::runtime_error("Cannot append Arrays of different types.");
  }

  storage.insert(storage.end(), other.storage.begin(), other.storage.end());
}

void Array::add_element() { storage.resize((1 + size()) * element_size_bytes); }

std::size_t Array::size() const { return storage.size() / element_size_bytes; }
void Array::clear() { storage.clear(); }
int Array::element_size() const { return element_size_bytes; }

Array::~Array() { clear(); }

std::vector<int64_t> getRangeValues(Array *in_array, const Range &in_range) {
  const bool is_fwd_range = in_range.step > 0;

  const auto convertIndex = [&](int64_t in_rawIdx) -> int64_t {
    if (in_rawIdx >= 0) {
      return in_rawIdx;
    }
    // Negative-based index:
    // in_rawIdx = -1 => size - 1 (last element)
    int64_t result = in_array->size() + in_rawIdx;
    if (result < 0) {
      throw std::invalid_argument("range");
    }
    return result;
  };

  // Convert to absolute index.
  const auto start_idx = convertIndex(in_range.start);
  const auto end_idx = convertIndex(in_range.end);
  // start == end
  if (start_idx == end_idx) {
    return {end_idx};
  }

  if (is_fwd_range) {
    if (start_idx > end_idx) {
      return {};
    }

    assert(in_range.step > 0);
    std::vector<int64_t> result;
    for (int64_t i = start_idx; i <= end_idx; i += in_range.step) {
      auto qubit = *reinterpret_cast<Qubit **>((*in_array)[i]);
      result.emplace_back(qubit->idx);
    }
    return result;
  }

  if (start_idx < end_idx) {
    return {};
  }

  std::vector<int64_t> result;
  assert(in_range.step < 0);
  for (int64_t i = start_idx; i >= end_idx; i += in_range.step) {
    auto qubit = *reinterpret_cast<Qubit **>((*in_array)[i]);
    result.emplace_back(qubit->idx);
  }
  return result;
}

namespace nvqir {
Array *vectorSizetToArray(std::vector<std::size_t> &);
}

extern "C" {

Array *__quantum__rt__array_create_1d(int32_t itemSizeInBytes,
                                      int64_t count_items) {
  return new Array(count_items, itemSizeInBytes);
}

int8_t *__quantum__rt__array_get_element_ptr_1d(Array *q, uint64_t idx) {
  Array &arr = *q;
  int8_t *ptr = arr[idx];
  return ptr;
}

int64_t __quantum__rt__array_get_size_1d(Array *state1) {
  if (state1 == nullptr)
    return 0;

  return state1->size();
}
Array *__quantum__rt__array_slice(Array *array, int32_t dim,
                                  int64_t range_start, int64_t range_step,
                                  int64_t range_end) {
  return quantum__rt__array_slice(array, dim,
                                  {range_start, range_step, range_end});
}

Array *__quantum__rt__array_slice_1d(Array *array, int64_t range_start,
                                     int64_t range_step, int64_t range_end) {
  return __quantum__rt__array_slice(array, 0, range_start, range_step,
                                    range_end);
}

Array *quantum__rt__array_slice(Array *array, int32_t dim, Range range) {
  const std::vector<int64_t> range_idxs = getRangeValues(array, range);
  std::vector<std::size_t> sliceIdxs;
  for (const auto &idx : range_idxs) {
    sliceIdxs.push_back(idx);
  }
  return nvqir::vectorSizetToArray(sliceIdxs);
}

Array *__quantum__rt__array_concatenate(Array *head, Array *tail) {
  if (head && tail) {
    auto resultArray = new Array(*head);
    resultArray->append(*tail);
    return resultArray;
  }

  return nullptr;
}
Array *__quantum__rt__array_copy(Array *array, bool forceNewInstance) {
  if (array && forceNewInstance) {
    return new Array(*array);
  }

  if (!array) {
    return nullptr;
  }

  return array;
}
void __quantum__rt__array_release(Array *a) { delete a; }
}