/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/QuditIdTracker.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <deque>
#include <map>
#include <memory>
#include <numeric>
#include <vector>

/// This file defines the various types that we will use
/// throughout the NVQIR library. Specifically, we will define
/// the QIR specification types here (Qubit, Array, TuplePtr, Result, etc.)

// The TuplePtr is just a i8 pointer.
using TuplePtr = int8_t *;

/// QIR Range type
struct Range {
  int64_t start;
  int64_t step;
  int64_t end;
};

/// The Qubit is a simple wrapper around
/// a Qubit unique identifier integer
struct Qubit {
  Qubit(std::size_t i) : idx(i) {}
  std::size_t idx;
};

/// General 1D array
class Array {
private:
  const int element_size_bytes;
  std::vector<int8_t> storage;
  std::atomic<int> ref_count;

public:
  // Get the element pointer at given index
  int8_t *operator[](std::size_t index);
  // Ctors
  // Default items are pointers.
  Array(std::size_t _nitems, int _item_size = sizeof(int8_t *));

  Array(const Array &other);

  void append(const Array &other);

  void add_element();

  std::size_t size() const;
  void clear();
  int element_size() const;

  ~Array();
};

/// Array Runtime Functions
extern "C" {
Array *__quantum__rt__array_create_1d(int32_t itemSizeInBytes,
                                      int64_t count_items);
void __quantum__rt__array_release(Array *);
int64_t __quantum__rt__array_get_size_1d(Array *array);
int8_t *__quantum__rt__array_get_element_ptr_1d(Array *q, uint64_t idx);
Array *__quantum__rt__array_copy(Array *array, bool forceNewInstance);
Array *__quantum__rt__array_concatenate(Array *head, Array *tail);
Array *__quantum__rt__array_slice(Array *array, int32_t dim,
                                  int64_t range_start, int64_t range_step,
                                  int64_t range_end);
Array *__quantum__rt__array_slice_1d(Array *array, int64_t range_start,
                                     int64_t range_step, int64_t range_end);
Array *quantum__rt__array_slice(Array *array, int32_t dim, Range range);
}

// Results
using Result = bool;
static const Result ResultZeroVal = false;
static const Result ResultOneVal = true;
inline Result *ResultZero = const_cast<Result *>(&ResultZeroVal);
inline Result *ResultOne = const_cast<Result *>(&ResultOneVal);

// Pauli type.
enum Pauli : int8_t {
  Pauli_I = 0,
  Pauli_X,
  Pauli_Z,
  Pauli_Y,
};
