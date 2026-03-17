/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <limits>

namespace cudaq {

extern "C" {
bool __nvqpp__MeasureResultBoolConversion(int);
}

/// We model the return type of a qubit measurement result via the
/// `measure_result` type. This allows us to keep track of when the result is
/// implicitly cast to a boolean (likely in the case of conditional feedback),
/// and affect the simulation accordingly.
class measure_result {
public:
  /// The intrinsic measurement value
  std::int64_t value = 0;

  /// Unique integer for measure result identification.
  /// INT64_MAX means unassigned; negative values are valid
  std::int64_t unique_id = std::numeric_limits<std::int64_t>::max();

  // Constructors
  measure_result() = default;
  explicit measure_result(int val) : value(val) {}
  explicit measure_result(int val, int id) : value(val), unique_id(id) {}

  // Operator overloads for conversions and comparisons
#ifdef CUDAQ_LIBRARY_MODE
  operator bool() const { return __nvqpp__MeasureResultBoolConversion(value); }
#else
  operator bool() const { return value == 1; }
#endif
  explicit operator int() const { return value; }
  explicit operator double() const { return static_cast<double>(value); }

  friend bool operator==(const measure_result &m1, const measure_result &m2) {
    return (m1.value == m2.value) && (m1.unique_id == m2.unique_id);
  }
  friend bool operator==(const measure_result &m, bool b) {
    return static_cast<bool>(m) == b;
  }
  friend bool operator==(bool b, const measure_result &m) {
    return b == static_cast<bool>(m);
  }

  friend bool operator!=(const measure_result &m1, const measure_result &m2) {
    return (m1.value != m2.value) || (m1.unique_id != m2.unique_id);
  }
  friend bool operator!=(const measure_result &m, bool b) {
    return static_cast<bool>(m) != b;
  }
  friend bool operator!=(bool b, const measure_result &m) {
    return b != static_cast<bool>(m);
  }
};

} // namespace cudaq
