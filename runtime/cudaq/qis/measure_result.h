/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq {

extern "C" {
bool __nvqpp__MeasureResultBoolConversion(int);
}

/// We model the return type of a qubit measurement result via the
/// `measure_result` type. This allows us to keep track of when the result is
/// implicitly cast to a boolean (likely in the case of conditional feedback),
/// and affect the simulation accordingly.
class measure_result {
private:
  /// The intrinsic measurement result
  int result = 0;

  /// Unique integer for measure result identification
  int uniqueId = -1; // unassigned

public:
  // Constructors
  measure_result() = default;
  measure_result(int res) : result(res) {}
  measure_result(int res, int id) : result(res), uniqueId(id) {}

  // Accessors
  int getResult() const { return result; }
  int getUniqueId() const { return uniqueId; }

  // Operator overloads for conversions and comparisons
#ifdef CUDAQ_LIBRARY_MODE
  operator bool() const { return __nvqpp__MeasureResultBoolConversion(result); }
#else
  operator bool() const { return result == 1; }
#endif
  explicit operator int() const { return result; }
  explicit operator double() const { return static_cast<double>(result); }

  friend bool operator==(const measure_result &m1, const measure_result &m2) {
    return (m1.result == m2.result) && (m1.uniqueId == m2.uniqueId);
  }
  friend bool operator==(const measure_result &m, bool b) {
    return static_cast<bool>(m) == b;
  }
  friend bool operator==(bool b, const measure_result &m) {
    return b == static_cast<bool>(m);
  }

  friend bool operator!=(const measure_result &m1, const measure_result &m2) {
    return static_cast<bool>(m1) != static_cast<bool>(m2);
  }
  friend bool operator!=(const measure_result &m, bool b) {
    return static_cast<bool>(m) != b;
  }
  friend bool operator!=(bool b, const measure_result &m) {
    return b != static_cast<bool>(m);
  }
};

} // namespace cudaq
