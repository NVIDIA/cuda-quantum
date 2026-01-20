/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <optional>
#include <vector>

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
  std::optional<std::size_t> uniqueId = std::nullopt;

public:
  measure_result() = default;
  measure_result(int res) : result(res) {}
  measure_result(int res, std::size_t id) : result(res), uniqueId(id) {}

  operator bool() const { return __nvqpp__MeasureResultBoolConversion(result); }
  explicit operator int() const { return result; }
  explicit operator double() const { return static_cast<double>(result); }

  friend bool operator==(const measure_result &m, bool b) {
    return static_cast<bool>(m) == b;
  }
};

} // namespace cudaq
