/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <complex>
#include <cstddef>
#include <cudaq.h>
#include <functional>
#include <iterator>
#include <string>
#include <vector>

namespace cudaq {

/// @brief Create a schedule for evaluating an operator expression at different
/// steps.
class Schedule {
private:
  std::vector<double> _steps;
  std::vector<std::string> _parameters;
  std::function<std::complex<double>(const std::string &, double)>
      _value_function;

public:
  /// @brief Range-based iterator begin function
  /// @return
  std::vector<double>::iterator begin() { return _steps.begin(); }

  /// @brief Range-based iterator end function
  /// @return
  std::vector<double>::iterator end() { return _steps.end(); }

  /// @brief Range-based constant iterator begin function
  /// @return
  std::vector<double>::const_iterator cbegin() const { return _steps.cbegin(); }

  /// @brief Range-based constant iterator end function
  /// @return
  std::vector<double>::const_iterator cend() const { return _steps.cend(); }

  /// @brief Range-based constant iterator begin function
  /// @return
  std::vector<double>::const_iterator begin() const { return cbegin(); }

  /// @brief Range-based constant iterator end function
  /// @return
  std::vector<double>::const_iterator end() const { return cend(); }

  const std::vector<std::string> &parameters() const { return _parameters; }

  std::function<std::complex<double>(const std::string &, double)>
  value_function() const {
    return _value_function;
  }
  /// @brief Constructor.
  /// @arg steps: The sequence of steps in the schedule. Restricted to a vector
  /// of complex values.
  /// @arg parameters: A sequence of strings representing the parameter names of
  /// an operator expression.
  /// @arg value_function: A function that takes the name of a parameter as well
  /// as an additional value ("step") of arbitrary type as argument and returns
  /// the complex value for that parameter at the given step.
  /// @details current_idx: Intializes the current index (_current_idx) to -1 to
  /// indicate that iteration has not yet begun. Once iteration starts,
  /// _current_idx will be used to track the position in the sequence of steps.
  Schedule(const std::vector<double> &steps,
           const std::vector<std::string> &parameters = {},
           std::function<std::complex<double>(const std::string &, double)>
               value_function = {});
  Schedule() = default;
};
} // namespace cudaq
