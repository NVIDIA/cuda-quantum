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
  pointer ptr;
  std::vector<std::complex<double>> steps;
  std::vector<std::string> parameters;
  std::function<std::complex<double>(const std::string &,
                                     const std::complex<double> &)>
      value_function;
  int current_idx;

public:
  Schedule(pointer ptr) : ptr(ptr){};

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
  Schedule(
      const std::vector<std::complex<double>> &steps,
      const std::vector<std::string> &parameters,
      const std::function<std::complex<double>(
          const std::string &, const std::complex<double> &)> &value_function);

  /// Below, I define what I believe are the minimal necessary methods needed
  /// for this to behave like an iterable. This should be revisited in the
  /// implementation phase.

  // Pointers.
  /// @brief `Dereference` operator to access the current step value.
  /// @return Reference to current complex step value.
  reference operator*() const;

  /// @brief Arrow operator to access the pointer the current step value.
  /// @return Pointer to the current complex step value.
  pointer operator->();

  // Prefix increment.
  /// @brief Prefix increment operator to move to the next step in the schedule.
  /// @return Reference to the updated Schedule object.
  Schedule &operator++();

  // Postfix increment.
  /// @brief `Postfix` increment operator to move to the next step in the
  /// schedule.
  /// @return Copy of the previous Schedule state.
  Schedule operator++(int);

  // Comparison.
  /// @brief Equality comparison operator.
  /// @param a: First Schedule object.
  /// @param b: Second Schedule object.
  /// @return True if both schedules point to the same step, false otherwise
  friend bool operator==(const Schedule &a, const Schedule &b);

  /// @brief Inequality comparison operator.
  /// @param a: First Schedule object.
  /// @param b: Second Schedule object.
  /// @return True if both schedules point to different steps, false otherwise
  friend bool operator!=(const Schedule &a, const Schedule &b);

  /// @brief Reset the schedule iterator to the beginning.
  void reset();

  /// @brief Get the current step in the schedule.
  /// @return The current complex step value as an optional. If no valid step,
  /// returns std::nullopt.
  std::optional<std::complex<double>> current_step() const;
};
} // namespace cudaq
