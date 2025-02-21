/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/schedule.h"
#include <optional>
#include <stdexcept>

namespace cudaq {

// Constructor
Schedule::Schedule(std::vector<std::complex<double>> steps,
                   std::vector<std::string> parameters,
                   std::function<std::complex<double>(
                       const std::string &, const std::complex<double> &)>
                       value_function)
    : _steps(steps), _parameters(parameters), _value_function(value_function),
      _current_idx(-1) {
  if (!_steps.empty()) {
    m_ptr = &_steps[0];
  } else {
    m_ptr = nullptr;
  }
}

// Dereference operator
Schedule::reference Schedule::operator*() const { return *m_ptr; }

// Arrow operator
Schedule::pointer Schedule::operator->() { return m_ptr; }

// Prefix increment
Schedule &Schedule::operator++() {
  if (_current_idx + 1 < static_cast<int>(_steps.size())) {
    _current_idx++;
    m_ptr = &_steps[_current_idx];
  } else {
    throw std::out_of_range("No more steps in the schedule.");
  }
  return *this;
}

// Postfix increment
Schedule Schedule::operator++(int) {
  Schedule tmp = *this;
  ++(*this);
  return tmp;
}

// Comparison operators
bool operator==(const Schedule &a, const Schedule &b) {
  return a.m_ptr == b.m_ptr;
};

bool operator!=(const Schedule &a, const Schedule &b) {
  return a.m_ptr != b.m_ptr;
};

// Reset schedule
void Schedule::reset() {
  _current_idx = -1;
  if (!_steps.empty()) {
    m_ptr = &_steps[0];
  } else {
    m_ptr = nullptr;
  }
}

// Get the current step
std::optional<std::complex<double>> Schedule::current_step() const {
  if (_current_idx >= 0 && _current_idx < static_cast<int>(_steps.size())) {
    return _steps[_current_idx];
  }
  return std::nullopt;
}

} // namespace cudaq
