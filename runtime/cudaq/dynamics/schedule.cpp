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
Schedule::Schedule(const std::vector<std::complex<double>> &steps,
                   const std::vector<std::string> &parameters,
                   const std::function<std::complex<double>(
                       const std::string &, const std::complex<double> &)> &value_function)
: steps(steps), parameters(parameters), value_function(value_function), current_idx(-1) {
  if (!steps.empty()) ptr = &this->steps[0];
  else ptr = nullptr;
}

// Dereference operator
Schedule::reference Schedule::operator*() const { return *ptr; }

// Arrow operator
Schedule::pointer Schedule::operator->() { return ptr; }

// Prefix increment
Schedule &Schedule::operator++() {
  if (current_idx + 1 < static_cast<int>(steps.size())) {
    current_idx++;
    ptr = &steps[current_idx];
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
  return a.ptr == b.ptr;
};

bool operator!=(const Schedule &a, const Schedule &b) {
  return a.ptr != b.ptr;
};

// Reset schedule
void Schedule::reset() {
  current_idx = -1;
  if (!steps.empty()) {
    ptr = &steps[0];
  } else {
    ptr = nullptr;
  }
}

// Get the current step
std::optional<std::complex<double>> Schedule::current_step() const {
  if (current_idx >= 0 && current_idx < static_cast<int>(steps.size())) {
    return steps[current_idx];
  }
  return std::nullopt;
}

} // namespace cudaq
