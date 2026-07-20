/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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
schedule::schedule(
    const std::vector<std::complex<double>> &steps,
    const std::vector<std::string> &parameters,
    const std::function<std::complex<double>(
        const std::string &, const std::complex<double> &)> &value_function)
    : steps(steps), parameters(parameters), value_function(value_function),
      current_idx(-1) {
  if (!steps.empty())
    ptr = &this->steps[0];
  else
    ptr = nullptr;
}

// Dereference operator
schedule::reference schedule::operator*() const { return *ptr; }

// Arrow operator
schedule::pointer schedule::operator->() { return ptr; }

// Prefix increment
schedule &schedule::operator++() {
  if (current_idx + 1 < steps.size()) {
    current_idx++;
    ptr = &steps[current_idx];
  } else {
    throw std::out_of_range("No more steps in the schedule.");
  }
  return *this;
}

// Postfix increment
schedule schedule::operator++(int) {
  schedule tmp = *this;
  ++(*this);
  return tmp;
}

// Comparison operators
bool operator==(const schedule &a, const schedule &b) {
  return a.ptr == b.ptr;
};

bool operator!=(const schedule &a, const schedule &b) {
  return a.ptr != b.ptr;
};

// Reset schedule
void schedule::reset() {
  current_idx = -1;
  if (!steps.empty()) {
    ptr = &steps[0];
  } else {
    ptr = nullptr;
  }
}

// Get the current step
std::optional<std::complex<double>> schedule::current_step() const {
  if (current_idx >= 0 && current_idx < steps.size())
    return steps[current_idx];

  return std::nullopt;
}

std::vector<std::complex<double>>::iterator schedule::begin() {
  return steps.begin();
}

std::vector<std::complex<double>>::iterator schedule::end() {
  return steps.end();
}

std::vector<std::complex<double>>::const_iterator schedule::begin() const {
  return steps.cbegin();
}

std::vector<std::complex<double>>::const_iterator schedule::end() const {
  return steps.cend();
}

// Get the parameters of the schedule.
const std::vector<std::string> &schedule::get_parameters() const {
  return parameters;
}

// Get the value function of the schedule.
const std::function<std::complex<double>(const std::string &,
                                         const std::complex<double> &)>
schedule::get_value_function() const {
  return value_function;
}

} // namespace cudaq
