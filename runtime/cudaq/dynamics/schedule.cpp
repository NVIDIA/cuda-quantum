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
Schedule::Schedule(
    const std::vector<double> &steps,
    const std::vector<std::string> &parameters,
    std::function<std::complex<double>(const std::string &, double)>
        value_function)
    : _steps(steps), _parameters(parameters), _value_function(value_function) {
  if (!_value_function) {
    _value_function = [&](const std::string &paramName,
                          double value) -> std::complex<double> {
      if (std::find(_parameters.begin(), _parameters.end(), paramName) ==
          _parameters.end())
        throw std::runtime_error("Unknown parameter named " + paramName);

      return value;
    };
  }
}
} // namespace cudaq
