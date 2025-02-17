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
    std::function<std::complex<double>(
        const std::unordered_map<std::string, std::complex<double>> &)>
        value_function)
    : _steps(steps), _parameters(parameters), _value_function(value_function) {}
} // namespace cudaq
