/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cmath>

// A simple state type
using TestState = double;

// Simple derivative function: dx/dt = -x (exponential decay)
inline TestState simple_derivative(const TestState &state, double t) {
    return -state;
}

// A complex function: dx/dt = sin(t)
inline TestState sine_derivative(const TestState &state, double t) {
    return std::sin(t);
}
