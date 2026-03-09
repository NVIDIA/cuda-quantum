/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <cudaq.h>

namespace cudaq {
/// @brief Implement Givens rotation at a specific angle
///
/// This kernel is equivalent to matrix exp(-i theta (YX - XY) / 2)
/// | 1, 0,  0, 0 |
/// | 0, c, -s, 0 |
/// | 0, s,  c, 0 |
/// | 0, 0,  0, 1 |
/// c = cos(theta); s = sin(theta)
/// @param theta Rotation angle (in rads)
/// @param q0 First qubit operand
/// @param q1 Second qubit operand
__qpu__ void givens_rotation(double theta, cudaq::qubit &q0, cudaq::qubit &q1) {
  exp_pauli(-0.5 * theta, "YX", q0, q1);
  exp_pauli(0.5 * theta, "XY", q0, q1);
}

namespace builder {
/// @brief Add Givens rotation kernel (theta angle as a QuakeValue) to the
/// kernel builder object
/// @tparam KernelBuilder
/// @param kernel
/// @param theta
/// @param q0
/// @param q1
template <typename KernelBuilder>
void givens_rotation(KernelBuilder &kernel, cudaq::QuakeValue theta,
                     cudaq::QuakeValue q0, cudaq::QuakeValue q1) {
  kernel.exp_pauli(-0.5 * theta, "YX", q0, q1);
  kernel.exp_pauli(0.5 * theta, "XY", q0, q1);
}

/// @brief Add Givens rotation kernel (fixed theta angle) to the kernel builder
/// object
/// @tparam KernelBuilder
/// @param kernel
/// @param theta
/// @param q0
/// @param q1
template <typename KernelBuilder>
void givens_rotation(KernelBuilder &kernel, double theta, cudaq::QuakeValue q0,
                     cudaq::QuakeValue q1) {
  givens_rotation(kernel, kernel.constantVal(theta), q0, q1);
}
} // namespace builder
} // namespace cudaq
