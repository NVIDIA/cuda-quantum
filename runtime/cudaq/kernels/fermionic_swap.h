/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq/builder/kernel_builder.h"
#include <cudaq.h>
namespace cudaq {
/// @brief Apply global phase (e^(i * theta)).
/// Note: since this is a global phase, the qubit operand can be selected
/// arbitrarily from the qubit register to which the global phase is applied.
/// @param theta Global phase value (in rads)
/// @param q Qubit operand
__qpu__ void global_phase(double theta, cudaq::qubit &q) {
  // R1(theta) and Rz(theta) are equivalent upto a global phase.
  // Hence, R1(phi) Rz(-phi) sequence results in a global phase of e^(i *
  // phi/2).
  // TODO: For hardware execution, where access to the state vector is
  // impossible, hence global phase is irrelevant, there are a couple of
  // optimization passes that we can do: (1) Remove this r1(angle) - rz (-angle)
  // pattern at the global scope when we know that no control modifier has been
  // applied. (2) Optimize the controlled gate decomposition (phase kickback on
  // the control qubits).
  r1(2.0 * theta, q);
  rz(-2.0 * theta, q);
}

/// @brief Fermionic SWAP rotation at a specific angle
///
/// This kernel performs a rotation in the adjacent Fermionic modes under the
/// Jordan-Wigner mapping represented by this unitary matrix:
///
///   |1  0                        0                        0         |
///   |0  e^(i*phi/2)cos(phi/2)    -i*e^(i*phi/2)sin(phi/2) 0         |
///   |0  -i*e^(i*phi/2)sin(phi/2) e^{i*phi/2}cos(phi/2)    0         |
///   |0  0                        0                        e^{i*phi} |
/// @param phi Rotation angle (in rads)
/// @param q0 First qubit operand
/// @param q1 Second qubit operand
__qpu__ void fermionic_swap(double phi, cudaq::qubit &q0, cudaq::qubit &q1) {
  h(q0);
  h(q1);

  x<cudaq::ctrl>(q0, q1);
  rz(phi / 2.0, q1);
  x<cudaq::ctrl>(q0, q1);

  h(q0);
  h(q1);

  rx(M_PI_2, q0);
  rx(M_PI_2, q1);

  x<cudaq::ctrl>(q0, q1);
  rz(phi / 2.0, q1);
  x<cudaq::ctrl>(q0, q1);

  rx(-M_PI_2, q0);
  rx(-M_PI_2, q1);
  rz(phi / 2.0, q0);
  rz(phi / 2.0, q1);

  // Global phase correction
  global_phase(phi / 2.0, q0);
}

namespace builder {
/// @brief Add Fermionic SWAP rotation kernel (phi angle as a QuakeValue) to the
/// kernel builder object
template <typename KernelBuilder>
void fermionic_swap(KernelBuilder &kernel, cudaq::QuakeValue phi,
                    cudaq::QuakeValue q0, cudaq::QuakeValue q1) {
  kernel.h(q0);
  kernel.h(q1);

  kernel.template x<cudaq::ctrl>(q0, q1);
  kernel.rz(phi / 2.0, q1);
  kernel.template x<cudaq::ctrl>(q0, q1);

  kernel.h(q0);
  kernel.h(q1);

  kernel.rx(M_PI_2, q0);
  kernel.rx(M_PI_2, q1);

  kernel.template x<cudaq::ctrl>(q0, q1);
  kernel.rz(phi / 2.0, q1);
  kernel.template x<cudaq::ctrl>(q0, q1);

  kernel.rx(-M_PI_2, q0);
  kernel.rx(-M_PI_2, q1);
  kernel.rz(phi / 2.0, q0);
  kernel.rz(phi / 2.0, q1);

  // Global phase correction
  kernel.r1(phi, q0);
  kernel.rz(-phi, q0);
}

/// @brief Add Fermionic SWAP rotation kernel (fixed phi angle) to the kernel
/// builder object
template <typename KernelBuilder>
void fermionic_swap(KernelBuilder &kernel, double phi, cudaq::QuakeValue q0,
                    cudaq::QuakeValue q1) {
  fermionic_swap(kernel, kernel.constantVal(phi), q0, q1);
}
} // namespace builder
} // namespace cudaq