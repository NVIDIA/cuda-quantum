/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ adjoint_op.cpp && ./a.out`

#include <cudaq.h>

int main() {
  // Allocate a qubit in a |0> state.
  cudaq::qubit qubit;
  // [Begin Adjoint Op]
  // Apply the unitary transformation defined by the matrix
  // T = | 1      0     |
  //     | 0  exp(iπ/4) |
  // to the state of the qubit `q`:
  t(qubit);

  // Apply its adjoint transformation defined by the matrix
  // T† = | 1      0     |
  //      | 0  exp(-iπ/4) |
  t<cudaq::adj>(qubit);
  // Qubit `q` is now again in the initial state |0>.

  return 0;
}
// [End Adjoint Op]
