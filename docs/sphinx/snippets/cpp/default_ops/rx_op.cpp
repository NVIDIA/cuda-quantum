/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ rx_op.cpp && ./a.out`

#include <cudaq.h>

int main() {
  cudaq::qubit qubit;

  // Apply the unitary transformation
  // Rx(θ) = |  cos(θ/2)  -isin(θ/2) |
  //         | -isin(θ/2)  cos(θ/2)  |
  rx(std::numbers::pi, qubit);

  return 0;
}

