/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ r1_op.cpp && ./a.out`

#include <cudaq.h>

int main() {
  cudaq::qubit qubit;
  // [Begin R1 Op]
  // Apply the unitary transformation
  // R1(λ) = | 1     0    |
  //         | 0  exp(iλ) |
  r1(std::numbers::pi, qubit);
  // [End R1 Op]
  return 0;
}

