/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ t_op.cpp && ./a.out`

int main() {
  cudaq::qubit qubit;

  // Apply the unitary transformation
  // T = | 1      0     |
  //     | 0  exp(iπ/4) |
  t(qubit);

  return 0;
}

