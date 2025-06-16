/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ swap_op.cpp && ./a.out`

int main() {
  cudaq::qubit qubit_1, qubit_2;

  // Apply the unitary transformation
  // Swap = | 1 0 0 0 |
  //        | 0 0 1 0 |
  //        | 0 1 0 0 |
  //        | 0 0 0 1 |
  swap(qubit_1, qubit_2);

  return 0;
}

