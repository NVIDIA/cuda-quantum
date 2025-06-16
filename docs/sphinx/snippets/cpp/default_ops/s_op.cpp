/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ s_op.cpp && ./a.out`

int main() {
  cudaq::qubit qubit;
  // [Begin S Op]
  // Apply the unitary transformation
  // S = | 1   0 |
  //     | 0   i |
  s(qubit);

  return 0;
}
// [End S Op]

