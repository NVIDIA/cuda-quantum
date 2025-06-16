/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ negated_control.cpp && ./a.out`

#include <cudaq.h>

int main() {
  cudaq::qubit c, q;
  // [Begin Negated Control]
  h(c);
  x<cudaq::ctrl>(!c, q);
  // The qubits c and q are in a state (|01> + |10>) / √2.

  return 0;
}
// [End Negated Control]

