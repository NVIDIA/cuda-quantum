/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Docs]
#include <cudaq.h>
#include <iostream>

__qpu__ void pauli_x_gate() {
  // Allocate a qubit.
  cudaq::qubit q;
  // Apply a Pauli-X gate to the qubit.
  x(q);
  // Measure the qubit.
  mz(q);
}

int main() {
  // Sample the qubit for 1000 shots to gather statistics.
  auto result = cudaq::sample(pauli_x_gate);
  result.dump();
  return 0;
}
// [End Docs]
