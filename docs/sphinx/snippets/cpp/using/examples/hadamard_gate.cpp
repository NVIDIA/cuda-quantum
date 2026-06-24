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

__qpu__ void hadamard_gate() {
  // Allocate a qubit.
  cudaq::qubit q;
  // Apply a hadamard gate to the qubit.
  h(q);
  // Measure the qubit.
  mz(q);
}

int main() {
  // Call the kernel to run the hadamard gate.
  auto result = cudaq::sample(hadamard_gate);

  // Default number of shots
  int shots = 1000;
  int count0 = result.count("0");
  int count1 = result.count("1");

  // result.dump();
  std::cout << "Measured |0> with probability "
            << static_cast<double>(count0) / shots << std::endl;
  std::cout << "Measured |1> with probability "
            << static_cast<double>(count1) / shots << std::endl;
  return 0;
}
// [End Docs]
