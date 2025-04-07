/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
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
  std::cout << "Measured |0> with probability "
            << static_cast<double>(result.count("0")) /
                   std::accumulate(result.begin(), result.end(), 0)
            << std::endl;
  std::cout << "Measured |1> with probability "
            << static_cast<double>(result.count("1")) /
                   std::accumulate(result.begin(), result.end(), 0)
            << std::endl;
  return 0;
}
// [End Docs]
