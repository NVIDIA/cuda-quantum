/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ first_kernel.cpp && ./a.out`
#include <cudaq.h>

// [Begin Documentation]
__qpu__ void kernel(int qubit_count) {
  // Allocate our qubits.
  cudaq::qvector qvector(qubit_count);
  // Place the first qubit in the superposition state.
  h(qvector[0]);
  // Loop through the allocated qubits and apply controlled-X,
  // or CNOT, operations between them.
  for (auto qubit : cudaq::range(qubit_count - 1)) {
    x<cudaq::ctrl>(qvector[qubit], qvector[qubit + 1]);
  }
  // Measure the qubits.
  mz(qvector);
}
// [End Documentation]

// Just for the CI:
int main() {
  auto test_result = cudaq::sample(kernel, 1, 1);
  test_result.dump();
}
