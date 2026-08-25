/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cstdio>
#include <cudaq.h>

__qpu__ void entangle_all(cudaq::qview<> qubits) {
  h(qubits[0]);
  for (std::size_t i = 1; i < qubits.size(); i++) {
    x<cudaq::ctrl>(qubits[0], qubits[i]);
  }
}

// Define a quantum kernel that returns an integer
__qpu__ int run_ghz(int num_qubits) {
  // Allocate qubits
  cudaq::qvector qubits(num_qubits);

  // Call kernel
  entangle_all(qubits);

  // Measure and return total number of qubits in state |1⟩
  int result = 0;
  for (int i = 0; i < num_qubits; i++) {
    if (mz(qubits[i])) {
      result += 1;
    }
  }
  return result;
}
