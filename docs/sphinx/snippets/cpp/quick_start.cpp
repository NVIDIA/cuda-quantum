/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ quick_start.cpp && ./a.out`
// [Begin Documentation]
#include <cudaq.h>

__qpu__ void kernel(int n_qubits) {
  cudaq::qvector qubits(n_qubits);
  h(qubits[0]);
  for (auto i = 1; i < n_qubits; ++i) {
    cx(qubits[0], qubits[i]);
  }
  mz(qubits);
}

int main(int argc, char *argv[]) {
  auto n_qubits = 1 < argc ? atoi(argv[1]) : 2;
  auto result = cudaq::sample(kernel, n_qubits);
  result.dump(); // Example: { 11:500 00:500 }
}
// [End Documentation]