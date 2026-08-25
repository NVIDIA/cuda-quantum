/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cstdio>
#include <cudaq.h>

// Declarations
__qpu__ void entangle_all(cudaq::qview<> qubits);
__qpu__ int run_ghz(int num_qubits);

// Define a kernel that just do measurements (no return)
__qpu__ void sample_app(int num_qubits) {
  // Allocate qubits
  cudaq::qvector qubits(num_qubits);

  // Call kernel
  entangle_all(qubits);

  // Measure all qubits
  mz(qubits);
}

__qpu__ int run_app(int a, int b) { return run_ghz(a) + run_ghz(b); }

int main() {
  // Execute/Run the kernel 20 times
  constexpr int num_shots = 20;
  auto results = cudaq::run(num_shots, run_app, 3, 5);
  assert(results.size() == num_shots);

  for (size_t i = 0; i < results.size(); ++i) {
    assert(results[i] == 0 || results[i] == 3 || results[i] == 5 ||
           results[i] == 8);
  }

  // Now, sample the kernel
  auto counts = cudaq::sample(num_shots, sample_app, 5);
  counts.dump();
  assert(counts.size() == 2);
  assert((counts.count("11111") + counts.count("00000")) == num_shots);
  return 0;
}
