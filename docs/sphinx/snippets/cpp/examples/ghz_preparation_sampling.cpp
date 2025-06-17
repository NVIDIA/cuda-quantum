/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ ghz_preparation_sampling.cpp && ./a.out`

#include <cudaq.h>
#include <stdio.h>

// [Begin GHZ State C++]
__qpu__ void ghz(const int n_qubits) { // Changed return to void as it's sampled
  cudaq::qvector q(n_qubits);
  h(q[0]);
  for (int i = 0; i < n_qubits - 1; ++i)
    // note use of ctrl modifier
    x<cudaq::ctrl>(q[i], q[i + 1]);

  mz(q);
}

int main() {
  // Sample the state produced by the ghz kernel
  auto counts = cudaq::sample(ghz, 10); // Pass n_qubits, e.g., 3
  // Corrected: cudaq::sample takes kernel_name, then args for kernel
  // Assuming the example meant to sample ghz with 10 qubits.
  // If 10 was shots, it would be cudaq::sample(10, ghz, num_qubits_for_ghz)
  // For this example, let's assume 10 is n_qubits for ghz.
  // The original RST implies ghz(10) is called.

  // Let's make it more explicit:
  int num_qubits_for_ghz = 4; // Example number of qubits
  printf("Sampling GHZ state for %d qubits:\n", num_qubits_for_ghz);
  auto counts_result = cudaq::sample(ghz, num_qubits_for_ghz);
  for (auto &[bits, count] : counts_result) { // Added & for structured binding
    printf("Observed %s %lu times.\n", bits.data(),
           count); // .data() for std::string from sample_result
  }
  return 0;
}
// [End GHZ State C++]

