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

// Define a kernel that just do measurements (no return)
__qpu__ void sample_ghz(int num_qubits) {
  // Allocate qubits
  cudaq::qvector qubits(num_qubits);

  // Call kernel
  entangle_all(qubits);

  // Measure all qubits
  mz(qubits);
}

int main() {
  // Execute/Run the kernel 20 times
  constexpr int num_shots = 20;
  auto results = cudaq::run(num_shots, run_ghz, 3);
  assert(results.size() == num_shots);

  // Print results
  printf("Executed %zu shots\n", results.size());
  printf("Results: [");
  for (size_t i = 0; i < results.size(); ++i) {
    printf("%d", results[i]);
    if (i < results.size() - 1) {
      printf(", ");
    }
  }
  printf("]\n");
  printf("Possible values: Either 0 or %d due to GHZ state properties\n", 3);
  // Count occurrences of each result
  std::map<int, int> value_counts;
  for (const auto &value : results) {
    value_counts[value]++;
  }
  printf("\nCounts of each result:\n");
  for (const auto &pair : value_counts) {
    printf("%d: %d times\n", pair.first, pair.second);
  }
  assert(value_counts.size() == 2);

  // Now, sample the kernel
  {
    auto counts = cudaq::sample(num_shots, sample_ghz, 5);
    counts.dump();
    assert(counts.size() == 2);
    assert((counts.count("11111") + counts.count("00000")) == num_shots);
  }

  // Again with different arguments
  {
    auto counts = cudaq::sample(num_shots, sample_ghz, 7);
    counts.dump();
    assert(counts.size() == 2);
    assert((counts.count("1111111") + counts.count("0000000")) == num_shots);
  }

  {
    auto results = cudaq::run(num_shots, run_ghz, 4);
    assert(results.size() == num_shots);
    for (auto result : results)
      assert(result == 0 || result == 4);
  }

  return 0;
}
