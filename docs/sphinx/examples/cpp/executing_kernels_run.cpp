/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Run]
#include <cstdio>
#include <cudaq.h>

// Define a quantum kernel that returns an integer
__qpu__ int simple_ghz(int num_qubits) {
  // Allocate qubits
  cudaq::qvector qubits(num_qubits);
  // Create GHZ state
  h(qubits[0]);
  for (int i = 1; i < num_qubits; i++) {
    x<cudaq::ctrl>(qubits[0], qubits[i]);
  }
  // Measure and return total number of qubits in state |1⟩
  int result = 0;
  for (int i = 0; i < num_qubits; i++) {
    if (mz(qubits[i])) {
      result += 1;
    }
  }
  return result;
}

int main() {
  // Execute the kernel 20 times
  auto results = cudaq::run(20, simple_ghz, 3);
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
  return 0;
}
// [End Run]
/* [Begin `RunOutput`]
Executed 20 shots
Results: [3, 0, 3, 0, 0, 0, 3, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 3, 0]
Possible values: Either 0 or 3 due to GHZ state properties

Counts of each result:
0: 13 times
3: 7 times
[End `RunOutput`] */
