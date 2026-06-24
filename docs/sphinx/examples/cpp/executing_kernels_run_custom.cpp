/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin `RunCustom`]
#include <cstdio>
#include <cudaq.h>

// Define a custom data structure to return from the quantum kernel
struct MeasurementResult {
  bool first_qubit;
  bool last_qubit;
  int total;
};

// Define a quantum kernel that returns the custom data structure
__qpu__ MeasurementResult bell_pair_with_data() {
  // Create a Bell pair
  cudaq::qvector qubits(2);
  h(qubits[0]);
  x<cudaq::ctrl>(qubits[0], qubits[1]);

  bool m0 = mz(qubits[0]);
  bool m1 = mz(qubits[1]);

  int total = 0;
  if (m0)
    total++;
  if (m1)
    total++;

  return {m0, m1, total};
}

int main() {
  auto results = cudaq::run(10, bell_pair_with_data);
  int correlated_count = 0;
  for (auto i = 0; i < results.size(); ++i) {
    printf("Shot %d: {%d, %d}	total ones=%d\n", i, results[i].first_qubit,
           results[i].last_qubit, results[i].total);
    if (results[i].first_qubit == results[i].last_qubit)
      correlated_count++;
  }
  printf("Correlated measurements: %d/%zu\n", correlated_count, results.size());
  return 0;
}
// [End `RunCustom`]
/* [Begin `RunCustomOutput`]
Shot 0: {1, 1}  total ones=2
Shot 1: {0, 0}  total ones=0
Shot 2: {0, 0}  total ones=0
Shot 3: {1, 1}  total ones=2
Shot 4: {1, 1}  total ones=2
Shot 5: {0, 0}  total ones=0
Shot 6: {1, 1}  total ones=2
Shot 7: {0, 0}  total ones=0
Shot 8: {1, 1}  total ones=2
Shot 9: {0, 0}  total ones=0
Correlated measurements: 10/10
[End `RunCustomOutput`] */
