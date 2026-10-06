/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Documentation]
#include <cudaq.h>
#include <iostream>

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

// [Begin Sample1]
int main() {

  int qubit_count = 2;
  auto result_0 = cudaq::sample(kernel, /* kernel args */ qubit_count);
  // Should see a roughly 50/50 distribution between the |00> and
  // |11> states. Example: {00: 505  11: 495}
  result_0.dump();
  // [End Sample1]

  // [Begin Sample2]
  // With an increased shots count, we will still see the same 50/50
  // distribution, but now with 10,000 total measurements instead of the default
  // 1000. Example: {00: 5005  11: 4995}
  int shots_count = 10000;
  auto result_1 = cudaq::sample(shots_count, kernel, qubit_count);
  result_1.dump();
  // [End Sample2]

  // [Begin Sample3]
  std::cout << result_1.most_probable() << "\n"; // prints: `00`
  std::cout << result_1.probability(result_1.most_probable())
            << "\n"; // prints: `0.5005`
}
// [End Sample3]
