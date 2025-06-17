/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with: `nvq++ types_qvector_example.cpp && ./a.out`

#include <cudaq.h>
#include <stdio.h>
#include <vector>

__qpu__ void qvector_usage_kernel_cpp() {
  // [Begin CppQvectorUsage]
  // Allocate 20 qubits, std::vector-like semantics
  cudaq::qvector q(20);
  // Get first qubit
  cudaq::qubit &first = q.front(); // Note: auto first = q.front(); in RST
  h(first);                        // Example operation

  // Get first 5 qubits
  cudaq::qview first_5 = q.front(5); // Note: auto first_5 = q.front(5); in RST
  for (auto &qubit_in_view : first_5) {
    h(qubit_in_view);
  }

  // Get last qubit
  cudaq::qubit &last = q.back(); // Note: auto last = q.back(); in RST
  x(last);                       // Example operation

  // Can loop over qubits with size() method
  for (std::size_t i = 0; i < q.size(); i++) {
    h(q[i]); // ... do something with q[i] ...
  }
  // Range based for loop supported
  for (auto &qb : q) {
    x(qb); // ... do something with qb ...
  }
  // [End CppQvectorUsage]
  mz(q); // Measure all for sampling
}

int main() {
  printf("C++ Qvector Usage Example:\n");
  auto counts = cudaq::sample(qvector_usage_kernel_cpp);
  counts.dump();
  return 0;
}

