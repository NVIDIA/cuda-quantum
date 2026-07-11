/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Observe]
#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithms/draw.h>

// Define a quantum kernel function.
__qpu__ void kernel(int qubit_count) {
  cudaq::qvector qvector(qubit_count);
  // 2-qubit GHZ state.
  h(qvector[0]);
  for (auto qubit : cudaq::range(qubit_count - 1)) {
    x<cudaq::ctrl>(qvector[qubit], qvector[qubit + 1]);
  }
}

int main() {
  int qubit_count = 2;

  // Define a Hamiltonian in terms of Pauli Spin operators.
  auto hamiltonian = cudaq::spin::z(0) + cudaq::spin::y(1) +
                     cudaq::spin::x(0) * cudaq::spin::z(0);

  // Compute the expectation value given the state prepared by the kernel.
  auto result = cudaq::observe(kernel, hamiltonian, qubit_count);
  printf("%.6lf\n", result.expectation());
  return 0;
}
// [End Observe]
/* [Begin `ObserveOutput`]
<H> = 0.000000
 [End `ObserveOutput`] */
