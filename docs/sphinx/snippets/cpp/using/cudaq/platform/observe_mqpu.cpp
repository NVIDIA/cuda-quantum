/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ observe_mqpu.cpp -o observe_mqpu.x -target nvidia-mqpu
// && ./observe_mqpu.x
// ```
#include <cudaq.h>

int main() {
  // [Begin Documentation]
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  // Get the quantum_platform singleton
  auto &platform = cudaq::get_platform();

  // Query the number of QPUs in the system
  auto num_qpus = platform.num_qpus();
  printf("Number of QPUs: %zu\n", num_qpus);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  double result = cudaq::observe<cudaq::parallel::thread>(ansatz, h, 0.59);
  printf("Expectation value: %lf\n", result);
  // [End Documentation]
  return 0;
}
