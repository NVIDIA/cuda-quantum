/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ observe_mqpu_mpi.cpp -o observe_mqpu_mpi.x --target nvidia
// --target-option mqpu mpirun -np <N> ./observe_mqpu_mpi.x
// ```
#include "cudaq.h"

int main() {
  // [Begin Documentation]
  cudaq::mpi::initialize();
  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  double result = cudaq::observe<cudaq::parallel::mpi>(ansatz, h, 0.59);
  if (cudaq::mpi::rank() == 0)
    printf("Expectation value: %lf\n", result);
  cudaq::mpi::finalize();
  // [End Documentation]

  return 0;
}
