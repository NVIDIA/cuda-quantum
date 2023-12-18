/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target remote-sim --remote-sim-auto-launch 1 %s -o %t && %t 
// RUN: nvq++ --enable-mlir --target remote-sim --remote-sim-auto-launch 1 %s -o %t && %t 
// clang-format on

#include <cudaq.h>

struct ansatz {
  auto operator()(double theta) __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  }
};

int main() {
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  // Observe takes the kernel, the spin_op, and the concrete
  // parameters for the kernel
  double energy = cudaq::observe(ansatz{}, h, .59);
  printf("Energy is %lf\n", energy);
#ifndef SYNTAX_CHECK
  assert(std::abs(energy + 1.748794) < 1e-3);
#endif
  return 0;
}
