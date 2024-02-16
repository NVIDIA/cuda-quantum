/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// REQUIRES: c++20
// FIXME: https://github.com/NVIDIA/cuda-quantum/issues/1111

// clang-format off
// RUN: nvq++ --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t 
// RUN: nvq++ --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include <cudaq.h>

void ghz(std::size_t N) __qpu__ {
  cudaq::qvector q(N);
  h(q[0]);
  for (int i = 0; i < N - 1; i++) {
    x<cudaq::ctrl>(q[i], q[i + 1]);
  }
  mz(q);
}

void ansatz(double theta) __qpu__ {
  cudaq::qvector q(2);
  x(q[0]);
  ry(theta, q[1]);
  x<cudaq::ctrl>(q[1], q[0]);
}

int main() {
  auto counts = cudaq::sample(ghz, 10);
  counts.dump();
  assert(counts.size() == 2);
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  double energy = cudaq::observe(ansatz, h, .59);
  printf("Energy is %lf\n", energy);
  assert(std::abs(energy + 1.748794) < 1e-3);
  return 0;
}
