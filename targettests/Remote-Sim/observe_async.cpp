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
// RUN: nvq++ --target remote-mqpu --remote-mqpu-auto-launch 3 %s -o %t && %t 
// RUN: nvq++ --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 3 %s -o %t && %t
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
  auto energyFuture = cudaq::observe_async(/*qpu_id=*/0, ansatz{}, h, .59);
  const double shift = 0.001;
  auto plusFuture =
      cudaq::observe_async(/*qpu_id=*/1, ansatz{}, h, .59 + shift);
  auto minusFuture =
      cudaq::observe_async(/*qpu_id=*/2, ansatz{}, h, .59 - shift);
  const auto energy = energyFuture.get().expectation();
  const double gradient =
      (plusFuture.get().expectation() - minusFuture.get().expectation()) /
      (2 * shift);
  printf("Energy is %lf\n", energy);
  printf("Gradient is %lf\n", gradient);
  assert(std::abs(energy + 1.748794) < 1e-3);
  return 0;
}
