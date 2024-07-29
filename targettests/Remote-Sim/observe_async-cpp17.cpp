/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// REQUIRES: c++17

// clang-format off
// RUN: nvq++ %cpp_std --target remote-mqpu --remote-mqpu-auto-launch 3 %s -o %t && %t 
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 3 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>

struct ansatz {
  auto operator()(double theta) __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    ry(theta, q[1]);
    cx(q[1], q[0]);
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
  REMOTE_TEST_ASSERT(std::abs(energy + 1.748794) < 1e-3);
  // Shots-based observe async. API
  cudaq::set_random_seed(13);
  auto energyFutureShots =
      cudaq::observe_async(/*shots=*/8192, /*qpu_id=*/0, ansatz{}, h, .59);
  const auto energyShots = energyFutureShots.get().expectation();
  printf("Energy (shots) is %lf\n", energyShots);
  REMOTE_TEST_ASSERT(std::abs(energyShots + 1.748794) < 0.1);
  return 0;
}
