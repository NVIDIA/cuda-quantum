/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++17
// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ %cpp_std --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t 
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/builder.h>
#include <cudaq/gradients.h>
#include <cudaq/optimizers.h>

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
  {
    // Simple `cudaq::observe` test
    double energy = cudaq::observe(ansatz{}, h, .59);
    printf("Energy is %lf\n", energy);
    REMOTE_TEST_ASSERT(std::abs(energy + 1.748794) < 1e-3);
  }
  {
    // Full VQE test with gradients
    auto argMapper = [&](std::vector<double> x) {
      return std::make_tuple(x[0]);
    };
    cudaq::gradients::parameter_shift gradient(ansatz{}, argMapper);
    gradient.shiftScalar = 1e-1;
    cudaq::optimizers::lbfgs optimizer_lbfgs;
    optimizer_lbfgs.max_line_search_trials = 10;
    auto [opt_val, opt_params] = optimizer_lbfgs.optimize(
        1, [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
          double e = cudaq::observe(ansatz{}, h, x[0]);
          gradient.compute(x, grad_vec, h, e);
          printf("<H>(%lf, %lf) = %lf\n", x[0], x[1], e);
          return e;
        });
    printf("Optimal value = %.16lf\n", opt_val);
    REMOTE_TEST_ASSERT(std::abs(opt_val + 1.748794) < 1e-3);
  }
  return 0;
}
