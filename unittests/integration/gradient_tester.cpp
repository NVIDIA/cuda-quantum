/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <cudaq/algorithms/gradients/central_difference.h>
#include <cudaq/optimizers.h>

#ifndef CUDAQ_BACKEND_DM
struct deuteron_n3_ansatz {
  void operator()(double x0, double x1) __qpu__ {
    cudaq::qvector q(3);
    x(q[0]);
    ry(x0, q[1]);
    ry(x1, q[2]);
    x<cudaq::ctrl>(q[2], q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    ry(-x0, q[1]);
    x<cudaq::ctrl>(q[0], q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  }
};

CUDAQ_TEST(GradientTester, checkSimple) {
  using namespace cudaq::spin;

  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);
  cudaq::spin_op h3 = h + 9.625 - 9.625 * z(2) - 3.913119 * x(1) * x(2) -
                      3.913119 * y(1) * y(2);

  // Default here is cobyla
  // Should see many more iterations
  printf("Optimize with no gradients.\n");
  cudaq::optimizers::cobyla optimizer;
  optimizer.optimize(
      2, [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
        double e = cudaq::observe(deuteron_n3_ansatz{}, h3, x[0], x[1]);
        printf("<H>(%lf, %lf) = %lf\n", x[0], x[1], e);
        return e;
      });

  // Change the default to l-bfgs which requires gradient calc
  // Should see fewer iterations
  printf("\nOptimize with gradients.\n");
  cudaq::gradients::central_difference gradient(
      deuteron_n3_ansatz{},
      [](std::vector<double> x) { return std::make_tuple(x[0], x[1]); });
  cudaq::optimizers::lbfgs optimizer_lbfgs;
  auto [opt_val_0, optpp] = optimizer_lbfgs.optimize(
      2, [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
        double e = cudaq::observe(deuteron_n3_ansatz{}, h3, x[0], x[1]);
        gradient.compute(x, grad_vec, h3, e);
        printf("<H>(%lf, %lf) = %lf\n", x[0], x[1], e);
        return e;
      });

  EXPECT_NEAR(-2.0453, opt_val_0, 1e-2);

  printf("\nOptimize with gradients, change a few options.\n");
  optimizer_lbfgs.initial_parameters = std::vector<double>{.1, .1};
  optimizer_lbfgs.max_eval = 10;
  auto [opt_val, optp] = optimizer_lbfgs.optimize(
      2, [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
        double e = cudaq::observe(deuteron_n3_ansatz{}, h3, x[0], x[1]);
        gradient.compute(x, grad_vec, h3, e);
        printf("<H>(%lf, %lf) = %lf\n", x[0], x[1], e);
        return e;
      });

  printf("Opt loop found %lf at [%lf, %lf]\n", opt_val, optp[0], optp[1]);

  EXPECT_NEAR(-2.0453, opt_val, 1e-2);
}

#endif
