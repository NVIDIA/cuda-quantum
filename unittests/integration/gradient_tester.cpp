/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <cudaq/algorithms/gradients/central_difference.h>
#include <cudaq/optimizers.h>

// Skip these gradient tests for slow backends to reduce test time.
// Note: CUDA-Q API level tests (e.g., `cudaq::observe`) should cover all
// backend-specific functionalities required to interface gradient modules.
#if !defined CUDAQ_BACKEND_DM && !defined CUDAQ_BACKEND_TENSORNET
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

  // Use l-bfgs optimizer which requires gradient calc
  // Since we have gradients, it should converge rather quickly (small number of
  // iterations is needed)
  printf("\nOptimize with gradients.\n");
  cudaq::gradients::central_difference gradient(
      deuteron_n3_ansatz{},
      [](std::vector<double> x) { return std::make_tuple(x[0], x[1]); });
  cudaq::optimizers::lbfgs optimizer_lbfgs;
  optimizer_lbfgs.max_line_search_trials = 3;
  auto [opt_val_0, optpp] = optimizer_lbfgs.optimize(
      2, [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
        double e = cudaq::observe(deuteron_n3_ansatz{}, h3, x[0], x[1]);
        gradient.compute(x, grad_vec, h3, e);
        printf("<H>(%lf, %lf) = %lf\n", x[0], x[1], e);
        return e;
      });

  EXPECT_NEAR(-2.0453, opt_val_0, 1e-3);

  printf("\nOptimize with gradients, change a few options.\n");
  optimizer_lbfgs.initial_parameters = std::vector<double>{.1, .1};
  optimizer_lbfgs.max_eval = 5;
  optimizer_lbfgs.max_line_search_trials = 3;
  auto [opt_val, optp] = optimizer_lbfgs.optimize(
      2, [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
        double e = cudaq::observe(deuteron_n3_ansatz{}, h3, x[0], x[1]);
        gradient.compute(x, grad_vec, h3, e);
        printf("<H>(%lf, %lf) = %lf\n", x[0], x[1], e);
        return e;
      });

  printf("Opt loop found %lf at [%lf, %lf]\n", opt_val, optp[0], optp[1]);

  EXPECT_NEAR(-2.0453, opt_val, 1e-3);
}

#endif
