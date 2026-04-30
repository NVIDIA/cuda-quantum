/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Compile and run with:
// ```
// nvq++ gradients.cpp -o gs.x && ./gs.x
// ```

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/gradients.h>
#include <cudaq/optimizers.h>

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

int main() {

  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  cudaq::spin_op h3 = h + 9.625 - 9.625 * cudaq::spin_op::z(2) -
                      3.913119 * cudaq::spin_op::x(1) * cudaq::spin_op::x(2) -
                      3.913119 * cudaq::spin_op::y(1) * cudaq::spin_op::y(2);

  // Default here is COBYLA
  // Should see many more iterations
  printf("Optimize with no gradients.\n");
  cudaq::optimizers::cobyla optimizer;
  auto [opt_val, opt_params] = optimizer.optimize(
      2, [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
        double e = cudaq::observe(deuteron_n3_ansatz{}, h3, x[0], x[1]);
        printf("<H>(%lf, %lf) = %lf\n", x[0], x[1], e);
        return e;
      });

  auto argMapper = [](std::vector<double> x) {
    return std::make_tuple(x[0], x[1]);
  };

  // Change the default to L-BFGS which requires gradient calculation
  // Should see fewer iterations
  printf("\nOptimize with gradients.\n");
  cudaq::gradients::parameter_shift gradient(deuteron_n3_ansatz{}, argMapper);
  gradient.shiftScalar = 1e-1;
  cudaq::optimizers::lbfgs optimizer_lbfgs;
  optimizer_lbfgs.max_line_search_trials = 10;
  auto [opt_val2, opt_params2] = optimizer_lbfgs.optimize(
      2, [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
        double e = cudaq::observe(deuteron_n3_ansatz{}, h3, x[0], x[1]);
        gradient.compute(x, grad_vec, h3, e);
        printf("<H>(%lf, %lf) = %lf\n", x[0], x[1], e);
        return e;
      });

  printf("\nOptimize with gradients, change a few options.\n");
  cudaq::gradients::central_difference gradient2(deuteron_n3_ansatz{},
                                                 argMapper);
  optimizer_lbfgs.initial_parameters = std::vector<double>{.1, .1};
  optimizer_lbfgs.max_eval = 10;
  auto [opt_val3, optp3] = optimizer_lbfgs.optimize(
      2, [&](const std::vector<double> &x, std::vector<double> &grad_vec) {
        double e = cudaq::observe(deuteron_n3_ansatz{}, h3, x[0], x[1]);
        gradient2.compute(x, grad_vec, h3, e);
        printf("<H>(%lf, %lf) = %lf\n", x[0], x[1], e);
        return e;
      });

  printf("Opt loop found %lf at [%lf, %lf]\n", opt_val3, optp3[0], optp3[1]);
  return 0;
}
