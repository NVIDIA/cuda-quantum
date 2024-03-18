/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Documentation]
#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <cudaq/gradients.h>
#include <cudaq/optimizers.h>
#include <iostream>

int main() {
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                     .21829 * z(0) - 6.125 * z(1);

  auto [ansatz, theta] = cudaq::make_kernel<double>();
  auto q = ansatz.qalloc();
  auto r = ansatz.qalloc();
  ansatz.x(q);
  ansatz.ry(theta, r);
  ansatz.x<cudaq::ctrl>(r, q);

  // Run VQE with a gradient-based optimizer.
  // Delegate cost function and gradient computation across different NVQC-based
  // QPUs.
  // Note: this needs to be compiled with `--nvqc-nqpus 3` create 3 virtual
  // QPUs.
  cudaq::optimizers::lbfgs optimizer;
  auto [opt_val, opt_params] = optimizer.optimize(
      /*dim=*/1, /*opt_function*/ [&](const std::vector<double> &params,
                                      std::vector<double> &grads) {
        // Queue asynchronous jobs to do energy evaluations across multiple QPUs
        auto energy_future =
            cudaq::observe_async(/*qpu_id=*/0, ansatz, h, params[0]);
        const double paramShift = M_PI_2;
        auto plus_future = cudaq::observe_async(/*qpu_id=*/1, ansatz, h,
                                                params[0] + paramShift);
        auto minus_future = cudaq::observe_async(/*qpu_id=*/2, ansatz, h,
                                                 params[0] - paramShift);
        grads[0] = (plus_future.get().expectation() -
                    minus_future.get().expectation()) /
                   2.0;
        return energy_future.get().expectation();
      });
  std::cout << "Minimum energy = " << opt_val << " (expected -1.74886).\n";
}
