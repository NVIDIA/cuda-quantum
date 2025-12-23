/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>

int main() {
   cudaq::spin_op h = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) - 
                     2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
                     .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

  {
    auto [ansatz, theta] = cudaq::make_kernel<double>();

    // // Allocate some qubits
    auto q = ansatz.qalloc(2);

    // Build up the circuit, use the acquired parameter
    ansatz.x(q[0]);
    ansatz.ry(theta, q[1]);
    ansatz.x<cudaq::ctrl>(q[1], q[0]);

    // Observe takes the kernel, the spin_op, and the concrete
    // parameters for the kernel
    double energy = cudaq::observe(ansatz, h, .59);
    printf("Energy is %lf\n", energy);
    REMOTE_TEST_ASSERT(std::abs(energy + 1.748794) < 1e-3);
  }
  {
    auto [ansatz, thetas] = cudaq::make_kernel<std::vector<double>>();

    // // Allocate some qubits
    auto q = ansatz.qalloc(2);

    // Build up the circuit, use the acquired parameter
    ansatz.x(q[0]);
    ansatz.ry(thetas[0], q[1]);
    ansatz.x<cudaq::ctrl>(q[1], q[0]);
    double energy = cudaq::observe(ansatz, h, std::vector<double>{.59});
    printf("Energy is %lf\n", energy);
    REMOTE_TEST_ASSERT(std::abs(energy + 1.748794) < 1e-3);
  }

  {
    cudaq::spin_op h = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) - 
                       2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
                       .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1) + 
                       9.625 - 9.625 * cudaq::spin_op::z(2) -
                       3.913119 * cudaq::spin_op::x(1) * cudaq::spin_op::x(2) - 
                       3.913119 * cudaq::spin_op::y(1) * cudaq::spin_op::y(2);
    auto [ansatz, theta, beta] = cudaq::make_kernel<double, double>();
    // Allocate some qubits
    auto q = ansatz.qalloc(3);
    // Build the kernel
    ansatz.x(q[0]);
    ansatz.ry(theta, q[1]);
    ansatz.ry(beta, q[2]);
    ansatz.x<cudaq::ctrl>(q[2], q[0]);
    ansatz.x<cudaq::ctrl>(q[0], q[1]);
    ansatz.ry(-theta, q[1]);
    ansatz.x<cudaq::ctrl>(q[0], q[1]);
    ansatz.x<cudaq::ctrl>(q[1], q[0]);
    const double energy = cudaq::observe(ansatz, h, 1.124, 2.3456);
    printf("Energy = %lf\n", energy);
  }

  return 0;
}
