/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: nvq++ --enable-mlir --platform default-qpud %s -o out_testqpudobserve.x && ./out_testqpudobserve.x | FileCheck %s && rm out_testqpudobserve.x

#include <cudaq.h>
#include <cudaq/algorithm.h>

// The example here shows a simple use case for the cudaq::observe()
// function in computing expected values of provided spin_ops.

// CHECK: Energy is -1.7487
// CHECK: Energy with shots is -1.

struct ansatz {
  auto operator()(double theta) __qpu__ {
    cudaq::qreg q(2);
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  }
};

struct ansatzVector {
  auto operator()(std::vector<double> theta) __qpu__ {
    cudaq::qreg q(2);
    x(q[0]);
    ry(theta[0], q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  }
};
int main() {

  // Build up your spin op algebraically
  using namespace cudaq::spin;
  cudaq::spin_op h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
                    .21829 * z(0) - 6.125 * z(1);

  {
    // Observe takes the kernel, the spin_op, and the concrete params for the
    // kernel
    double energy = cudaq::observe(ansatz{}, h, .59);
    printf("Energy is %lf\n", energy);

    // Set shots high enough that we're accurate to -1.7
    cudaq::set_shots(10000);
    auto result = cudaq::observe(ansatz{}, h, .59);
    printf("Energy with shots is %lf\n", result.exp_val_z());

    auto z1Counts = result.counts(z(1));
    assert(z1Counts.size() == 2);
    assert(z1Counts.count("0") && z1Counts.count("1"));
  }

  {
    // Observe takes the kernel, the spin_op, and the concrete params for the
    // kernel
    double energy = cudaq::observe(ansatzVector{}, h, std::vector<double>{.59});
    printf("Energy is %lf\n", energy);

    // Set shots high enough that we're accurate to -1.7
    cudaq::set_shots(10000);
    auto result = cudaq::observe(ansatzVector{}, h, std::vector<double>{.59});
    printf("Energy with shots is %lf\n", result.exp_val_z());

    auto z1Counts = result.counts(z(1));
    assert(z1Counts.size() == 2);
    assert(z1Counts.count("0") && z1Counts.count("1"));
  }
  return 0;
}
