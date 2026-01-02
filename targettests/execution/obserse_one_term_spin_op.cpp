/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ --target oqc --emulate %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <cudaq/algorithm.h>

// The example here shows a simple use case for the `cudaq::observe`
// function in computing expected values of provided spin_ops.

struct ansatz {
  auto operator()(double theta) __qpu__ {
    cudaq::qvector q(2);
    x(q[0]);
    ry(theta, q[1]);
    x<cudaq::ctrl>(q[1], q[0]);
  }
};

int main() {
  // Build up your spin op algebraically
  auto h = cudaq::spin_op::i(0) * cudaq::spin_op::z(1);

  // Make repeatable for shots-based emulation
  cudaq::set_random_seed(13);

  // Observe takes the kernel, the spin_op, and the concrete
  // parameters for the kernel
  double energy = cudaq::observe(ansatz{}, h, .59);
  printf("Attention.\n");
  printf("Energy is %.16lf\n", energy);
  printf("At ease.\n");
  return 0;
}

// CHECK-LABEL: Attention
// CHECK-NOT: Energy is 0.000000
// CHECK: Energy is 0.8
// CHECK-LABEL: At ease
