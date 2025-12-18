/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target infleqtion      --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target anyon           --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target ionq            --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target iqm             --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt  %t | FileCheck %s
// RUN: nvq++ --target oqc             --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum      --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %quantum_machines_avail; then nvq++ --target quantum_machines --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

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
   cudaq::spin_op h = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
                     2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
                     .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);

  // Make repeatable for shots-based emulation
  cudaq::set_random_seed(13);

  // Observe takes the kernel, the spin_op, and the concrete
  // parameters for the kernel
  double energy = cudaq::observe(ansatz{}, h, .59);
  printf("Energy is %.16lf\n", energy);
  return 0;
}

// Note: seeds 2 and 12 will push this to -2 instead of -1. All other seeds in
// 1-100 range will be -1.x.

// CHECK: Energy is -1.
