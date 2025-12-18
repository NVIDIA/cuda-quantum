/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target infleqtion --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target ionq       --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target iqm        --emulate %s -o %t && IQM_QPU_QA=%iqm_tests_dir/Crystal_5.txt  %t | FileCheck %s
// RUN: nvq++ --target oqc        --emulate %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// RUN: if %braket_avail; then nvq++ --target braket --emulate %s -o %t && %t | FileCheck %s; fi
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithm.h>
#include <math.h>

bool isClose(double a, double b, double tol = 1e-1) {
  return std::abs(a - b) <= tol;
}

// The example here shows a simple use case for the `cudaq::observe`
// function in computing expected values of provided spin_ops.

// The test validate that the expectation results for each pauli
// term in {X,Y,Z} are correct for simulators and quantum devices.

struct ansatz_x {
  auto operator()() __qpu__ { cudaq::qvector q(1); }
};

struct ansatz_y {
  auto operator()() __qpu__ {
    cudaq::qvector q(4);
    x(q[0]);
  }
};

struct ansatz_z {
  auto operator()() __qpu__ {
    cudaq::qvector q(2);
    x(q);
  }
};

int main() {

  {
    auto h = cudaq::spin_op::x(0);
    cudaq::set_random_seed(13);

    // Observe the kernel and make sure we get the expected energy
    // This tests implementation of observing X op.
    double energy = cudaq::observe(10000, ansatz_x{}, h);
    if (isClose(energy, 0.0))
      printf("Energy is %d.\n", (int)(energy + 0.5));
    else
      printf("Observe of X failed. Energy is %.16lf\n", energy);
  }

  {
    auto h = cudaq::spin_op::y(3);
    cudaq::set_random_seed(13);

    // Observe the kernel and make sure we get the expected energy
    // This tests implementation of observing Y op.
    double energy = cudaq::observe(10000, ansatz_y{}, h);
    if (isClose(energy, 0.0))
      printf("Energy is %d.\n", (int)(energy + 0.5));
    else
      printf("Observe of Y failed. Energy is %.16lf\n", energy);
  }

  {
    auto h = cudaq::spin_op::z(0) * cudaq::spin_op::z(1);
    cudaq::set_random_seed(13);

    // Observe the kernel and make sure we get the expected energy.
    // This tests implementation of observing Z op,
    // which should be the same as sample expectation.
    double energy = cudaq::observe(10000, ansatz_z{}, h);
    double expectation = cudaq::sample(10000, ansatz_z{}).expectation();

    if (isClose(energy, expectation)) {
      printf("Energy is %d.\n", (int)(energy + 0.5));
      printf("Expectation is %d.\n", (int)(expectation + 0.5));
    } else
      printf("Observe of Z failed. "
             " Energy is %.16lf, expectation is %.16lf\n",
             energy, expectation);
  }

  return 0;
}

// CHECK: Energy is 0.
// CHECK: Energy is 0.
// CHECK: Energy is 1.
// CHECK: Expectation is 1.
