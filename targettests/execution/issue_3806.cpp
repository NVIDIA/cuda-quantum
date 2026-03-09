/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && %t | FileCheck %s
// RUN: nvq++ --library-mode %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target quantinuum --quantinuum-machine Helios-1SC --emulate %s -o %t && %t | FileCheck %s
// RUN: if %qci_avail; then nvq++ --target qci --emulate %s -o %t && %t | FileCheck %s; fi
// clang-format on

#include <cudaq.h>
#include <iostream>

struct kernel_with_conditional {
  std::vector<bool> operator()() __qpu__ {
    std::vector<bool> var(8);
    cudaq::qubit q0, q1, q2, q3, q4, q5, q6, q7;
    var[0] = mz(q0);
    var[1] = mz(q1);
    x(q2);
    var[2] = mz(q2);
    var[3] = mz(q3);
    x(q4);
    var[4] = mz(q4);
    var[5] = mz(q5);
    x(q6);
    var[6] = mz(q6);
    var[7] = mz(q7);
    return var;
  }
};

int main() {
  auto results = cudaq::run(2, kernel_with_conditional{});
  std::cout << "First result: " << results[0][0] << " " << results[0][1]
            << " " << results[0][2] << " " << results[0][3]
            << " " << results[0][4] << " " << results[0][5]
            << " " << results[0][6] << " " << results[0][7]
            << "\n";
  std::cout << "Second result: " << results[1][0] << " " << results[1][1]
            << " " << results[1][2] << " " << results[1][3]
            << " " << results[1][4] << " " << results[1][5]
            << " " << results[1][6] << " " << results[1][7]
            << "\n";
  return 0;
}

// CHECK: First result: 0 0 1 0 1 0 1 0
// CHECK: Second result: 0 0 1 0 1 0 1 0