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
    std::vector<bool> var(2);
    cudaq::qubit q0, q1;
    var[0] = mz(q0);
    var[1] = mz(q1);
    return var;
  }
};

int main() {
  auto results = cudaq::run(2, kernel_with_conditional{});
  std::cout << "First result: " << results[0][0] << " " << results[0][1]
            << "\n";
  std::cout << "Second result: " << results[1][0] << " " << results[1][1]
            << "\n";
  return 0;
}

// CHECK: First result: 0 0
// CHECK: Second result: 0 0
