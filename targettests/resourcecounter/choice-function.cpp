/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/


// Compile and run with:
// ```
// RUN: nvq++ %cpp_std --target quantinuum               --emulate %s -o %t && %t | FileCheck %s
// ```

#include <cudaq.h>

// Basic check that the choice function works to determine the path taken
struct mykernel {
  auto operator()() __qpu__ {
    cudaq::qubit q;
    cudaq::qubit p;

    h(q);

    auto m1 = mz(q);

    if (m1)
      x(p);

    mz(p);
  }
};

int main() {
  auto kernel = mykernel{};
  auto gateCountsTrue = cudaq::count_resources([](){ return true; }, kernel);
  auto gateCountsFalse = cudaq::count_resources([](){ return false; }, kernel);

  printf("True path\n");
  gateCountsTrue.dump();
  // CHECK-LABEL: True path
  // CHECK-DAG: h :  1
  // CHECK-DAG: x :  1
  printf("False path\n");
  gateCountsFalse.dump();
  // CHECK-LABEL: False path
  // CHECK-DAG: h :  1

  return 0;
}
