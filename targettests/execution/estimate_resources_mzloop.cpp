/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/


// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>

// Tests that the choice function works properly for a loop condition
struct mykernel {
  auto operator()() __qpu__ {

    cudaq::qubit q;
    h(q);
    while (mz(q)) {
      rz(0.1, q);
      h(q);
    }
  }
};

int main() {
  auto kernel = mykernel{};
  int i = 0;
  // Should cause 5 loops
  auto counts = cudaq::estimate_resources([&](){ return i++ < 5; }, kernel);
  counts.dump();

  // CHECK: Total # of gates: 11
  // CHECK-DAG: h :  6
  // CHECK-DAG: rz :  5

  return 0;
}
