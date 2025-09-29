/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>

// Tests that the choice function works properly for a loop condition
struct mykernel {
  auto operator()() __qpu__ {

    cudaq::qubit q;
    h(q);
    for (size_t i = 0; i < 100; i++) {
      x(q);
      auto bit = mz(q);
      reset(q);
      if (bit)
        break;
    }
  }
};

int main() {
  auto kernel = mykernel{};
  int i = 0;
  auto counts = cudaq::estimate_resources([&]() { return ++i >= 10; }, kernel);
  counts.dump();

  // CHECK: Total # of gates: 21
  // CHECK-DAG: h :  1
  // CHECK-DAG: x :  10
  // CHECK-DAG: reset :  10

  return 0;
}
