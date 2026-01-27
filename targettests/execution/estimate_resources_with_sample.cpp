/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/


// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s

#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>

// Tests that estimate_resources works with sample
struct mykernel {
  auto operator()() __qpu__ {
    cudaq::qubit q;

    x(q);

    auto m1 = mz(q);
  }
};

int main() {
  auto kernel = mykernel{};
  auto counts1 = cudaq::sample(5, kernel);
  auto gateCounts = cudaq::estimate_resources(kernel);
  auto counts2 = cudaq::sample(10, kernel);

  
  counts1.dump();
  // CHECK: m1 : { 1:5 }
  gateCounts.dump();
  // CHECK: Total # of gates: 1, total # of qubits: 1
  // CHECK: x :  1
  counts2.dump();
  // CHECK: m1 : { 1:10 }

  return 0;
}
