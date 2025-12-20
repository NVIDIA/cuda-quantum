/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// clang-format off
// RUN: nvq++ --target remote-mqpu                             %s -o %t && %t | FileCheck %s
// RUN: nvq++ --target remote-mqpu --remote-mqpu-auto-launch 2 %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>

struct mykernel {
  auto operator()() __qpu__ {
    cudaq::qubit q;
    
    rx(.0,q);
    ry(.0,q);
    rz(.0,q);
    h(q);
    x(q);
    y(q);
    z(q);
    s(q);
    t(q);
  }
};

int main() {
  auto kernel = mykernel{};
  auto gateCounts = cudaq::estimate_resources(kernel);

  gateCounts.dump();
  // CHECK: Total # of gates: 9, total # of qubits: 1
  // Note: This is a little fragile with filecheck, it's important to have `rx :  1`
  //       before `x :  1` or else `x :  1` will match `rx :  1` and `rx :  1` will
  //       have no matches
  // CHECK-DAG: rx :  1
  // CHECK-DAG: ry :  1
  // CHECK-DAG: rz :  1
  // CHECK-DAG: h :  1
  // CHECK-DAG: x :  1
  // CHECK-DAG: y :  1
  // CHECK-DAG: z :  1
  // CHECK-DAG: s :  1
  // CHECK-DAG: t :  1

  return 0;
}
