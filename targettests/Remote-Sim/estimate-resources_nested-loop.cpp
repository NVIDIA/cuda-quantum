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

template <std::size_t N>
struct mykernel {
  auto operator()() __qpu__ {

    cudaq::qarray<N> q;
    for (size_t i = 0; i < N - 1; i++) {
      h(q[i]);
      for (size_t j = i + 1; j < N; j++) {
        x<cudaq::ctrl>(q[i], q[j]);
      }
    }
    mz(q);
  }
};

int main() {
  auto kernel = mykernel<10>{};
  auto counts = cudaq::estimate_resources(kernel);

  counts.dump();
  // CHECK: Total # of gates: 54
  // CHECK-DAG: h :  9
  // CHECK-DAG: cx :  45

  return 0;
}
