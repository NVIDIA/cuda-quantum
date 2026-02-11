/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %s -o %t && %t 2>&1 | FileCheck %s
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t 2>&1 | FileCheck %s

#include <cudaq.h>

struct test_kernel {
  auto operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    auto res = mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(test_kernel{});
  counts.dump();
  return 0;
}

// CHECK: WARNING: Kernel "test_kernel" uses named measurement results
