/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %s -o %t && env CUDAQ_MLIR_NUM_THREADS=4 %t | FileCheck %s
// RUN: env CUDAQ_MLIR_NUM_THREADS=abc not %t 2>&1 | FileCheck %s --check-prefix=INVALID
// clang-format on

#include <algorithm>
#include <cudaq.h>
#include <iostream>
#include <string>
#include <vector>

int main() {
  try {
    // Builder kernels are compiled at runtime, so their pass pipelines run on
    // the thread pool selected by CUDAQ_MLIR_NUM_THREADS.
    auto kernel = cudaq::make_kernel();
    auto q = kernel.qalloc(3);
    kernel.h(q[0]);
    kernel.x<cudaq::ctrl>(q[0], q[1]);
    kernel.x<cudaq::ctrl>(q[1], q[2]);
    kernel.mz(q);

    auto counts = cudaq::sample(kernel);
    std::vector<std::string> outcomes;
    for (auto &&[bits, count] : counts)
      outcomes.push_back(bits);
    std::sort(outcomes.begin(), outcomes.end());
    for (auto &bits : outcomes)
      std::cout << bits << '\n';
  } catch (const std::exception &e) {
    std::cout << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}

// CHECK: 000
// CHECK-NEXT: 111
// CHECK-NOT: {{[01]+}}

// INVALID: error: Invalid CUDAQ_MLIR_NUM_THREADS value: 'abc'
