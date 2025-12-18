/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target=qpp-cpu %s -o=%t
// RUN: nvq++ --target qpp-cpu %s -o %t && CUDAQ_LOG_LEVEL=info %t | FileCheck --check-prefix=CHECK-QPP %s
// RUN: CUDAQ_DEFAULT_SIMULATOR="density-matrix-cpu" nvq++ %s -o %t && CUDAQ_LOG_LEVEL=info %t | FileCheck --check-prefix=CHECK-DM %s
// RUN: CUDAQ_DEFAULT_SIMULATOR="foo" nvq++ %s -o %t && CUDAQ_LOG_LEVEL=info %t | FileCheck %s
// RUN: CUDAQ_DEFAULT_SIMULATOR="qpp-cpu" nvq++ --target quantinuum --emulate %s -o %t && CUDAQ_LOG_LEVEL=info %t | FileCheck --check-prefix=CHECK-QPP %s
// clang-format on

#include <cudaq.h>

struct ghz {
  auto operator()(int N) __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(ghz{}, 4);
  counts.dump();
  return 0;
}

// CHECK-QPP: [info] [NVQIR.cpp:{{[0-9]+}}] Creating the qpp backend.
// CHECK-QPP: [info] [DefaultExecutionManager.cpp:{{[0-9]+}}] [DefaultExecutionManager] Creating the qpp backend.

// CHECK-DM: [info] [NVQIR.cpp:{{[0-9]+}}] Creating the dm backend.
// CHECK-DM: [info] [DefaultExecutionManager.cpp:{{[0-9]+}}] [DefaultExecutionManager] Creating the dm backend.

// CHECK-NOT: foo
