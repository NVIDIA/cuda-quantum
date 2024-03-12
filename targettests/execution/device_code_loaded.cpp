/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -std=c++17 --enable-mlir %s -o %t && %t | FileCheck %s
// RUN: if [ $(echo %cpp_std | cut -c4- ) -ge 20 ]; then \
// RUN:   nvq++ --enable-mlir %s -o %t && %t | FileCheck %s; \
// RUN: fi

#include <cudaq.h>

// CHECK: { [[B0:.*]]:[[C0:.*]] [[B1:.*]]:[[C1:.*]] }
// CHECK-NEXT: module {{.*}} func.func @__nvqpp__mlirgen__ghz{{.*}}(%arg0: i32{{.*}}) attributes {

// Define a quantum kernel
struct ghz {
  auto operator()(const int N) __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  auto counts = cudaq::sample(ghz{}, 3);
  counts.dump();

  printf("%s\n", cudaq::get_quake(ghz{}).data());
  return 0;
}
