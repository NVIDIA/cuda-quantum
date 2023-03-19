/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: nvq++ --enable-mlir --platform default-qpud %s -o out_testqpudsample.x && ./out_testqpudsample.x | FileCheck %s && rm out_testqpudsample.x

#include <cudaq.h>

// CHECK: { [[B0:.*]]:[[C0:.*]] [[B1:.*]]:[[C1:.*]] }

// Define a quantum kernel
struct ghz {
  auto operator()(const int N) __qpu__ {
    cudaq::qreg q(N);
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
  return 0;
}
