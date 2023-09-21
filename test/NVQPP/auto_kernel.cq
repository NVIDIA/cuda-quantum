/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// FIXME: enable this for all architectures,
// see https://github.com/NVIDIA/cuda-quantum/issues/553.

// REQUIRES: x64_86-registered-target
// RUN: nvq++ --enable-mlir -v %s -o out_auto_kernel.x && ./out_auto_kernel.x | FileCheck %s

#include <cudaq.h>

// CHECK: size 3
// CHECK: 0: {{[tf]}}
// CHECK: 1: {{[tf]}}
// CHECK: 2: {{[tf]}}

struct ak2 {
  auto operator()() __qpu__ {
    cudaq::qreg<3> q;
    x(q[0]);
    h(q[1]);
    x<cudaq::ctrl>(q[1], q[2]);
    x<cudaq::ctrl>(q[0], q[1]);
    h(q[0]);
    x(q[1]);
    y(q[2]);
    return mz(q);
  }
};

int main() {
  auto counts = ak2{}();

  printf("size %zu\n", counts.size());
  for (std::size_t i = 0, I = counts.size(); i < I; i++) {
    printf("%zu: %s\n", i, (counts[i] ? "true" : "false"));
  }
  return 0;
}
