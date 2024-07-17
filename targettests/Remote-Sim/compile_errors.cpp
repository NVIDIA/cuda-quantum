/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// clang-format off
// RUN: nvq++ %cpp_std --target remote-mqpu %s 2>&1 | FileCheck %s
// clang-format on

#include <cudaq.h>

struct hGateTest {
  // Multi-level nested vectors (> 2) is not supported by the library-mode args
  // wrapper.
  auto operator()(
      const std::vector<std::vector<std::vector<double>>> &angles) __qpu__ {
    cudaq::qubit q;
    ry(angles[0][0][0], q);
    rx(angles[1][0][0], q);
    h(q);
  }
};

int main() {
  auto counts = cudaq::sample(
      hGateTest{},
      std::vector<std::vector<std::vector<double>>>{{{M_PI_2}}, {{M_PI}}});
  // CHECK: Argument type can't be serialized.
  counts.dump();
  return 0;
}
