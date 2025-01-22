/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ %cpp_std --target quera %s -o %t.x
// RUN: not %t.x 2>&1 | FileCheck %s
// RUN: nvq++ %cpp_std --target quera --emulate %s -o %t.x
// RUN: not %t.x 2>&1 | FileCheck %s

#include <cudaq.h>

auto bell = []() __qpu__ {
  cudaq::qvector q(2);
  h(q[0]);
  x<cudaq::ctrl>(q[0], q[1]);
};

int main() {
  auto counts = cudaq::sample(bell);
  counts.dump();
  return 0;
}

// CHECK: Arbitrary kernel execution is not supported on this target
