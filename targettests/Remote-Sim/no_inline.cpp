/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test remote simulation without inlining

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ --enable-mlir -fno-aggressive-inline --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>
#include <iostream>

struct baz {
  __qpu__ void operator()(cudaq::qubit &q) { h(q); }
};

__qpu__ void bar(cudaq::qubit &q) { z(q); }

struct foo {
  template <typename CallableKernel>
  __qpu__ void operator()(CallableKernel &&func, int size) {
    cudaq::qvector q(size);
    // Check inlining of callable arguments or global scope functions.
    func(q[0]);
    bar(q[0]);
    h(q[0]);
    auto result = mz(q[0]);
  }
};

int main() {
  auto result = cudaq::sample(1000, foo{}, baz{}, /*qreg size*/ 1);
  std::cout << result.most_probable() << '\n';
  REMOTE_TEST_ASSERT("1" == result.most_probable());
  return 0;
}
