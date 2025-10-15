/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ %cpp_std %s |& FileCheck %s -check-prefix=FAIL
// clang-format on

#include <cudaq.h>
#include <iostream>

struct Foo {
  int bar;
  std::vector<bool> baz;
};

struct Quark {
  Foo operator()() __qpu__ {
    cudaq::qvector q(3);
    return {747, mz(q)};
  }
};

int main() {
  const auto results = cudaq::run(3, Quark{});
  return 0;
}

// FAIL: error: kernel result type not supported
