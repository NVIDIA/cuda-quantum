/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: nvq++ --target quantinuum --emulate %s -o %t && %t | FileCheck %s
// XFAIL: darwin-arm64
// clang-format on

// Note: This test fails on macOS ARM64 due to a known LLVM bug where C++
// exceptions thrown from JIT-compiled code cannot be caught. This is caused
// by libunwind issues on Darwin ARM64.
// See: https://github.com/llvm/llvm-project/issues/49036

#include <cstdio>
#include <cudaq.h>
#include <cudaq/algorithms/resource_estimation.h>

struct mykernel {
  auto operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
    auto m1 = mz(q);
  }
};

int main() {
  auto kernel = mykernel{};
  std::function<bool()> choice = [&]() {
    auto counts1 = cudaq::sample(5, kernel);
    counts1.dump();
    return true;
  };
  auto exception_thrown = false;
  try {
    auto gateCounts = cudaq::estimate_resources(choice, kernel);
    gateCounts.dump();
  } catch (...) {
    exception_thrown = true;
  }
  if (exception_thrown)
    printf("success\n");
  else
    printf("FAILED!\n");

  return 0;
}

// CHECK: success
