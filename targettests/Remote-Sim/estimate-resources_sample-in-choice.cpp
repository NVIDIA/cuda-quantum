/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// UNSUPPORTED: darwin-arm64
// clang-format off
// RUN: nvq++ --target remote-mqpu                             %s -o %t && %t
// RUN: nvq++ --target remote-mqpu --remote-mqpu-auto-launch 2 %s -o %t && %t
// clang-format on

// Note: This test is marked UNSUPPORTED (not XFAIL) on macOS ARM64 due to a
// known LLVM bug where C++ exceptions thrown from JIT-compiled code cannot be
// caught. When the exception crashes the main process, the spawned cudaq-qpud
// daemon processes are left orphaned and running, which can cause resource
// leaks and issues in the test infrastructure.
// See: https://github.com/llvm/llvm-project/issues/49036

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
  std::function<bool()> choice = [&](){
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
  assert(exception_thrown);

  return 0;
}
