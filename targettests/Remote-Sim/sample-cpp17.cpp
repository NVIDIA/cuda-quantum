/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: c++17
// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ %cpp_std --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t 
// RUN: nvq++ %cpp_std --enable-mlir --target remote-mqpu --remote-mqpu-auto-launch 1 %s -o %t && %t
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>

template <std::size_t N>
struct ghz {
  auto operator()() __qpu__ {
    cudaq::qarray<N> q;
    h(q[0]);
    for (int i = 0; i < N - 1; i++) {
      cx(q[i], q[i + 1]);
    }
    mz(q);
  }
};

int main() {
  {
    auto kernel = ghz<10>{};
    auto counts = cudaq::sample(kernel);
    counts.dump();
    REMOTE_TEST_ASSERT(counts.size() == 2);
  }
  {
    // Kernels as lambda functions
    const auto ghz = [](std::size_t N) __qpu__ {
      cudaq::qvector q(N);
      h(q[0]);
      for (int i = 0; i < N - 1; i++) {
        cx(q[i], q[i + 1]);
      }
      mz(q);
    };
    auto counts = cudaq::sample(ghz, 3);
    counts.dump();
    REMOTE_TEST_ASSERT(counts.size() == 2);
  }
  return 0;
}
