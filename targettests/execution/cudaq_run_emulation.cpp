/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim
// REQUIRES: c++20

// clang-format off
// RUN: nvq++ -fenable-cudaq-run %cpp_std --target quantinuum --emulate  %s -o %t && %t | FileCheck %s
// clang-format on

#include <cudaq.h>

__qpu__ int test_kernel(int count) {
  unsigned result = 0;
  cudaq::qvector v(count);
  h(v[0]);
  for (int i = 0; i < count - 1; i++)
    cx(v[i], v[i + 1]);
  for (int i = 0; i < count; i++)
    if (mz(v[i]))
      result += 1;
  return result;
}

int main() {
  int c = 0;
  {
    constexpr int numQubits = 4;
    auto results = cudaq::run(100, test_kernel, numQubits);
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results) {
        printf("%d: %d\n", c++, i);
        if (i != 0 && i != 4)
          break;
      }
      if (c == 100)
        printf("success!\n");
    }
  }

  return 0;
}

// CHECK: success!
