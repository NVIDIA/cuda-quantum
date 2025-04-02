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

// TODO: merge this with the `cudaq_run.cpp` test once the QIR lowering pipeline
// can handle non-constant expressions.

#include <cudaq.h>

__qpu__ int test_kernel(int count) {
  unsigned result = 0;
  cudaq::qvector v(count);
  h(v);
  // Note: Hardware QIR lowering pipeline cannot handle non-constant
  // expressions.
  for (int i = 0; i < count; i++)
    result += 1;
  return result;
}

int main() {
  int c = 0;
  {
    std::vector<int> results =
        cudaq::run<int>(100, std::function<int(int)>{test_kernel}, 4);
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results) {
        printf("%d: %d\n", c++, i);
        if (i != 4)
          break;
      }
      if (c == 100)
        printf("success!\n");
    }
  }

  return 0;
}

// CHECK: success!
