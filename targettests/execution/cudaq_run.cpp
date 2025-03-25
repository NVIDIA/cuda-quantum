/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: nvq++ -fenable-cudaq-run %cpp_std %s -o %t && %t | FileCheck %s

#include <cudaq.h>

__qpu__ int nullary_test() {
  unsigned result = 0;
  cudaq::qvector v(8);
  h(v);
  z(v);
  for (int i = 0; i < 8; i++) {
    bool w = mz(v[i]);
    result |= ((unsigned)w) << (8 - 1 - i);
  }
  return result;
}

__qpu__ int unary_test(int count) {
  unsigned result = 0;
  cudaq::qvector v(count);
  h(v);
  z(v);
  for (int i = 0; i < count; i++) {
    bool w = mz(v[i]);
    result |= ((unsigned)w) << (count - 1 - i);
  }
  return result;
}

int main() {
  int c = 0;
  {
    std::vector<int> results =
        cudaq::run<int>(100, std::function<int()>{nullary_test});
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }

  {
    std::vector<int> results =
        cudaq::run<int>(50, std::function<int(int)>{unary_test}, 4);
    c = 0;
    if (results.size() != 50) {
      printf("FAILED! Expected 50 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }

  return 0;
}

// CHECK: success!
// CHECK: success!
