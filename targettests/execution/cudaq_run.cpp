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
    const auto results = cudaq::run(100, nullary_test);
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }

  {
    const auto results = cudaq::run(50, unary_test, 4);
    c = 0;
    if (results.size() != 50) {
      printf("FAILED! Expected 50 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }

  // Run async
  {
    const auto results =
        cudaq::run_async(/*qpu_id=*/0, 100, nullary_test).get();
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      for (auto i : results)
        printf("%d: %d\n", c++, i);
      printf("success!\n");
    }
  }

  {
    const auto results =
        cudaq::run_async(/*qpu_id=*/0, 50, unary_test, 4).get();
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
// CHECK: success!
// CHECK: success!
