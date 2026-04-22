/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// REQUIRES: remote-sim

// clang-format off
// RUN: nvq++ --target remote-mqpu --remote-mqpu-auto-launch 4 %s -o %t && %t | FileCheck %s
// clang-format on

#include "remote_test_assert.h"
#include <cudaq.h>

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
  for (int q = 0; q < 4; q++) {
    const auto results =
        cudaq::run_async(/*qpu_id=*/q, 10, unary_test, 4).get();
    int c = 0;
    if (results.size() != 10) {
      printf("FAILED! Expected 10 shots. Got %lu\n", results.size());
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
