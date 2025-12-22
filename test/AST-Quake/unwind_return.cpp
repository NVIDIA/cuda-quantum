/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>
#include <iostream>

__qpu__ int test_kernel(int count) {
  cudaq::qvector v(count);
  h(v[0]);
  for (int i = 0; i < count - 1; i++)
    cx(v[i], v[i + 1]);
  auto result = mz(v);
  if (result[0])
    return 1;
  else
    return 2;
}

int main() {
  {
    constexpr int numQubits = 4;
    auto results = cudaq::run(100, test_kernel, numQubits);
    if (results.size() != 100) {
      printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
    } else {
      for (auto result : results) {
        std::cout << "Result: " << result << "\n";
      }
    }
  }

  return 0;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test_kernel._Z11test_kerneli(
// CHECK-SAME:      %[[VAL_0:.*]]: i32) -> i32 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           cc.if(%{{.*}}) {
// CHECK:             cc.unwind_return %[[VAL_2]] : i32
// CHECK:           } else {
// CHECK:             cc.unwind_return %[[VAL_1]] : i32
// CHECK:           }
// CHECK:           %[[VAL_28:.*]] = cc.undef i32
// CHECK:           return %[[VAL_28]] : i32
// CHECK:         }
