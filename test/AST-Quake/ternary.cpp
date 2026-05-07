/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
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
  auto results = mz(v);
  int acc = 0;
  for (auto result : results)
    acc += (result ? 1 : 0);
  return acc;
}

int main() {
  constexpr int numQubits = 4;
  auto results = cudaq::run(100, test_kernel, numQubits);
  if (results.size() != 100) {
    printf("FAILED! Expected 100 shots. Got %lu\n", results.size());
  } else {
    for (auto result : results)
      std::cout << "Result: " << result << "\n";
  }
  return 0;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test_kernel._Z11test_kerneli(
// CHECK-DAG:        %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK-DAG:        %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:             cc.scope {
// CHECK:               %[[VAL_35:.*]] = cc.alloca i1
// CHECK:               %[[VAL_36:.*]] = cc.load %[[VAL_35]] : !cc.ptr<i1>
// CHECK:               %[[VAL_37:.*]] = cc.if(%[[VAL_36]]) -> i32 {
// CHECK:                 cc.continue %[[VAL_3]] : i32
// CHECK:               } else {
// CHECK:                 cc.continue %[[VAL_4]] : i32
// CHECK:               }
// CHECK:             }
// CHECK:             cc.continue %{{.*}} : i64
