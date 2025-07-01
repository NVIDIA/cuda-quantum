/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | FileCheck %s

#include <cudaq.h>

struct simple_pod {
  long long a;
  long long b;
};

// Test POD default constructor - should just allocate memory
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test_pod_constructor._Z20test_pod_constructorv
// CHECK:           () -> !cc.struct<"simple_pod" {i64, i64} [128,8]>
// CHECK:           %[[ALLOC:.*]] = cc.alloca !cc.struct<"simple_pod" {i64, i64} [128,8]>
// CHECK-NOT:       call @_ZN10simple_podC1Ev
// CHECK:           return %[[RESULT:.*]] : !cc.struct<"simple_pod" {i64, i64} [128,8]>
simple_pod test_pod_constructor() __qpu__ {
  simple_pod result;
  result.a = 9;
  result.b = 10;
  return result;
}

int main() {
  auto result = cudaq::run(5, test_pod_constructor);

  return 0;
}
