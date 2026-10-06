/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | FileCheck %s

// A kernel that calls another kernel owns the heap storage of the dynamic
// result that it gets back. It moves it to its own stack and frees the heap
// storage. For a result that is nested, such as a vector of vectors, that is
// done for each of the inner vectors as well as for the outer one. Otherwise,
// the storage of the inner vectors would be leaked.

#include <cudaq.h>

std::vector<int> flat(int n) __qpu__ {
  std::vector<int> v(n);
  for (int i = 0; i < n; ++i)
    v[i] = i;
  return v;
}

int callerFlat() __qpu__ { return flat(4)[3]; }

std::vector<std::vector<int>> nested() __qpu__ { return {{1, 2}, {3, 4}}; }

int callerNested() __qpu__ { return nested()[1][1]; }

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_callerFlat.
// CHECK:           %[[R:.*]] = call @__nvqpp__mlirgen__function_flat.{{.*}} : (i32) -> !cc.sequence<i32>
// CHECK:           %[[N:.*]] = cc.sequence_size %[[R]] : (!cc.sequence<i32>) -> i64
// CHECK:           %[[BUF:.*]] = cc.alloca i32{{\[}}%[[N]] : i64]
// CHECK:           call @__nvqpp_vectorCopyToStack(
// CHECK-NOT:       cc.loop
// CHECK:           cc.compute_ptr %[[BUF]][3]

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_callerNested.
// CHECK:           %[[RR:.*]] = call @__nvqpp__mlirgen__function_nested.{{.*}} : () -> !cc.sequence<!cc.sequence<i32>>
// CHECK:           %[[RRN:.*]] = cc.sequence_size %[[RR]] : (!cc.sequence<!cc.sequence<i32>>) -> i64
// CHECK:           cc.alloca !cc.sequence<i32>{{\[}}%[[RRN]] : i64]
// CHECK:           cc.loop while
// CHECK:           cc.alloca i32{{\[}}
// CHECK:           call @__nvqpp_vectorCopyToStack(
// CHECK:           call @free(
