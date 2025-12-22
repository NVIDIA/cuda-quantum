/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct VectorIntReturn {
  std::vector<int> operator()() __qpu__ { return {142, 243}; }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorIntReturn() -> !cc.stdvec<i32> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 4 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 142 : i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 243 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_4:.*]] = cc.alloca !cc.array<i32 x 2>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<i32 x 2>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_4]][1] : (!cc.ptr<!cc.array<i32 x 2>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<i32 x 2>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_8:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_7]], %[[VAL_3]], %[[VAL_0]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_9:.*]] = cc.stdvec_init %[[VAL_8]], %[[VAL_3]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i32>
// CHECK:           return %[[VAL_9]] : !cc.stdvec<i32>
// CHECK:         }
// clang-format on

struct VectorIntResult {
  std::vector<int> operator()() __qpu__ {
    std::vector<int> result(2);
    result[0] = 42;
    return result;
  }
};

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__VectorIntResult() -> !cc.stdvec<i32> attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 4 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 42 : i32
// CHECK-DAG:       %[[VAL_3:.*]] = cc.alloca !cc.array<i32 x 2>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<i32 x 2>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<i32 x 2>>) -> !cc.ptr<i8>
// CHECK:           %[[VAL_6:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_5]], %[[VAL_0]], %[[VAL_1]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_init %[[VAL_6]], %[[VAL_0]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i32>
// CHECK:           return %[[VAL_7]] : !cc.stdvec<i32>
// CHECK:         }
// clang-format on
