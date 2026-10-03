/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Test that a single element initializer_list is materialized as an array.

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

__qpu__ void testSingletonVector() {
  cudaq::qarray<3> q;
  std::vector<int> index = {2};
  ry(M_PI_2, q[index[0]]);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_testSingletonVector._Z19testSingletonVectorv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 1.5707963267948966 : f64
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.array<i32 x 1>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<i32 x 1>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<i32 x 1>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.cast signed %[[VAL_6]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = quake.extract_ref %[[VAL_2]]{{\[}}%[[VAL_7]]] : (!quake.veq<3>, i64) -> !quake.ref
// CHECK:           quake.ry (%[[VAL_0]]) %[[VAL_8]] : (f64, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ void testSingletonArray() {
  cudaq::qarray<3> q;
  int index[1] = {2};
  ry(M_PI_2, q[index[0]]);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_testSingletonArray._Z18testSingletonArrayv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 1.5707963267948966 : f64
// CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 2 : i32
// CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<3>
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.array<i32 x 1>
// CHECK:           %[[VAL_4:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<i32 x 1>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.cast %[[VAL_3]] : (!cc.ptr<!cc.array<i32 x 1>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.cast signed %[[VAL_6]] : (i32) -> i64
// CHECK:           %[[VAL_8:.*]] = quake.extract_ref %[[VAL_2]]{{\[}}%[[VAL_7]]] : (!quake.veq<3>, i64) -> !quake.ref
// CHECK:           quake.ry (%[[VAL_0]]) %[[VAL_8]] : (f64, !quake.ref) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on
