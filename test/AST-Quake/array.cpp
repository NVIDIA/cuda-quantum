/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

struct T {
   void operator()(int N) __qpu__ {
      cudaq::qreg Q(N);
      x(Q);
   }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__T(
// CHECK-SAME:         %[[VAL_0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = arith.extsi %[[VAL_4]] : i32 to i64
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_5]] : i64]
// CHECK:           %[[VAL_7:.*]] = quake.veq_size %[[VAL_6]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_7]] : i64 to index
// CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_2]]) -> (index)) {
// CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_8]] : index
// CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_12:.*]]: index):
// CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_6]]{{\[}}%[[VAL_12]]] : (!quake.veq<?>, index) -> !quake.ref
// CHECK:             quake.x %[[VAL_13]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_12]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_14:.*]]: index):
// CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_1]] : index
// CHECK:             cc.continue %[[VAL_15]] : index
// CHECK:           } {invariant}
// CHECK:           return
// CHECK:         }

struct S {
   void operator()() __qpu__ {
      int arr[3] = {4, 8, 10};
      T{}(arr[0]);
      T{}(arr[1]);
      T{}(arr[2]);
   }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.array<i32 x 3>
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_3]][0] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_3]][1] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_3]][2] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.alloca !cc.struct<"T" {}>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_3]][0] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           call @__nvqpp__mlirgen__T(%[[VAL_9]]) : (i32) -> ()
// CHECK:           %[[VAL_10:.*]] = cc.alloca !cc.struct<"T" {}>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_3]][1] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = cc.load %[[VAL_11]] : !cc.ptr<i32>
// CHECK:           call @__nvqpp__mlirgen__T(%[[VAL_12]]) : (i32) -> ()
// CHECK:           %[[VAL_13:.*]] = cc.alloca !cc.struct<"T" {}>
// CHECK:           %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_3]][2] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<i32>
// CHECK:           call @__nvqpp__mlirgen__T(%[[VAL_15]]) : (i32) -> ()
// CHECK:           return
// CHECK:         }

struct S1 {
   void operator()() __qpu__ {
      int arr[] = {4, 8, 10};
      T{}(arr[0]);
      T{}(arr[1]);
      T{}(arr[2]);
   }
};

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__S1() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
// CHECK:           %[[VAL_0:.*]] = arith.constant 4 : i32
// CHECK:           %[[VAL_1:.*]] = arith.constant 8 : i32
// CHECK:           %[[VAL_2:.*]] = arith.constant 10 : i32
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.array<i32 x 3>
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_3]][0] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.compute_ptr %[[VAL_3]][1] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_5]] : !cc.ptr<i32>
// CHECK:           %[[VAL_6:.*]] = cc.compute_ptr %[[VAL_3]][2] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_6]] : !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.alloca !cc.struct<"T" {}>
// CHECK:           %[[VAL_8:.*]] = cc.compute_ptr %[[VAL_3]][0] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           call @__nvqpp__mlirgen__T(%[[VAL_9]]) : (i32) -> ()
// CHECK:           %[[VAL_10:.*]] = cc.alloca !cc.struct<"T" {}>
// CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_3]][1] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_12:.*]] = cc.load %[[VAL_11]] : !cc.ptr<i32>
// CHECK:           call @__nvqpp__mlirgen__T(%[[VAL_12]]) : (i32) -> ()
// CHECK:           %[[VAL_13:.*]] = cc.alloca !cc.struct<"T" {}>
// CHECK:           %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_3]][2] : (!cc.ptr<!cc.array<i32 x 3>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<i32>
// CHECK:           call @__nvqpp__mlirgen__T(%[[VAL_15]]) : (i32) -> ()
// CHECK:           return
// CHECK:         }

