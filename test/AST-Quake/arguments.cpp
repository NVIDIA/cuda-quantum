/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include <cudaq.h>

__qpu__ void kernel_a(long a, unsigned long b) {}

__qpu__ void kernel_b(short a, unsigned short b) {}

__qpu__ void test_kernel() {
  int a0 = 4;
  int a1 = 7;
  kernel_a(a0, a1);
  kernel_a(-4, -7);
  int a2 = 42;
  int a3 = 72;
  kernel_b(a2, a3);
  kernel_b(-42, -72);
  unsigned a4 = 37u;
  unsigned a5 = 88u;
  kernel_a(a4, a5);
  kernel_a(0xDeadBeefu, 0xCafeBabeu);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test_kernel._Z11test_kernelv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 3405691582 : i64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 3735928559 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant -72 : i16
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant -42 : i16
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant -7 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant -4 : i64
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 88 : i32
// CHECK-DAG:       %[[VAL_7:.*]] = arith.constant 37 : i32
// CHECK-DAG:       %[[VAL_8:.*]] = arith.constant 72 : i32
// CHECK-DAG:       %[[VAL_9:.*]] = arith.constant 42 : i32
// CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 7 : i32
// CHECK-DAG:       %[[VAL_11:.*]] = arith.constant 4 : i32
// CHECK-DAG:       %[[VAL_12:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_11]], %[[VAL_12]] : !cc.ptr<i32>
// CHECK:           %[[VAL_13:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_10]], %[[VAL_13]] : !cc.ptr<i32>
// CHECK:           %[[VAL_14:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i32>
// CHECK:           %[[VAL_15:.*]] = cc.cast signed %[[VAL_14]] : (i32) -> i64
// CHECK:           %[[VAL_16:.*]] = cc.load %[[VAL_13]] : !cc.ptr<i32>
// CHECK:           %[[VAL_17:.*]] = cc.cast signed %[[VAL_16]] : (i32) -> i64
// CHECK:           call @__nvqpp__mlirgen__function_kernel_a._Z8kernel_alm(%[[VAL_15]], %[[VAL_17]]) : (i64, i64) -> ()
// CHECK:           call @__nvqpp__mlirgen__function_kernel_a._Z8kernel_alm(%[[VAL_5]], %[[VAL_4]]) : (i64, i64) -> ()
// CHECK:           %[[VAL_18:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_9]], %[[VAL_18]] : !cc.ptr<i32>
// CHECK:           %[[VAL_19:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_8]], %[[VAL_19]] : !cc.ptr<i32>
// CHECK:           %[[VAL_20:.*]] = cc.load %[[VAL_18]] : !cc.ptr<i32>
// CHECK:           %[[VAL_21:.*]] = cc.cast %[[VAL_20]] : (i32) -> i16
// CHECK:           %[[VAL_22:.*]] = cc.load %[[VAL_19]] : !cc.ptr<i32>
// CHECK:           %[[VAL_23:.*]] = cc.cast %[[VAL_22]] : (i32) -> i16
// CHECK:           call @__nvqpp__mlirgen__function_kernel_b._Z8kernel_bst(%[[VAL_21]], %[[VAL_23]]) : (i16, i16) -> ()
// CHECK:           call @__nvqpp__mlirgen__function_kernel_b._Z8kernel_bst(%[[VAL_3]], %[[VAL_2]]) : (i16, i16) -> ()
// CHECK:           %[[VAL_24:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_7]], %[[VAL_24]] : !cc.ptr<i32>
// CHECK:           %[[VAL_25:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_6]], %[[VAL_25]] : !cc.ptr<i32>
// CHECK:           %[[VAL_26:.*]] = cc.load %[[VAL_24]] : !cc.ptr<i32>
// CHECK:           %[[VAL_27:.*]] = cc.cast unsigned %[[VAL_26]] : (i32) -> i64
// CHECK:           %[[VAL_28:.*]] = cc.load %[[VAL_25]] : !cc.ptr<i32>
// CHECK:           %[[VAL_29:.*]] = cc.cast unsigned %[[VAL_28]] : (i32) -> i64
// CHECK:           call @__nvqpp__mlirgen__function_kernel_a._Z8kernel_alm(%[[VAL_27]], %[[VAL_29]]) : (i64, i64) -> ()
// CHECK:           call @__nvqpp__mlirgen__function_kernel_a._Z8kernel_alm(%[[VAL_1]], %[[VAL_0]]) : (i64, i64) -> ()
// CHECK:           return
// CHECK:         }

