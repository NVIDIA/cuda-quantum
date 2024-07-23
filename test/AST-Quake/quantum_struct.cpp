/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %cpp_std %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

struct test {
  int i;
  double d;
  cudaq::qview<> q;
};

__qpu__ void hello(cudaq::qubit &q) { h(q); }

__qpu__ void kernel(test t) {
  h(t.q);
  hello(t.q[0]);
}

__qpu__ void entry(int i) {
  cudaq::qvector q(i);
  test tt{1, 2.2, q};
  kernel(tt);
}

int main() { cudaq::sample(entry, 4); }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_hello._Z5helloRN5cudaq5quditILm2EEE(
// CHECK-SAME:                                                                              %[[VAL_0:.*]]: !quake.ref) attributes {"cudaq-kernel", no_this} {
// CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_kernel._Z6kernel4test(
// CHECK-SAME:                                                                %[[VAL_0:.*]]: !cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>) attributes {"cudaq-kernel", no_this} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
// CHECK:           %[[VAL_3:.*]] = cc.alloca !cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>>
// CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_3]][2] : (!cc.ptr<!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>>) -> !cc.ptr<!quake.veq<?>>
// CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<!quake.veq<?>>
// CHECK:           %[[VAL_6:.*]] = quake.veq_size %[[VAL_5]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_7:.*]] = cc.loop while ((%[[VAL_8:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_6]] : i64
// CHECK:             cc.condition %[[VAL_9]](%[[VAL_8]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
// CHECK:             %[[VAL_11:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_10]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             quake.h %[[VAL_11]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_10]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_12:.*]]: i64):
// CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_13]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_3]][2] : (!cc.ptr<!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>>) -> !cc.ptr<!quake.veq<?>>
// CHECK:           %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<!quake.veq<?>>
// CHECK:           %[[VAL_16:.*]] = quake.extract_ref %[[VAL_15]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__function_hello._Z5helloRN5cudaq5quditILm2EEE(%[[VAL_16]]) : (!quake.ref) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_entry._Z5entryi(
// CHECK-SAME:                                                          %[[VAL_0:.*]]: i32) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_1:.*]] = arith.constant 2.200000e+00 : f64
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[VAL_3:.*]] = cc.alloca i32
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i32>
// CHECK:           %[[VAL_5:.*]] = cc.cast signed %[[VAL_4]] : (i32) -> i64
// CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_5]] : i64]
// CHECK:           %[[VAL_7:.*]] = cc.alloca !cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>>) -> !cc.ptr<i32>
// CHECK:           cc.store %[[VAL_2]], %[[VAL_8]] : !cc.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = cc.compute_ptr %[[VAL_7]][1] : (!cc.ptr<!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>>) -> !cc.ptr<f64>
// CHECK:           cc.store %[[VAL_1]], %[[VAL_9]] : !cc.ptr<f64>
// CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_7]][2] : (!cc.ptr<!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>>) -> !cc.ptr<!quake.veq<?>>
// CHECK:           cc.store %[[VAL_6]], %[[VAL_10]] : !cc.ptr<!quake.veq<?>>
// CHECK:           %[[VAL_11:.*]] = cc.load %[[VAL_7]] : !cc.ptr<!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>>
// CHECK:           call @__nvqpp__mlirgen__function_kernel._Z6kernel4test(%[[VAL_11]]) : (!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_Z5entryi(
// CHECK-SAME:                         %[[VAL_0:.*]]: i32) attributes {no_this} {
// CHECK:           return
// CHECK:         }