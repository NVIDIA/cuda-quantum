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
// CHECK:           %[[VAL_3:.*]] = cc.extract_value %[[VAL_0]][2] : (!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>) -> !quake.veq<?>
// CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_3]] : (!quake.veq<?>) -> i64
// CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (i64)) {
// CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_4]] : i64
// CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : i64)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_8:.*]]: i64):
// CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:             quake.h %[[VAL_9]] : (!quake.ref) -> ()
// CHECK:             cc.continue %[[VAL_8]] : i64
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
// CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_1]] : i64
// CHECK:             cc.continue %[[VAL_11]] : i64
// CHECK:           } {invariant}
// CHECK:           %[[VAL_12:.*]] = cc.extract_value %[[VAL_0]][2] : (!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>) -> !quake.veq<?>
// CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_12]][0] : (!quake.veq<?>) -> !quake.ref
// CHECK:           call @__nvqpp__mlirgen__function_hello._Z5helloRN5cudaq5quditILm2EEE(%[[VAL_13]]) : (!quake.ref) -> ()
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
// CHECK:           %[[VAL_7:.*]] = cc.undef !cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>
// CHECK:           %[[VAL_8:.*]] = cc.insert_value %[[VAL_2]], %[[VAL_7]][0] : (!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>, i32) -> !cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>
// CHECK:           %[[VAL_9:.*]] = cc.insert_value %[[VAL_1]], %[[VAL_8]][1] : (!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>, f64) -> !cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>
// CHECK:           %[[VAL_10:.*]] = cc.insert_value %[[VAL_6]], %[[VAL_9]][2] : (!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>, !quake.veq<?>) -> !cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>
// CHECK:           call @__nvqpp__mlirgen__function_kernel._Z6kernel4test(%[[VAL_10]]) : (!cc.struct<"test" {i32, f64, !quake.veq<?>} [256,8]>) -> ()
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @_Z5entryi(
// CHECK-SAME:                         %[[VAL_0:.*]]: i32) attributes {no_this} {
// CHECK:           return
// CHECK:         }