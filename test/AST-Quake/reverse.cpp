/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt -cse | FileCheck %s

#include <cudaq.h>

__qpu__ int std_reverse_std_vector_int() {
  std::vector<int> qr(10);

  std::reverse(qr.begin(), qr.end());
  int sum;
  for (auto &q : qr) {
    sum += q;
  }
  return sum;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_std_reverse_std_vector_int
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 10 : index
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 8 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca !cc.array<i32 x 10>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<i32 x 10>>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_6]][10] : (!cc.ptr<i32>) -> !cc.ptr<i32>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<i32>) -> i64
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<i32>) -> i64
// CHECK:           %[[VAL_10:.*]] = arith.subi %[[VAL_8]], %[[VAL_9]] : i64
// CHECK:           %[[VAL_11:.*]] = arith.divsi %[[VAL_10]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<i32>) -> !cc.ptr<!cc.array<i32 x ?>>
// CHECK:           %[[VAL_13:.*]] = arith.index_cast %[[VAL_11]] : i64 to index
// CHECK:           %[[VAL_14:.*]] = cc.loop while ((%[[VAL_15:.*]] = %[[VAL_3]]) -> (index)) {
// CHECK:             %[[VAL_16:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_13]] : index
// CHECK:             cc.condition %[[VAL_16]](%[[VAL_15]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_17:.*]]: index):
// CHECK:             %[[VAL_18:.*]] = arith.index_cast %[[VAL_17]] : index to i64
// CHECK:             %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_12]]{{\[}}%[[VAL_18]]] : (!cc.ptr<!cc.array<i32 x ?>>, i64) -> !cc.ptr<i32>
// CHECK:             %[[VAL_20:.*]] = arith.subi %[[VAL_11]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_18]] : i64
// CHECK:             %[[VAL_22:.*]] = cc.compute_ptr %[[VAL_12]]{{\[}}%[[VAL_21]]] : (!cc.ptr<!cc.array<i32 x ?>>, i64) -> !cc.ptr<i32>
// CHECK:             %[[VAL_23:.*]] = cc.load %[[VAL_19]] : !cc.ptr<i32>
// CHECK:             %[[VAL_24:.*]] = cc.load %[[VAL_22]] : !cc.ptr<i32>
// CHECK:             cc.store %[[VAL_23]], %[[VAL_22]] : !cc.ptr<i32>
// CHECK:             cc.store %[[VAL_24]], %[[VAL_19]] : !cc.ptr<i32>
// CHECK:             cc.continue %[[VAL_17]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_25:.*]]: index):
// CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_2]] : index
// CHECK:             cc.continue %[[VAL_26]] : index
// CHECK:           } {invariant}
// CHECK:           %[[VAL_27:.*]] = cc.alloca i32
// CHECK:           %[[VAL_28:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<i32 x 10>>) -> !cc.ptr<!cc.array<i32 x ?>>
// CHECK:           %[[VAL_29:.*]] = cc.loop while ((%[[VAL_30:.*]] = %[[VAL_3]]) -> (index)) {
// CHECK:             %[[VAL_31:.*]] = arith.cmpi slt, %[[VAL_30]], %[[VAL_0]] : index
// CHECK:             cc.condition %[[VAL_31]](%[[VAL_30]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_32:.*]]: index):
// CHECK:             %[[VAL_33:.*]] = arith.index_cast %[[VAL_32]] : index to i64
// CHECK:             %[[VAL_34:.*]] = cc.compute_ptr %[[VAL_28]]{{\[}}%[[VAL_33]]] : (!cc.ptr<!cc.array<i32 x ?>>, i64) -> !cc.ptr<i32>
// CHECK:             %[[VAL_35:.*]] = cc.load %[[VAL_34]] : !cc.ptr<i32>
// CHECK:             %[[VAL_36:.*]] = cc.load %[[VAL_27]] : !cc.ptr<i32>
// CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_36]], %[[VAL_35]] : i32
// CHECK:             cc.store %[[VAL_37]], %[[VAL_27]] : !cc.ptr<i32>
// CHECK:             cc.continue %[[VAL_32]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_38:.*]]: index):
// CHECK:             %[[VAL_39:.*]] = arith.addi %[[VAL_38]], %[[VAL_2]] : index
// CHECK:             cc.continue %[[VAL_39]] : index
// CHECK:           } {invariant}
// CHECK:           %[[VAL_40:.*]] = cc.load %[[VAL_27]] : !cc.ptr<i32>
// CHECK:           return %[[VAL_40]] : i32

__qpu__ double std_reverse_std_vector_double() {
  std::vector<double> qr(7);

  std::reverse(qr.begin(), qr.end());
  double sum;
  for (auto &q : qr) {
    sum += q;
  }
  return sum;
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_std_reverse_std_vector_double
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 7 : index
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 16 : i64
// CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca !cc.array<f64 x 7>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<f64 x 7>>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_7:.*]] = cc.compute_ptr %[[VAL_6]][7] : (!cc.ptr<f64>) -> !cc.ptr<f64>
// CHECK:           %[[VAL_8:.*]] = cc.cast %[[VAL_7]] : (!cc.ptr<f64>) -> i64
// CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<f64>) -> i64
// CHECK:           %[[VAL_10:.*]] = arith.subi %[[VAL_8]], %[[VAL_9]] : i64
// CHECK:           %[[VAL_11:.*]] = arith.divsi %[[VAL_10]], %[[VAL_4]] : i64
// CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_6]] : (!cc.ptr<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_13:.*]] = arith.index_cast %[[VAL_11]] : i64 to index
// CHECK:           %[[VAL_14:.*]] = cc.loop while ((%[[VAL_15:.*]] = %[[VAL_3]]) -> (index)) {
// CHECK:             %[[VAL_16:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_13]] : index
// CHECK:             cc.condition %[[VAL_16]](%[[VAL_15]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_17:.*]]: index):
// CHECK:             %[[VAL_18:.*]] = arith.index_cast %[[VAL_17]] : index to i64
// CHECK:             %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_12]]{{\[}}%[[VAL_18]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK:             %[[VAL_20:.*]] = arith.subi %[[VAL_11]], %[[VAL_1]] : i64
// CHECK:             %[[VAL_21:.*]] = arith.subi %[[VAL_20]], %[[VAL_18]] : i64
// CHECK:             %[[VAL_22:.*]] = cc.compute_ptr %[[VAL_12]]{{\[}}%[[VAL_21]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK:             %[[VAL_23:.*]] = cc.load %[[VAL_19]] : !cc.ptr<f64>
// CHECK:             %[[VAL_24:.*]] = cc.load %[[VAL_22]] : !cc.ptr<f64>
// CHECK:             cc.store %[[VAL_23]], %[[VAL_22]] : !cc.ptr<f64>
// CHECK:             cc.store %[[VAL_24]], %[[VAL_19]] : !cc.ptr<f64>
// CHECK:             cc.continue %[[VAL_17]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_25:.*]]: index):
// CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_2]] : index
// CHECK:             cc.continue %[[VAL_26]] : index
// CHECK:           } {invariant}
// CHECK:           %[[VAL_27:.*]] = cc.alloca f64
// CHECK:           %[[VAL_28:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<f64 x 7>>) -> !cc.ptr<!cc.array<f64 x ?>>
// CHECK:           %[[VAL_29:.*]] = cc.loop while ((%[[VAL_30:.*]] = %[[VAL_3]]) -> (index)) {
// CHECK:             %[[VAL_31:.*]] = arith.cmpi slt, %[[VAL_30]], %[[VAL_0]] : index
// CHECK:             cc.condition %[[VAL_31]](%[[VAL_30]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_32:.*]]: index):
// CHECK:             %[[VAL_33:.*]] = arith.index_cast %[[VAL_32]] : index to i64
// CHECK:             %[[VAL_34:.*]] = cc.compute_ptr %[[VAL_28]]{{\[}}%[[VAL_33]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
// CHECK:             %[[VAL_35:.*]] = cc.load %[[VAL_34]] : !cc.ptr<f64>
// CHECK:             %[[VAL_36:.*]] = cc.load %[[VAL_27]] : !cc.ptr<f64>
// CHECK:             %[[VAL_37:.*]] = arith.addf %[[VAL_36]], %[[VAL_35]] : f64
// CHECK:             cc.store %[[VAL_37]], %[[VAL_27]] : !cc.ptr<f64>
// CHECK:             cc.continue %[[VAL_32]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_38:.*]]: index):
// CHECK:             %[[VAL_39:.*]] = arith.addi %[[VAL_38]], %[[VAL_2]] : index
// CHECK:             cc.continue %[[VAL_39]] : index
// CHECK:           } {invariant}
// CHECK:           %[[VAL_40:.*]] = cc.load %[[VAL_27]] : !cc.ptr<f64>
// CHECK:           return %[[VAL_40]] : f64
