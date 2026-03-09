/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// clang-format off
// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s
// clang-format on

#include "cudaq.h"

__qpu__ void apply_pauli(cudaq::qview<> qubits, const std::vector<char> &word) {
  for (std::size_t i = 0; i < word.size(); i++) {
    if (word[i] == 'X') {
      x(qubits[i]);
    }
    if (word[i] == 'Y') {
      y(qubits[i]);
    }
    if (word[i] == 'Z') {
      z(qubits[i]);
    }
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_apply_pauli
// CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !cc.stdvec<i8>) attributes
// CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 90 : i32
// CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 89 : i32
// CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 88 : i32
// CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 0 : i64
// CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 1 : i64
// CHECK:           cc.scope {
// CHECK:             %[[VAL_7:.*]] = cc.alloca i64
// CHECK:             cc.store %[[VAL_5]], %[[VAL_7]] : !cc.ptr<i64>
// CHECK:             cc.loop while {
// CHECK:               %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
// CHECK:               %[[VAL_9:.*]] = cc.stdvec_size %[[VAL_1]] : (!cc.stdvec<i8>) -> i64
// CHECK:               %[[VAL_10:.*]] = arith.cmpi ult, %[[VAL_8]], %[[VAL_9]] : i64
// CHECK:               cc.condition %[[VAL_10]]
// CHECK:             } do {
// CHECK:               %[[VAL_11:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
// CHECK:               %[[VAL_12:.*]] = cc.stdvec_data %[[VAL_1]] : (!cc.stdvec<i8>) -> !cc.ptr<!cc.array<i8 x ?>>
// CHECK:               %[[VAL_13:.*]] = cc.compute_ptr %[[VAL_12]][%[[VAL_11]]] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
// CHECK:               %[[VAL_14:.*]] = cc.load %[[VAL_13]] : !cc.ptr<i8>
// CHECK:               %[[VAL_15:.*]] = cc.cast {{.*}}%[[VAL_14]] : (i8) -> i32
// CHECK:               %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_15]], %[[VAL_4]] : i32
// CHECK:               cc.if(%[[VAL_16]]) {
// CHECK:                 %[[VAL_17:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
// CHECK:                 %[[VAL_18:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_17]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.x %[[VAL_18]] : (!quake.ref) -> ()
// CHECK:               }
// CHECK:               %[[VAL_19:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
// CHECK:               %[[VAL_20:.*]] = cc.stdvec_data %[[VAL_1]] : (!cc.stdvec<i8>) -> !cc.ptr<!cc.array<i8 x ?>>
// CHECK:               %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_20]][%[[VAL_19]]] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
// CHECK:               %[[VAL_22:.*]] = cc.load %[[VAL_21]] : !cc.ptr<i8>
// CHECK:               %[[VAL_23:.*]] = cc.cast {{.*}}%[[VAL_22]] : (i8) -> i32
// CHECK:               %[[VAL_24:.*]] = arith.cmpi eq, %[[VAL_23]], %[[VAL_3]] : i32
// CHECK:               cc.if(%[[VAL_24]]) {
// CHECK:                 %[[VAL_25:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
// CHECK:                 %[[VAL_26:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_25]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.y %[[VAL_26]] : (!quake.ref) -> ()
// CHECK:               }
// CHECK:               %[[VAL_27:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
// CHECK:               %[[VAL_28:.*]] = cc.stdvec_data %[[VAL_1]] : (!cc.stdvec<i8>) -> !cc.ptr<!cc.array<i8 x ?>>
// CHECK:               %[[VAL_29:.*]] = cc.compute_ptr %[[VAL_28]][%[[VAL_27]]] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
// CHECK:               %[[VAL_30:.*]] = cc.load %[[VAL_29]] : !cc.ptr<i8>
// CHECK:               %[[VAL_31:.*]] = cc.cast {{.*}}%[[VAL_30]] : (i8) -> i32
// CHECK:               %[[VAL_32:.*]] = arith.cmpi eq, %[[VAL_31]], %[[VAL_2]] : i32
// CHECK:               cc.if(%[[VAL_32]]) {
// CHECK:                 %[[VAL_33:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
// CHECK:                 %[[VAL_34:.*]] = quake.extract_ref %[[VAL_0]][%[[VAL_33]]] : (!quake.veq<?>, i64) -> !quake.ref
// CHECK:                 quake.z %[[VAL_34]] : (!quake.ref) -> ()
// CHECK:               }
// CHECK:               cc.continue
// CHECK:             } step {
// CHECK:               %[[VAL_35:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
// CHECK:               %[[VAL_36:.*]] = arith.addi %[[VAL_35]], %[[VAL_6]] : i64
// CHECK:               cc.store %[[VAL_36]], %[[VAL_7]] : !cc.ptr<i64>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

