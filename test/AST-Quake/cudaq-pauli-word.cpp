/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt | FileCheck %s

#include "cudaq.h"

__qpu__ void test0(double theta, std::vector<cudaq::pauli_word> paulis) {
  cudaq::qvector q(2);
  for (auto &p : paulis)
    exp_pauli(theta, q, p);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test0._Z5test0dSt6vectorIN5cudaq10pauli_wordESaIS1_EE(
// CHECK-SAME:                                                                                                %[[VAL_0:.*]]: f64,
// CHECK-SAME:                                                                                                %[[VAL_1:.*]]: !cc.stdvec<!cc.ptr<i8>>) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_4:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<2>
// CHECK:           %[[VAL_6:.*]] = cc.stdvec_size %[[VAL_1]] : (!cc.stdvec<!cc.ptr<i8>>) -> i64
// CHECK:           %[[VAL_7:.*]] = cc.stdvec_data %[[VAL_1]] : (!cc.stdvec<!cc.ptr<i8>>) -> !cc.ptr<!cc.array<!cc.ptr<i8> x ?>>
// CHECK:           %[[VAL_8:.*]] = arith.index_cast %[[VAL_6]] : i64 to index
// CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_3]]) -> (index)) {
// CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_8]] : index
// CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : index)
// CHECK:           } do {
// CHECK:           ^bb0(%[[VAL_12:.*]]: index):
// CHECK:             %[[VAL_13:.*]] = arith.index_cast %[[VAL_12]] : index to i64
// CHECK:             %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_13]]] : (!cc.ptr<!cc.array<!cc.ptr<i8> x ?>>, i64) -> !cc.ptr<!cc.ptr<i8>>
// CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_4]] : !cc.ptr<f64>
// CHECK:             %[[VAL_16:.*]] = cc.load %[[VAL_14]] : !cc.ptr<!cc.ptr<i8>>
// CHECK:             quake.exp_pauli(%[[VAL_15]]) %[[VAL_5]], %[[VAL_16]] : (f64, !quake.veq<2>, !cc.ptr<i8>) -> ()
// CHECK:             cc.continue %[[VAL_12]] : index
// CHECK:           } step {
// CHECK:           ^bb0(%[[VAL_17:.*]]: index):
// CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_2]] : index
// CHECK:             cc.continue %[[VAL_18]] : index
// CHECK:           } {invariant}
// CHECK:           return
// CHECK:         }

__qpu__ void test1(double theta, cudaq::qview<> q, cudaq::pauli_word term) {
  exp_pauli(theta, q, term);
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test1._Z5test1dN5cudaq5qviewILm2EEENS_10pauli_wordE(
// CHECK-SAME:                                                                                              %[[VAL_0:.*]]: f64,
// CHECK-SAME:                                                                                              %[[VAL_1:.*]]: !quake.veq<?>,
// CHECK-SAME:                                                                                              %[[VAL_2:.*]]: !cc.ptr<i8>) attributes {"cudaq-kernel", no_this} {
// CHECK:           %[[VAL_3:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<f64>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_3]] : !cc.ptr<f64>
// CHECK:           quake.exp_pauli(%[[VAL_4]]) %[[VAL_1]], %[[VAL_2]] : (f64, !quake.veq<?>, !cc.ptr<i8>) -> ()
// CHECK:           return
// CHECK:         }