/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: cudaq-quake %s | cudaq-opt -canonicalize | FileCheck %s

#include <cudaq.h>

__qpu__ void pauli_word_vec(std::vector<cudaq::pauli_word> words,
                            double theta) {
  cudaq::qvector v(4);
  exp_pauli(theta, v, words[0]);
}

// clang-format off
// CHECK-LABEL:   func.func
// @__nvqpp__mlirgen__function_pauli_word_vec._Z14pauli_word_vecSt6vectorIN5cudaq10pauli_wordESaIS1_EEd(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<!cc.charspan>,
// CHECK-SAME:      %[[VAL_1:.*]]: f64) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_2:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           %[[VAL_5:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<!cc.charspan>) -> !cc.ptr<!cc.array<!cc.charspan x ?>>
// CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<!cc.charspan x ?>>) -> !cc.ptr<!cc.charspan>
// CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_6]] : !cc.ptr<!cc.charspan>
// CHECK:           quake.exp_pauli (%[[VAL_4]]) %[[VAL_3]] to %[[VAL_7]] : (f64, !quake.veq<4>, !cc.charspan) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on

__qpu__ void pauli_word(cudaq::pauli_word wordle, double theta) {
  cudaq::qvector v(4);
  exp_pauli(theta, v, wordle);
}

// clang-format off
// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_pauli_word._Z10pauli_wordN5cudaq10pauli_wordEd(
// CHECK-SAME:      %[[VAL_0:.*]]: !cc.charspan,
// CHECK-SAME:      %[[VAL_1:.*]]: f64) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
// CHECK:           %[[VAL_2:.*]] = cc.alloca f64
// CHECK:           cc.store %[[VAL_1]], %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<4>
// CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_2]] : !cc.ptr<f64>
// CHECK:           quake.exp_pauli (%[[VAL_4]]) %[[VAL_3]] to %[[VAL_0]] : (f64, !quake.veq<4>, !cc.charspan) -> ()
// CHECK:           return
// CHECK:         }
// clang-format on
