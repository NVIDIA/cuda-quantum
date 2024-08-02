# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s


import numpy as np

import cudaq


def test_qft():

    @cudaq.kernel
    def iqft(qubits: cudaq.qview):
        N = qubits.size()
        for i in range(N // 2):
            swap(qubits[i], qubits[N - i - 1])

        for i in range(N - 1):
            h(qubits[i])
            j = i + 1
            for y in range(i, -1, -1):
                r1.ctrl(-np.pi / 2**(j - y), qubits[j], qubits[y])

        h(qubits[N - 1])

    print(iqft)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__iqft(
# CHECK-SAME:                                      %[[VAL_0:.*]]: !quake.veq<?>) {
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant -3.1415926535897931 : f64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant -1 : i64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_6:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_7:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_6]], %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_9:.*]] = arith.floordivsi %[[VAL_8]], %[[VAL_5]] : i64
# CHECK:           %[[VAL_10:.*]] = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_9]] : i64
# CHECK:             cc.condition %[[VAL_12]](%[[VAL_11]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_13:.*]]: i64):
# CHECK:             %[[VAL_14:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             %[[VAL_16:.*]] = arith.subi %[[VAL_15]], %[[VAL_13]] : i64
# CHECK:             %[[VAL_17:.*]] = arith.subi %[[VAL_16]], %[[VAL_3]] : i64
# CHECK:             %[[VAL_18:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_17]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.swap %[[VAL_14]], %[[VAL_18]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_13]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_19:.*]]: i64):
# CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_20]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_21:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_22:.*]] = arith.subi %[[VAL_21]], %[[VAL_3]] : i64
# CHECK:           %[[VAL_23:.*]] = cc.loop while ((%[[VAL_24:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_24]], %[[VAL_22]] : i64
# CHECK:             cc.condition %[[VAL_25]](%[[VAL_24]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_26:.*]]: i64):
# CHECK:             %[[VAL_27:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_26]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_27]] : (!quake.ref) -> ()
# CHECK:             %[[VAL_28:.*]] = arith.addi %[[VAL_26]], %[[VAL_3]] : i64
# CHECK:             %[[VAL_29:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_28]], %[[VAL_29]] : !cc.ptr<i64>
# CHECK:             %[[VAL_30:.*]] = cc.loop while ((%[[VAL_31:.*]] = %[[VAL_26]]) -> (i64)) {
# CHECK:               %[[VAL_32:.*]] = arith.cmpi sgt, %[[VAL_31]], %[[VAL_2]] : i64
# CHECK:               cc.condition %[[VAL_32]](%[[VAL_31]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_33:.*]]: i64):
# CHECK:               %[[VAL_34:.*]] = cc.load %[[VAL_29]] : !cc.ptr<i64>
# CHECK:               %[[VAL_35:.*]] = arith.subi %[[VAL_34]], %[[VAL_33]] : i64
# CHECK:               %[[VAL_36:.*]] = math.ipowi %[[VAL_5]], %[[VAL_35]] : i64
# CHECK:               %[[VAL_37:.*]] = arith.sitofp %[[VAL_36]] : i64 to f64
# CHECK:               %[[VAL_38:.*]] = arith.divf %[[VAL_1]], %[[VAL_37]] : f64
# CHECK:               %[[VAL_39:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_34]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               %[[VAL_40:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_33]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.r1 (%[[VAL_38]]) {{\[}}%[[VAL_39]]] %[[VAL_40]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:               cc.continue %[[VAL_33]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_41:.*]]: i64):
# CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_41]], %[[VAL_2]] : i64
# CHECK:               cc.continue %[[VAL_42]] : i64
# CHECK:             } {invariant}
# CHECK:             cc.continue %[[VAL_26]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_43:.*]]: i64):
# CHECK:             %[[VAL_44:.*]] = arith.addi %[[VAL_43]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_44]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_45:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_46:.*]] = arith.subi %[[VAL_45]], %[[VAL_3]] : i64
# CHECK:           %[[VAL_47:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_46]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:           quake.h %[[VAL_47]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }
