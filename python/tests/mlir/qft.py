# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__iqft..
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant -3.1415926535897931 : f64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant -1 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_7:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_8:.*]] = cc.undef i64
# CHECK:           %[[VAL_9:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_10:.*]] = arith.floordivsi %[[VAL_9]], %[[VAL_3]] : i64
# CHECK:           %[[VAL_11:.*]]:2 = cc.loop while ((%[[VAL_12:.*]] = %[[VAL_5]], %[[VAL_13:.*]] = %[[VAL_8]]) -> (i64, i64)) {
# CHECK:             %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_10]] : i64
# CHECK:             cc.condition %[[VAL_14]](%[[VAL_12]], %[[VAL_13]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_15:.*]]: i64, %[[VAL_16:.*]]: i64):
# CHECK:             %[[VAL_17:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_15]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_18:.*]] = arith.subi %[[VAL_9]], %[[VAL_15]] : i64
# CHECK:             %[[VAL_19:.*]] = arith.subi %[[VAL_18]], %[[VAL_4]] : i64
# CHECK:             %[[VAL_20:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.swap %[[VAL_17]], %[[VAL_20]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_15]], %[[VAL_15]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_21:.*]]: i64, %[[VAL_22:.*]]: i64):
# CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_21]], %[[VAL_4]] : i64
# CHECK:             cc.continue %[[VAL_23]], %[[VAL_22]] : i64, i64
# CHECK:           }
# CHECK:           %[[VAL_24:.*]] = arith.subi %[[VAL_9]], %[[VAL_4]] : i64
# CHECK:           %[[VAL_25:.*]]:4 = cc.loop while ((%[[VAL_26:.*]] = %[[VAL_5]], %[[VAL_27:.*]] = %[[VAL_28:.*]]#1, %[[VAL_29:.*]] = %[[VAL_7]], %[[VAL_30:.*]] = %[[VAL_6]]) -> (i64, i64, i64, i64)) {
# CHECK:             %[[VAL_31:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_24]] : i64
# CHECK:             cc.condition %[[VAL_31]](%[[VAL_26]], %[[VAL_27]], %[[VAL_29]], %[[VAL_30]] : i64, i64, i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_32:.*]]: i64, %[[VAL_33:.*]]: i64, %[[VAL_34:.*]]: i64, %[[VAL_35:.*]]: i64):
# CHECK:             %[[VAL_36:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_32]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_36]] : (!quake.ref) -> ()
# CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_32]], %[[VAL_4]] : i64
# CHECK:             %[[VAL_38:.*]]:2 = cc.loop while ((%[[VAL_39:.*]] = %[[VAL_32]], %[[VAL_40:.*]] = %[[VAL_35]]) -> (i64, i64)) {
# CHECK:               %[[VAL_41:.*]] = arith.cmpi sgt, %[[VAL_39]], %[[VAL_2]] : i64
# CHECK:               cc.condition %[[VAL_41]](%[[VAL_39]], %[[VAL_40]] : i64, i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_42:.*]]: i64, %[[VAL_43:.*]]: i64):
# CHECK:               %[[VAL_44:.*]] = arith.subi %[[VAL_37]], %[[VAL_42]] : i64
# CHECK:               %[[VAL_45:.*]] = math.ipowi %[[VAL_3]], %[[VAL_44]] : i64
# CHECK:               %[[VAL_46:.*]] = cc.cast signed %[[VAL_45]] : (i64) -> f64
# CHECK:               %[[VAL_47:.*]] = arith.divf %[[VAL_1]], %[[VAL_46]] : f64
# CHECK:               %[[VAL_48:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_37]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               %[[VAL_49:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_42]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.r1 (%[[VAL_47]]) {{\[}}%[[VAL_48]]] %[[VAL_49]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:               cc.continue %[[VAL_42]], %[[VAL_42]] : i64, i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_50:.*]]: i64, %[[VAL_51:.*]]: i64):
# CHECK:               %[[VAL_52:.*]] = arith.addi %[[VAL_50]], %[[VAL_2]] : i64
# CHECK:               cc.continue %[[VAL_52]], %[[VAL_51]] : i64, i64
# CHECK:             }
# CHECK:             cc.continue %[[VAL_32]], %[[VAL_32]], %[[VAL_37]], %[[VAL_53:.*]]#1 : i64, i64, i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_54:.*]]: i64, %[[VAL_55:.*]]: i64, %[[VAL_56:.*]]: i64, %[[VAL_57:.*]]: i64):
# CHECK:             %[[VAL_58:.*]] = arith.addi %[[VAL_54]], %[[VAL_4]] : i64
# CHECK:             cc.continue %[[VAL_58]], %[[VAL_55]], %[[VAL_56]], %[[VAL_57]] : i64, i64, i64, i64
# CHECK:           }
# CHECK:           %[[VAL_59:.*]] = arith.subi %[[VAL_9]], %[[VAL_4]] : i64
# CHECK:           %[[VAL_60:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_59]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:           quake.h %[[VAL_60]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }
