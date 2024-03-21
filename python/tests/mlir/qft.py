# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest
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
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant -1 : i64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_6:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_7:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_6]], %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_9:.*]] = arith.floordivsi %[[VAL_8]], %[[VAL_5]] : i64
# CHECK:           %[[VAL_10:.*]] = math.absi %[[VAL_9]] : i64
# CHECK:           %[[VAL_11:.*]] = cc.alloca i64{{\[}}%[[VAL_10]] : i64]
# CHECK:           %[[VAL_12:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_4]], %[[VAL_12]] : !cc.ptr<i64>
# CHECK:           %[[VAL_13:.*]] = cc.loop while ((%[[VAL_14:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_9]] : i64
# CHECK:             cc.condition %[[VAL_15]](%[[VAL_14]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64):
# CHECK:             %[[VAL_17:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i64>
# CHECK:             %[[VAL_18:.*]] = cc.compute_ptr %[[VAL_11]]{{\[}}%[[VAL_17]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             cc.store %[[VAL_17]], %[[VAL_18]] : !cc.ptr<i64>
# CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_2]] : i64
# CHECK:             cc.store %[[VAL_19]], %[[VAL_12]] : !cc.ptr<i64>
# CHECK:             cc.continue %[[VAL_16]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_20:.*]]: i64):
# CHECK:             %[[VAL_21:.*]] = arith.addi %[[VAL_20]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_21]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_22:.*]] = cc.loop while ((%[[VAL_23:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_24:.*]] = arith.cmpi slt, %[[VAL_23]], %[[VAL_10]] : i64
# CHECK:             cc.condition %[[VAL_24]](%[[VAL_23]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_25:.*]]: i64):
# CHECK:             %[[VAL_26:.*]] = cc.compute_ptr %[[VAL_11]]{{\[}}%[[VAL_25]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_27:.*]] = cc.load %[[VAL_26]] : !cc.ptr<i64>
# CHECK:             %[[VAL_28:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_27]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_29:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             %[[VAL_30:.*]] = arith.subi %[[VAL_29]], %[[VAL_27]] : i64
# CHECK:             %[[VAL_31:.*]] = arith.subi %[[VAL_30]], %[[VAL_2]] : i64
# CHECK:             %[[VAL_32:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_31]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.swap %[[VAL_28]], %[[VAL_32]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_25]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_33:.*]]: i64):
# CHECK:             %[[VAL_34:.*]] = arith.addi %[[VAL_33]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_34]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_35:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_36:.*]] = arith.subi %[[VAL_35]], %[[VAL_2]] : i64
# CHECK:           %[[VAL_37:.*]] = math.absi %[[VAL_36]] : i64
# CHECK:           %[[VAL_38:.*]] = cc.alloca i64{{\[}}%[[VAL_37]] : i64]
# CHECK:           %[[VAL_39:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_4]], %[[VAL_39]] : !cc.ptr<i64>
# CHECK:           %[[VAL_40:.*]] = cc.loop while ((%[[VAL_41:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_42:.*]] = arith.cmpi slt, %[[VAL_41]], %[[VAL_36]] : i64
# CHECK:             cc.condition %[[VAL_42]](%[[VAL_41]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_43:.*]]: i64):
# CHECK:             %[[VAL_44:.*]] = cc.load %[[VAL_39]] : !cc.ptr<i64>
# CHECK:             %[[VAL_45:.*]] = cc.compute_ptr %[[VAL_38]]{{\[}}%[[VAL_44]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             cc.store %[[VAL_44]], %[[VAL_45]] : !cc.ptr<i64>
# CHECK:             %[[VAL_46:.*]] = arith.addi %[[VAL_44]], %[[VAL_2]] : i64
# CHECK:             cc.store %[[VAL_46]], %[[VAL_39]] : !cc.ptr<i64>
# CHECK:             cc.continue %[[VAL_43]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_47:.*]]: i64):
# CHECK:             %[[VAL_48:.*]] = arith.addi %[[VAL_47]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_48]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_49:.*]] = cc.loop while ((%[[VAL_50:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_51:.*]] = arith.cmpi slt, %[[VAL_50]], %[[VAL_37]] : i64
# CHECK:             cc.condition %[[VAL_51]](%[[VAL_50]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_52:.*]]: i64):
# CHECK:             %[[VAL_53:.*]] = cc.compute_ptr %[[VAL_38]]{{\[}}%[[VAL_52]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_54:.*]] = cc.load %[[VAL_53]] : !cc.ptr<i64>
# CHECK:             %[[VAL_55:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_54]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_55]] : (!quake.ref) -> ()
# CHECK:             %[[VAL_56:.*]] = arith.addi %[[VAL_54]], %[[VAL_2]] : i64
# CHECK:             %[[VAL_57:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_56]], %[[VAL_57]] : !cc.ptr<i64>
# CHECK:             %[[VAL_58:.*]] = arith.subi %[[VAL_3]], %[[VAL_54]] : i64
# CHECK:             %[[VAL_59:.*]] = math.absi %[[VAL_58]] : i64
# CHECK:             %[[VAL_60:.*]] = cc.alloca i64{{\[}}%[[VAL_59]] : i64]
# CHECK:             %[[VAL_61:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_4]], %[[VAL_61]] : !cc.ptr<i64>
# CHECK:             %[[VAL_62:.*]] = cc.loop while ((%[[VAL_63:.*]] = %[[VAL_54]]) -> (i64)) {
# CHECK:               %[[VAL_64:.*]] = arith.cmpi sgt, %[[VAL_63]], %[[VAL_3]] : i64
# CHECK:               cc.condition %[[VAL_64]](%[[VAL_63]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_65:.*]]: i64):
# CHECK:               %[[VAL_66:.*]] = cc.load %[[VAL_61]] : !cc.ptr<i64>
# CHECK:               %[[VAL_67:.*]] = arith.muli %[[VAL_66]], %[[VAL_3]] : i64
# CHECK:               %[[VAL_68:.*]] = arith.addi %[[VAL_54]], %[[VAL_67]] : i64
# CHECK:               %[[VAL_69:.*]] = cc.compute_ptr %[[VAL_60]]{{\[}}%[[VAL_66]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:               cc.store %[[VAL_68]], %[[VAL_69]] : !cc.ptr<i64>
# CHECK:               %[[VAL_70:.*]] = arith.addi %[[VAL_66]], %[[VAL_2]] : i64
# CHECK:               cc.store %[[VAL_70]], %[[VAL_61]] : !cc.ptr<i64>
# CHECK:               cc.continue %[[VAL_65]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_71:.*]]: i64):
# CHECK:               %[[VAL_72:.*]] = arith.addi %[[VAL_71]], %[[VAL_3]] : i64
# CHECK:               cc.continue %[[VAL_72]] : i64
# CHECK:             } {invariant}
# CHECK:             %[[VAL_73:.*]] = cc.loop while ((%[[VAL_74:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:               %[[VAL_75:.*]] = arith.cmpi slt, %[[VAL_74]], %[[VAL_59]] : i64
# CHECK:               cc.condition %[[VAL_75]](%[[VAL_74]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_76:.*]]: i64):
# CHECK:               %[[VAL_77:.*]] = cc.compute_ptr %[[VAL_60]]{{\[}}%[[VAL_76]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:               %[[VAL_78:.*]] = cc.load %[[VAL_77]] : !cc.ptr<i64>
# CHECK:               %[[VAL_79:.*]] = cc.load %[[VAL_57]] : !cc.ptr<i64>
# CHECK:               %[[VAL_80:.*]] = arith.subi %[[VAL_79]], %[[VAL_78]] : i64
# CHECK:               %[[VAL_81:.*]] = math.ipowi %[[VAL_5]], %[[VAL_80]] : i64
# CHECK:               %[[VAL_82:.*]] = arith.sitofp %[[VAL_81]] : i64 to f64
# CHECK:               %[[VAL_83:.*]] = arith.divf %[[VAL_1]], %[[VAL_82]] : f64
# CHECK:               %[[VAL_84:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_79]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               %[[VAL_85:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_78]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.r1 (%[[VAL_83]]) {{\[}}%[[VAL_84]]] %[[VAL_85]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:               cc.continue %[[VAL_76]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_86:.*]]: i64):
# CHECK:               %[[VAL_87:.*]] = arith.addi %[[VAL_86]], %[[VAL_2]] : i64
# CHECK:               cc.continue %[[VAL_87]] : i64
# CHECK:             } {invariant}
# CHECK:             cc.continue %[[VAL_52]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_88:.*]]: i64):
# CHECK:             %[[VAL_89:.*]] = arith.addi %[[VAL_88]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_89]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_90:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_91:.*]] = arith.subi %[[VAL_90]], %[[VAL_2]] : i64
# CHECK:           %[[VAL_92:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_91]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:           quake.h %[[VAL_92]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }