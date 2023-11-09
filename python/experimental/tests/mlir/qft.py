# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../../../python_packages/cudaq pytest -rP  %s | FileCheck %s

import os

import pytest
import numpy as np

import cudaq

def test_qft():
    @cudaq.kernel(jit=True)
    def iqft(qubits: cudaq.qview):
        N = qubits.size()
        for i in range(N//2):
            swap(qubits[i], qubits[N-i-1])

        for i in range(N-1):
            h(qubits[i])
            j = i + 1
            for y in range(i, -1, -1):
                r1.ctrl(-np.pi / 2**(j-y), qubits[j], qubits[y])

        h(qubits[N-1])
    print(iqft)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__iqft(
# CHECK-SAME:                                      %[[VAL_0:.*]]: !quake.veq<?>) {
# CHECK:           %[[VAL_1:.*]] = arith.constant -3.1415926535897931 : f64
# CHECK:           %[[VAL_2:.*]] = arith.constant -1 : i64
# CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_5:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_6:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_7:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_6]], %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_9:.*]] = arith.floordivsi %[[VAL_8]], %[[VAL_5]] : i64
# CHECK:           %[[VAL_10:.*]] = math.absi %[[VAL_9]] : i64
# CHECK:           %[[VAL_11:.*]] = cc.alloca i64{{\[}}%[[VAL_10]] : i64]
# CHECK:           %[[VAL_12:.*]] = cc.loop while ((%[[VAL_13:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_9]] : i64
# CHECK:             cc.condition %[[VAL_14]](%[[VAL_13]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_15:.*]]: i64):
# CHECK:             %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_11]]{{\[}}%[[VAL_15]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             cc.store %[[VAL_15]], %[[VAL_16]] : !cc.ptr<i64>
# CHECK:             cc.continue %[[VAL_15]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_17:.*]]: i64):
# CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_18]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_19:.*]] = cc.loop while ((%[[VAL_20:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_10]] : i64
# CHECK:             cc.condition %[[VAL_21]](%[[VAL_20]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_22:.*]]: i64):
# CHECK:             %[[VAL_23:.*]] = cc.compute_ptr %[[VAL_11]]{{\[}}%[[VAL_22]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_24:.*]] = cc.load %[[VAL_23]] : !cc.ptr<i64>
# CHECK:             %[[VAL_25:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_24]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_26:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             %[[VAL_27:.*]] = arith.subi %[[VAL_26]], %[[VAL_24]] : i64
# CHECK:             %[[VAL_28:.*]] = arith.subi %[[VAL_27]], %[[VAL_3]] : i64
# CHECK:             %[[VAL_29:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_28]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.swap %[[VAL_25]], %[[VAL_29]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_22]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_30:.*]]: i64):
# CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_31]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_32:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_33:.*]] = arith.subi %[[VAL_32]], %[[VAL_3]] : i64
# CHECK:           %[[VAL_34:.*]] = math.absi %[[VAL_33]] : i64
# CHECK:           %[[VAL_35:.*]] = cc.alloca i64{{\[}}%[[VAL_34]] : i64]
# CHECK:           %[[VAL_36:.*]] = cc.loop while ((%[[VAL_37:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_38:.*]] = arith.cmpi slt, %[[VAL_37]], %[[VAL_33]] : i64
# CHECK:             cc.condition %[[VAL_38]](%[[VAL_37]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_39:.*]]: i64):
# CHECK:             %[[VAL_40:.*]] = cc.compute_ptr %[[VAL_35]]{{\[}}%[[VAL_39]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             cc.store %[[VAL_39]], %[[VAL_40]] : !cc.ptr<i64>
# CHECK:             cc.continue %[[VAL_39]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_41:.*]]: i64):
# CHECK:             %[[VAL_42:.*]] = arith.addi %[[VAL_41]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_42]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_43:.*]] = cc.loop while ((%[[VAL_44:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_45:.*]] = arith.cmpi slt, %[[VAL_44]], %[[VAL_34]] : i64
# CHECK:             cc.condition %[[VAL_45]](%[[VAL_44]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_46:.*]]: i64):
# CHECK:             %[[VAL_47:.*]] = cc.compute_ptr %[[VAL_35]]{{\[}}%[[VAL_46]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_48:.*]] = cc.load %[[VAL_47]] : !cc.ptr<i64>
# CHECK:             %[[VAL_49:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_48]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_49]] : (!quake.ref) -> ()
# CHECK:             %[[VAL_50:.*]] = arith.addi %[[VAL_48]], %[[VAL_3]] : i64
# CHECK:             %[[VAL_51:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_50]], %[[VAL_51]] : !cc.ptr<i64>
# CHECK:             %[[VAL_52:.*]] = arith.subi %[[VAL_2]], %[[VAL_48]] : i64
# CHECK:             %[[VAL_53:.*]] = math.absi %[[VAL_52]] : i64
# CHECK:             %[[VAL_54:.*]] = cc.alloca i64{{\[}}%[[VAL_53]] : i64]
# CHECK:             %[[VAL_55:.*]] = cc.loop while ((%[[VAL_56:.*]] = %[[VAL_48]]) -> (i64)) {
# CHECK:               %[[VAL_57:.*]] = arith.cmpi sgt, %[[VAL_56]], %[[VAL_2]] : i64
# CHECK:               cc.condition %[[VAL_57]](%[[VAL_56]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_58:.*]]: i64):
# CHECK:               %[[VAL_59:.*]] = arith.muli %[[VAL_58]], %[[VAL_2]] : i64
# CHECK:               %[[VAL_60:.*]] = arith.addi %[[VAL_48]], %[[VAL_59]] : i64
# CHECK:               %[[VAL_61:.*]] = cc.compute_ptr %[[VAL_54]]{{\[}}%[[VAL_58]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:               cc.store %[[VAL_60]], %[[VAL_61]] : !cc.ptr<i64>
# CHECK:               cc.continue %[[VAL_58]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_62:.*]]: i64):
# CHECK:               %[[VAL_63:.*]] = arith.addi %[[VAL_62]], %[[VAL_2]] : i64
# CHECK:               cc.continue %[[VAL_63]] : i64
# CHECK:             } {invariant}
# CHECK:             %[[VAL_64:.*]] = cc.loop while ((%[[VAL_65:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:               %[[VAL_66:.*]] = arith.cmpi slt, %[[VAL_65]], %[[VAL_53]] : i64
# CHECK:               cc.condition %[[VAL_66]](%[[VAL_65]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_67:.*]]: i64):
# CHECK:               %[[VAL_68:.*]] = cc.compute_ptr %[[VAL_54]]{{\[}}%[[VAL_67]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:               %[[VAL_69:.*]] = cc.load %[[VAL_68]] : !cc.ptr<i64>
# CHECK:               %[[VAL_70:.*]] = cc.load %[[VAL_51]] : !cc.ptr<i64>
# CHECK:               %[[VAL_71:.*]] = arith.subi %[[VAL_70]], %[[VAL_69]] : i64
# CHECK:               %[[VAL_72:.*]] = math.ipowi %[[VAL_5]], %[[VAL_71]] : i64
# CHECK:               %[[VAL_73:.*]] = arith.sitofp %[[VAL_72]] : i64 to f64
# CHECK:               %[[VAL_74:.*]] = arith.divf %[[VAL_1]], %[[VAL_73]] : f64
# CHECK:               %[[VAL_75:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_70]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               %[[VAL_76:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_69]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.r1 (%[[VAL_74]]) {{\[}}%[[VAL_75]]] %[[VAL_76]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:               cc.continue %[[VAL_67]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_77:.*]]: i64):
# CHECK:               %[[VAL_78:.*]] = arith.addi %[[VAL_77]], %[[VAL_3]] : i64
# CHECK:               cc.continue %[[VAL_78]] : i64
# CHECK:             } {invariant}
# CHECK:             cc.continue %[[VAL_46]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_79:.*]]: i64):
# CHECK:             %[[VAL_80:.*]] = arith.addi %[[VAL_79]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_80]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_81:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_82:.*]] = arith.subi %[[VAL_81]], %[[VAL_3]] : i64
# CHECK:           %[[VAL_83:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_82]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:           quake.h %[[VAL_83]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }