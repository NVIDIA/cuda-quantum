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
# CHECK:             %[[VAL_30:.*]] = arith.subi %[[VAL_2]], %[[VAL_26]] : i64
# CHECK:             %[[VAL_31:.*]] = math.absi %[[VAL_30]] : i64
# CHECK:             %[[VAL_32:.*]] = cc.alloca i64{{\[}}%[[VAL_31]] : i64]
# CHECK:             %[[VAL_33:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_4]], %[[VAL_33]] : !cc.ptr<i64>
# CHECK:             %[[VAL_34:.*]] = cc.loop while ((%[[VAL_35:.*]] = %[[VAL_26]]) -> (i64)) {
# CHECK:               %[[VAL_36:.*]] = arith.cmpi sgt, %[[VAL_35]], %[[VAL_2]] : i64
# CHECK:               cc.condition %[[VAL_36]](%[[VAL_35]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_37:.*]]: i64):
# CHECK:               %[[VAL_38:.*]] = cc.load %[[VAL_33]] : !cc.ptr<i64>
# CHECK:               %[[VAL_39:.*]] = arith.muli %[[VAL_38]], %[[VAL_2]] : i64
# CHECK:               %[[VAL_40:.*]] = arith.addi %[[VAL_26]], %[[VAL_39]] : i64
# CHECK:               %[[VAL_41:.*]] = cc.compute_ptr %[[VAL_32]]{{\[}}%[[VAL_38]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:               cc.store %[[VAL_40]], %[[VAL_41]] : !cc.ptr<i64>
# CHECK:               %[[VAL_42:.*]] = arith.addi %[[VAL_38]], %[[VAL_3]] : i64
# CHECK:               cc.store %[[VAL_42]], %[[VAL_33]] : !cc.ptr<i64>
# CHECK:               cc.continue %[[VAL_37]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_43:.*]]: i64):
# CHECK:               %[[VAL_44:.*]] = arith.addi %[[VAL_43]], %[[VAL_2]] : i64
# CHECK:               cc.continue %[[VAL_44]] : i64
# CHECK:             } {invariant}
# CHECK:             %[[VAL_45:.*]] = cc.loop while ((%[[VAL_46:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:               %[[VAL_47:.*]] = arith.cmpi slt, %[[VAL_46]], %[[VAL_31]] : i64
# CHECK:               cc.condition %[[VAL_47]](%[[VAL_46]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_48:.*]]: i64):
# CHECK:               %[[VAL_49:.*]] = cc.compute_ptr %[[VAL_32]]{{\[}}%[[VAL_48]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:               %[[VAL_50:.*]] = cc.load %[[VAL_49]] : !cc.ptr<i64>
# CHECK:               %[[VAL_51:.*]] = cc.load %[[VAL_29]] : !cc.ptr<i64>
# CHECK:               %[[VAL_52:.*]] = arith.subi %[[VAL_51]], %[[VAL_50]] : i64
# CHECK:               %[[VAL_53:.*]] = math.ipowi %[[VAL_5]], %[[VAL_52]] : i64
# CHECK:               %[[VAL_54:.*]] = arith.sitofp %[[VAL_53]] : i64 to f64
# CHECK:               %[[VAL_55:.*]] = arith.divf %[[VAL_1]], %[[VAL_54]] : f64
# CHECK:               %[[VAL_56:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_51]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               %[[VAL_57:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_50]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.r1 (%[[VAL_55]]) {{\[}}%[[VAL_56]]] %[[VAL_57]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:               cc.continue %[[VAL_48]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_58:.*]]: i64):
# CHECK:               %[[VAL_59:.*]] = arith.addi %[[VAL_58]], %[[VAL_3]] : i64
# CHECK:               cc.continue %[[VAL_59]] : i64
# CHECK:             } {invariant}
# CHECK:             cc.continue %[[VAL_26]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_60:.*]]: i64):
# CHECK:             %[[VAL_61:.*]] = arith.addi %[[VAL_60]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_61]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_62:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           %[[VAL_63:.*]] = arith.subi %[[VAL_62]], %[[VAL_3]] : i64
# CHECK:           %[[VAL_64:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_63]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:           quake.h %[[VAL_64]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }