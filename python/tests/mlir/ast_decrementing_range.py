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


def test_decrementing_range():

    @cudaq.kernel(jit=True)
    def test(q:int, p:int):
        qubits = cudaq.qvector(5) 
        for k in range(q, p, -1):
            x(qubits[k])

    print(test)
    test(2, 0)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test(
# CHECK-SAME:                                      %[[VAL_0:.*]]: i64,
# CHECK-SAME:                                      %[[VAL_1:.*]]: i64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = arith.constant -1 : i64
# CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_5:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<i64>
# CHECK:           %[[VAL_6:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_1]], %[[VAL_6]] : !cc.ptr<i64>
# CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<5>
# CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i64>
# CHECK:           %[[VAL_9:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i64>
# CHECK:           %[[VAL_10:.*]] = arith.subi %[[VAL_9]], %[[VAL_8]] : i64
# CHECK:           %[[VAL_11:.*]] = math.absi %[[VAL_10]] : i64
# CHECK:           %[[VAL_12:.*]] = cc.alloca i64{{\[}}%[[VAL_11]] : i64]
# CHECK:           %[[VAL_13:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_2]], %[[VAL_13]] : !cc.ptr<i64>
# CHECK:           %[[VAL_14:.*]] = cc.loop while ((%[[VAL_15:.*]] = %[[VAL_8]]) -> (i64)) {
# CHECK:             %[[VAL_16:.*]] = arith.cmpi sgt, %[[VAL_15]], %[[VAL_9]] : i64
# CHECK:             cc.condition %[[VAL_16]](%[[VAL_15]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_17:.*]]: i64):
# CHECK:             %[[VAL_18:.*]] = cc.load %[[VAL_13]] : !cc.ptr<i64>
# CHECK:             %[[VAL_19:.*]] = arith.muli %[[VAL_18]], %[[VAL_3]] : i64
# CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_8]], %[[VAL_19]] : i64
# CHECK:             %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_12]]{{\[}}%[[VAL_18]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             cc.store %[[VAL_20]], %[[VAL_21]] : !cc.ptr<i64>
# CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_18]], %[[VAL_4]] : i64
# CHECK:             cc.store %[[VAL_22]], %[[VAL_13]] : !cc.ptr<i64>
# CHECK:             cc.continue %[[VAL_17]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_23:.*]]: i64):
# CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_24]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_25:.*]] = cc.loop while ((%[[VAL_26:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_11]] : i64
# CHECK:             cc.condition %[[VAL_27]](%[[VAL_26]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_28:.*]]: i64):
# CHECK:             %[[VAL_29:.*]] = cc.compute_ptr %[[VAL_12]]{{\[}}%[[VAL_28]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_30:.*]] = cc.load %[[VAL_29]] : !cc.ptr<i64>
# CHECK:             %[[VAL_31:.*]] = quake.extract_ref %[[VAL_7]]{{\[}}%[[VAL_30]]] : (!quake.veq<5>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_31]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_28]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_32:.*]]: i64):
# CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_32]], %[[VAL_4]] : i64
# CHECK:             cc.continue %[[VAL_33]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }