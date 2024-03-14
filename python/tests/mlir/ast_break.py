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


def test_break():

    @cudaq.kernel(verbose=False)
    def kernel(x: float):
        q = cudaq.qvector(4)
        for i in range(10):
            x = x + x**2
            if x > 5:
                break
            ry(x, q[i % 4])

    print(kernel)
    kernel(1.2)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel(
# CHECK-SAME:                                        %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 5.000000e+00 : f64
# CHECK:           %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_3:.*]] = arith.constant 10 : i64
# CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_6:.*]] = arith.constant 4 : i64
# CHECK:           %[[VAL_7:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_7]] : !cc.ptr<f64>
# CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_9:.*]] = cc.alloca !cc.array<i64 x 10>
# CHECK:           %[[VAL_10:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_5]], %[[VAL_10]] : !cc.ptr<i64>
# CHECK:           %[[VAL_11:.*]] = cc.loop while ((%[[VAL_12:.*]] = %[[VAL_5]]) -> (i64)) {
# CHECK:             %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_3]] : i64
# CHECK:             cc.condition %[[VAL_13]](%[[VAL_12]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_14:.*]]: i64):
# CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_10]] : !cc.ptr<i64>
# CHECK:             %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_9]]{{\[}}%[[VAL_15]]] : (!cc.ptr<!cc.array<i64 x 10>>, i64) -> !cc.ptr<i64>
# CHECK:             cc.store %[[VAL_15]], %[[VAL_16]] : !cc.ptr<i64>
# CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_15]], %[[VAL_2]] : i64
# CHECK:             cc.store %[[VAL_17]], %[[VAL_10]] : !cc.ptr<i64>
# CHECK:             cc.continue %[[VAL_14]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_18:.*]]: i64):
# CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_19]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_20:.*]] = cc.loop while ((%[[VAL_21:.*]] = %[[VAL_5]]) -> (i64)) {
# CHECK:             %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_21]], %[[VAL_3]] : i64
# CHECK:             cc.condition %[[VAL_22]](%[[VAL_21]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_23:.*]]: i64):
# CHECK:             %[[VAL_24:.*]] = cc.compute_ptr %[[VAL_9]]{{\[}}%[[VAL_23]]] : (!cc.ptr<!cc.array<i64 x 10>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_25:.*]] = cc.load %[[VAL_24]] : !cc.ptr<i64>
# CHECK:             %[[VAL_26:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_27:.*]] = math.fpowi %[[VAL_26]], %[[VAL_4]] : f64, i64
# CHECK:             %[[VAL_28:.*]] = arith.addf %[[VAL_26]], %[[VAL_27]] : f64
# CHECK:             cc.store %[[VAL_28]], %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_29:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_30:.*]] = arith.cmpf ogt, %[[VAL_29]], %[[VAL_1]] : f64
# CHECK:             cc.if(%[[VAL_30]]) {
# CHECK:               cc.unwind_break %[[VAL_23]] : i64
# CHECK:             }
# CHECK:             %[[VAL_31:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_32:.*]] = arith.remui %[[VAL_25]], %[[VAL_6]] : i64
# CHECK:             %[[VAL_33:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_32]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_31]]) %[[VAL_33]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_23]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_34:.*]]: i64):
# CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_34]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_35]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }
