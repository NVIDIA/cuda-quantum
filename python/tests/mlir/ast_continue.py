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


def test_continue():

    @cudaq.kernel(verbose=True)
    def kernel(x: float):
        q = cudaq.qvector(4)
        for i in range(10):
            x = x + x**2
            if x > 10:
                x(q[i % 4])
                continue
            ry(x, q[i % 4])

    print(kernel)
    kernel(1.2)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel(
# CHECK-SAME:                                        %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint"} {
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1.000000e+01 : f64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 10 : i64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 2 : i64
# CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 0 : i64
# CHECK-DAG:           %[[VAL_6:.*]] = arith.constant 4 : i64
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
# CHECK:               %[[VAL_31:.*]] = arith.remui %[[VAL_25]], %[[VAL_6]] : i64
# CHECK:               %[[VAL_32:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_31]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.x %[[VAL_32]] : (!quake.ref) -> ()
# CHECK:               cc.unwind_continue %[[VAL_23]] : i64
# CHECK:             }
# CHECK:             %[[VAL_33:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_34:.*]] = arith.remui %[[VAL_25]], %[[VAL_6]] : i64
# CHECK:             %[[VAL_35:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_34]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_33]]) %[[VAL_35]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_23]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_36:.*]]: i64):
# CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_36]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_37]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }
