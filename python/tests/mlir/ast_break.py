# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../../.. pytest -rP  %s | FileCheck %s

import os

import pytest
import numpy as np

import cudaq

def test_break():
    
    @cudaq.kernel(jit=True, verbose=False)
    def kernel(x : float): 
        q = cudaq.qvector(4)
        for i in range(10):
            x = x + x**2
            if x > 5:
                break
            ry(x, q[i%4])
        
    print(kernel)
    kernel(1.2)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel(
# CHECK-SAME:                                        %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 5.000000e+00 : f64
# CHECK:           %[[VAL_2:.*]] = arith.constant 10 : i64
# CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_4:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_5:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_6:.*]] = arith.constant 4 : i64
# CHECK:           %[[VAL_7:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_7]] : !cc.ptr<f64>
# CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_9:.*]] = cc.alloca !cc.array<i64 x 10>
# CHECK:           %[[VAL_10:.*]] = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_5]]) -> (i64)) {
# CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_2]] : i64
# CHECK:             cc.condition %[[VAL_12]](%[[VAL_11]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_13:.*]]: i64):
# CHECK:             %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_9]]{{\[}}%[[VAL_13]]] : (!cc.ptr<!cc.array<i64 x 10>>, i64) -> !cc.ptr<i64>
# CHECK:             cc.store %[[VAL_13]], %[[VAL_14]] : !cc.ptr<i64>
# CHECK:             cc.continue %[[VAL_13]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_15:.*]]: i64):
# CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_4]] : i64
# CHECK:             cc.continue %[[VAL_16]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_17:.*]] = cc.loop while ((%[[VAL_18:.*]] = %[[VAL_5]]) -> (i64)) {
# CHECK:             %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_2]] : i64
# CHECK:             cc.condition %[[VAL_19]](%[[VAL_18]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_20:.*]]: i64):
# CHECK:             %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_9]]{{\[}}%[[VAL_20]]] : (!cc.ptr<!cc.array<i64 x 10>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_22:.*]] = cc.load %[[VAL_21]] : !cc.ptr<i64>
# CHECK:             %[[VAL_23:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_24:.*]] = math.fpowi %[[VAL_23]], %[[VAL_3]] : f64, i64
# CHECK:             %[[VAL_25:.*]] = arith.addf %[[VAL_23]], %[[VAL_24]] : f64
# CHECK:             cc.store %[[VAL_25]], %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_26:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_27:.*]] = arith.cmpf ogt, %[[VAL_26]], %[[VAL_1]] : f64
# CHECK:             cc.if(%[[VAL_27]]) {
# CHECK:               cc.unwind_break %[[VAL_20]] : i64
# CHECK:             }
# CHECK:             %[[VAL_28:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_29:.*]] = arith.remui %[[VAL_22]], %[[VAL_6]] : i64
# CHECK:             %[[VAL_30:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_29]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_28]]) %[[VAL_30]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_20]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_31:.*]]: i64):
# CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_31]], %[[VAL_4]] : i64
# CHECK:             cc.continue %[[VAL_32]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }