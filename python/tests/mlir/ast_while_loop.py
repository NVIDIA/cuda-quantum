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


def test_while():

    @cudaq.kernel
    def cost():
        q = cudaq.qvector(6)
        i = 5
        while i > 0:
            ry(np.pi, q[i])
            i -= 1

    print(cost)
    # cost()


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_1:.*]] = arith.constant 3.1415926535897931 : f64
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = arith.constant 5 : i64
# CHECK:           %[[VAL_4:.*]] = quake.alloca !quake.veq<6>
# CHECK:           %[[VAL_5:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_3]], %[[VAL_5]] : !cc.ptr<i64>
# CHECK:           cc.loop while {
# CHECK:             %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i64>
# CHECK:             %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_2]] : i64
# CHECK:             cc.condition %[[VAL_7]]
# CHECK:           } do {
# CHECK:             %[[VAL_8:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i64>
# CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_8]]] : (!quake.veq<6>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_1]]) %[[VAL_9]] : (f64, !quake.ref) -> ()
# CHECK:             %[[VAL_10:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i64>
# CHECK:             %[[VAL_11:.*]] = arith.subi %[[VAL_10]], %[[VAL_0]] : i64
# CHECK:             cc.store %[[VAL_11]], %[[VAL_5]] : !cc.ptr<i64>
# CHECK:             cc.continue
# CHECK:           }
# CHECK:           return
# CHECK:         }


def test_complex_conditional():

    @cudaq.kernel
    def cost():
        q = cudaq.qvector(6)
        i = 5
        while i > 0 and i < 14 and i != 2:
            ry(np.pi, q[i])
            i -= 1

    print(cost)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_1:.*]] = arith.constant 3.1415926535897931 : f64
# CHECK:           %[[VAL_2:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_3:.*]] = arith.constant 14 : i64
# CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_5:.*]] = arith.constant 5 : i64
# CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<6>
# CHECK:           %[[VAL_7:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_5]], %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           cc.loop while {
# CHECK:             %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_4]] : i64
# CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_3]] : i64
# CHECK:             %[[VAL_11:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_2]] : i64
# CHECK:             %[[VAL_12:.*]] = arith.andi %[[VAL_11]], %[[VAL_10]] : i1
# CHECK:             %[[VAL_13:.*]] = arith.andi %[[VAL_12]], %[[VAL_9]] : i1
# CHECK:             cc.condition %[[VAL_13]]
# CHECK:           } do {
# CHECK:             %[[VAL_14:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_6]]{{\[}}%[[VAL_14]]] : (!quake.veq<6>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_1]]) %[[VAL_15]] : (f64, !quake.ref) -> ()
# CHECK:             %[[VAL_16:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             %[[VAL_17:.*]] = arith.subi %[[VAL_16]], %[[VAL_0]] : i64
# CHECK:             cc.store %[[VAL_17]], %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             cc.continue
# CHECK:           }
# CHECK:           return
# CHECK:         }
