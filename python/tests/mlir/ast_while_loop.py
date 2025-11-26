# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

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

    counts = cudaq.sample(cost)
    assert len(counts) == 1
    assert '011111' in counts
    print(cost)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
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

    counts = cudaq.sample(cost)
    assert len(counts) == 1
    assert '000111' in counts
    print(cost)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 3.1415926535897931 : f64
# CHECK-DAG:           %[[VAL_10:.*]] = arith.constant 2 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 14 : i64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant false
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 5 : i64
# CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<6>
# CHECK:           %[[VAL_7:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_5]], %[[VAL_7]] : !cc.ptr<i64>
# CHECK:           cc.loop while {
# CHECK:             %[[VAL_8:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             %[[VAL_9:.*]] = arith.cmpi sle, %[[VAL_8]], %[[VAL_4]] : i64
# CHECK:             %[[VAL_11:.*]] = cc.if(%[[VAL_9]]) -> i1 {
# CHECK:               cc.continue %[[VAL_3]] : i1
# CHECK:             } else {
# CHECK:               %[[VAL_12:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:               %[[VAL_13:.*]] = arith.cmpi sge, %[[VAL_12]], %[[VAL_2]] : i64
# CHECK:               %[[VAL_20:.*]] = cc.if(%[[VAL_13]]) -> i1 {
# CHECK:                 cc.continue %[[VAL_3]] : i1
# CHECK:               } else {
# CHECK:                 %[[VAL_21:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:                 %[[VAL_22:.*]] = arith.cmpi ne, %[[VAL_21]], %[[VAL_10]] : i64
# CHECK:                 cc.continue %[[VAL_22]] : i1
# CHECK:               }
# CHECK:               cc.continue %[[VAL_20:.*]]
# CHECK:             }
# CHECK:             cc.condition %[[VAL_11:.*]]
# CHECK:           } do {
# CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             %[[VAL_16:.*]] = quake.extract_ref %[[VAL_6]]{{\[}}%[[VAL_15]]] : (!quake.veq<6>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_1]]) %[[VAL_16]] : (f64, !quake.ref) -> ()
# CHECK:             %[[VAL_17:.*]] = cc.load %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             %[[VAL_18:.*]] = arith.subi %[[VAL_17]], %[[VAL_0]] : i64
# CHECK:             cc.store %[[VAL_18]], %[[VAL_7]] : !cc.ptr<i64>
# CHECK:             cc.continue
# CHECK:           }
# CHECK:           return
# CHECK:         }
