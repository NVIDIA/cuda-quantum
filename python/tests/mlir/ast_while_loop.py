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


def test_while():

    @cudaq.kernel
    def trowe():
        q = cudaq.qvector(6)
        i = 5
        while i > 0:
            ry(np.pi, q[i])
            i -= 1

    print(trowe)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__trowe..
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 3.1415926535897931 : f64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 5 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.veq<6>
# CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_7:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_2]] : i64
# CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_8:.*]]: i64):
# CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_8]]] : (!quake.veq<6>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_1]]) %[[VAL_9]] : (f64, !quake.ref) -> ()
# CHECK:             %[[VAL_10:.*]] = arith.subi %[[VAL_8]], %[[VAL_0]] : i64
# CHECK:             cc.continue %[[VAL_10]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_4]] : !quake.veq<6>
# CHECK:           return
# CHECK:         }


def test_complex_conditional():

    @cudaq.kernel
    def costco():
        q = cudaq.qvector(6)
        i = 5
        while i > 0 and i < 14 and i != 2:
            ry(np.pi, q[i])
            i -= 1

    print(costco)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__costco..
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 3.1415926535897931 : f64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 14 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant false
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 5 : i64
# CHECK-DAG:       %[[VAL_7:.*]] = quake.alloca !quake.veq<6>
# CHECK:           %[[VAL_8:.*]] = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_6]]) -> (i64)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi sle, %[[VAL_9]], %[[VAL_5]] : i64
# CHECK:             %[[VAL_11:.*]] = cc.if(%[[VAL_10]]) -> i1 {
# CHECK:               cc.continue %[[VAL_4]] : i1
# CHECK:             } else {
# CHECK:               %[[VAL_12:.*]] = arith.cmpi sge, %[[VAL_9]], %[[VAL_3]] : i64
# CHECK:               %[[VAL_13:.*]] = cc.if(%[[VAL_12]]) -> i1 {
# CHECK:                 cc.continue %[[VAL_4]] : i1
# CHECK:               } else {
# CHECK:                 %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_2]] : i64
# CHECK:                 cc.continue %[[VAL_14]] : i1
# CHECK:               }
# CHECK:               cc.continue %[[VAL_15:.*]] : i1
# CHECK:             }
# CHECK:             cc.condition %[[VAL_16:.*]](%[[VAL_9]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_17:.*]]: i64):
# CHECK:             %[[VAL_18:.*]] = quake.extract_ref %[[VAL_7]]{{\[}}%[[VAL_17]]] : (!quake.veq<6>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_1]]) %[[VAL_18]] : (f64, !quake.ref) -> ()
# CHECK:             %[[VAL_19:.*]] = arith.subi %[[VAL_17]], %[[VAL_0]] : i64
# CHECK:             cc.continue %[[VAL_19]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_20:.*]]: i64):
# CHECK:             cc.continue %[[VAL_20]] : i64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_7]] : !quake.veq<6>
# CHECK:           return
# CHECK:         }
