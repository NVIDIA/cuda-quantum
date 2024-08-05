# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s


import numpy as np

import cudaq


def test_elif():

    @cudaq.kernel(verbose=True)
    def cost(thetas: np.ndarray):  # can pass 1D ndarray or list
        q = cudaq.qvector(4)
        for i, theta in enumerate(thetas):
            if i % 2.0:  # asserting we convert 2.0 to 2
                ry(theta, q[i % 4])
            else:
                rx(theta, q[i % 4])

    print(cost)
    cost(np.asarray([1., 2., 3., 4., 5., 6.]))
    cost([1., 2., 3., 4., 5., 6.])


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost(
# CHECK-SAME:                                      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint"} {
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 2 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 0 : i64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 4 : i64
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_6:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
# CHECK:           %[[VAL_7:.*]] = cc.alloca !cc.struct<{i64, f64}>{{\[}}%[[VAL_6]] : i64]
# CHECK:           %[[VAL_8:.*]] = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_6]] : i64
# CHECK:             cc.condition %[[VAL_10]](%[[VAL_9]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
# CHECK:             %[[VAL_12:.*]] = cc.undef !cc.struct<{i64, f64}>
# CHECK:             %[[VAL_13:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:             %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_13]][%[[VAL_11]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<f64>
# CHECK:             %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_11]]] : (!cc.ptr<!cc.array<!cc.struct<{i64, f64}> x ?>>, i64) -> !cc.ptr<!cc.struct<{i64, f64}>>
# CHECK:             %[[VAL_17:.*]] = cc.insert_value %[[VAL_11]], %[[VAL_12]][0] : (!cc.struct<{i64, f64}>, i64) -> !cc.struct<{i64, f64}>
# CHECK:             %[[VAL_18:.*]] = cc.insert_value %[[VAL_15]], %[[VAL_17]][1] : (!cc.struct<{i64, f64}>, f64) -> !cc.struct<{i64, f64}>
# CHECK:             cc.store %[[VAL_18]], %[[VAL_16]] : !cc.ptr<!cc.struct<{i64, f64}>>
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_19:.*]]: i64):
# CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_20]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_21:.*]] = cc.loop while ((%[[VAL_22:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_6]] : i64
# CHECK:             cc.condition %[[VAL_23]](%[[VAL_22]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_24:.*]]: i64):
# CHECK:             %[[VAL_25:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_24]]] : (!cc.ptr<!cc.array<!cc.struct<{i64, f64}> x ?>>, i64) -> !cc.ptr<!cc.struct<{i64, f64}>>
# CHECK:             %[[VAL_26:.*]] = cc.load %[[VAL_25]] : !cc.ptr<!cc.struct<{i64, f64}>>
# CHECK:             %[[VAL_27:.*]] = cc.extract_value %[[VAL_26]][0] : (!cc.struct<{i64, f64}>) -> i64
# CHECK:             %[[VAL_28:.*]] = cc.extract_value %[[VAL_26]][1] : (!cc.struct<{i64, f64}>) -> f64
# CHECK:             %[[VAL_29:.*]] = arith.remui %[[VAL_27]], %[[VAL_1]] : i64
# CHECK:             %[[VAL_30:.*]] = arith.cmpi ne, %[[VAL_29]], %[[VAL_3]] : i64
# CHECK:             cc.if(%[[VAL_30]]) {
# CHECK:               %[[VAL_31:.*]] = arith.remui %[[VAL_27]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_32:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_31]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.ry (%[[VAL_28]]) %[[VAL_32]] : (f64, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_33:.*]] = arith.remui %[[VAL_27]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_34:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_33]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.rx (%[[VAL_28]]) %[[VAL_34]] : (f64, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_24]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_35:.*]]: i64):
# CHECK:             %[[VAL_36:.*]] = arith.addi %[[VAL_35]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_36]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }
