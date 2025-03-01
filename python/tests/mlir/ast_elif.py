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
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 4 : i64
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_1]] : i64]
# CHECK:           %[[VAL_3:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
# CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_3]] : i64
# CHECK:             cc.condition %[[VAL_8]](%[[VAL_7]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:             %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_10]]{{\[}}%[[VAL_9]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_12:.*]] = arith.constant 2.000000e+00 : f64
# CHECK:             %[[VAL_13:.*]] = arith.fptosi %[[VAL_12]] : f64 to i64
# CHECK:             %[[VAL_14:.*]] = arith.remui %[[VAL_9]], %[[VAL_13]] : i64
# CHECK:             %[[VAL_15:.*]] = arith.constant 0 : i64
# CHECK:             %[[VAL_16:.*]] = arith.cmpi ne, %[[VAL_14]], %[[VAL_15]] : i64
# CHECK:             cc.if(%[[VAL_16]]) {
# CHECK:               %[[VAL_17:.*]] = cc.load %[[VAL_11]] : !cc.ptr<f64>
# CHECK:               %[[VAL_18:.*]] = arith.constant 4 : i64
# CHECK:               %[[VAL_19:.*]] = arith.remui %[[VAL_9]], %[[VAL_18]] : i64
# CHECK:               %[[VAL_20:.*]] = quake.extract_ref %[[VAL_2]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.ry (%[[VAL_17]]) %[[VAL_20]] : (f64, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_21:.*]] = cc.load %[[VAL_11]] : !cc.ptr<f64>
# CHECK:               %[[VAL_22:.*]] = arith.constant 4 : i64
# CHECK:               %[[VAL_23:.*]] = arith.remui %[[VAL_9]], %[[VAL_22]] : i64
# CHECK:               %[[VAL_24:.*]] = quake.extract_ref %[[VAL_2]]{{\[}}%[[VAL_23]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.rx (%[[VAL_21]]) %[[VAL_24]] : (f64, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_9]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_25:.*]]: i64):
# CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_5]] : i64
# CHECK:             cc.continue %[[VAL_26]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_6:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
# CHECK:           %[[VAL_7:.*]] = cc.loop while ((%[[VAL_8:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_6]] : i64
# CHECK:             cc.condition %[[VAL_9]](%[[VAL_8]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
# CHECK:             %[[VAL_11:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:             %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_11]]{{\[}}%[[VAL_10]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_13:.*]] = arith.remui %[[VAL_10]], %[[VAL_1]] : i64
# CHECK:             %[[VAL_14:.*]] = arith.cmpi ne, %[[VAL_13]], %[[VAL_3]] : i64
# CHECK:             cc.if(%[[VAL_14]]) {
# CHECK:               %[[VAL_15:.*]] = cc.load %[[VAL_12]] : !cc.ptr<f64>
# CHECK:               %[[VAL_16:.*]] = arith.remui %[[VAL_10]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_17:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_16]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.ry (%[[VAL_15]]) %[[VAL_17]] : (f64, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_18:.*]] = cc.load %[[VAL_12]] : !cc.ptr<f64>
# CHECK:               %[[VAL_19:.*]] = arith.remui %[[VAL_10]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_20:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_19]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.rx (%[[VAL_18]]) %[[VAL_20]] : (f64, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_10]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_21:.*]]: i64):
# CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_22]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }

