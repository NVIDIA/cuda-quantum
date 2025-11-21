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
            if i % 2.0:  # asserting we convert i to float
                ry(theta, q[i % 4])
            else:
                rx(theta, q[i % 4])

    cost(np.asarray([1., 2., 3., 4., 5., 6.]))
    cost([1., 2., 3., 4., 5., 6.])
    print(cost)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 4 : i64
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_1]] : i64]
# CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_5:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_3:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
# CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_3]] : i64
# CHECK:             cc.condition %[[VAL_8]](%[[VAL_7]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:             %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_10]]{{\[}}%[[VAL_9]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_17:.*]] = cc.load %[[VAL_11]] : !cc.ptr<f64>
# CHECK:             %[[VAL_30:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_9]], %[[VAL_30]] : !cc.ptr<i64>
# CHECK:             %[[VAL_31:.*]] = cc.alloca f64
# CHECK:             cc.store %[[VAL_17]], %[[VAL_31]] : !cc.ptr<f64>
# CHECK:             %[[VAL_32:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i64>
# CHECK:             %[[VAL_12:.*]] = arith.constant 2.000000e+00 : f64
# CHECK:             %[[VAL_13:.*]] = cc.cast signed %[[VAL_32]] : (i64) -> f64
# CHECK:             %[[VAL_14:.*]] = arith.remf %[[VAL_13]], %[[VAL_12]] : f64
# CHECK:             %[[VAL_15:.*]] = arith.constant 0.000000e+00 : f64
# CHECK:             %[[VAL_16:.*]] = arith.cmpf une, %[[VAL_14]], %[[VAL_15]] : f64
# CHECK:             cc.if(%[[VAL_16]]) {
# CHECK:               %[[VAL_33:.*]] = cc.load %[[VAL_31]] : !cc.ptr<f64>
# CHECK:               %[[VAL_34:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i64>
# CHECK:               %[[VAL_18:.*]] = arith.constant 4 : i64
# CHECK:               %[[VAL_19:.*]] = arith.remui %[[VAL_34]], %[[VAL_18]] : i64
# CHECK:               %[[VAL_20:.*]] = quake.extract_ref %[[VAL_2]]{{\[}}%[[VAL_19]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.ry (%[[VAL_33]]) %[[VAL_20]] : (f64, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_33:.*]] = cc.load %[[VAL_31]] : !cc.ptr<f64>
# CHECK:               %[[VAL_34:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i64>
# CHECK:               %[[VAL_22:.*]] = arith.constant 4 : i64
# CHECK:               %[[VAL_23:.*]] = arith.remui %[[VAL_34]], %[[VAL_22]] : i64
# CHECK:               %[[VAL_24:.*]] = quake.extract_ref %[[VAL_2]]{{\[}}%[[VAL_23]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.rx (%[[VAL_33]]) %[[VAL_24]] : (f64, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_9]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_25:.*]]: i64):
# CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_5]] : i64
# CHECK:             cc.continue %[[VAL_26]] : i64
# CHECK:           }
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL:.*]] = arith.constant 0.000000e+00 : f64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2.000000e+00 : f64
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
# CHECK:             %[[VAL_16:.*]] = cc.load %[[VAL_12]] : !cc.ptr<f64>
# CHECK:             %[[VAL_30:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_10]], %[[VAL_30]] : !cc.ptr<i64>
# CHECK:             %[[VAL_31:.*]] = cc.alloca f64
# CHECK:             cc.store %[[VAL_16]], %[[VAL_31]] : !cc.ptr<f64>
# CHECK:             %[[VAL_32:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i64>
# CHECK:             %[[VAL_13:.*]] = cc.cast signed %[[VAL_32]] : (i64) -> f64
# CHECK:             %[[VAL_14:.*]] = arith.remf %[[VAL_13]], %[[VAL_1]] : f64
# CHECK:             %[[VAL_15:.*]] = arith.cmpf une, %[[VAL_14]], %[[VAL]] : f64
# CHECK:             cc.if(%[[VAL_15]]) {
# CHECK:               %[[VAL_33:.*]] = cc.load %[[VAL_31]] : !cc.ptr<f64>
# CHECK:               %[[VAL_34:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i64>
# CHECK:               %[[VAL_17:.*]] = arith.remui %[[VAL_34]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_18:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_17]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.ry (%[[VAL_33]]) %[[VAL_18]] : (f64, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_33:.*]] = cc.load %[[VAL_31]] : !cc.ptr<f64>
# CHECK:               %[[VAL_34:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i64>
# CHECK:               %[[VAL_20:.*]] = arith.remui %[[VAL_34]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_21:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_20]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.rx (%[[VAL_33]]) %[[VAL_21]] : (f64, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_10]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_22:.*]]: i64):
# CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_23]] : i64
# CHECK:           }
# CHECK:           return
# CHECK:         }