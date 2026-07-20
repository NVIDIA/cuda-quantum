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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost..
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2.000000e+00 : f64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = cc.undef f64
# CHECK-DAG:       %[[VAL_7:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_8:.*]] = quake.alloca !quake.veq<4>
# CHECK-DAG:       %[[VAL_9:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
# CHECK:           %[[VAL_10:.*]]:3 = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_4]], %[[VAL_12:.*]] = %[[VAL_7]], %[[VAL_13:.*]] = %[[VAL_6]]) -> (i64, i64, f64)) {
# CHECK:             %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_9]] : i64
# CHECK:             cc.condition %[[VAL_14]](%[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : i64, i64, f64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_15:.*]]: i64, %[[VAL_16:.*]]: i64, %[[VAL_17:.*]]: f64):
# CHECK:             %[[VAL_18:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:             %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_18]]{{\[}}%[[VAL_15]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_20:.*]] = cc.load %[[VAL_19]] : !cc.ptr<f64>
# CHECK:             %[[VAL_21:.*]] = cc.cast signed %[[VAL_15]] : (i64) -> f64
# CHECK:             %[[VAL_22:.*]] = arith.remf %[[VAL_21]], %[[VAL_2]] : f64
# CHECK:             %[[VAL_23:.*]] = arith.cmpf une, %[[VAL_22]], %[[VAL_1]] : f64
# CHECK:             cc.if(%[[VAL_23]]) {
# CHECK:               %[[VAL_24:.*]] = arith.remui %[[VAL_15]], %[[VAL_5]] : i64
# CHECK:               %[[VAL_25:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_24]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.ry (%[[VAL_20]]) %[[VAL_25]] : (f64, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_26:.*]] = arith.remui %[[VAL_15]], %[[VAL_5]] : i64
# CHECK:               %[[VAL_27:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_26]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.rx (%[[VAL_20]]) %[[VAL_27]] : (f64, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_15]], %[[VAL_15]], %[[VAL_20]] : i64, i64, f64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_28:.*]]: i64, %[[VAL_29:.*]]: i64, %[[VAL_30:.*]]: f64):
# CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_28]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_31]], %[[VAL_29]], %[[VAL_30]] : i64, i64, f64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_8]] : !quake.veq<4>
# CHECK:           return
# CHECK:         }
