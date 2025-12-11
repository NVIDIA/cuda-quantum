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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost
# CHECK-SAME:      (%[[VAL_0:.*]]: !cc.stdvec<f64>)
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2.000000e+00 : f64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_7:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
# CHECK:           %[[VAL_8:.*]] = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_7]] : i64
# CHECK:             cc.condition %[[VAL_10]](%[[VAL_9]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
# CHECK:             %[[VAL_12:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:             %[[VAL_13:.*]] = cc.compute_ptr %[[VAL_12]]{{\[}}%[[VAL_11]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_14:.*]] = cc.cast signed %[[VAL_11]] : (i64) -> f64
# CHECK:             %[[VAL_15:.*]] = arith.remf %[[VAL_14]], %[[VAL_2]] : f64
# CHECK:             %[[VAL_16:.*]] = arith.cmpf une, %[[VAL_15]], %[[VAL_1]] : f64
# CHECK:             cc.if(%[[VAL_16]]) {
# CHECK:               %[[VAL_17:.*]] = cc.load %[[VAL_13]] : !cc.ptr<f64>
# CHECK:               %[[VAL_18:.*]] = arith.remui %[[VAL_11]], %[[VAL_5]] : i64
# CHECK:               %[[VAL_19:.*]] = quake.extract_ref %[[VAL_6]]{{\[}}%[[VAL_18]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.ry (%[[VAL_17]]) %[[VAL_19]] : (f64, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_20:.*]] = cc.load %[[VAL_13]] : !cc.ptr<f64>
# CHECK:               %[[VAL_21:.*]] = arith.remui %[[VAL_11]], %[[VAL_5]] : i64
# CHECK:               %[[VAL_22:.*]] = quake.extract_ref %[[VAL_6]]{{\[}}%[[VAL_21]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.rx (%[[VAL_20]]) %[[VAL_22]] : (f64, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_23:.*]]: i64):
# CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_24]] : i64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_6]] : !quake.veq<4>
# CHECK:           return
# CHECK:         }
