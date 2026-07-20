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


def test_for_ndarray():

    @cudaq.kernel
    def cost(thetas: np.ndarray):  # can pass 1D ndarray or list
        q = cudaq.qvector(4)
        i = 0
        for theta in thetas:
            ry(theta, q[i])
            i += 1

    print(cost)
    cost(np.asarray([1., 2., 3., 4.]))


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost..
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = cc.undef f64
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.veq<4>
# CHECK-DAG:       %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
# CHECK:           %[[VAL_6:.*]]:3 = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_2]], %[[VAL_8:.*]] = %[[VAL_3]], %[[VAL_9:.*]] = %[[VAL_2]]) -> (i64, f64, i64)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_5]] : i64
# CHECK:             cc.condition %[[VAL_10]](%[[VAL_7]], %[[VAL_8]], %[[VAL_9]] : i64, f64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64, %[[VAL_12:.*]]: f64, %[[VAL_13:.*]]: i64):
# CHECK:             %[[VAL_14:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:             %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_14]]{{\[}}%[[VAL_11]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_16:.*]] = cc.load %[[VAL_15]] : !cc.ptr<f64>
# CHECK:             %[[VAL_17:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_13]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_16]]) %[[VAL_17]] : (f64, !quake.ref) -> ()
# CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_13]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_11]], %[[VAL_16]], %[[VAL_18]] : i64, f64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_19:.*]]: i64, %[[VAL_20:.*]]: f64, %[[VAL_21:.*]]: i64):
# CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_19]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_22]], %[[VAL_20]], %[[VAL_21]] : i64, f64, i64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_4]] : !quake.veq<4>
# CHECK:           return
