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

    @cudaq.kernel
    def cost(thetas: np.ndarray):  # can pass 1D ndarray or list
        q = cudaq.qvector(4)
        i = 0
        for theta in thetas:
            ry(theta, q[i])
            i += 1

    print(cost)
    cost(np.asarray([1., 2., 3., 4.]))


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__cost(
# CHECK-SAME:                                      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint"} {
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_4:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_2]], %[[VAL_4]] : !cc.ptr<i64>
# CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
# CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_5]] : i64
# CHECK:             cc.condition %[[VAL_8]](%[[VAL_7]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:             %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_10]][%[[VAL_9]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_12:.*]] = cc.load %[[VAL_11]] : !cc.ptr<f64>
# CHECK:             %[[VAL_13:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
# CHECK:             %[[VAL_14:.*]] = quake.extract_ref %[[VAL_3]][%[[VAL_13]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_12]]) %[[VAL_14]] : (f64, !quake.ref) -> ()
# CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
# CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_1]] : i64
# CHECK:             cc.store %[[VAL_16]], %[[VAL_4]] : !cc.ptr<i64>
# CHECK:             cc.continue %[[VAL_9]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_17:.*]]: i64):
# CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_18]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }
