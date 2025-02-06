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
# CHECK-SAME:                                      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 2 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 0 : i64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 4 : i64
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_6:.*]] = cc.stdvec_size %[[VAL_0]] : (!cc.stdvec<f64>) -> i64
# CHECK:           %[[VAL_21:.*]] = cc.loop while ((%[[VAL_22:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_6]] : i64
# CHECK:             cc.condition %[[VAL_23]](%[[VAL_22]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_24:.*]]: i64):
# CHECK:             %[[VAL_7:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<i64 x ?>>
# CHECK:             %[[VAL_25:.*]] = cc.compute_ptr %[[VAL_7]][%[[VAL_24]]] : (!cc.ptr<!cc.array<f64> x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_29:.*]] = arith.remui %[[VAL_24]], %[[VAL_1]] : i64
# CHECK:             %[[VAL_30:.*]] = arith.cmpi ne, %[[VAL_29]], %[[VAL_3]] : i64
# CHECK:             %[[VAL_28:.*]] = cc.load %[[VAL_25]] : !cc.ptr<f64>
# CHECK:             cc.if(%[[VAL_30]]) {
# CHECK:               %[[VAL_31:.*]] = arith.remui %[[VAL_24]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_32:.*]] = quake.extract_ref %[[VAL_5]][%[[VAL_31]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.ry (%[[VAL_28]]) %[[VAL_32]] : (f64, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_33:.*]] = arith.remui %[[VAL_24]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_34:.*]] = quake.extract_ref %[[VAL_5]][%[[VAL_33]]] : (!quake.veq<4>, i64) -> !quake.ref
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
