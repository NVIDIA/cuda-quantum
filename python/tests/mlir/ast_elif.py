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
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.sequence<f64>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0.000000e+00 : f64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2.000000e+00 : f64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = cc.undef f64
# CHECK-DAG:       %[[VAL_7:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_8:.*]] = quake.alloca !quake.veq<4>
# CHECK-DAG:       %[[VAL_9:.*]] = cc.sequence_size %[[VAL_0]] : (!cc.sequence<f64>) -> i64
# CHECK:           %[[VAL_10:.*]]:3 = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_4]], %[[VAL_12:.*]] = %[[VAL_7]], %[[VAL_13:.*]] = %[[VAL_6]]) -> (i64, i64, f64)) {
# CHECK:             %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_9]] : i64
# CHECK:             cc.condition %[[VAL_14]](%[[VAL_11]], %[[VAL_12]], %[[VAL_13]] : i64, i64, f64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_15:.*]]: i64, %[[VAL_16:.*]]: i64, %[[VAL_17:.*]]: f64):
# CHECK:             %[[VAL_18:.*]] = cc.sequence_data %[[VAL_0]] : (!cc.sequence<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:             %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_18]]{{\[}}%[[VAL_15]]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_20:.*]] = cc.load %[[VAL_19]] : !cc.ptr<f64>
# CHECK:             %[[VAL_21:.*]] = cc.cast signed %[[VAL_15]] : (i64) -> f64
# CHECK:             %[[VAL_22:.*]] = arith.remf %[[VAL_21]], %[[VAL_2]] : f64
# CHECK:             %[[VAL_23:.*]] = arith.cmpf une, %[[VAL_22]], %[[VAL_1]] : f64
# CHECK:             %[[VAL_24:.*]] = arith.cmpf olt, %[[VAL_22]], %[[VAL_1]] : f64
# CHECK:             %[[VAL_25:.*]] = arith.andi %[[VAL_23]], %[[VAL_24]] : i1
# CHECK:             %[[VAL_26:.*]] = arith.addf %[[VAL_22]], %[[VAL_2]] : f64
# CHECK:             %[[VAL_27:.*]] = arith.select %[[VAL_25]], %[[VAL_26]], %[[VAL_22]] : f64
# CHECK:             %[[VAL_28:.*]] = arith.select %[[VAL_23]], %[[VAL_27]], %[[VAL_1]] : f64
# CHECK:             %[[VAL_29:.*]] = arith.cmpf une, %[[VAL_28]], %[[VAL_1]] : f64
# CHECK:             cc.if(%[[VAL_29]]) {
# CHECK:               %[[VAL_30:.*]] = arith.remsi %[[VAL_15]], %[[VAL_5]] : i64
# CHECK:               %[[VAL_31:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_32:.*]] = arith.xori %[[VAL_30]], %[[VAL_5]] : i64
# CHECK:               %[[VAL_33:.*]] = arith.cmpi slt, %[[VAL_32]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_34:.*]] = arith.andi %[[VAL_31]], %[[VAL_33]] : i1
# CHECK:               %[[VAL_35:.*]] = arith.addi %[[VAL_30]], %[[VAL_5]] : i64
# CHECK:               %[[VAL_36:.*]] = arith.select %[[VAL_34]], %[[VAL_35]], %[[VAL_30]] : i64
# CHECK:               %[[VAL_37:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_36]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.ry (%[[VAL_20]]) %[[VAL_37]] : (f64, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_38:.*]] = arith.remsi %[[VAL_15]], %[[VAL_5]] : i64
# CHECK:               %[[VAL_39:.*]] = arith.cmpi ne, %[[VAL_38]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_40:.*]] = arith.xori %[[VAL_38]], %[[VAL_5]] : i64
# CHECK:               %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_40]], %[[VAL_4]] : i64
# CHECK:               %[[VAL_42:.*]] = arith.andi %[[VAL_39]], %[[VAL_41]] : i1
# CHECK:               %[[VAL_43:.*]] = arith.addi %[[VAL_38]], %[[VAL_5]] : i64
# CHECK:               %[[VAL_44:.*]] = arith.select %[[VAL_42]], %[[VAL_43]], %[[VAL_38]] : i64
# CHECK:               %[[VAL_45:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_44]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.rx (%[[VAL_20]]) %[[VAL_45]] : (f64, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_15]], %[[VAL_15]], %[[VAL_20]] : i64, i64, f64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_46:.*]]: i64, %[[VAL_47:.*]]: i64, %[[VAL_48:.*]]: f64):
# CHECK:             %[[VAL_49:.*]] = arith.addi %[[VAL_46]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_49]], %[[VAL_47]], %[[VAL_48]] : i64, i64, f64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_8]] : !quake.veq<4>
# CHECK:           return
# CHECK:         }
