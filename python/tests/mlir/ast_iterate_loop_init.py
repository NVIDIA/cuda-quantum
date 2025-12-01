# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_iterate_list_init():

    @cudaq.kernel
    def kernel(x: float):
        q = cudaq.qvector(4)
        for i in [0, 1, 2, 3]:
            x = x + i
            ry(x, q[i % 4])

    print(kernel)
    kernel(1.2)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel(
# CHECK-SAME:                                        %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 3 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 2 : i64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:           %[[VAL_5:.*]] = arith.constant 4 : i64
# CHECK:           %[[VAL_6:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_6]] : !cc.ptr<f64>
# CHECK:           %[[VAL_7:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_8:.*]] = cc.alloca !cc.array<i64 x 4>
# CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_8]] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_4]], %[[VAL_9]] : !cc.ptr<i64>
# CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_8]][1] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_3]], %[[VAL_10]] : !cc.ptr<i64>
# CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_8]][2] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_2]], %[[VAL_11]] : !cc.ptr<i64>
# CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_8]][3] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_1]], %[[VAL_12]] : !cc.ptr<i64>
# CHECK:           %[[VAL_13:.*]] = cc.loop while ((%[[VAL_14:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_5]] : i64
# CHECK:             cc.condition %[[VAL_15]](%[[VAL_14]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64):
# CHECK:             %[[VAL_18:.*]] = cc.compute_ptr %[[VAL_8]][%[[VAL_16]]] : (!cc.ptr<!cc.array<i64 x 4>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_19:.*]] = cc.load %[[VAL_18]] : !cc.ptr<i64>
# CHECK:             %[[VAL_30:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_19]], %[[VAL_30]] : !cc.ptr<i64>
# CHECK:             %[[VAL_20:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
# CHECK:             %[[VAL_31:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i64>
# CHECK:             %[[VAL_21:.*]] = cc.cast signed %[[VAL_31]] : (i64) -> f64
# CHECK:             %[[VAL_22:.*]] = arith.addf %[[VAL_20]], %[[VAL_21]] : f64
# CHECK:             cc.store %[[VAL_22]], %[[VAL_6]] : !cc.ptr<f64>
# CHECK:             %[[VAL_23:.*]] = cc.load %[[VAL_6]] : !cc.ptr<f64>
# CHECK:             %[[VAL_32:.*]] = cc.load %[[VAL_30]] : !cc.ptr<i64>
# CHECK:             %[[VAL_24:.*]] = arith.remui %[[VAL_32]], %[[VAL_5]] : i64
# CHECK:             %[[VAL_25:.*]] = quake.extract_ref %[[VAL_7]]{{\[}}%[[VAL_24]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_23]]) %[[VAL_25]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_16]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_26:.*]]: i64):
# CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_26]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_27]] : i64
# CHECK:           }
# CHECK:           return
# CHECK:         }
