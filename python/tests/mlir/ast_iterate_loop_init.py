# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel..
# CHECK-SAME:      %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 3 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_7:.*]] = quake.alloca !quake.veq<4>
# CHECK-DAG:       %[[VAL_8:.*]] = cc.alloca !cc.array<i64 x 4>
# CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_8]] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_4]], %[[VAL_9]] : !cc.ptr<i64>
# CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_8]][1] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_3]], %[[VAL_10]] : !cc.ptr<i64>
# CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_8]][2] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_2]], %[[VAL_11]] : !cc.ptr<i64>
# CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_8]][3] : (!cc.ptr<!cc.array<i64 x 4>>) -> !cc.ptr<i64>
# CHECK:           cc.store %[[VAL_1]], %[[VAL_12]] : !cc.ptr<i64>
# CHECK:           %[[VAL_13:.*]]:3 = cc.loop while ((%[[VAL_14:.*]] = %[[VAL_4]], %[[VAL_15:.*]] = %[[VAL_6]], %[[VAL_16:.*]] = %[[VAL_0]]) -> (i64, i64, f64)) {
# CHECK:             %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_5]] : i64
# CHECK:             cc.condition %[[VAL_17]](%[[VAL_14]], %[[VAL_15]], %[[VAL_16]] : i64, i64, f64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_18:.*]]: i64, %[[VAL_19:.*]]: i64, %[[VAL_20:.*]]: f64):
# CHECK:             %[[VAL_21:.*]] = cc.compute_ptr %[[VAL_8]]{{\[}}%[[VAL_18]]] : (!cc.ptr<!cc.array<i64 x 4>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_22:.*]] = cc.load %[[VAL_21]] : !cc.ptr<i64>
# CHECK:             %[[VAL_23:.*]] = cc.cast signed %[[VAL_22]] : (i64) -> f64
# CHECK:             %[[VAL_24:.*]] = arith.addf %[[VAL_20]], %[[VAL_23]] : f64
# CHECK:             %[[VAL_25:.*]] = arith.remui %[[VAL_22]], %[[VAL_5]] : i64
# CHECK:             %[[VAL_26:.*]] = quake.extract_ref %[[VAL_7]]{{\[}}%[[VAL_25]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_24]]) %[[VAL_26]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_18]], %[[VAL_22]], %[[VAL_24]] : i64, i64, f64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_27:.*]]: i64, %[[VAL_28:.*]]: i64, %[[VAL_29:.*]]: f64):
# CHECK:             %[[VAL_30:.*]] = arith.addi %[[VAL_27]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_30]], %[[VAL_28]], %[[VAL_29]] : i64, i64, f64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_7]] : !quake.veq<4>
# CHECK:           return
# CHECK:         }
