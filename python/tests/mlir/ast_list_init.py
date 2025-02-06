# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_list_init():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(6)
        params = [1., 2., 3., 4.]
        for i, p in enumerate(params):
            ry(p, q[i])

    print(kernel)
    kernel()


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 2.000000e+00 : f64
# CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1.000000e+00 : f64
# CHECK-DAG:       %[[VAL_7:.*]] = quake.alloca !quake.veq<6>
# CHECK-DAG:       %[[VAL_8:.*]] = cc.alloca !cc.array<f64 x 4>
# CHECK:           %[[VAL_91:.*]] = cc.cast %[[VAL_8]] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_8]] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
# CHECK:           cc.store %[[VAL_6]], %[[VAL_9]] : !cc.ptr<f64>
# CHECK:           %[[VAL_10:.*]] = cc.compute_ptr %[[VAL_8]][1] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
# CHECK:           cc.store %[[VAL_5]], %[[VAL_10]] : !cc.ptr<f64>
# CHECK:           %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_8]][2] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
# CHECK:           cc.store %[[VAL_4]], %[[VAL_11]] : !cc.ptr<f64>
# CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_8]][3] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
# CHECK:           cc.store %[[VAL_3]], %[[VAL_12]] : !cc.ptr<f64>
# CHECK:           %[[VAL_13:.*]] = cc.stdvec_init %[[VAL_91]], %[[VAL_2]] : (!cc.ptr<!cc.array<f64 x ?>>, i64) -> !cc.stdvec<f64>
# CHECK:           %[[VAL_14:.*]] = cc.alloca !cc.stdvec<f64>
# CHECK:           cc.store %[[VAL_13]], %[[VAL_14]] : !cc.ptr<!cc.stdvec<f64>>
# CHECK:           %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<!cc.stdvec<f64>>
# CHECK:           %[[VAL_16:.*]] = cc.stdvec_size %[[VAL_15]] : (!cc.stdvec<f64>) -> i64
# CHECK:           %[[VAL_31:.*]] = cc.loop while ((%[[VAL_32:.*]] = %[[VAL_1]]) -> (i64)) {
# CHECK:             %[[VAL_33:.*]] = arith.cmpi slt, %[[VAL_32]], %[[VAL_16]] : i64
# CHECK:             cc.condition %[[VAL_33]](%[[VAL_32]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_34:.*]]: i64):
# CHECK:             %[[VAL_17:.*]] = cc.stdvec_data %[[VAL_15]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:             %[[VAL_35:.*]] = cc.compute_ptr %[[VAL_17]][%[[VAL_34]]] : (!cc.ptr<!cc.array<f64> x ?>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_36:.*]] = cc.load %[[VAL_35]] : !cc.ptr<f64>
# CHECK:             %[[VAL_39:.*]] = quake.extract_ref %[[VAL_7]][%[[VAL_34]]] : (!quake.veq<6>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_36]]) %[[VAL_39]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_34]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_40:.*]]: i64):
# CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_40]], %[[VAL_0]] : i64
# CHECK:             cc.continue %[[VAL_41]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }
