# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq
import numpy


def test_list_init():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(6)
        params = [1., 2., 3., 4.]
        for i, p in enumerate(params):
            ry(p, q[i])

    print(kernel)
    kernel()


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel..
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 4.000000e+00 : f64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 3.000000e+00 : f64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 2.000000e+00 : f64
# CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 1.000000e+00 : f64
# CHECK-DAG:       %[[VAL_7:.*]] = cc.undef f64
# CHECK-DAG:       %[[VAL_8:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_9:.*]] = quake.alloca !quake.veq<6>
# CHECK-DAG:       %[[VAL_10:.*]] = cc.alloca !cc.array<f64 x 4>
# CHECK:           %[[VAL_11:.*]] = cc.cast %[[VAL_10]] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
# CHECK:           cc.store %[[VAL_6]], %[[VAL_11]] : !cc.ptr<f64>
# CHECK:           %[[VAL_12:.*]] = cc.compute_ptr %[[VAL_10]][1] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
# CHECK:           cc.store %[[VAL_5]], %[[VAL_12]] : !cc.ptr<f64>
# CHECK:           %[[VAL_13:.*]] = cc.compute_ptr %[[VAL_10]][2] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
# CHECK:           cc.store %[[VAL_4]], %[[VAL_13]] : !cc.ptr<f64>
# CHECK:           %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_10]][3] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
# CHECK:           cc.store %[[VAL_3]], %[[VAL_14]] : !cc.ptr<f64>
# CHECK:           %[[VAL_15:.*]]:3 = cc.loop while ((%[[VAL_16:.*]] = %[[VAL_1]], %[[VAL_17:.*]] = %[[VAL_8]], %[[VAL_18:.*]] = %[[VAL_7]]) -> (i64, i64, f64)) {
# CHECK:             %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_2]] : i64
# CHECK:             cc.condition %[[VAL_19]](%[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : i64, i64, f64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_20:.*]]: i64, %[[VAL_21:.*]]: i64, %[[VAL_22:.*]]: f64):
# CHECK:             %[[VAL_23:.*]] = cc.compute_ptr %[[VAL_10]]{{\[}}%[[VAL_20]]] : (!cc.ptr<!cc.array<f64 x 4>>, i64) -> !cc.ptr<f64>
# CHECK:             %[[VAL_24:.*]] = cc.load %[[VAL_23]] : !cc.ptr<f64>
# CHECK:             %[[VAL_25:.*]] = quake.extract_ref %[[VAL_9]]{{\[}}%[[VAL_20]]] : (!quake.veq<6>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_24]]) %[[VAL_25]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_20]], %[[VAL_20]], %[[VAL_24]] : i64, i64, f64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_26:.*]]: i64, %[[VAL_27:.*]]: i64, %[[VAL_28:.*]]: f64):
# CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_26]], %[[VAL_0]] : i64
# CHECK:             cc.continue %[[VAL_29]], %[[VAL_27]], %[[VAL_28]] : i64, i64, f64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_9]] : !quake.veq<6>
# CHECK:           return
# CHECK:         }


def test_list_conversion_fail():

    try:

        @cudaq.kernel
        def test1(params: list[complex]):
            angles = numpy.array(params, dtype=float)
            q = cudaq.qubit()
            for a in angles:
                rz(a, q)

        print(test1)
    except Exception as e:
        print("Failure for test1:")
        print(e)


# CHECK-LABEL:  Failure for test1:
# CHECK:        cannot convert value of type complex<f64> to the requested type f64
# CHECK-NEXT:   (offending source -> numpy.array(params, dtype=float))
