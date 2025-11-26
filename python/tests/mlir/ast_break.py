# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq
import numpy


def test_break():

    @cudaq.kernel(verbose=False)
    def kernel(x: float):
        q = cudaq.qvector(4)
        for i in range(10):
            x = x + x**2
            if x > 5:
                break
            ry(x, q[i % 4])

    kernel(1.2)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 5.000000e+00 : f64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 10 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_7:.*]] = cc.alloca f64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_7]] : !cc.ptr<f64>
# CHECK:           %[[VAL_8:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_5]] : i64
# CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_12:.*]]: i64):
# CHECK:             %[[VAL_23:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_12]], %[[VAL_23]] : !cc.ptr<i64>
# CHECK:             %[[VAL_13:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_14:.*]] = math.fpowi %[[VAL_13]], %[[VAL_2]] : f64, i64
# CHECK:             %[[VAL_15:.*]] = arith.addf %[[VAL_13]], %[[VAL_14]] : f64
# CHECK:             cc.store %[[VAL_15]], %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_16:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_17:.*]] = arith.cmpf ogt, %[[VAL_16]], %[[VAL_1]] : f64
# CHECK:             cf.cond_br %[[VAL_17]], ^bb1, ^bb2
# CHECK:           ^bb1:
# CHECK:             cc.break %[[VAL_12]] : i64
# CHECK:           ^bb2:
# CHECK:             %[[VAL_18:.*]] = cc.load %[[VAL_7]] : !cc.ptr<f64>
# CHECK:             %[[VAL_24:.*]] = cc.load %[[VAL_23]] : !cc.ptr<i64>
# CHECK:             %[[VAL_19:.*]] = arith.remui %[[VAL_24]], %[[VAL_6]] : i64
# CHECK:             %[[VAL_20:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_19]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_18]]) %[[VAL_20]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_12]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_21:.*]]: i64):
# CHECK:             %[[VAL_22:.*]] = arith.addi %[[VAL_21]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_22]] : i64
# CHECK:           }
# CHECK:           return
# CHECK:         }


def test_break2():

    @cudaq.kernel(verbose=False)
    def kernel(x: float):
        qs = cudaq.qvector(6)
        for idx, q in enumerate(qs):
            if idx > 2:
                break
            ry(x, q)

    res = cudaq.sample(kernel, numpy.pi)
    assert len(res) == 1 and '111000' in res
    print(f"output test1: {res}")


# CHECK-LABEL:  output test1: { 111000:1000 }
