# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_break():

    @cudaq.kernel(verbose=False)
    def kernel(x: float):
        q = cudaq.qvector(4)
        for i in range(10):
            x = x + x**2
            if x > 5:
                break
            ry(x, q[i % 4])

    print(kernel)
    kernel(1.2)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel..
# CHECK-SAME:      %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 5.000000e+00 : f64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 10 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_8:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_9:.*]]:2 = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_5]], %[[VAL_12:.*]] = %[[VAL_0]]) -> (i64, f64)) {
# CHECK:             %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_3]] : i64
# CHECK:             cc.condition %[[VAL_13]](%[[VAL_10]], %[[VAL_12]] : i64, f64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_14:.*]]: i64, %[[VAL_16:.*]]: f64):
# CHECK:             %[[VAL_17:.*]] = math.fpowi %[[VAL_16]], %[[VAL_2]] : f64, i64
# CHECK:             %[[VAL_18:.*]] = arith.addf %[[VAL_16]], %[[VAL_17]] : f64
# CHECK:             %[[VAL_19:.*]] = arith.cmpf ogt, %[[VAL_18]], %[[VAL_1]] : f64
# CHECK:             cf.cond_br %[[VAL_19]], ^bb1, ^bb2
# CHECK:           ^bb1:
# CHECK:             cc.break %[[VAL_14]], %[[VAL_18]] : i64, f64
# CHECK:           ^bb2:
# CHECK:             %[[VAL_24:.*]] = arith.remsi %[[VAL_14]], %[[VAL_6]] : i64
# CHECK:             %[[VAL_25:.*]] = arith.cmpi ne, %[[VAL_24]], %[[VAL_5]] : i64
# CHECK:             %[[VAL_26:.*]] = arith.xori %[[VAL_24]], %[[VAL_6]] : i64
# CHECK:             %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_5]] : i64
# CHECK:             %[[VAL_28:.*]] = arith.andi %[[VAL_25]], %[[VAL_27]] : i1
# CHECK:             %[[VAL_29:.*]] = arith.addi %[[VAL_24]], %[[VAL_6]] : i64
# CHECK:             %[[VAL_30:.*]] = arith.select %[[VAL_28]], %[[VAL_29]], %[[VAL_24]] : i64
# CHECK:             %[[VAL_31:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_30]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_18]]) %[[VAL_31]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_14]], %[[VAL_18]] : i64, f64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_32:.*]]: i64, %[[VAL_34:.*]]: f64):
# CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_32]], %[[VAL_4]] : i64
# CHECK:             cc.continue %[[VAL_35]], %[[VAL_34]] : i64, f64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_8]] : !quake.veq<4>
# CHECK:           return
# CHECK:         }
