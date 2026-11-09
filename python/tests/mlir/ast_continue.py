# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_continue():

    @cudaq.kernel
    def kernel_ok(xmen: float):
        q = cudaq.qvector(4)
        for i in range(10):
            xmen = xmen + xmen**2
            if xmen > 10:
                x(q[i % 4])
                continue
            ry(xmen, q[i % 4])

    kernel_ok(1.2)
    print(kernel_ok)

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_ok..
    # CHECK-SAME:      %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
    # CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1.000000e+01 : f64
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
    # CHECK:             %[[VAL_22:.*]] = arith.remsi %[[VAL_14]], %[[VAL_6]] : i64
    # CHECK:             %[[VAL_23:.*]] = arith.cmpi ne, %[[VAL_22]], %[[VAL_5]] : i64
    # CHECK:             %[[VAL_24:.*]] = arith.xori %[[VAL_22]], %[[VAL_6]] : i64
    # CHECK:             %[[VAL_25:.*]] = arith.cmpi slt, %[[VAL_24]], %[[VAL_5]] : i64
    # CHECK:             %[[VAL_26:.*]] = arith.andi %[[VAL_23]], %[[VAL_25]] : i1
    # CHECK:             %[[VAL_27:.*]] = arith.addi %[[VAL_22]], %[[VAL_6]] : i64
    # CHECK:             %[[VAL_28:.*]] = arith.select %[[VAL_26]], %[[VAL_27]], %[[VAL_22]] : i64
    # CHECK:             %[[VAL_29:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_28]]] : (!quake.veq<4>, i64) -> !quake.ref
    # CHECK:             quake.x %[[VAL_29]] : (!quake.ref) -> ()
    # CHECK:             cc.continue %[[VAL_14]], %[[VAL_18]] : i64, f64
    # CHECK:           ^bb2:
    # CHECK:             %[[VAL_30:.*]] = arith.remsi %[[VAL_14]], %[[VAL_6]] : i64
    # CHECK:             %[[VAL_31:.*]] = arith.cmpi ne, %[[VAL_30]], %[[VAL_5]] : i64
    # CHECK:             %[[VAL_32:.*]] = arith.xori %[[VAL_30]], %[[VAL_6]] : i64
    # CHECK:             %[[VAL_33:.*]] = arith.cmpi slt, %[[VAL_32]], %[[VAL_5]] : i64
    # CHECK:             %[[VAL_34:.*]] = arith.andi %[[VAL_31]], %[[VAL_33]] : i1
    # CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_30]], %[[VAL_6]] : i64
    # CHECK:             %[[VAL_36:.*]] = arith.select %[[VAL_34]], %[[VAL_35]], %[[VAL_30]] : i64
    # CHECK:             %[[VAL_37:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_36]]] : (!quake.veq<4>, i64) -> !quake.ref
    # CHECK:             quake.ry (%[[VAL_18]]) %[[VAL_37]] : (f64, !quake.ref) -> ()
    # CHECK:             cc.continue %[[VAL_14]], %[[VAL_18]] : i64, f64
    # CHECK:           } step {
    # CHECK:           ^bb0(%[[VAL_38:.*]]: i64, %[[VAL_40:.*]]: f64):
    # CHECK:             %[[VAL_41:.*]] = arith.addi %[[VAL_38]], %[[VAL_4]] : i64
    # CHECK:             cc.continue %[[VAL_41]], %[[VAL_40]] : i64, f64
    # CHECK:           }
    # CHECK:           quake.dealloc %[[VAL_8]] : !quake.veq<4>
    # CHECK:           return

    try:

        @cudaq.kernel
        def kernel(x: float):
            q = cudaq.qvector(4)
            for i in range(10):
                x = x + x**2
                if x > 10:
                    x(q[i % 4])
                    continue
                ry(x, q[i % 4])

        kernel(1.2)
    except Exception as e:
        print("kernel:")
        print(e)


# CHECK-LABEL: kernel:
# CHECK-NEXT: object is not callable
# CHECK-NEXT: offending source -> x(q
