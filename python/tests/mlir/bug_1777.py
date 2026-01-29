# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_bug_1777():

    @cudaq.kernel
    def test():
        qubits = cudaq.qvector(2)

        res = True
        for i in range(2):
            res = mz(qubits[i])
            if res == False:
                inner_mz = mz(qubits)

        if res == True:
            outer_mz = mz(qubits)

    print(test)
    result = cudaq.sample(test)
    print("Results!")
    print(result)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test..
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel", qubitMeasurementFeedback = true} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant false
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant true
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_6:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_7:.*]]:3 = cc.loop while ((%[[VAL_8:.*]] = %[[VAL_2]], %[[VAL_9:.*]] = %[[VAL_5]], %[[VAL_10:.*]] = %[[VAL_3]]) -> (i64, i64, i1)) {
# CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_4]] : i64
# CHECK:             cc.condition %[[VAL_11]](%[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : i64, i64, i1)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_12:.*]]: i64, %[[VAL_13:.*]]: i64, %[[VAL_14:.*]]: i1):
# CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_6]]{{\[}}%[[VAL_12]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             %[[VAL_16:.*]] = quake.mz %[[VAL_15]] name "res" : (!quake.ref) -> !quake.measure
# CHECK:             %[[VAL_17:.*]] = quake.discriminate %[[VAL_16]] : (!quake.measure) -> i1
# CHECK:             %[[VAL_18:.*]] = arith.cmpi eq, %[[VAL_17]], %[[VAL_0]] : i1
# CHECK:             cc.if(%[[VAL_18]]) {
# CHECK:               %[[VAL_19:.*]] = quake.mz %[[VAL_6]] name "inner_mz" : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
# CHECK:             } else {
# CHECK:             }
# CHECK:             cc.continue %[[VAL_12]], %[[VAL_12]], %[[VAL_17]] : i64, i64, i1
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_20:.*]]: i64, %[[VAL_21:.*]]: i64, %[[VAL_22:.*]]: i1):
# CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_20]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_23]], %[[VAL_21]], %[[VAL_22]] : i64, i64, i1
# CHECK:           }
# CHECK:           %[[VAL_24:.*]] = arith.cmpi eq, %[[VAL_25:.*]]#2, %[[VAL_3]] : i1
# CHECK:           cc.if(%[[VAL_24]]) {
# CHECK:             %[[VAL_26:.*]] = quake.mz %[[VAL_6]] name "outer_mz" : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
# CHECK:           } else {
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_6]] : !quake.veq<2>
# CHECK:           return
# CHECK:         }

# CHECK-LABEL: Results!
# CHECK:         {
# CHECK-DAG:         __global__ : { 00:1000 }
# CHECK-DAG:          inner_mz : { 0000:1000 }
# CHECK-DAG:          res : { 0:2000 }
# CHECK:         }
