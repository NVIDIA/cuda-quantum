# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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
    print(result)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test
# CHECK-SAME:      () attributes {"cudaq-entrypoint", "cudaq-kernel", qubitMeasurementFeedback = true} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant false
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant true
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_6:.*]]:2 = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_2]], %[[VAL_8:.*]] = %[[VAL_3]]) -> (i64, i1)) {
# CHECK:             %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_4]] : i64
# CHECK:             cc.condition %[[VAL_9]](%[[VAL_7]], %[[VAL_8]] : i64, i1)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64, %[[VAL_11:.*]]: i1):
# CHECK:             %[[VAL_12:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_10]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             %[[VAL_13:.*]] = quake.mz %[[VAL_12]] name "res" : (!quake.ref) -> !quake.measure
# CHECK:             %[[VAL_14:.*]] = quake.discriminate %[[VAL_13]] : (!quake.measure) -> i1
# CHECK:             %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_14]], %[[VAL_0]] : i1
# CHECK:             cc.if(%[[VAL_15]]) {
# CHECK:               %[[VAL_16:.*]] = quake.mz %[[VAL_5]] name "inner_mz" : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
# CHECK:             }
# CHECK:             cc.continue %[[VAL_10]], %[[VAL_14]] : i64, i1
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_17:.*]]: i64, %[[VAL_18:.*]]: i1):
# CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_17]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_19]], %[[VAL_18]] : i64, i1
# CHECK:           }
# CHECK:           %[[VAL_20:.*]] = arith.cmpi eq, %[[VAL_21:.*]]#1, %[[VAL_3]] : i1
# CHECK:           cc.if(%[[VAL_20]]) {
# CHECK:             %[[VAL_22:.*]] = quake.mz %[[VAL_5]] name "outer_mz" : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_5]] : !quake.veq<2>
# CHECK:           return
# CHECK:         }

# CHECK:         {
# CHECK-DAG:         __global__ : { 00:1000 }
# CHECK-DAG:          inner_mz : { 0000:1000 }
# CHECK-DAG:          res : { 0:2000 }
# CHECK:         }
