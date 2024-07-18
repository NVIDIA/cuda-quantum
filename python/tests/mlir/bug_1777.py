# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq

import pytest


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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test() attributes {"cudaq-entrypoint", qubitMeasurementFeedback = true} {
# CHECK:           %[[VAL_0:.*]] = arith.constant false
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = arith.constant true
# CHECK:           %[[VAL_4:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_6:.*]] = cc.alloca i1
# CHECK:           cc.store %[[VAL_3]], %[[VAL_6]] : !cc.ptr<i1>
# CHECK:           %[[VAL_7:.*]] = cc.loop while ((%[[VAL_8:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_4]] : i64
# CHECK:             cc.condition %[[VAL_9]](%[[VAL_8]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
# CHECK:             %[[VAL_11:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_10]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             %[[VAL_12:.*]] = quake.mz %[[VAL_11]] name "res" : (!quake.ref) -> !quake.measure
# CHECK:             %[[VAL_13:.*]] = quake.discriminate %[[VAL_12]] : (!quake.measure) -> i1
# CHECK:             cc.store %[[VAL_13]], %[[VAL_6]] : !cc.ptr<i1>
# CHECK:             %[[VAL_14:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i1>
# CHECK:             %[[VAL_15:.*]] = arith.cmpi eq, %[[VAL_14]], %[[VAL_0]] : i1
# CHECK:             cc.if(%[[VAL_15]]) {
# CHECK:               %[[VAL_16:.*]] = quake.mz %[[VAL_5]] name "inner_mz" : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
# CHECK:               %[[VAL_17:.*]] = quake.discriminate %[[VAL_16]] : (!cc.stdvec<!quake.measure>) -> !cc.stdvec<i1>
# CHECK:               %[[VAL_18:.*]] = cc.alloca !cc.stdvec<i1>
# CHECK:               cc.store %[[VAL_17]], %[[VAL_18]] : !cc.ptr<!cc.stdvec<i1>>
# CHECK:             }
# CHECK:             cc.continue %[[VAL_10]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_19:.*]]: i64):
# CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_20]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_21:.*]] = cc.load %[[VAL_6]] : !cc.ptr<i1>
# CHECK:           %[[VAL_22:.*]] = arith.cmpi eq, %[[VAL_21]], %[[VAL_3]] : i1
# CHECK:           cc.if(%[[VAL_22]]) {
# CHECK:             %[[VAL_23:.*]] = quake.mz %[[VAL_5]] name "outer_mz" : (!quake.veq<2>) -> !cc.stdvec<!quake.measure>
# CHECK:             %[[VAL_24:.*]] = quake.discriminate %[[VAL_23]] : (!cc.stdvec<!quake.measure>) -> !cc.stdvec<i1>
# CHECK:             %[[VAL_25:.*]] = cc.alloca !cc.stdvec<i1>
# CHECK:             cc.store %[[VAL_24]], %[[VAL_25]] : !cc.ptr<!cc.stdvec<i1>>
# CHECK:           }
# CHECK:           return
# CHECK:         }

# CHECK:         __global__ : { 00:1000 }
# CHECK:          inner_mz : { 0000:1000 }
# CHECK:          res : { 0:2000 }
