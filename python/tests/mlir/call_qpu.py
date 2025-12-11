# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest

import cudaq


# Check that we have `__nvqpp_vectorCopyToStack` on returned vectors from a QPU call.
def test_qpu_call_return_vector():

    @cudaq.kernel
    def func_achat(qv: cudaq.qvector) -> list[bool]:
        # measure the entire register
        return mz(qv)

    print(func_achat)

    @cudaq.kernel
    def func_shiim(qv: cudaq.qvector) -> int:
        vs = qv[1:3]
        bs = func_achat(vs)
        i = 0
        for b in bs:
            if b:
                i += 1
        return i

    print(func_shiim)

    @cudaq.kernel
    def main_kernel() -> int:
        qv = cudaq.qvector(8)
        x(qv)
        count = func_shiim(qv)
        return count

    results = cudaq.run(main_kernel)
    # All should be 2 since we slice out 2 qubits.
    for v in results:
        assert v == 2


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__func_achat
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>) -> !cc.stdvec<i1> attributes {"cudaq-kernel", qubitMeasurementFeedback = true} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant false
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 8 : i64
# CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
# CHECK:           %[[VAL_4:.*]] = quake.discriminate %[[VAL_3]] : (!cc.stdvec<!quake.measure>) -> !cc.stdvec<i1>
# CHECK:           %[[VAL_5:.*]] = cc.stdvec_data %[[VAL_4]] : (!cc.stdvec<i1>) -> !cc.ptr<!cc.array<i8 x ?>>
# CHECK:           %[[VAL_6:.*]] = cc.stdvec_size %[[VAL_4]] : (!cc.stdvec<i1>) -> i64
# CHECK:           %[[VAL_7:.*]] = cc.cast %[[VAL_5]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
# CHECK:           %[[VAL_8:.*]] = arith.muli %[[VAL_6]], %[[VAL_2]] : i64
# CHECK:           %[[VAL_9:.*]] = call @malloc(%[[VAL_8]]) : (i64) -> !cc.ptr<i8>
# CHECK:           call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_9]], %[[VAL_7]], %[[VAL_8]], %[[VAL_1]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
# CHECK:           %[[VAL_10:.*]] = cc.stdvec_init %[[VAL_9]], %[[VAL_6]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
# CHECK:           return %[[VAL_10]] : !cc.stdvec<i1>
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__func_shiim
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>,
# CHECK-SAME:      %[[VAL_1:.*]]: !cc.callable<(!quake.veq<?>) -> !cc.stdvec<i1>> {quake.pylifted}) -> i64 attributes {"cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_4:.*]] = quake.subveq %[[VAL_0]], 1, 2 : (!quake.veq<?>) -> !quake.veq<2>
# CHECK:           %[[VAL_5:.*]] = quake.relax_size %[[VAL_4]] : (!quake.veq<2>) -> !quake.veq<?>
# CHECK:           %[[VAL_6:.*]] = cc.call_callable %[[VAL_1]], %[[VAL_5]] : (!cc.callable<(!quake.veq<?>) -> !cc.stdvec<i1>>, !quake.veq<?>) -> !cc.stdvec<i1> {symbol = "func_achat"}
# CHECK:           %[[VAL_7:.*]] = cc.stdvec_size %[[VAL_6]] : (!cc.stdvec<i1>) -> i64
# CHECK:           %[[VAL_8:.*]]:2 = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_2]], %[[VAL_10:.*]] = %[[VAL_2]]) -> (i64, i64)) {
# CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_7]] : i64
# CHECK:             cc.condition %[[VAL_11]](%[[VAL_9]], %[[VAL_10]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_12:.*]]: i64, %[[VAL_13:.*]]: i64):
# CHECK:             %[[VAL_14:.*]] = cc.stdvec_data %[[VAL_6]] : (!cc.stdvec<i1>) -> !cc.ptr<!cc.array<i8 x ?>>
# CHECK:             %[[VAL_15:.*]] = cc.compute_ptr %[[VAL_14]]{{\[}}%[[VAL_12]]] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
# CHECK:             %[[VAL_16:.*]] = cc.load %[[VAL_15]] : !cc.ptr<i8>
# CHECK:             %[[VAL_17:.*]] = cc.cast %[[VAL_16]] : (i8) -> i1
# CHECK:             %[[VAL_18:.*]] = cc.if(%[[VAL_17]]) -> i64 {
# CHECK:               %[[VAL_19:.*]] = arith.addi %[[VAL_13]], %[[VAL_3]] : i64
# CHECK:               cc.continue %[[VAL_19]] : i64
# CHECK:             } else {
# CHECK:               cc.continue %[[VAL_13]] : i64
# CHECK:             }
# CHECK:             cc.continue %[[VAL_12]], %[[VAL_20:.*]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_21:.*]]: i64, %[[VAL_22:.*]]: i64):
# CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_21]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_23]], %[[VAL_22]] : i64, i64
# CHECK:           }
# CHECK:           return %[[VAL_24:.*]]#1 : i64
# CHECK:         }
