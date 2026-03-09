# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest

import cudaq


# Check that we have `__nvqpp_vectorCopyToStack` on returned vectors from a QPU
# call.
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

    print(main_kernel)
    results = cudaq.run(main_kernel)
    # All should be 2 since we slice out 2 qubits.
    for v in results:
        assert v == 2


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__func_achat..
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>) -> !cc.stdvec<i1> attributes {"cudaq-kernel", qubitMeasurementFeedback = true} {
# CHECK:           %[[VAL_1:.*]] = arith.constant false
# CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
# CHECK:           %[[VAL_3:.*]] = quake.discriminate %[[VAL_2]] : (!cc.stdvec<!quake.measure>) -> !cc.stdvec<i1>
# CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<i1>) -> !cc.ptr<!cc.array<i8 x ?>>
# CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_3]] : (!cc.stdvec<i1>) -> i64
# CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
# CHECK:           %[[VAL_7:.*]] = call @malloc(%[[VAL_5]]) : (i64) -> !cc.ptr<i8>
# CHECK:           call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_7]], %[[VAL_6]], %[[VAL_5]], %[[VAL_1]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
# CHECK:           %[[VAL_8:.*]] = cc.stdvec_init %[[VAL_7]], %[[VAL_5]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
# CHECK:           return %[[VAL_8]] : !cc.stdvec<i1>
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__func_shiim..
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>,
# CHECK-SAME:      %[[VAL_1:.*]]: !cc.callable<(!quake.veq<?>) -> !cc.stdvec<i1>> {quake.pylifted}) -> i64 attributes {"cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant false
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i8
# CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = cc.undef i1
# CHECK:           %[[VAL_7:.*]] = quake.subveq %[[VAL_0]], 1, 2 : (!quake.veq<?>) -> !quake.veq<2>
# CHECK:           %[[VAL_8:.*]] = quake.relax_size %[[VAL_7]] : (!quake.veq<2>) -> !quake.veq<?>
# CHECK:           %[[VAL_9:.*]] = cc.call_callable %[[VAL_1]], %[[VAL_8]] : (!cc.callable<(!quake.veq<?>) -> !cc.stdvec<i1>>, !quake.veq<?>) -> !cc.stdvec<i1> {symbol = "func_achat"}
# CHECK:           %[[VAL_10:.*]] = cc.stdvec_data %[[VAL_9]] : (!cc.stdvec<i1>) -> !cc.ptr<!cc.array<i8 x ?>>
# CHECK:           %[[VAL_11:.*]] = cc.stdvec_size %[[VAL_9]] : (!cc.stdvec<i1>) -> i64
# CHECK:           %[[VAL_12:.*]] = cc.cast %[[VAL_10]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
# CHECK:           %[[VAL_13:.*]] = cc.alloca i8{{\[}}%[[VAL_11]] : i64]
# CHECK:           %[[VAL_14:.*]] = cc.cast %[[VAL_13]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
# CHECK:           call @llvm.memcpy.p0i8.p0i8.i64(%[[VAL_14]], %[[VAL_12]], %[[VAL_11]], %[[VAL_2]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64, i1) -> ()
# CHECK:           call @free(%[[VAL_12]]) : (!cc.ptr<i8>) -> ()
# CHECK:           %[[VAL_15:.*]]:3 = cc.loop while ((%[[VAL_16:.*]] = %[[VAL_5]], %[[VAL_17:.*]] = %[[VAL_6]], %[[VAL_18:.*]] = %[[VAL_5]]) -> (i64, i1, i64)) {
# CHECK:             %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_11]] : i64
# CHECK:             cc.condition %[[VAL_19]](%[[VAL_16]], %[[VAL_17]], %[[VAL_18]] : i64, i1, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_20:.*]]: i64, %[[VAL_21:.*]]: i1, %[[VAL_22:.*]]: i64):
# CHECK:             %[[VAL_23:.*]] = cc.compute_ptr %[[VAL_13]]{{\[}}%[[VAL_20]]] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
# CHECK:             %[[VAL_24:.*]] = cc.load %[[VAL_23]] : !cc.ptr<i8>
# CHECK:             %[[VAL_25:.*]] = arith.cmpi ne, %[[VAL_24]], %[[VAL_4]] : i8
# CHECK:             %[[VAL_26:.*]] = cc.if(%[[VAL_25]]) -> i64 {
# CHECK:               %[[VAL_27:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : i64
# CHECK:               cc.continue %[[VAL_27]] : i64
# CHECK:             } else {
# CHECK:               cc.continue %[[VAL_22]] : i64
# CHECK:             }
# CHECK:             cc.continue %[[VAL_20]], %[[VAL_25]], %[[VAL_28:.*]] : i64, i1, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_29:.*]]: i64, %[[VAL_30:.*]]: i1, %[[VAL_31:.*]]: i64):
# CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_29]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_32]], %[[VAL_30]], %[[VAL_31]] : i64, i1, i64
# CHECK:           }
# CHECK:           return %[[VAL_33:.*]]#2 : i64
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__main_kernel..
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.callable<(!quake.veq<?>) -> i64> {quake.pylifted}) -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 8 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.veq<8>
# CHECK:           %[[VAL_5:.*]] = quake.relax_size %[[VAL_4]] : (!quake.veq<8>) -> !quake.veq<?>
# CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_8]](%[[VAL_7]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_9]]] : (!quake.veq<8>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_10]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_9]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
# CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_12]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_13:.*]] = cc.call_callable %[[VAL_0]], %[[VAL_5]] : (!cc.callable<(!quake.veq<?>) -> i64>, !quake.veq<?>) -> i64 {symbol = "func_shiim"}
# CHECK:           quake.dealloc %[[VAL_4]] : !quake.veq<8>
# CHECK:           return %[[VAL_13]] : i64
# CHECK:         }
