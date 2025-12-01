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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__func_achat(
# CHECK-SAME:                                            %[[VAL_0:.*]]: !quake.veq<?>) -> !cc.stdvec<i1> attributes {"cudaq-kernel", qubitMeasurementFeedback = true} {
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_0]] : (!quake.veq<?>) -> !cc.stdvec<!quake.measure>
# CHECK:           %[[VAL_3:.*]] = quake.discriminate %[[VAL_2]] : (!cc.stdvec<!quake.measure>) -> !cc.stdvec<i1>
# CHECK:           %[[VAL_4:.*]] = cc.stdvec_data %[[VAL_3]] : (!cc.stdvec<i1>) -> !cc.ptr<!cc.array<i8 x ?>>
# CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_3]] : (!cc.stdvec<i1>) -> i64
# CHECK:           %[[VAL_6:.*]] = cc.cast %[[VAL_4]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
# CHECK:           %[[VAL_7:.*]] = call @__nvqpp_vectorCopyCtor(%[[VAL_6]], %[[VAL_5]], %[[VAL_1]]) : (!cc.ptr<i8>, i64, i64) -> !cc.ptr<i8>
# CHECK:           %[[VAL_8:.*]] = cc.stdvec_init %[[VAL_7]], %[[VAL_5]] : (!cc.ptr<i8>, i64) -> !cc.stdvec<i1>
# CHECK:           return %[[VAL_8]] : !cc.stdvec<i1>
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__func_shiim(
# CHECK-SAME:                                            %[[VAL_0:.*]]: !quake.veq<?>) -> i64 attributes {"cudaq-kernel", qubitMeasurementFeedback = true} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_10:.*]] = arith.constant 0 : i8
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = quake.subveq %[[VAL_0]], 1, 2 : (!quake.veq<?>) -> !quake.veq<2>
# CHECK:           %[[VAL_4:.*]] = quake.relax_size %[[VAL_3]] : (!quake.veq<2>) -> !quake.veq<?>
# CHECK:           %[[VAL_5:.*]] = call @__nvqpp__mlirgen__func_achat(%[[VAL_4]]) : (!quake.veq<?>) -> !cc.stdvec<i1>
# CHECK:           %[[VAL_6:.*]] = cc.stdvec_data %[[VAL_5]] : (!cc.stdvec<i1>) -> !cc.ptr<i8>
# CHECK:           %[[VAL_7:.*]] = cc.stdvec_size %[[VAL_5]] : (!cc.stdvec<i1>) -> i64
# CHECK:           %[[VAL_8:.*]] = cc.alloca i8{{\[}}%[[VAL_7]] : i64]
# CHECK:           %[[VAL_9:.*]] = cc.cast %[[VAL_8]] : (!cc.ptr<!cc.array<i8 x ?>>) -> !cc.ptr<i8>
# CHECK:           call @__nvqpp_vectorCopyToStack(%[[VAL_9]], %[[VAL_6]], %[[VAL_7]]) : (!cc.ptr<i8>, !cc.ptr<i8>, i64) -> ()
# CHECK:           %[[VAL_12:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_2]], %[[VAL_12]] : !cc.ptr<i64>
# CHECK:           %[[VAL_15:.*]] = cc.loop while ((%[[VAL_16:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_17:.*]] = arith.cmpi slt, %[[VAL_16]], %[[VAL_7]] : i64
# CHECK:             cc.condition %[[VAL_17]](%[[VAL_16]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_18:.*]]: i64):
# CHECK:             %[[VAL_20:.*]] = cc.compute_ptr %[[VAL_8]]{{\[}}%[[VAL_18]]] : (!cc.ptr<!cc.array<i8 x ?>>, i64) -> !cc.ptr<i8>
# CHECK:             %[[VAL_21:.*]] = cc.load %[[VAL_20]] : !cc.ptr<i8>
# CHECK:             %[[VAL_11:.*]] = arith.cmpi ne, %[[VAL_21]], %[[VAL_10]] : i8
# CHECK:             %[[VAL_13:.*]] = cc.alloca i1
# CHECK:             cc.store %[[VAL_11]], %[[VAL_13]] : !cc.ptr<i1>
# CHECK:             %[[VAL_14:.*]] = cc.load %[[VAL_13]] : !cc.ptr<i1>
# CHECK:             cc.if(%[[VAL_14]]) {
# CHECK:               %[[VAL_23:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i64>
# CHECK:               %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_1]] : i64
# CHECK:               cc.store %[[VAL_24]], %[[VAL_12]] : !cc.ptr<i64>
# CHECK:             }
# CHECK:             cc.continue %[[VAL_18]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_25:.*]]: i64):
# CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_26]] : i64
# CHECK:           }
# CHECK:           %[[VAL_27:.*]] = cc.load %[[VAL_12]] : !cc.ptr<i64>
# CHECK:           return %[[VAL_27]] : i64
# CHECK:         }
