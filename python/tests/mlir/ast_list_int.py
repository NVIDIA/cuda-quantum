# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s



import cudaq


def test_list_int():

    @cudaq.kernel
    def oracle(register: cudaq.qview, auxillary_qubit: cudaq.qubit,
               hidden_bitstring: list[int]):
        # Also test out len() here, should convert to stdvec_size
        x = len(hidden_bitstring)
        for index, bit in enumerate(hidden_bitstring):
            if bit == 1:
                x.ctrl(register[index], auxillary_qubit)

    print(oracle)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__oracle(
# CHECK-SAME:                                        %[[VAL_0:.*]]: !quake.veq<?>,
# CHECK-SAME:                                        %[[VAL_1:.*]]: !quake.ref,
# CHECK-SAME:                                        %[[VAL_2:.*]]: !cc.stdvec<i64>) {
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_2]] : (!cc.stdvec<i64>) -> i64
# CHECK:           %[[VAL_6:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_5]], %[[VAL_6]] : !cc.ptr<i64>
# CHECK:           %[[VAL_7:.*]] = cc.alloca !cc.struct<{i64, i64}>{{\[}}%[[VAL_5]] : i64]
# CHECK:           %[[VAL_8:.*]] = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_5]] : i64
# CHECK:             cc.condition %[[VAL_10]](%[[VAL_9]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
# CHECK:             %[[VAL_12:.*]] = cc.undef !cc.struct<{i64, i64}>
# CHECK:             %[[VAL_13:.*]] = cc.stdvec_data %[[VAL_2]] : (!cc.stdvec<i64>) -> !cc.ptr<!cc.array<i64 x ?>>
# CHECK:             %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_13]][%[[VAL_11]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<i64>
# CHECK:             %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_11]]] : (!cc.ptr<!cc.array<!cc.struct<{i64, i64}> x ?>>, i64) -> !cc.ptr<!cc.struct<{i64, i64}>>
# CHECK:             %[[VAL_17:.*]] = cc.insert_value %[[VAL_11]], %[[VAL_12]][0] : (!cc.struct<{i64, i64}>, i64) -> !cc.struct<{i64, i64}>
# CHECK:             %[[VAL_18:.*]] = cc.insert_value %[[VAL_15]], %[[VAL_17]][1] : (!cc.struct<{i64, i64}>, i64) -> !cc.struct<{i64, i64}>
# CHECK:             cc.store %[[VAL_18]], %[[VAL_16]] : !cc.ptr<!cc.struct<{i64, i64}>>
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_19:.*]]: i64):
# CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_19]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_20]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_21:.*]] = cc.loop while ((%[[VAL_22:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_23:.*]] = arith.cmpi slt, %[[VAL_22]], %[[VAL_5]] : i64
# CHECK:             cc.condition %[[VAL_23]](%[[VAL_22]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_24:.*]]: i64):
# CHECK:             %[[VAL_25:.*]] = cc.compute_ptr %[[VAL_7]]{{\[}}%[[VAL_24]]] : (!cc.ptr<!cc.array<!cc.struct<{i64, i64}> x ?>>, i64) -> !cc.ptr<!cc.struct<{i64, i64}>>
# CHECK:             %[[VAL_26:.*]] = cc.load %[[VAL_25]] : !cc.ptr<!cc.struct<{i64, i64}>>
# CHECK:             %[[VAL_27:.*]] = cc.extract_value %[[VAL_26]][0] : (!cc.struct<{i64, i64}>) -> i64
# CHECK:             %[[VAL_28:.*]] = cc.extract_value %[[VAL_26]][1] : (!cc.struct<{i64, i64}>) -> i64
# CHECK:             %[[VAL_29:.*]] = arith.cmpi eq, %[[VAL_28]], %[[VAL_3]] : i64
# CHECK:             cc.if(%[[VAL_29]]) {
# CHECK:               %[[VAL_30:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_27]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_30]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_24]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_31:.*]]: i64):
# CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_31]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_32]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }
