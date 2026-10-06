# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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
        # Also test out len() here, should convert to sequence_size
        n = len(hidden_bitstring)
        for index, bit in enumerate(hidden_bitstring):
            if bit == 1:
                x.ctrl(register[index], auxillary_qubit)

    print(oracle)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__oracle..
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.ref, %[[VAL_2:.*]]: !cc.sequence<i64>) attributes {"cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_6:.*]] = cc.sequence_size %[[VAL_2]] : (!cc.sequence<i64>) -> i64
# CHECK:           %[[VAL_7:.*]] = cc.loop while ((%[[VAL_8:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_6]] : i64
# CHECK:             cc.condition %[[VAL_10]](%[[VAL_8]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
# CHECK:             %[[VAL_13:.*]] = cc.sequence_data %[[VAL_2]] : (!cc.sequence<i64>) -> !cc.ptr<!cc.array<i64 x ?>>
# CHECK:             %[[VAL_14:.*]] = cc.compute_ptr %[[VAL_13]]{{\[}}%[[VAL_11]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_15:.*]] = cc.load %[[VAL_14]] : !cc.ptr<i64>
# CHECK:             %[[VAL_16:.*]] = arith.cmpi eq, %[[VAL_15]], %[[VAL_3]] : i64
# CHECK:             cc.if(%[[VAL_16]]) {
# CHECK:               %[[VAL_17:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_11]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_17]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:             }
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_18:.*]]: i64):
# CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_18]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_20]] : i64
# CHECK:           }
# CHECK:           return
# CHECK:         }
