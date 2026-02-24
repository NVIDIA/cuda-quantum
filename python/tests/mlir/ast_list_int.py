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
        # Also test out len() here, should convert to stdvec_size
        n = len(hidden_bitstring)
        for index, bit in enumerate(hidden_bitstring):
            if bit == 1:
                x.ctrl(register[index], auxillary_qubit)

    print(oracle)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__oracle..
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.ref, %[[VAL_2:.*]]: !cc.stdvec<i64>) attributes {"cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_6:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_7:.*]] = cc.stdvec_size %[[VAL_2]] : (!cc.stdvec<i64>) -> i64
# CHECK:           %[[VAL_8:.*]]:3 = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_4]], %[[VAL_10:.*]] = %[[VAL_6]], %[[VAL_11:.*]] = %[[VAL_5]]) -> (i64, i64, i64)) {
# CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_9]], %[[VAL_7]] : i64
# CHECK:             cc.condition %[[VAL_12]](%[[VAL_9]], %[[VAL_10]], %[[VAL_11]] : i64, i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_13:.*]]: i64, %[[VAL_14:.*]]: i64, %[[VAL_15:.*]]: i64):
# CHECK:             %[[VAL_16:.*]] = cc.stdvec_data %[[VAL_2]] : (!cc.stdvec<i64>) -> !cc.ptr<!cc.array<i64 x ?>>
# CHECK:             %[[VAL_17:.*]] = cc.compute_ptr %[[VAL_16]]{{\[}}%[[VAL_13]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_18:.*]] = cc.load %[[VAL_17]] : !cc.ptr<i64>
# CHECK:             %[[VAL_19:.*]] = arith.cmpi eq, %[[VAL_18]], %[[VAL_3]] : i64
# CHECK:             cc.if(%[[VAL_19]]) {
# CHECK:               %[[VAL_20:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_20]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:             }
# CHECK:             cc.continue %[[VAL_13]], %[[VAL_13]], %[[VAL_18]] : i64, i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_21:.*]]: i64, %[[VAL_22:.*]]: i64, %[[VAL_23:.*]]: i64):
# CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_21]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_24]], %[[VAL_22]], %[[VAL_23]] : i64, i64, i64
# CHECK:           }
# CHECK:           return
# CHECK:         }
