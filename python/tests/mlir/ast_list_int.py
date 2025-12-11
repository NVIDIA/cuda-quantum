# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__oracle
# CHECK-SAME:      (%[[VAL_0:.*]]: !quake.veq<?>, %[[VAL_1:.*]]: !quake.ref, %[[VAL_2:.*]]: !cc.stdvec<i64>) attributes {"cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_5:.*]] = cc.stdvec_size %[[VAL_2]] : (!cc.stdvec<i64>) -> i64
# CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_5]] : i64
# CHECK:             cc.condition %[[VAL_8]](%[[VAL_7]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = cc.stdvec_data %[[VAL_2]] : (!cc.stdvec<i64>) -> !cc.ptr<!cc.array<i64 x ?>>
# CHECK:             %[[VAL_11:.*]] = cc.compute_ptr %[[VAL_10]]{{\[}}%[[VAL_9]]] : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.ptr<i64>
# CHECK:             %[[VAL_12:.*]] = cc.load %[[VAL_11]] : !cc.ptr<i64>
# CHECK:             %[[VAL_13:.*]] = arith.cmpi eq, %[[VAL_12]], %[[VAL_3]] : i64
# CHECK:             cc.if(%[[VAL_13]]) {
# CHECK:               %[[VAL_14:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_9]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_14]]] %[[VAL_1]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_9]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_15:.*]]: i64):
# CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_16]] : i64
# CHECK:           return
