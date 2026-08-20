# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_control_kernel():

    @cudaq.kernel
    def reflect(qubits: cudaq.qview):
        ctrls = qubits.front(qubits.size() - 1)
        last = qubits.back()

        def compute():
            h(qubits)
            x(qubits)

        cudaq.compute_action(compute, lambda: z.ctrl(ctrls, last))

    print(reflect)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__reflect..0x
# CHECK-SAME:      %[[ARG0:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
# CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[CONSTANT_2:.*]] = arith.constant 1 : i64
# CHECK:           %[[VEQ_SIZE_0:.*]] = quake.veq_size %[[ARG0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[SUBI_0:.*]] = arith.subi %[[VEQ_SIZE_0]], %[[CONSTANT_0]] : i64
# CHECK:           %[[SUBVEQ_0:.*]] = quake.subveq %[[ARG0]], 0, %[[SUBI_0]] : (!quake.veq<?>, i64) -> !quake.veq<?>
# CHECK:           %[[VEQ_SIZE_1:.*]] = quake.veq_size %[[ARG0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[SUBI_1:.*]] = arith.subi %[[VEQ_SIZE_1]], %[[CONSTANT_2]] : i64
# CHECK:           %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[SUBI_1]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:           %[[VEQ_SIZE_2:.*]] = quake.veq_size %[[ARG0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[LOOP_0:.*]] = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_1]]) -> (i64)) {
# CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[VEQ_SIZE_2]] : i64
# CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_1:.*]]: i64):
# CHECK:             %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[VAL_1]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[EXTRACT_REF_1]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_1]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_2:.*]]: i64):
# CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_2]] : i64
# CHECK:             cc.continue %[[ADDI_0]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VEQ_SIZE_3:.*]] = quake.veq_size %[[ARG0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[LOOP_1:.*]] = cc.loop while ((%[[VAL_3:.*]] = %[[CONSTANT_1]]) -> (i64)) {
# CHECK:             %[[CMPI_1:.*]] = arith.cmpi slt, %[[VAL_3]], %[[VEQ_SIZE_3]] : i64
# CHECK:             cc.condition %[[CMPI_1]](%[[VAL_3]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_4:.*]]: i64):
# CHECK:             %[[EXTRACT_REF_2:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[VAL_4]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x %[[EXTRACT_REF_2]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_4]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_5:.*]]: i64):
# CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_5]], %[[CONSTANT_2]] : i64
# CHECK:             cc.continue %[[ADDI_1]] : i64
# CHECK:           } {invariant}
# CHECK:           quake.z {{\[}}%[[SUBVEQ_0]]] %[[EXTRACT_REF_0]] : (!quake.veq<?>, !quake.ref) -> ()
# CHECK:           %[[VEQ_SIZE_4:.*]] = quake.veq_size %[[ARG0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[CMPI_4:.*]] = arith.cmpi sgt, %[[VEQ_SIZE_4]], %[[CONSTANT_1]] : i64
# CHECK:           %[[SELECT_2:.*]] = arith.select %[[CMPI_4]], %[[VEQ_SIZE_4]], %[[CONSTANT_1]] : i64
# CHECK:           %[[VEQ_SIZE_5:.*]] = quake.veq_size %[[ARG0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[CMPI_5:.*]] = arith.cmpi sgt, %[[VEQ_SIZE_5]], %[[CONSTANT_1]] : i64
# CHECK:           %[[SELECT_3:.*]] = arith.select %[[CMPI_5]], %[[VEQ_SIZE_5]], %[[CONSTANT_1]] : i64
# CHECK:           %[[CMPI_6:.*]] = arith.cmpi sgt, %[[SELECT_3]], %[[CONSTANT_1]] : i64
# CHECK:           %[[SELECT_4:.*]] = arith.select %[[CMPI_6]], %[[SELECT_3]], %[[CONSTANT_1]] : i64
# CHECK:           %[[SUBI_2:.*]] = arith.subi %[[SELECT_4]], %[[CONSTANT_2]] : i64
# CHECK:           %[[LOOP_2:.*]]:2 = cc.loop while ((%[[VAL_6:.*]] = %[[SUBI_2]], %[[VAL_7:.*]] = %[[SELECT_4]]) -> (i64, i64)) {
# CHECK:             %[[CMPI_7:.*]] = arith.cmpi sgt, %[[VAL_7]], %[[CONSTANT_1]] : i64
# CHECK:             cc.condition %[[CMPI_7]](%[[VAL_6]], %[[VAL_7]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_8:.*]]: i64, %[[VAL_9:.*]]: i64):
# CHECK:             %[[EXTRACT_REF_3:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x %[[EXTRACT_REF_3]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_8]], %[[VAL_9]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64, %[[VAL_11:.*]]: i64):
# CHECK:             %[[SUBI_3:.*]] = arith.subi %[[VAL_10]], %[[CONSTANT_2]] : i64
# CHECK:             %[[SUBI_4:.*]] = arith.subi %[[VAL_11]], %[[CONSTANT_2]] : i64
# CHECK:             cc.continue %[[SUBI_3]], %[[SUBI_4]] : i64, i64
# CHECK:           }
# CHECK:           %[[CMPI_8:.*]] = arith.cmpi sgt, %[[SELECT_2]], %[[CONSTANT_1]] : i64
# CHECK:           %[[SELECT_5:.*]] = arith.select %[[CMPI_8]], %[[SELECT_2]], %[[CONSTANT_1]] : i64
# CHECK:           %[[SUBI_5:.*]] = arith.subi %[[SELECT_5]], %[[CONSTANT_2]] : i64
# CHECK:           %[[LOOP_3:.*]]:2 = cc.loop while ((%[[VAL_12:.*]] = %[[SUBI_5]], %[[VAL_13:.*]] = %[[SELECT_5]]) -> (i64, i64)) {
# CHECK:             %[[CMPI_9:.*]] = arith.cmpi sgt, %[[VAL_13]], %[[CONSTANT_1]] : i64
# CHECK:             cc.condition %[[CMPI_9]](%[[VAL_12]], %[[VAL_13]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_14:.*]]: i64, %[[VAL_15:.*]]: i64):
# CHECK:             %[[EXTRACT_REF_4:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[VAL_14]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[EXTRACT_REF_4]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_14]], %[[VAL_15]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64, %[[VAL_17:.*]]: i64):
# CHECK:             %[[SUBI_6:.*]] = arith.subi %[[VAL_16]], %[[CONSTANT_2]] : i64
# CHECK:             %[[SUBI_7:.*]] = arith.subi %[[VAL_17]], %[[CONSTANT_2]] : i64
# CHECK:             cc.continue %[[SUBI_6]], %[[SUBI_7]] : i64, i64
# CHECK:           }
# CHECK:           return
# CHECK:         }
