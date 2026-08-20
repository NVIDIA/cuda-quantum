# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import numpy as np

import cudaq


def test_qft():

    @cudaq.kernel
    def iqft(qubits: cudaq.qview):
        N = qubits.size()
        for i in range(N // 2):
            swap(qubits[i], qubits[N - i - 1])

        for i in range(N - 1):
            h(qubits[i])
            j = i + 1
            for y in range(i, -1, -1):
                r1.ctrl(-np.pi / 2**(j - y), qubits[j], qubits[y])

        h(qubits[N - 1])

    print(iqft)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__iqft..
# CHECK-SAME:      (%[[ARG0:.*]]: !quake.veq<?>) attributes {"cudaq-kernel"} {
# CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant -3.1415926535897931 : f64
# CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant -1 : i64
# CHECK-DAG:       %[[CONSTANT_2:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[CONSTANT_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[CONSTANT_4:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[UNDEF_0:.*]] = cc.undef i64
# CHECK-DAG:       %[[UNDEF_1:.*]] = cc.undef i64
# CHECK:           %[[VEQ_SIZE_0:.*]] = quake.veq_size %[[ARG0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[FLOORDIVSI_0:.*]] = arith.floordivsi %[[VEQ_SIZE_0]], %[[CONSTANT_2]] : i64
# CHECK:           %[[LOOP_0:.*]] = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_4]]) -> (i64)) {
# CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[FLOORDIVSI_0]] : i64
# CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_1:.*]]: i64):
# CHECK:             %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[VAL_1]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[SUBI_0:.*]] = arith.subi %[[VEQ_SIZE_0]], %[[VAL_1]] : i64
# CHECK:             %[[SUBI_1:.*]] = arith.subi %[[SUBI_0]], %[[CONSTANT_3]] : i64
# CHECK:             %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[SUBI_1]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.swap %[[EXTRACT_REF_0]], %[[EXTRACT_REF_1]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_1]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_2:.*]]: i64):
# CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_3]] : i64
# CHECK:             cc.continue %[[ADDI_0]] : i64
# CHECK:           } {normalized}
# CHECK:           %[[SUBI_2:.*]] = arith.subi %[[VEQ_SIZE_0]], %[[CONSTANT_3]] : i64
# CHECK:           %[[LOOP_1:.*]]:3 = cc.loop while ((%[[VAL_3:.*]] = %[[CONSTANT_4]], %[[VAL_4:.*]] = %[[UNDEF_1]], %[[VAL_5:.*]] = %[[UNDEF_0]]) -> (i64, i64, i64)) {
# CHECK:             %[[CMPI_1:.*]] = arith.cmpi slt, %[[VAL_3]], %[[SUBI_2]] : i64
# CHECK:             cc.condition %[[CMPI_1]](%[[VAL_3]], %[[VAL_4]], %[[VAL_5]] : i64, i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_6:.*]]: i64, %[[VAL_7:.*]]: i64, %[[VAL_8:.*]]: i64):
# CHECK:             %[[EXTRACT_REF_2:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[VAL_6]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[EXTRACT_REF_2]] : (!quake.ref) -> ()
# CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_6]], %[[CONSTANT_3]] : i64
# CHECK:             %[[SUBI_3:.*]] = arith.subi %[[CONSTANT_1]], %[[VAL_6]] : i64
# CHECK:             %[[DIVSI_0:.*]] = arith.divsi %[[SUBI_3]], %[[CONSTANT_1]] : i64
# CHECK:             %[[CMPI_2:.*]] = arith.cmpi sgt, %[[DIVSI_0]], %[[CONSTANT_4]] : i64
# CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPI_2]], %[[DIVSI_0]], %[[CONSTANT_4]] : i64
# CHECK:             %[[LOOP_2:.*]] = cc.loop while ((%[[VAL_9:.*]] = %[[CONSTANT_4]]) -> (i64)) {
# CHECK:               %[[CMPI_3:.*]] = arith.cmpi ne, %[[VAL_9]], %[[SELECT_0]] : i64
# CHECK:               cc.condition %[[CMPI_3]](%[[VAL_9]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_10:.*]]: i64):
# CHECK:               %[[SUBI_4:.*]] = arith.subi %[[VAL_6]], %[[VAL_10]] : i64
# CHECK:               %[[SUBI_5:.*]] = arith.subi %[[ADDI_1]], %[[SUBI_4]] : i64
# CHECK:               %[[IPOWI_0:.*]] = math.ipowi %[[CONSTANT_2]], %[[SUBI_5]] : i64
# CHECK:               %[[CAST_0:.*]] = cc.cast signed %[[IPOWI_0]] : (i64) -> f64
# CHECK:               %[[DIVF_0:.*]] = arith.divf %[[CONSTANT_0]], %[[CAST_0]] : f64
# CHECK:               %[[EXTRACT_REF_3:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[ADDI_1]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               %[[EXTRACT_REF_4:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[SUBI_4]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.r1 (%[[DIVF_0]]) {{\[}}%[[EXTRACT_REF_3]]] %[[EXTRACT_REF_4]] : (f64, !quake.ref, !quake.ref) -> ()
# CHECK:               cc.continue %[[VAL_10]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_11:.*]]: i64):
# CHECK:               %[[ADDI_2:.*]] = arith.addi %[[VAL_11]], %[[CONSTANT_3]] : i64
# CHECK:               cc.continue %[[ADDI_2]] : i64
# CHECK:             } {normalized}
# CHECK:             %[[SUBI_6:.*]] = arith.subi %[[LOOP_2]], %[[CONSTANT_3]] : i64
# CHECK:             %[[CMPI_4:.*]] = arith.cmpi eq, %[[LOOP_2]], %[[CONSTANT_4]] : i64
# CHECK:             %[[SELECT_1:.*]] = arith.select %[[CMPI_4]], %[[VAL_8]], %[[SUBI_6]] : i64
# CHECK:             cc.continue %[[VAL_6]], %[[ADDI_1]], %[[SELECT_1]] : i64, i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_12:.*]]: i64, %[[VAL_13:.*]]: i64, %[[VAL_14:.*]]: i64):
# CHECK:             %[[ADDI_3:.*]] = arith.addi %[[VAL_12]], %[[CONSTANT_3]] : i64
# CHECK:             cc.continue %[[ADDI_3]], %[[VAL_13]], %[[VAL_14]] : i64, i64, i64
# CHECK:           } {normalized}
# CHECK:           %[[SUBI_7:.*]] = arith.subi %[[VEQ_SIZE_0]], %[[CONSTANT_3]] : i64
# CHECK:           %[[EXTRACT_REF_5:.*]] = quake.extract_ref %[[ARG0]]{{\[}}%[[SUBI_7]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:           quake.h %[[EXTRACT_REF_5]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }
