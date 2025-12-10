# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_ghz():

    @cudaq.kernel
    def ghz(N: int):
        q = cudaq.qvector(N)
        h(q[0])
        for i in range(N - 1):
            x.ctrl(q[i], q[i + 1])

    print(ghz)

    @cudaq.kernel
    def simple(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits - 1)):
            x.ctrl(qubit, qubits[i + 1])

    print(simple)

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__ghz
# CHECK-SAME:      %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_0]] : i64]
# CHECK:           %[[VAL_4:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<?>) -> !quake.ref
# CHECK:           quake.h %[[VAL_4]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_5:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : i64
# CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_5]] : i64
# CHECK:             cc.condition %[[VAL_8]](%[[VAL_7]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_9]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : i64
# CHECK:             %[[VAL_12:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_11]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[VAL_10]]] %[[VAL_12]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_9]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_13:.*]]: i64):
# CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_14]] : i64
# CHECK:           } {invariant}
# CHECK:           quake.dealloc %[[VAL_3]] : !quake.veq<?>
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__simple
# CHECK-SAME:      %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_0]] : i64]
# CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<?>) -> !quake.ref
# CHECK:           quake.h %[[VAL_5]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_6:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : i64
# CHECK:           %[[VAL_7:.*]] = quake.subveq %[[VAL_4]], 0, %[[VAL_6]] : (!quake.veq<?>, i64) -> !quake.veq<?>
# CHECK:           %[[VAL_8:.*]] = quake.veq_size %[[VAL_7]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_8]] : i64
# CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_12:.*]]: i64):
# CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_7]]{{\[}}%[[VAL_12]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_2]] : i64
# CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_14]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[VAL_13]]] %[[VAL_15]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_12]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64):
# CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_17]] : i64
# CHECK:           } {invariant}
# CHECK:           quake.dealloc %[[VAL_4]] : !quake.veq<?>
# CHECK:           return
# CHECK:         }
