# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__ghz..
# CHECK-SAME:      %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_0]] : i64]
# CHECK:           %[[VAL_5:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<?>) -> !quake.ref
# CHECK:           quake.h %[[VAL_5]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_6:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : i64
# CHECK:           %[[VAL_7:.*]]:2 = cc.loop while ((%[[VAL_8:.*]] = %[[VAL_2]], %[[VAL_9:.*]] = %[[VAL_3]]) -> (i64, i64)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_6]] : i64
# CHECK:             cc.condition %[[VAL_10]](%[[VAL_8]], %[[VAL_9]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64, %[[VAL_12:.*]]: i64):
# CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_11]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_11]], %[[VAL_1]] : i64
# CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_14]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[VAL_13]]] %[[VAL_15]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_11]], %[[VAL_11]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64, %[[VAL_17:.*]]: i64):
# CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_16]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_18]], %[[VAL_17]] : i64, i64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_4]] : !quake.veq<?>
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__simple..
# CHECK-SAME:      %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_5:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_0]] : i64]
# CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_5]][0] : (!quake.veq<?>) -> !quake.ref
# CHECK:           quake.h %[[VAL_6]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_7:.*]] = arith.subi %[[VAL_0]], %[[VAL_1]] : i64
# CHECK:           %[[VAL_8:.*]] = quake.subveq %[[VAL_5]], 0, %[[VAL_7]] : (!quake.veq<?>, i64) -> !quake.veq<?>
# CHECK:           %[[VAL_9:.*]] = quake.veq_size %[[VAL_8]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_10:.*]]:2 = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_3]], %[[VAL_12:.*]] = %[[VAL_4]]) -> (i64, i64)) {
# CHECK:             %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_9]] : i64
# CHECK:             cc.condition %[[VAL_13]](%[[VAL_11]], %[[VAL_12]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_14:.*]]: i64, %[[VAL_15:.*]]: i64):
# CHECK:             %[[VAL_16:.*]] = quake.extract_ref %[[VAL_8]]{{\[}}%[[VAL_14]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_14]], %[[VAL_2]] : i64
# CHECK:             %[[VAL_18:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_17]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[VAL_16]]] %[[VAL_18]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_14]], %[[VAL_14]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_19:.*]]: i64, %[[VAL_20:.*]]: i64):
# CHECK:             %[[VAL_21:.*]] = arith.addi %[[VAL_19]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_21]], %[[VAL_20]] : i64, i64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_5]] : !quake.veq<?>
# CHECK:           return
# CHECK:         }
