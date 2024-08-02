# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
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

    # CHECK-LABEL:   func.func @__nvqpp__mlirgen__ghz(
    # CHECK-SAME:                                     %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint"} {
    # CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1 : i64
    # CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 0 : i64
    # CHECK:           %[[VAL_3:.*]] = cc.alloca i64
    # CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i64>
    # CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
    # CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_4]] : i64]
    # CHECK:           %[[VAL_6:.*]] = quake.extract_ref %[[VAL_5]][0] : (!quake.veq<?>) -> !quake.ref
    # CHECK:           quake.h %[[VAL_6]] : (!quake.ref) -> ()
    # CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
    # CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_7]], %[[VAL_1]] : i64
    # CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_2]]) -> (i64)) {
    # CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_8]] : i64
    # CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : i64)
    # CHECK:           } do {
    # CHECK:           ^bb0(%[[VAL_12:.*]]: i64):
    # CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_12]]] : (!quake.veq<?>, i64) -> !quake.ref
    # CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : i64
    # CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_14]]] : (!quake.veq<?>, i64) -> !quake.ref
    # CHECK:             quake.x {{\[}}%[[VAL_13]]] %[[VAL_15]] : (!quake.ref, !quake.ref) -> ()
    # CHECK:             cc.continue %[[VAL_12]] : i64
    # CHECK:           } step {
    # CHECK:           ^bb0(%[[VAL_16:.*]]: i64):
    # CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_1]] : i64
    # CHECK:             cc.continue %[[VAL_17]] : i64
    # CHECK:           } {invariant}
    # CHECK:           return
    # CHECK:         }

    @cudaq.kernel
    def simple(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        h(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits - 1)):
            x.ctrl(qubit, qubits[i + 1])

    print(simple)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__simple(
# CHECK-SAME:                                        %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint"} {
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 2 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_4:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_4]] : !cc.ptr<i64>
# CHECK:           %[[VAL_5:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
# CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_5]] : i64]
# CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_6]][0] : (!quake.veq<?>) -> !quake.ref
# CHECK:           quake.h %[[VAL_7]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_8:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
# CHECK:           %[[VAL_9:.*]] = arith.subi %[[VAL_8]], %[[VAL_1]] : i64
# CHECK:           %[[VAL_10:.*]] = quake.subveq %[[VAL_6]], %[[VAL_3]], %[[VAL_9]] : (!quake.veq<?>, i64, i64) -> !quake.veq<?>
# CHECK:           %[[VAL_11:.*]] = quake.veq_size %[[VAL_10]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_12:.*]] = cc.alloca !cc.struct<{i64, !quake.ref}>{{\[}}%[[VAL_11]] : i64]
# CHECK:           %[[VAL_13:.*]] = cc.loop while ((%[[VAL_14:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_11]] : i64
# CHECK:             cc.condition %[[VAL_15]](%[[VAL_14]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64):
# CHECK:             %[[VAL_17:.*]] = cc.undef !cc.struct<{i64, !quake.ref}>
# CHECK:             %[[VAL_18:.*]] = quake.extract_ref %[[VAL_10]]{{\[}}%[[VAL_16]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             %[[VAL_19:.*]] = cc.compute_ptr %[[VAL_12]]{{\[}}%[[VAL_16]]] : (!cc.ptr<!cc.array<!cc.struct<{i64, !quake.ref}> x ?>>, i64) -> !cc.ptr<!cc.struct<{i64, !quake.ref}>>
# CHECK:             %[[VAL_20:.*]] = cc.insert_value %[[VAL_16]], %[[VAL_17]][0] : (!cc.struct<{i64, !quake.ref}>, i64) -> !cc.struct<{i64, !quake.ref}>
# CHECK:             %[[VAL_21:.*]] = cc.insert_value %[[VAL_18]], %[[VAL_20]][1] : (!cc.struct<{i64, !quake.ref}>, !quake.ref) -> !cc.struct<{i64, !quake.ref}>
# CHECK:             cc.store %[[VAL_21]], %[[VAL_19]] : !cc.ptr<!cc.struct<{i64, !quake.ref}>>
# CHECK:             cc.continue %[[VAL_16]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_22:.*]]: i64):
# CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_23]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_24:.*]] = cc.loop while ((%[[VAL_25:.*]] = %[[VAL_3]]) -> (i64)) {
# CHECK:             %[[VAL_26:.*]] = arith.cmpi slt, %[[VAL_25]], %[[VAL_11]] : i64
# CHECK:             cc.condition %[[VAL_26]](%[[VAL_25]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_27:.*]]: i64):
# CHECK:             %[[VAL_28:.*]] = cc.compute_ptr %[[VAL_12]]{{\[}}%[[VAL_27]]] : (!cc.ptr<!cc.array<!cc.struct<{i64, !quake.ref}> x ?>>, i64) -> !cc.ptr<!cc.struct<{i64, !quake.ref}>>
# CHECK:             %[[VAL_29:.*]] = cc.load %[[VAL_28]] : !cc.ptr<!cc.struct<{i64, !quake.ref}>>
# CHECK:             %[[VAL_30:.*]] = cc.extract_value %[[VAL_29]][0] : (!cc.struct<{i64, !quake.ref}>) -> i64
# CHECK:             %[[VAL_31:.*]] = cc.extract_value %[[VAL_29]][1] : (!cc.struct<{i64, !quake.ref}>) -> !quake.ref
# CHECK:             %[[VAL_32:.*]] = arith.addi %[[VAL_30]], %[[VAL_2]] : i64
# CHECK:             %[[VAL_33:.*]] = quake.extract_ref %[[VAL_6]]{{\[}}%[[VAL_32]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x {{\[}}%[[VAL_31]]] %[[VAL_33]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_27]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_34:.*]]: i64):
# CHECK:             %[[VAL_35:.*]] = arith.addi %[[VAL_34]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_35]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }
