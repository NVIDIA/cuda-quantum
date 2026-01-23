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

    @cudaq.kernel(verbose=True)
    def reflect(qubits: cudaq.qview):
        ctrls = qubits.front(qubits.size() - 1)
        last = qubits.back()
        compute = lambda: (h(qubits), x(qubits))
        cudaq.compute_action(compute, lambda: z.ctrl(ctrls, last))

    print(reflect)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__reflect
# CHECK-SAME:      (%[[VAL_0:.*]]: !quake.veq<?>)
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_5:.*]] = arith.subi %[[VAL_4]], %[[VAL_1]] : i64
# CHECK:           %[[VAL_6:.*]] = quake.subveq %[[VAL_0]], 0, %[[VAL_5]] : (!quake.veq<?>, i64) -> !quake.veq<?>
# CHECK:           %[[VAL_7:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_8:.*]] = arith.subi %[[VAL_7]], %[[VAL_3]] : i64
# CHECK:           %[[VAL_9:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_8]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:           %[[VAL_10:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_11:.*]] = cc.loop while ((%[[VAL_12:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_10]] : i64
# CHECK:             cc.condition %[[VAL_13]](%[[VAL_12]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_14:.*]]: i64):
# CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_14]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_15]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_14]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64):
# CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_17]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_18:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_19:.*]] = cc.loop while ((%[[VAL_20:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_21:.*]] = arith.cmpi slt, %[[VAL_20]], %[[VAL_18]] : i64
# CHECK:             cc.condition %[[VAL_21]](%[[VAL_20]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_22:.*]]: i64):
# CHECK:             %[[VAL_23:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_22]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_23]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_22]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_24:.*]]: i64):
# CHECK:             %[[VAL_25:.*]] = arith.addi %[[VAL_24]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_25]] : i64
# CHECK:           } {invariant}
# CHECK:           quake.z {{\[}}%[[VAL_6]]] %[[VAL_9]] : (!quake.veq<?>, !quake.ref) -> ()
# CHECK:           %[[VAL_26:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_27:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_28:.*]] = arith.cmpi sgt, %[[VAL_27]], %[[VAL_2]] : i64
# CHECK:           %[[VAL_29:.*]] = arith.select %[[VAL_28]], %[[VAL_27]], %[[VAL_2]] : i64
# CHECK:           %[[VAL_30:.*]] = arith.subi %[[VAL_29]], %[[VAL_3]] : i64
# CHECK:           %[[VAL_31:.*]]:2 = cc.loop while ((%[[VAL_32:.*]] = %[[VAL_30]], %[[VAL_33:.*]] = %[[VAL_29]]) -> (i64, i64)) {
# CHECK:             %[[VAL_34:.*]] = arith.cmpi sgt, %[[VAL_33]], %[[VAL_2]] : i64
# CHECK:             cc.condition %[[VAL_34]](%[[VAL_32]], %[[VAL_33]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_35:.*]]: i64, %[[VAL_36:.*]]: i64):
# CHECK:             %[[VAL_37:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_35]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_37]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_35]], %[[VAL_36]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_38:.*]]: i64, %[[VAL_39:.*]]: i64):
# CHECK:             %[[VAL_40:.*]] = arith.subi %[[VAL_38]], %[[VAL_3]] : i64
# CHECK:             %[[VAL_41:.*]] = arith.subi %[[VAL_39]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_40]], %[[VAL_41]] : i64, i64
# CHECK:           }
# CHECK:           %[[VAL_42:.*]] = arith.cmpi sgt, %[[VAL_26]], %[[VAL_2]] : i64
# CHECK:           %[[VAL_43:.*]] = arith.select %[[VAL_42]], %[[VAL_26]], %[[VAL_2]] : i64
# CHECK:           %[[VAL_44:.*]] = arith.subi %[[VAL_43]], %[[VAL_3]] : i64
# CHECK:           %[[VAL_45:.*]]:2 = cc.loop while ((%[[VAL_46:.*]] = %[[VAL_44]], %[[VAL_47:.*]] = %[[VAL_43]]) -> (i64, i64)) {
# CHECK:             %[[VAL_48:.*]] = arith.cmpi sgt, %[[VAL_47]], %[[VAL_2]] : i64
# CHECK:             cc.condition %[[VAL_48]](%[[VAL_46]], %[[VAL_47]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_49:.*]]: i64, %[[VAL_50:.*]]: i64):
# CHECK:             %[[VAL_51:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_49]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_51]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_49]], %[[VAL_50]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_52:.*]]: i64, %[[VAL_53:.*]]: i64):
# CHECK:             %[[VAL_54:.*]] = arith.subi %[[VAL_52]], %[[VAL_3]] : i64
# CHECK:             %[[VAL_55:.*]] = arith.subi %[[VAL_53]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_54]], %[[VAL_55]] : i64, i64
# CHECK:           }
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func private @__nvqpp__lifted.lambda.1(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>,
# CHECK-SAME:      %[[VAL_1:.*]]: !quake.ref) attributes {"cudaq-kernel"} {
# CHECK:           quake.z {{\[}}%[[VAL_0]]] %[[VAL_1]] : (!quake.veq<?>, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func private @__nvqpp__lifted.lambda.0(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.veq<?>,
# CHECK-SAME:      %[[VAL_1:.*]]: i64,
# CHECK-SAME:      %[[VAL_2:.*]]: i64) attributes {"cudaq-kernel"} {
# CHECK:           %[[VAL_3:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_3]] : i64
# CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_7:.*]]: i64):
# CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_7]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_8]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_7]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_10]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_11:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_12:.*]] = cc.loop while ((%[[VAL_13:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_14:.*]] = arith.cmpi slt, %[[VAL_13]], %[[VAL_11]] : i64
# CHECK:             cc.condition %[[VAL_14]](%[[VAL_13]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_15:.*]]: i64):
# CHECK:             %[[VAL_16:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_15]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_16]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_15]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_17:.*]]: i64):
# CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_18]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }
