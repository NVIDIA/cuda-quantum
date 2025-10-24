# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__reflect(
# CHECK-SAME:                      %[[VAL_0:.*]]: !quake.veq<?>)
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 2 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK:           %[[VAL_4:.*]] = quake.veq_size %[[VAL_0]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_5:.*]] = arith.subi %[[VAL_4]], %[[VAL_1]] : i64
# CHECK:           %[[VAL_6:.*]] = quake.subveq %[[VAL_0]], 0, %[[VAL_5]] : (!quake.veq<?>, i64) -> !quake.veq<?>
# CHECK:           %[[VAL_7:.*]] = arith.subi %[[VAL_4]], %[[VAL_3]] : i64
# CHECK:           %[[VAL_8:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_7]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:           %[[VAL_9:.*]] = cc.create_lambda {
# CHECK:             %[[VAL_10:.*]] = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:               %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_4]] : i64
# CHECK:               cc.condition %[[VAL_12]](%[[VAL_11]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_13:.*]]: i64):
# CHECK:               %[[VAL_14:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_13]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.h %[[VAL_14]] : (!quake.ref) -> ()
# CHECK:               cc.continue %[[VAL_13]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_15:.*]]: i64):
# CHECK:               %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_3]] : i64
# CHECK:               cc.continue %[[VAL_16]] : i64
# CHECK:             }
# CHECK:             %[[VAL_17:.*]] = cc.loop while ((%[[VAL_18:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:               %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_4]] : i64
# CHECK:               cc.condition %[[VAL_19]](%[[VAL_18]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_20:.*]]: i64):
# CHECK:               %[[VAL_21:.*]] = quake.extract_ref %[[VAL_0]]{{\[}}%[[VAL_20]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:               quake.x %[[VAL_21]] : (!quake.ref) -> ()
# CHECK:               cc.continue %[[VAL_20]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_22:.*]]: i64):
# CHECK:               %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_3]] : i64
# CHECK:               cc.continue %[[VAL_23]] : i64
# CHECK:             }
# CHECK:           } : !cc.callable<() -> ()>
# CHECK:           %[[VAL_24:.*]] = cc.create_lambda {
# CHECK:             quake.z {{\[}}%[[VAL_6]]] %[[VAL_8]] : (!quake.veq<?>, !quake.ref) -> ()
# CHECK:           } : !cc.callable<() -> ()>
# CHECK:           quake.compute_action %[[VAL_9]], %[[VAL_24]] : !cc.callable<() -> ()>, !cc.callable<() -> ()>
# CHECK:           return
# CHECK:         }
