# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_banjo():

    @cudaq.kernel
    def bar():
        ancilla = cudaq.qubit()
        qubits = cudaq.qvector(4)
        qubit_num = qubits.size()

        for i in range(qubit_num):
            if i == 0:
                x.ctrl(ancilla, qubits[0])
            else:
                x.ctrl([ancilla, *qubits[0:i]], qubits[i])

    shots = 10000
    print('sample bar:')
    x = cudaq.sample(bar, shots_count=shots)
    print(x)
    print(bar)


# CHECK-LABEL: sample bar:
# CHECK: { 00000:10000 }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__bar
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_8:.*]]: i64):
# CHECK:             %[[VAL_9:.*]] = arith.cmpi eq, %[[VAL_8]], %[[VAL_2]] : i64
# CHECK:             cc.if(%[[VAL_9]]) {
# CHECK:               %[[VAL_10:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<4>) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_3]]] %[[VAL_10]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_11:.*]] = arith.subi %[[VAL_8]], %[[VAL_1]] : i64
# CHECK:               %[[VAL_12:.*]] = quake.subveq %[[VAL_4]], 0, %[[VAL_11]] : (!quake.veq<4>, i64) -> !quake.veq<?>
# CHECK:               %[[VAL_13:.*]] = quake.concat %[[VAL_3]], %[[VAL_12]] : (!quake.ref, !quake.veq<?>) -> !quake.veq<?>
# CHECK:               %[[VAL_14:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_8]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_13]]] %[[VAL_14]] : (!quake.veq<?>, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_8]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_15:.*]]: i64):
# CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_16]] : i64
# CHECK:           } {invariant}
# CHECK-DAG:       quake.dealloc %[[VAL_4]] : !quake.veq<4>
# CHECK-DAG:       quake.dealloc %[[VAL_3]] : !quake.ref
# CHECK:           return
# CHECK:         }
