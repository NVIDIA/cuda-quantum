# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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
    print(bar)
    x = cudaq.sample(bar, shots_count=shots)
    print('sample bar:')
    print(x)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__bar..
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_5:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_6:.*]]:2 = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_2]], %[[VAL_8:.*]] = %[[VAL_3]]) -> (i64, i64)) {
# CHECK:             %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_9]](%[[VAL_7]], %[[VAL_8]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64, %[[VAL_11:.*]]: i64):
# CHECK:             %[[VAL_12:.*]] = arith.cmpi eq, %[[VAL_10]], %[[VAL_2]] : i64
# CHECK:             cc.if(%[[VAL_12]]) {
# CHECK:               %[[VAL_13:.*]] = quake.extract_ref %[[VAL_5]][0] : (!quake.veq<4>) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_4]]] %[[VAL_13]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_14:.*]] = arith.subi %[[VAL_10]], %[[VAL_1]] : i64
# CHECK:               %[[VAL_15:.*]] = quake.subveq %[[VAL_5]], 0, %[[VAL_14]] : (!quake.veq<4>, i64) -> !quake.veq<?>
# CHECK:               %[[VAL_16:.*]] = quake.concat %[[VAL_4]], %[[VAL_15]] : (!quake.ref, !quake.veq<?>) -> !quake.veq<?>
# CHECK:               %[[VAL_17:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_10]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_16]]] %[[VAL_17]] : (!quake.veq<?>, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_10]], %[[VAL_10]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_18:.*]]: i64, %[[VAL_19:.*]]: i64):
# CHECK:             %[[VAL_20:.*]] = arith.addi %[[VAL_18]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_20]], %[[VAL_19]] : i64, i64
# CHECK:           }
# CHECK-DAG:       quake.dealloc %[[VAL_5]] : !quake.veq<4>
# CHECK-DAG:       quake.dealloc %[[VAL_4]] : !quake.ref
# CHECK:           return
# CHECK:         }

# CHECK-LABEL: sample bar:
# CHECK: { 00000:10000 }
