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

    print(bar)
    shots = 10000
    print('sample bar:')
    x = cudaq.sample(bar, shots_count=shots)
    print(x)


# CHECK-LABEL:   func.func
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.veq<4>
# CHECK-DAG:       %[[VAL_5:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_5]] : !cc.ptr<i64>
# CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_5]] : !cc.ptr<i64>
# CHECK:           %[[VAL_7:.*]] = cc.loop while ((%[[VAL_8:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_6]] : i64
# CHECK:             cc.condition %[[VAL_9]](%[[VAL_8]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
# CHECK:             %[[VAL_20:.*]] = cc.alloca i64
# CHECK:             cc.store %[[VAL_10]], %[[VAL_20]] : !cc.ptr<i64>
# CHECK:             %[[VAL_21:.*]] = cc.load %[[VAL_20]] : !cc.ptr<i64>
# CHECK:             %[[VAL_11:.*]] = arith.cmpi eq, %[[VAL_21]], %[[VAL_2]] : i64
# CHECK:             cc.if(%[[VAL_11]]) {
# CHECK:               %[[VAL_12:.*]] = quake.extract_ref %[[VAL_4]][0] : (!quake.veq<4>) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_3]]] %[[VAL_12]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[VAL_22:.*]] = cc.load %[[VAL_20]] : !cc.ptr<i64>
# CHECK:               %[[VAL_13:.*]] = arith.subi %[[VAL_22]], %[[VAL_1]] : i64
# CHECK:               %[[VAL_14:.*]] = quake.subveq %[[VAL_4]], 0, %[[VAL_13]] : (!quake.veq<4>, i64) -> !quake.veq<?>
# CHECK:               %[[VAL_15:.*]] = quake.concat %[[VAL_3]], %[[VAL_14]] : (!quake.ref, !quake.veq<?>) -> !quake.veq<?>
# CHECK:               %[[VAL_16:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_22]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[VAL_15]]] %[[VAL_16]] : (!quake.veq<?>, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_10]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_17:.*]]: i64):
# CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_18]] : i64
# CHECK:           }
# CHECK:           return
# CHECK:         }

# CHECK-LABEL: sample bar:
# CHECK: { 00000:10000 }
