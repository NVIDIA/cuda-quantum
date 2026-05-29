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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__bar..0x
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[CONSTANT_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[UNDEF_0:.*]] = cc.undef i64
# CHECK-DAG:       %[[ALLOCA_0:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[ALLOCA_1:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[LOOP_0:.*]]:2 = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_2]], %[[VAL_1:.*]] = %[[UNDEF_0]]) -> (i64, i64)) {
# CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[CONSTANT_0]] : i64
# CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]], %[[VAL_1]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_2:.*]]: i64, %[[VAL_3:.*]]: i64):
# CHECK:             %[[CMPI_1:.*]] = arith.cmpi eq, %[[VAL_2]], %[[CONSTANT_2]] : i64
# CHECK:             cc.if(%[[CMPI_1]]) {
# CHECK:               %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_1]][0] : (!quake.veq<4>) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[ALLOCA_0]]] %[[EXTRACT_REF_0]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             } else {
# CHECK:               %[[SUBI_0:.*]] = arith.subi %[[VAL_2]], %[[CONSTANT_1]] : i64
# CHECK:               %[[CMPI_2:.*]] = arith.cmpi sge, %[[SUBI_0]], %[[CONSTANT_2]] : i64
# CHECK:               %[[IF_0:.*]] = cc.if(%[[CMPI_2]]) -> !quake.veq<?> {
# CHECK:                 %[[SUBVEQ_0:.*]] = quake.subveq %[[ALLOCA_1]], 0, %[[SUBI_0]] : (!quake.veq<4>, i64) -> !quake.veq<?>
# CHECK:                 cc.continue %[[SUBVEQ_0]] : !quake.veq<?>
# CHECK:               } else {
# CHECK:                 %[[UNDEF_1:.*]] = cc.undef !quake.veq<?>
# CHECK:                 cc.continue %[[UNDEF_1]] : !quake.veq<?>
# CHECK:               }
# CHECK:               %[[CONCAT_0:.*]] = quake.concat %[[ALLOCA_0]], %[[IF_0]] : (!quake.ref, !quake.veq<?>) -> !quake.veq<?>
# CHECK:               %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ALLOCA_1]]{{\[}}%[[VAL_2]]] : (!quake.veq<4>, i64) -> !quake.ref
# CHECK:               quake.x {{\[}}%[[CONCAT_0]]] %[[EXTRACT_REF_1]] : (!quake.veq<?>, !quake.ref) -> ()
# CHECK:             }
# CHECK:             cc.continue %[[VAL_2]], %[[VAL_2]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: i64):
# CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_4]], %[[CONSTANT_1]] : i64
# CHECK:             cc.continue %[[ADDI_0]], %[[VAL_5]] : i64, i64
# CHECK:           }
# CHECK-DAG:       quake.dealloc %[[ALLOCA_1]] : !quake.veq<4>
# CHECK-DAG:       quake.dealloc %[[ALLOCA_0]] : !quake.ref
# CHECK:           return
# CHECK:         }

# CHECK-LABEL: sample bar:
# CHECK: { 00000:10000 }
