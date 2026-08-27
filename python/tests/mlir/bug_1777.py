# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_bug_1777():

    @cudaq.kernel
    def test():
        qubits = cudaq.qvector(2)

        res = True
        for i in range(2):
            res = mz(qubits[i])
            if res == False:
                inner_mz = mz(qubits)

        if res == True:
            outer_mz = mz(qubits)

    print(test)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test..
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel", qubitMeasurementFeedback = true} {
# CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant false
# CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[CONSTANT_2:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[CONSTANT_3:.*]] = arith.constant true
# CHECK-DAG:       %[[CONSTANT_4:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[UNDEF_0:.*]] = cc.undef !cc.sequence<!cc.measure_handle>
# CHECK-DAG:       %[[UNDEF_1:.*]] = cc.undef !cc.sequence<!cc.measure_handle>
# CHECK:           %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[LOOP_0:.*]]:3 = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_2]], %[[VAL_1:.*]] = %[[UNDEF_1]], %[[VAL_2:.*]] = %[[CONSTANT_3]]) -> (i64, !cc.sequence<!cc.measure_handle>, i1)) {
# CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[CONSTANT_4]] : i64
# CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]], %[[VAL_1]], %[[VAL_2]] : i64, !cc.sequence<!cc.measure_handle>, i1)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_3:.*]]: i64, %[[VAL_4:.*]]: !cc.sequence<!cc.measure_handle>, %[[VAL_5:.*]]: i1):
# CHECK:             %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_0]]{{\[}}%[[VAL_3]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             %[[MZ_0:.*]] = quake.mz %[[EXTRACT_REF_0]] name "res" : (!quake.ref) -> !cc.measure_handle
# CHECK:             %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[MZ_0]] : (!cc.measure_handle) -> i1
# CHECK:             %[[CMPI_1:.*]] = arith.cmpi eq, %[[DISCRIMINATE_0]], %[[CONSTANT_0]] : i1
# CHECK:             %[[IF_0:.*]] = cc.if(%[[CMPI_1]]) -> !cc.sequence<!cc.measure_handle> {
# CHECK:               %[[MZ_1:.*]] = quake.mz %[[ALLOCA_0]] name "inner_mz" : (!quake.veq<2>) -> !cc.sequence<!cc.measure_handle>
# CHECK:               cc.continue %[[MZ_1]] : !cc.sequence<!cc.measure_handle>
# CHECK:             } else {
# CHECK:               cc.continue %[[VAL_4]] : !cc.sequence<!cc.measure_handle>
# CHECK:             }
# CHECK:             cc.continue %[[VAL_3]], %[[IF_0]], %[[DISCRIMINATE_0]] : i64, !cc.sequence<!cc.measure_handle>, i1
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_6:.*]]: i64, %[[VAL_7:.*]]: !cc.sequence<!cc.measure_handle>, %[[VAL_8:.*]]: i1):
# CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_6]], %[[CONSTANT_1]] : i64
# CHECK:             cc.continue %[[ADDI_0]], %[[VAL_7]], %[[VAL_8]] : i64, !cc.sequence<!cc.measure_handle>, i1
# CHECK:           }
# CHECK:           %[[IF_1:.*]] = cc.if(%[[LOOP_0]]#2) -> !cc.sequence<!cc.measure_handle> {
# CHECK:             %[[MZ_2:.*]] = quake.mz %[[ALLOCA_0]] name "outer_mz" : (!quake.veq<2>) -> !cc.sequence<!cc.measure_handle>
# CHECK:             cc.continue %[[MZ_2]] : !cc.sequence<!cc.measure_handle>
# CHECK:           } else {
# CHECK:             cc.continue %[[UNDEF_0]] : !cc.sequence<!cc.measure_handle>
# CHECK:           }
# CHECK:           quake.log_output %[[ALLOCA_0]] : (!quake.veq<2>) -> ()
# CHECK:           quake.dealloc %[[ALLOCA_0]] : !quake.veq<2>
# CHECK:           return
# CHECK:         }
