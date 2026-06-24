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
# CHECK-DAG:       %[[UNDEF_0:.*]] = cc.undef !cc.stdvec<!cc.measure_handle>
# CHECK-DAG:       %[[UNDEF_1:.*]] = cc.undef !cc.stdvec<!cc.measure_handle>
# CHECK-DAG:       %[[UNDEF_2:.*]] = cc.undef i64
# CHECK:           %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[LOOP_0:.*]]:4 = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_2]], %[[VAL_1:.*]] = %[[UNDEF_2]], %[[VAL_2:.*]] = %[[CONSTANT_3]], %[[VAL_3:.*]] = %[[UNDEF_1]]) -> (i64, i64, i1, !cc.stdvec<!cc.measure_handle>)) {
# CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[CONSTANT_4]] : i64
# CHECK:             cc.condition %[[CMPI_0]](%[[VAL_0]], %[[VAL_1]], %[[VAL_2]], %[[VAL_3]] : i64, i64, i1, !cc.stdvec<!cc.measure_handle>)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_4:.*]]: i64, %[[VAL_5:.*]]: i64, %[[VAL_6:.*]]: i1, %[[VAL_7:.*]]: !cc.stdvec<!cc.measure_handle>):
# CHECK:             %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_0]]{{\[}}%[[VAL_4]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             %[[MZ_0:.*]] = quake.mz %[[EXTRACT_REF_0]] name "res" : (!quake.ref) -> !cc.measure_handle
# CHECK:             %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[MZ_0]] : (!cc.measure_handle) -> i1
# CHECK:             %[[CMPI_1:.*]] = arith.cmpi eq, %[[DISCRIMINATE_0]], %[[CONSTANT_0]] : i1
# CHECK:             %[[IF_0:.*]] = cc.if(%[[CMPI_1]]) -> !cc.stdvec<!cc.measure_handle> {
# CHECK:               %[[MZ_1:.*]] = quake.mz %[[ALLOCA_0]] name "inner_mz" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:               cc.continue %[[MZ_1]] : !cc.stdvec<!cc.measure_handle>
# CHECK:             } else {
# CHECK:               cc.continue %[[VAL_7]] : !cc.stdvec<!cc.measure_handle>
# CHECK:             }
# CHECK:             cc.continue %[[VAL_4]], %[[VAL_4]], %[[DISCRIMINATE_0]], %[[IF_0]] : i64, i64, i1, !cc.stdvec<!cc.measure_handle>
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_8:.*]]: i64, %[[VAL_9:.*]]: i64, %[[VAL_10:.*]]: i1, %[[VAL_11:.*]]: !cc.stdvec<!cc.measure_handle>):
# CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_8]], %[[CONSTANT_1]] : i64
# CHECK:             cc.continue %[[ADDI_0]], %[[VAL_9]], %[[VAL_10]], %[[VAL_11]] : i64, i64, i1, !cc.stdvec<!cc.measure_handle>
# CHECK:           }
# CHECK:           %[[IF_1:.*]] = cc.if(%[[VAL_12:.*]]#2) -> !cc.stdvec<!cc.measure_handle> {
# CHECK:             %[[MZ_2:.*]] = quake.mz %[[ALLOCA_0]] name "outer_mz" : (!quake.veq<2>) -> !cc.stdvec<!cc.measure_handle>
# CHECK:             cc.continue %[[MZ_2]] : !cc.stdvec<!cc.measure_handle>
# CHECK:           } else {
# CHECK:             cc.continue %[[UNDEF_0]] : !cc.stdvec<!cc.measure_handle>
# CHECK:           }
# CHECK:           quake.dealloc %[[ALLOCA_0]] : !quake.veq<2>
# CHECK:           return
# CHECK:         }
