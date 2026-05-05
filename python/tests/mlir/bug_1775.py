# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_bug_1775():

    @cudaq.kernel
    def test():
        qubit = cudaq.qubit()
        res = mz(qubit)
        h(qubit)

        res = mz(qubit)
        Flag = res

        if Flag == True:
            true_res = mz(qubit)
        else:
            false_res = mz(qubit)

    print(test)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test
# CHECK-SAME:      () attributes {"cudaq-entrypoint", "cudaq-kernel", qubitMeasurementFeedback = true} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant true
# CHECK-DAG:       %[[VAL_1:.*]] = cc.undef !cc.measure_handle
# CHECK-DAG:       %[[VAL_2:.*]] = cc.undef !cc.measure_handle
# CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_3]] name "res" : (!quake.ref) -> !cc.measure_handle
# CHECK:           quake.h %[[VAL_3]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_3]] name "res" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_6:.*]] = quake.discriminate %[[VAL_5]] : (!cc.measure_handle) -> i1
# CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_6]], %[[VAL_0]] : i1
# CHECK:           %[[VAL_8:.*]]:2 = cc.if(%[[VAL_7]]) -> (!cc.measure_handle, !cc.measure_handle) {
# CHECK:             %[[VAL_9:.*]] = quake.mz %[[VAL_3]] name "true_res" : (!quake.ref) -> !cc.measure_handle
# CHECK:             cc.continue %[[VAL_9]], %[[VAL_1]] : !cc.measure_handle, !cc.measure_handle
# CHECK:           } else {
# CHECK:             %[[VAL_10:.*]] = quake.mz %[[VAL_3]] name "false_res" : (!quake.ref) -> !cc.measure_handle
# CHECK:             cc.continue %[[VAL_2]], %[[VAL_10]] : !cc.measure_handle, !cc.measure_handle
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_3]] : !quake.ref
# CHECK:           return
# CHECK:         }
