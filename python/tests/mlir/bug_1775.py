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
# CHECK-DAG:       %[[VAL_0:.*]] = cc.undef !cc.measure_handle
# CHECK-DAG:       %[[VAL_1:.*]] = cc.undef !cc.measure_handle
# CHECK-DAG:       %[[VAL_2:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_3:.*]] = quake.mz %[[VAL_2]] name "res" : (!quake.ref) -> !cc.measure_handle
# CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_2]] name "res" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_5:.*]] = quake.discriminate %[[VAL_4]] : (!cc.measure_handle) -> i1
# CHECK:           %[[VAL_6:.*]]:2 = cc.if(%[[VAL_5]]) -> (!cc.measure_handle, !cc.measure_handle) {
# CHECK:             %[[VAL_7:.*]] = quake.mz %[[VAL_2]] name "true_res" : (!quake.ref) -> !cc.measure_handle
# CHECK:             cc.continue %[[VAL_7]], %[[VAL_0]] : !cc.measure_handle, !cc.measure_handle
# CHECK:           } else {
# CHECK:             %[[VAL_8:.*]] = quake.mz %[[VAL_2]] name "false_res" : (!quake.ref) -> !cc.measure_handle
# CHECK:             cc.continue %[[VAL_1]], %[[VAL_8]] : !cc.measure_handle, !cc.measure_handle
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_2]] : !quake.ref
# CHECK:           return
# CHECK:         }
