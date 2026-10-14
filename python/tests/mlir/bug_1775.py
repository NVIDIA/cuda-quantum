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
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.mz %[[VAL_0]] name "res" : (!quake.ref) -> !cc.measure_handle
# CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_0]] name "res" : (!quake.ref) -> !cc.measure_handle
# CHECK:           %[[VAL_3:.*]] = quake.discriminate %[[VAL_2]] : (!cc.measure_handle) -> i1
# CHECK:           cc.if(%[[VAL_3]]) {
# CHECK:             %[[VAL_4:.*]] = quake.mz %[[VAL_0]] name "true_res" : (!quake.ref) -> !cc.measure_handle
# CHECK:           } else {
# CHECK:             %[[VAL_5:.*]] = quake.mz %[[VAL_0]] name "false_res" : (!quake.ref) -> !cc.measure_handle
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_0]] : !quake.ref
# CHECK:           return
# CHECK:         }
