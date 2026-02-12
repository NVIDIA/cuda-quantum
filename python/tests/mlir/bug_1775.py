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

    @cudaq.kernel(defer_compilation=False)
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
# CHECK-DAG:       %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_1]] name "res" : (!quake.ref) -> !quake.measure
# CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_4:.*]] = quake.mz %[[VAL_1]] name "res" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_5:.*]] = quake.discriminate %[[VAL_4]] : (!quake.measure) -> i1
# CHECK:           %[[VAL_7:.*]] = arith.cmpi eq, %[[VAL_5]], %[[VAL_0]] : i1
# CHECK:           cc.if(%[[VAL_7]]) {
# CHECK:             %[[VAL_8:.*]] = quake.mz %[[VAL_1]] name "true_res" : (!quake.ref) -> !quake.measure
# CHECK:           } else {
# CHECK:             %[[VAL_9:.*]] = quake.mz %[[VAL_1]] name "false_res" : (!quake.ref) -> !quake.measure
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_1]]
# CHECK:           return
# CHECK:         }
