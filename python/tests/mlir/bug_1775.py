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
    result = cudaq.sample(test)

    print(result)
    reg_names = result.register_names

    assert 'true_res' in reg_names
    assert 'false_res' in reg_names
    assert '__global__' in reg_names
    assert 'res' in reg_names

    assert '1' in result.get_register_counts(
        'true_res') and '0' not in result.get_register_counts('true_res')
    assert '0' in result.get_register_counts(
        'false_res') and '1' not in result.get_register_counts('false_res')


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
