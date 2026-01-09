# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_bug_1875():

    @cudaq.kernel
    def kernel_break():
        ancilla_a = cudaq.qubit()
        ancilla_b = cudaq.qubit()
        q = cudaq.qubit()

        h(ancilla_a)
        h(ancilla_b)
        x(q)

        aux_1 = mz(ancilla_a)
        aux_2 = mz(ancilla_b)
        if aux_1 == 0 and aux_2 == 0:
            x.ctrl(ancilla_a, q)
            a = mz(q)

    print(kernel_break)
    result = cudaq.sample(kernel_break, shots_count=1000)

    assert 'a' in result.register_names


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel_break..
# CHECK-SAME: () attributes {"cudaq-entrypoint", "cudaq-kernel", qubitMeasurementFeedback = true} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant false
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.ref
# CHECK:           quake.h %[[VAL_2]] : (!quake.ref) -> ()
# CHECK:           quake.h %[[VAL_3]] : (!quake.ref) -> ()
# CHECK:           quake.x %[[VAL_4]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_5:.*]] = quake.mz %[[VAL_2]] name "aux_1" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_6:.*]] = quake.discriminate %[[VAL_5]] : (!quake.measure) -> i1
# CHECK:           %[[VAL_7:.*]] = quake.mz %[[VAL_3]] name "aux_2" : (!quake.ref) -> !quake.measure
# CHECK:           %[[VAL_8:.*]] = quake.discriminate %[[VAL_7]] : (!quake.measure) -> i1
# CHECK:           %[[VAL_9:.*]] = cc.cast unsigned %[[VAL_6]] : (i1) -> i64
# CHECK:           %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_1]] : i64
# CHECK:           %[[VAL_11:.*]] = cc.if(%[[VAL_10]]) -> i1 {
# CHECK:             cc.continue %[[VAL_0]] : i1
# CHECK:           } else {
# CHECK:             %[[VAL_12:.*]] = cc.cast unsigned %[[VAL_8]] : (i1) -> i64
# CHECK:             %[[VAL_13:.*]] = arith.cmpi eq, %[[VAL_12]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_13]] : i1
# CHECK:           }
# CHECK:           cc.if(%[[VAL_14:.*]]) {
# CHECK:             quake.x {{\[}}%[[VAL_2]]] %[[VAL_4]] : (!quake.ref, !quake.ref) -> ()
# CHECK:             %[[VAL_15:.*]] = quake.mz %[[VAL_4]] name "a" : (!quake.ref) -> !quake.measure
# CHECK:           }
# CHECK-DAG:       quake.dealloc %[[VAL_3]] : !quake.ref
# CHECK-DAG:       quake.dealloc %[[VAL_2]] : !quake.ref
# CHECK-DAG:       quake.dealloc %[[VAL_4]] : !quake.ref
# CHECK:           return
