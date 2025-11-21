# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_qreg_iter():

    @cudaq.kernel
    def foo(N: int):
        q = cudaq.qvector(N)
        for r in q:
            x(r)

    print(foo)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__foo(
# CHECK-SAME:                                     %[[VAL_0:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i64>
# CHECK:           %[[VAL_4:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<?>{{\[}}%[[VAL_4]] : i64]
# CHECK:           %[[VAL_6:.*]] = quake.veq_size %[[VAL_5]] : (!quake.veq<?>) -> i64
# CHECK:           %[[VAL_7:.*]] = cc.loop while ((%[[VAL_8:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_9:.*]] = arith.cmpi slt, %[[VAL_8]], %[[VAL_6]] : i64
# CHECK:             cc.condition %[[VAL_9]](%[[VAL_8]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
# CHECK:             %[[VAL_11:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_10]]] : (!quake.veq<?>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_11]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_10]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_12:.*]]: i64):
# CHECK:             %[[VAL_13:.*]] = arith.addi %[[VAL_12]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_13]] : i64
# CHECK:           }
# CHECK:           return
# CHECK:         }
