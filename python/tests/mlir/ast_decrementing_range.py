# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s



import cudaq


def test_decrementing_range():

    @cudaq.kernel
    def test(q: int, p: int):
        qubits = cudaq.qvector(5)
        for k in range(q, p, -1):
            x(qubits[k])

    print(test)
    test(2, 0)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test(
# CHECK-SAME:                                      %[[VAL_0:.*]]: i64,
# CHECK-SAME:                                      %[[VAL_1:.*]]: i64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_2:.*]] = arith.constant -1 : i64
# CHECK:           %[[VAL_3:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_0]], %[[VAL_3]] : !cc.ptr<i64>
# CHECK:           %[[VAL_4:.*]] = cc.alloca i64
# CHECK:           cc.store %[[VAL_1]], %[[VAL_4]] : !cc.ptr<i64>
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<5>
# CHECK:           %[[VAL_6:.*]] = cc.load %[[VAL_3]] : !cc.ptr<i64>
# CHECK:           %[[VAL_7:.*]] = cc.load %[[VAL_4]] : !cc.ptr<i64>
# CHECK:           %[[VAL_8:.*]] = cc.loop while ((%[[VAL_9:.*]] = %[[VAL_6]]) -> (i64)) {
# CHECK:             %[[VAL_10:.*]] = arith.cmpi sgt, %[[VAL_9]], %[[VAL_7]] : i64
# CHECK:             cc.condition %[[VAL_10]](%[[VAL_9]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
# CHECK:             %[[VAL_12:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_11]]] : (!quake.veq<5>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_12]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_13:.*]]: i64):
# CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_13]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_14]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }
