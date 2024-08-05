# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s



import cudaq


def test_list_comprehension():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(6)
        [h(r) for r in q]
        x(q[0])
        x.ctrl(q[1], q[2])

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint"} {
# CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 6 : i64
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<6>
# CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_7:.*]]: i64):
# CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_7]]] : (!quake.veq<6>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_8]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_7]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_10]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_11:.*]] = quake.extract_ref %[[VAL_3]][0] : (!quake.veq<6>) -> !quake.ref
# CHECK:           quake.x %[[VAL_11]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_12:.*]] = quake.extract_ref %[[VAL_3]][1] : (!quake.veq<6>) -> !quake.ref
# CHECK:           %[[VAL_13:.*]] = quake.extract_ref %[[VAL_3]][2] : (!quake.veq<6>) -> !quake.ref
# CHECK:           quake.x {{\[}}%[[VAL_12]]] %[[VAL_13]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }
