# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import cudaq


def test_var_scope():

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(2)
        var_int = 42
        var_bool = True

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__kernel
# CHECK-SAME:      () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<2>
# CHECK:           quake.dealloc %[[VAL_2]]
# CHECK:           return
# CHECK:         }


def test_variable_name():

    @cudaq.kernel
    def slice():
        q = cudaq.qvector(4)
        # The next statement is not an error. `slice` is a local variable that
        # hides the name of the kernel.
        slice = q[2:]
        x(slice)

    print(slice)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__slice
# CHECK-SAME:      () attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 2 : i64
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = quake.alloca !quake.veq<4>
# CHECK:           %[[VAL_4:.*]] = quake.subveq %[[VAL_3]], 2, 3 : (!quake.veq<4>) -> !quake.veq<2>
# CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_1]]) -> (i64)) {
# CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_8:.*]]: i64):
# CHECK:             %[[VAL_9:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_8]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_9]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_8]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_10:.*]]: i64):
# CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_11]] : i64
# CHECK:           } {invariant}
# CHECK:           quake.dealloc %[[VAL_3]] : !quake.veq<4>
# CHECK:           return
# CHECK:         }
