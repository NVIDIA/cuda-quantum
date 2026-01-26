# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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

    test(2, 0)
    print(test)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test..
# CHECK-SAME:      %[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant -1 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_4:.*]] = quake.alloca !quake.veq<5>
# CHECK:           %[[VAL_5:.*]]:2 = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_0]], %[[VAL_7:.*]] = %[[VAL_3]]) -> (i64, i64)) {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi sgt, %[[VAL_6]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_8]](%[[VAL_6]], %[[VAL_7]] : i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64, %[[VAL_10:.*]]: i64):
# CHECK:             %[[VAL_11:.*]] = quake.extract_ref %[[VAL_4]]{{\[}}%[[VAL_9]]] : (!quake.veq<5>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_11]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_9]], %[[VAL_9]] : i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_12:.*]]: i64, %[[VAL_13:.*]]: i64):
# CHECK:             %[[VAL_14:.*]] = arith.addi %[[VAL_12]], %[[VAL_2]] : i64
# CHECK:             cc.continue %[[VAL_14]], %[[VAL_13]] : i64, i64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_4]] : !quake.veq<5>
# CHECK:           return
# CHECK:         }


def test_should_we_really_support_this_in_cuda_q():

    # The following restriction on step being constant is NOT from Python! It
    # is a limitation imposed by the Python bridge. Does it really even make
    # sense? Why does the step have to be a literal constant?
    try:

        @cudaq.kernel
        def test_bad(q: int, p: int, m: int):
            qubits = cudaq.qvector(5)
            for k in range(q, p, m):
                k = 4
                h(qubits[k])

        test_bad(2, 0, -1)
    except Exception as e:
        print("test_bad:")
        print(e)

    # Do we really want to support assignment to the induction symbol in loops?
    # This will result in a Fortran like iterator, where the number of
    # iterations of the loop is determined before any of the loop iterations is
    # ever executed and the assignment in the body has no effect on the actual
    # loop induction.
    @cudaq.kernel
    def fortranigans(q: int, p: int):
        qubits = cudaq.qvector(5)
        for k in range(q, p, -1):
            k = 4
            h(qubits[k])

    fortranigans(2, 0)
    print(fortranigans)


# CHECK-LABEL: test_bad:
# CHECK: range step value must be a constant

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__fortranigans..
# CHECK-SAME:      %[[VAL_0:.*]]: i64, %[[VAL_1:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 4 : i64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant -1 : i64
# CHECK-DAG:       %[[VAL_4:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_5:.*]] = cc.undef i64
# CHECK-DAG:       %[[VAL_6:.*]] = quake.alloca !quake.veq<5>
# CHECK:           %[[VAL_7:.*]]:3 = cc.loop while ((%[[VAL_8:.*]] = %[[VAL_0]], %[[VAL_9:.*]] = %[[VAL_5]], %[[VAL_10:.*]] = %[[VAL_4]]) -> (i64, i64, i64)) {
# CHECK:             %[[VAL_11:.*]] = arith.cmpi sgt, %[[VAL_8]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_11]](%[[VAL_8]], %[[VAL_9]], %[[VAL_10]] : i64, i64, i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_12:.*]]: i64, %[[VAL_13:.*]]: i64, %[[VAL_14:.*]]: i64):
# CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_6]][4] : (!quake.veq<5>) -> !quake.ref
# CHECK:             quake.h %[[VAL_15]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_12]], %[[VAL_12]], %[[VAL_2]] : i64, i64, i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64, %[[VAL_17:.*]]: i64, %[[VAL_18:.*]]: i64):
# CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_16]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_19]], %[[VAL_17]], %[[VAL_18]] : i64, i64, i64
# CHECK:           }
# CHECK:           quake.dealloc %[[VAL_6]] : !quake.veq<5>
# CHECK:           return
