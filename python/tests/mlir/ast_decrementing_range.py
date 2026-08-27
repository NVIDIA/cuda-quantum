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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__test..0x
# CHECK-SAME:      (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[CONSTANT_2:.*]] = arith.constant -1 : i64
# CHECK-DAG:       %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<5>
# CHECK:           %[[SUBI_0:.*]] = arith.subi %[[ARG1]], %[[CONSTANT_2]] : i64
# CHECK:           %[[SUBI_1:.*]] = arith.subi %[[SUBI_0]], %[[ARG0]] : i64
# CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SUBI_1]], %[[CONSTANT_2]] : i64
# CHECK:           %[[DIVSI_0:.*]] = arith.divsi %[[ADDI_0]], %[[CONSTANT_2]] : i64
# CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[DIVSI_0]], %[[CONSTANT_1]] : i64
# CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[DIVSI_0]], %[[CONSTANT_1]] : i64
# CHECK:           %[[LOOP_0:.*]] = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_1]]) -> (i64)) {
# CHECK:             %[[CMPI_1:.*]] = arith.cmpi ne, %[[VAL_0]], %[[SELECT_0]] : i64
# CHECK:             cc.condition %[[CMPI_1]](%[[VAL_0]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_2:.*]]: i64):
# CHECK:             %[[SUBI_2:.*]] = arith.subi %[[ARG0]], %[[VAL_2]] : i64
# CHECK:             %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_0]]{{\[}}%[[SUBI_2]]] : (!quake.veq<5>, i64) -> !quake.ref
# CHECK:             quake.x %[[EXTRACT_REF_0]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_2]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_4:.*]]: i64):
# CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_4]], %[[CONSTANT_0]] : i64
# CHECK:             cc.continue %[[ADDI_1]] : i64
# CHECK:           } {normalized}
# CHECK:           quake.dealloc %[[ALLOCA_0]] : !quake.veq<5>
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

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__fortranigans..0
# CHECK-SAME:      (%[[ARG0:.*]]: i64, %[[ARG1:.*]]: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
# CHECK-DAG:       %[[CONSTANT_0:.*]] = arith.constant 1 : i64
# CHECK-DAG:       %[[CONSTANT_1:.*]] = arith.constant 0 : i64
# CHECK-DAG:       %[[CONSTANT_3:.*]] = arith.constant -1 : i64
# CHECK-DAG:       %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<5>
# CHECK:           %[[SUBI_0:.*]] = arith.subi %[[ARG1]], %[[CONSTANT_3]] : i64
# CHECK:           %[[SUBI_1:.*]] = arith.subi %[[SUBI_0]], %[[ARG0]] : i64
# CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SUBI_1]], %[[CONSTANT_3]] : i64
# CHECK:           %[[DIVSI_0:.*]] = arith.divsi %[[ADDI_0]], %[[CONSTANT_3]] : i64
# CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[DIVSI_0]], %[[CONSTANT_1]] : i64
# CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[DIVSI_0]], %[[CONSTANT_1]] : i64
# CHECK:           %[[LOOP_0:.*]] = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_1]]) -> (i64)) {
# CHECK:             %[[CMPI_1:.*]] = arith.cmpi ne, %[[VAL_0]], %[[SELECT_0]] : i64
# CHECK:             cc.condition %[[CMPI_1]](%[[VAL_0]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_2:.*]]: i64):
# CHECK:             %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_0]][4] : (!quake.veq<5>) -> !quake.ref
# CHECK:             quake.h %[[EXTRACT_REF_0]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_2]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_4:.*]]: i64):
# CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_4]], %[[CONSTANT_0]] : i64
# CHECK:             cc.continue %[[ADDI_1]] : i64
# CHECK:           } {normalized}
# CHECK:           quake.dealloc %[[ALLOCA_0]] : !quake.veq<5>
