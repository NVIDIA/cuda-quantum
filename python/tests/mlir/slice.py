# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os
import pytest
import cudaq


def test_static_case():

    @cudaq.kernel
    def static_empty_slice() -> int:
        q = cudaq.qvector(5)
        index = 0

        # Evaluates to q[0:0]. Expected: no-op.
        x(q[0:index])

        return int(mz(q[0])) + int(mz(q[1])) + int(mz(q[2])) + int(mz(
            q[3])) + int(mz(q[4]))

    print(static_empty_slice)

    result = cudaq.run(static_empty_slice, shots_count=10)
    print(result)


def test_another_case():

    @cudaq.kernel
    def static_notempty_slice() -> int:
        q = cudaq.qvector(5)
        index = 1

        # Evaluates to q[0:1]. Expected: 1 qubit
        x(q[0:index])

        return int(mz(q[0])) + int(mz(q[1])) + int(mz(q[2])) + int(mz(
            q[3])) + int(mz(q[4]))

    print(static_notempty_slice)

    result = cudaq.run(static_notempty_slice, shots_count=10)
    print(result)


def test_dynamic_case():

    @cudaq.kernel
    def dynamic_empty_slice() -> int:
        q = cudaq.qvector(5)

        # mz(q[4]) is deterministically 0
        index = int(mz(q[4]))

        # Evaluates to q[0:0] logically, so is a dynamic NOP.
        x(q[0:index])

        return int(mz(q[0])) + int(mz(q[1])) + int(mz(q[2])) + int(mz(
            q[3])) + int(mz(q[4]))

    print(dynamic_empty_slice)

    result = cudaq.run(dynamic_empty_slice, shots_count=10)
    print(result)


def test_another_dynamic_case():

    @cudaq.kernel
    def dynamic_notempty_slice() -> int:
        q = cudaq.qvector(5)

        x(q)

        # mz(q[4]) is deterministically 1
        index = int(mz(q[4]))

        # Evaluates to q[0:1] logically. 1 qubit.
        x(q[0:index])

        return int(mz(q[0])) + int(mz(q[1])) + int(mz(q[2])) + int(mz(
            q[3])) + int(mz(q[4]))

    print(dynamic_notempty_slice)

    result = cudaq.run(dynamic_notempty_slice, shots_count=10)
    print(result)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__static_empty_slice..0x
# CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
# CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
# CHECK:           %[[SCOPE_0:.*]] = cc.scope -> (i64) {
# CHECK:             %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<5>
# CHECK:             %[[POISON_0:.*]] = cc.undef !quake.veq<?>
# CHECK:             %[[LOOP_0:.*]] = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_0]]) -> (i64)) {
# CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[CONSTANT_0]] : i64
# CHECK:               cc.condition %[[CMPI_0]](%[[VAL_0]] : i64)
# CHECK:             } do {

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__static_notempty_slice..0x
# CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i64
# CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i64
# CHECK:           %[[SCOPE_0:.*]] = cc.scope -> (i64) {
# CHECK:             %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<5>
# CHECK:             %[[LOOP_0:.*]] = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_1]]) -> (i64)) {
# CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[CONSTANT_0]] : i64
# CHECK:               cc.condition %[[CMPI_0]](%[[VAL_0]] : i64)
# CHECK:             } do {

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__dynamic_empty_slice..0x
# CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
# CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
# CHECK:           %[[SCOPE_0:.*]] = cc.scope -> (i64) {
# CHECK:             %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<5>
# CHECK:             %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_0]][4] : (!quake.veq<5>) -> !quake.ref
# CHECK:             %[[MZ_0:.*]] = quake.mz %[[EXTRACT_REF_0]] name "index" : (!quake.ref) -> !cc.measure_handle
# CHECK:             %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[MZ_0]] : (!cc.measure_handle) -> i1
# CHECK:             %[[CAST_0:.*]] = cc.cast unsigned %[[DISCRIMINATE_0]] : (i1) -> i64
# CHECK:             %[[SUBI_0:.*]] = arith.subi %[[CAST_0]], %[[CONSTANT_1]] : i64
# CHECK:             %[[CMPI_0:.*]] = arith.cmpi sge, %[[SUBI_0]], %[[CONSTANT_0]] : i64
# CHECK:             %[[IF_0:.*]]:2 = cc.if(%[[CMPI_0]]) -> (!quake.veq<?>, i64) {
# CHECK:               %[[SUBVEQ_0:.*]] = quake.subveq %[[ALLOCA_0]], 0, %[[SUBI_0]] : (!quake.veq<5>, i64) -> !quake.veq<?>
# CHECK:               %[[VEQ_SIZE_0:.*]] = quake.veq_size %[[SUBVEQ_0]] : (!quake.veq<?>) -> i64
# CHECK:               cc.continue %[[SUBVEQ_0]], %[[VEQ_SIZE_0]] : !quake.veq<?>, i64
# CHECK:             } else {
# CHECK:               %[[UNDEF_0:.*]] = cc.undef !quake.veq<?>
# CHECK:               cc.continue %[[UNDEF_0]], %[[CONSTANT_0]] : !quake.veq<?>, i64
# CHECK:             }
# CHECK:             %[[LOOP_0:.*]] = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_0]]) -> (i64)) {
# CHECK:               %[[CMPI_1:.*]] = arith.cmpi slt, %[[VAL_0]], %[[IF_0]]#1 : i64
# CHECK:               cc.condition %[[CMPI_1]](%[[VAL_0]] : i64)
# CHECK:             } do {

# CHECK-LABEL:   func.func @__nvqpp__mlirgen__dynamic_notempty_slice..0x
# CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i64
# CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i64
# CHECK:           %[[CONSTANT_2:.*]] = arith.constant 5 : i64
# CHECK:           %[[SCOPE_0:.*]] = cc.scope -> (i64) {
# CHECK:             %[[ALLOCA_0:.*]] = quake.alloca !quake.veq<5>
# CHECK:             %[[LOOP_0:.*]] = cc.loop while ((%[[VAL_0:.*]] = %[[CONSTANT_0]]) -> (i64)) {
# CHECK:               %[[CMPI_0:.*]] = arith.cmpi slt, %[[VAL_0]], %[[CONSTANT_2]] : i64
# CHECK:               cc.condition %[[CMPI_0]](%[[VAL_0]] : i64)
# CHECK:             } do {
# CHECK:             ^bb0(%[[VAL_1:.*]]: i64):
# CHECK:               %[[EXTRACT_REF_0:.*]] = quake.extract_ref %[[ALLOCA_0]]{{\[}}%[[VAL_1]]] : (!quake.veq<5>, i64) -> !quake.ref
# CHECK:               quake.x %[[EXTRACT_REF_0]] : (!quake.ref) -> ()
# CHECK:               cc.continue %[[VAL_1]] : i64
# CHECK:             } step {
# CHECK:             ^bb0(%[[VAL_2:.*]]: i64):
# CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_1]] : i64
# CHECK:               cc.continue %[[ADDI_0]] : i64
# CHECK:             } {invariant}
# CHECK:             %[[EXTRACT_REF_1:.*]] = quake.extract_ref %[[ALLOCA_0]][4] : (!quake.veq<5>) -> !quake.ref
# CHECK:             %[[MZ_0:.*]] = quake.mz %[[EXTRACT_REF_1]] name "index" : (!quake.ref) -> !cc.measure_handle
# CHECK:             %[[DISCRIMINATE_0:.*]] = quake.discriminate %[[MZ_0]] : (!cc.measure_handle) -> i1
# CHECK:             %[[CAST_0:.*]] = cc.cast unsigned %[[DISCRIMINATE_0]] : (i1) -> i64
# CHECK:             %[[SUBI_0:.*]] = arith.subi %[[CAST_0]], %[[CONSTANT_1]] : i64
# CHECK:             %[[CMPI_1:.*]] = arith.cmpi sge, %[[SUBI_0]], %[[CONSTANT_0]] : i64
# CHECK:             %[[IF_0:.*]]:2 = cc.if(%[[CMPI_1]]) -> (!quake.veq<?>, i64) {
# CHECK:               %[[SUBVEQ_0:.*]] = quake.subveq %[[ALLOCA_0]], 0, %[[SUBI_0]] : (!quake.veq<5>, i64) -> !quake.veq<?>
# CHECK:               %[[VEQ_SIZE_0:.*]] = quake.veq_size %[[SUBVEQ_0]] : (!quake.veq<?>) -> i64
# CHECK:               cc.continue %[[SUBVEQ_0]], %[[VEQ_SIZE_0]] : !quake.veq<?>, i64
# CHECK:             } else {
# CHECK:               %[[UNDEF_0:.*]] = cc.undef !quake.veq<?>
# CHECK:               cc.continue %[[UNDEF_0]], %[[CONSTANT_0]] : !quake.veq<?>, i64
# CHECK:             }
# CHECK:             %[[LOOP_1:.*]] = cc.loop while ((%[[VAL_3:.*]] = %[[CONSTANT_0]]) -> (i64)) {
# CHECK:               %[[CMPI_2:.*]] = arith.cmpi slt, %[[VAL_3]], %[[IF_0]]#1 : i64
# CHECK:               cc.condition %[[CMPI_2]](%[[VAL_3]] : i64)
# CHECK:             } do {
