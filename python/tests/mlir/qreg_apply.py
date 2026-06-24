# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest

import cudaq


def test_kernel_qreg():
    """
    Test the `cudaq.Kernel` on each non-parameterized single qubit gate.
    Each gate is applied to both qubits in a 2-qubit register.
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 2.
    qreg = kernel.qalloc(2)
    # Apply each gate to entire register.
    # Test both with and without keyword arguments.
    kernel.h(target=qreg)
    kernel.x(target=qreg)
    kernel.y(target=qreg)
    kernel.z(qreg)
    kernel.t(qreg)
    kernel.s(qreg)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen__PythonKernelBuilderInstance
# CHECK-SAME:       () attributes {"cudaq-entrypoint"
# CHECK-DAG:           %[[VAL_0:.*]] = arith.constant 2 : i64
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_7:.*]]: i64):
# CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_7]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             quake.h %[[VAL_8]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_7]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_10]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_11:.*]] = cc.loop while ((%[[VAL_12:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_13]](%[[VAL_12]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_14:.*]]: i64):
# CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_14]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             quake.x %[[VAL_15]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_14]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64):
# CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_17]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_18:.*]] = cc.loop while ((%[[VAL_19:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_20]](%[[VAL_19]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_21:.*]]: i64):
# CHECK:             %[[VAL_22:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_21]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             quake.y %[[VAL_22]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_21]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_23:.*]]: i64):
# CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_24]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_25:.*]] = cc.loop while ((%[[VAL_26:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_27]](%[[VAL_26]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_28:.*]]: i64):
# CHECK:             %[[VAL_29:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_28]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             quake.z %[[VAL_29]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_28]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_30:.*]]: i64):
# CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_31]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_32:.*]] = cc.loop while ((%[[VAL_33:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_34:.*]] = arith.cmpi slt, %[[VAL_33]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_34]](%[[VAL_33]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_35:.*]]: i64):
# CHECK:             %[[VAL_36:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_35]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             quake.t %[[VAL_36]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_35]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_37:.*]]: i64):
# CHECK:             %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_38]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_39:.*]] = cc.loop while ((%[[VAL_40:.*]] = %[[VAL_2]]) -> (i64)) {
# CHECK:             %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_40]], %[[VAL_0]] : i64
# CHECK:             cc.condition %[[VAL_41]](%[[VAL_40]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_42:.*]]: i64):
# CHECK:             %[[VAL_43:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_42]]] : (!quake.veq<2>, i64) -> !quake.ref
# CHECK:             quake.s %[[VAL_43]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_42]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_44:.*]]: i64):
# CHECK:             %[[VAL_45:.*]] = arith.addi %[[VAL_44]], %[[VAL_1]] : i64
# CHECK:             cc.continue %[[VAL_45]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
