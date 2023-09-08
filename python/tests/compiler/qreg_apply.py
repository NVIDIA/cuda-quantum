# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import os

import pytest
import numpy as np

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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = arith.constant 2 : index
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_7:.*]]: index):
# CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_7]]] : (!quake.veq<2>, index) -> !quake.ref
# CHECK:             quake.h %[[VAL_8]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_7]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_9:.*]]: index):
# CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_10]] : index
# CHECK:           } {invariant}
# CHECK:           %[[VAL_11:.*]] = cc.loop while ((%[[VAL_12:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_13]](%[[VAL_12]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_14:.*]]: index):
# CHECK:             %[[VAL_15:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_14]]] : (!quake.veq<2>, index) -> !quake.ref
# CHECK:             quake.x %[[VAL_15]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_14]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_16:.*]]: index):
# CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_17]] : index
# CHECK:           } {invariant}
# CHECK:           %[[VAL_18:.*]] = cc.loop while ((%[[VAL_19:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_20]](%[[VAL_19]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_21:.*]]: index):
# CHECK:             %[[VAL_22:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_21]]] : (!quake.veq<2>, index) -> !quake.ref
# CHECK:             quake.y %[[VAL_22]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_21]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_23:.*]]: index):
# CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_24]] : index
# CHECK:           } {invariant}
# CHECK:           %[[VAL_25:.*]] = cc.loop while ((%[[VAL_26:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_27]](%[[VAL_26]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_28:.*]]: index):
# CHECK:             %[[VAL_29:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_28]]] : (!quake.veq<2>, index) -> !quake.ref
# CHECK:             quake.z %[[VAL_29]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_28]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_30:.*]]: index):
# CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_31]] : index
# CHECK:           } {invariant}
# CHECK:           %[[VAL_32:.*]] = cc.loop while ((%[[VAL_33:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_34:.*]] = arith.cmpi slt, %[[VAL_33]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_34]](%[[VAL_33]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_35:.*]]: index):
# CHECK:             %[[VAL_36:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_35]]] : (!quake.veq<2>, index) -> !quake.ref
# CHECK:             quake.t %[[VAL_36]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_35]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_37:.*]]: index):
# CHECK:             %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_38]] : index
# CHECK:           } {invariant}
# CHECK:           %[[VAL_39:.*]] = cc.loop while ((%[[VAL_40:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_40]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_41]](%[[VAL_40]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_42:.*]]: index):
# CHECK:             %[[VAL_43:.*]] = quake.extract_ref %[[VAL_3]]{{\[}}%[[VAL_42]]] : (!quake.veq<2>, index) -> !quake.ref
# CHECK:             quake.s %[[VAL_43]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_42]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_44:.*]]: index):
# CHECK:             %[[VAL_45:.*]] = arith.addi %[[VAL_44]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_45]] : index
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
