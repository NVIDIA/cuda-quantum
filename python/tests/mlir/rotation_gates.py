# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

import pytest

import cudaq


def test_control_list_rotation():
    """Tests the controlled rotation gates, provided a list of controls."""
    kernel, value = cudaq.make_kernel(float)
    target = kernel.qalloc()
    q1 = kernel.qalloc()
    q2 = kernel.qalloc()

    controls = [q1, q2]
    controls_reversed = [q2, q1]

    kernel.crx(value, controls, target)
    kernel.crx(1.0, controls_reversed, target)

    kernel.cry(value, controls_reversed, target)
    kernel.cry(2.0, controls, target)

    kernel.crz(value, controls, target)
    kernel.crz(3.0, controls_reversed, target)

    kernel.cr1(value, controls_reversed, target)
    kernel.cr1(4.0, controls, target)

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint"} {
# CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 4.000000e+00 : f64
# CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 3.000000e+00 : f64
# CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 2.000000e+00 : f64
# CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1.000000e+00 : f64
# CHECK-DAG:       %[[VAL_5:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_6:.*]] = quake.alloca !quake.ref
# CHECK-DAG:       %[[VAL_7:.*]] = quake.alloca !quake.ref
# CHECK:           quake.rx (%[[VAL_0]]) {{\[}}%[[VAL_6]], %[[VAL_7]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.rx (%[[VAL_4]]) {{\[}}%[[VAL_7]], %[[VAL_6]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.ry (%[[VAL_0]]) {{\[}}%[[VAL_7]], %[[VAL_6]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.ry (%[[VAL_3]]) {{\[}}%[[VAL_6]], %[[VAL_7]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.rz (%[[VAL_0]]) {{\[}}%[[VAL_6]], %[[VAL_7]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.rz (%[[VAL_2]]) {{\[}}%[[VAL_7]], %[[VAL_6]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.r1 (%[[VAL_0]]) {{\[}}%[[VAL_7]], %[[VAL_6]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           quake.r1 (%[[VAL_1]]) {{\[}}%[[VAL_6]], %[[VAL_7]]] %[[VAL_5]] : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }


def test_rotation_qreg():
    """Tests that our rotation gates may accept a qreg as a target."""
    kernel, value = cudaq.make_kernel(float)
    targets = kernel.qalloc(3)

    kernel.rx(value, targets)
    kernel.rx(1.0, targets)

    kernel.ry(value, targets)
    kernel.ry(1.0, targets)

    kernel.rz(value, targets)
    kernel.rz(1.0, targets)

    kernel.r1(value, targets)
    kernel.r1(1.0, targets)

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:                                                                 %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint"} {
# CHECK-DAG:           %[[VAL_1:.*]] = arith.constant 3 : i64
# CHECK-DAG:           %[[VAL_2:.*]] = arith.constant 1.000000e+00 : f64
# CHECK-DAG:           %[[VAL_3:.*]] = arith.constant 1 : i64
# CHECK-DAG:           %[[VAL_4:.*]] = arith.constant 0 : i64
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<3>
# CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_8]](%[[VAL_7]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_9:.*]]: i64):
# CHECK:             %[[VAL_10:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_9]]] : (!quake.veq<3>, i64) -> !quake.ref
# CHECK:             quake.rx (%[[VAL_0]]) %[[VAL_10]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_9]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_11:.*]]: i64):
# CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_12]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_13:.*]] = cc.loop while ((%[[VAL_14:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_15:.*]] = arith.cmpi slt, %[[VAL_14]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_15]](%[[VAL_14]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_16:.*]]: i64):
# CHECK:             %[[VAL_17:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_16]]] : (!quake.veq<3>, i64) -> !quake.ref
# CHECK:             quake.rx (%[[VAL_2]]) %[[VAL_17]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_16]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_18:.*]]: i64):
# CHECK:             %[[VAL_19:.*]] = arith.addi %[[VAL_18]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_19]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_20:.*]] = cc.loop while ((%[[VAL_21:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_22:.*]] = arith.cmpi slt, %[[VAL_21]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_22]](%[[VAL_21]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_23:.*]]: i64):
# CHECK:             %[[VAL_24:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_23]]] : (!quake.veq<3>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_0]]) %[[VAL_24]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_23]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_25:.*]]: i64):
# CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_26]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_27:.*]] = cc.loop while ((%[[VAL_28:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_29:.*]] = arith.cmpi slt, %[[VAL_28]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_29]](%[[VAL_28]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_30:.*]]: i64):
# CHECK:             %[[VAL_31:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_30]]] : (!quake.veq<3>, i64) -> !quake.ref
# CHECK:             quake.ry (%[[VAL_2]]) %[[VAL_31]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_30]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_32:.*]]: i64):
# CHECK:             %[[VAL_33:.*]] = arith.addi %[[VAL_32]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_33]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_34:.*]] = cc.loop while ((%[[VAL_35:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_36:.*]] = arith.cmpi slt, %[[VAL_35]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_36]](%[[VAL_35]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_37:.*]]: i64):
# CHECK:             %[[VAL_38:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_37]]] : (!quake.veq<3>, i64) -> !quake.ref
# CHECK:             quake.rz (%[[VAL_0]]) %[[VAL_38]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_37]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_39:.*]]: i64):
# CHECK:             %[[VAL_40:.*]] = arith.addi %[[VAL_39]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_40]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_41:.*]] = cc.loop while ((%[[VAL_42:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_43:.*]] = arith.cmpi slt, %[[VAL_42]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_43]](%[[VAL_42]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_44:.*]]: i64):
# CHECK:             %[[VAL_45:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_44]]] : (!quake.veq<3>, i64) -> !quake.ref
# CHECK:             quake.rz (%[[VAL_2]]) %[[VAL_45]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_44]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_46:.*]]: i64):
# CHECK:             %[[VAL_47:.*]] = arith.addi %[[VAL_46]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_47]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_48:.*]] = cc.loop while ((%[[VAL_49:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_50:.*]] = arith.cmpi slt, %[[VAL_49]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_50]](%[[VAL_49]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_51:.*]]: i64):
# CHECK:             %[[VAL_52:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_51]]] : (!quake.veq<3>, i64) -> !quake.ref
# CHECK:             quake.r1 (%[[VAL_0]]) %[[VAL_52]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_51]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_53:.*]]: i64):
# CHECK:             %[[VAL_54:.*]] = arith.addi %[[VAL_53]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_54]] : i64
# CHECK:           } {invariant}
# CHECK:           %[[VAL_55:.*]] = cc.loop while ((%[[VAL_56:.*]] = %[[VAL_4]]) -> (i64)) {
# CHECK:             %[[VAL_57:.*]] = arith.cmpi slt, %[[VAL_56]], %[[VAL_1]] : i64
# CHECK:             cc.condition %[[VAL_57]](%[[VAL_56]] : i64)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_58:.*]]: i64):
# CHECK:             %[[VAL_59:.*]] = quake.extract_ref %[[VAL_5]]{{\[}}%[[VAL_58]]] : (!quake.veq<3>, i64) -> !quake.ref
# CHECK:             quake.r1 (%[[VAL_2]]) %[[VAL_59]] : (f64, !quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_58]] : i64
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_60:.*]]: i64):
# CHECK:             %[[VAL_61:.*]] = arith.addi %[[VAL_60]], %[[VAL_3]] : i64
# CHECK:             cc.continue %[[VAL_61]] : i64
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
