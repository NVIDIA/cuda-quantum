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


def test_make_kernel_no_input():
    """
    Test `cudaq.make_kernel` without any inputs.
    """
    # Empty constructor (no kernel type).
    kernel = cudaq.make_kernel()
    # Kernel arguments should be an empty list.
    assert kernel.arguments == []
    # Kernel should have 0 parameters.
    assert kernel.argument_count == 0
    # Print the quake string to the terminal. FileCheck will ensure that
    # the MLIR doesn't contain any instructions or register allocations.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           return
# CHECK:         }


def test_make_kernel_float():
    """
    Test `cudaq.make_kernel` with one float parameter.
    """
    kernel, parameter = cudaq.make_kernel(float)
    # Kernel should only have 1 argument and parameter.
    got_arguments = kernel.arguments
    got_argument_count = kernel.argument_count
    assert len(got_arguments) == 1
    assert got_argument_count == 1
    # Dump the MLIR for FileCheck.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:    %[[VAL_0:.*]]: f64) {
# CHECK:           return
# CHECK:         }


def test_make_kernel_list():
    """
    Test `cudaq.make_kernel` with a list of floats as parameters.
    """
    kernel, parameter = cudaq.make_kernel(list)
    # Kernel should only have 1 argument and parameter.
    got_arguments = kernel.arguments
    got_argument_count = kernel.argument_count
    assert len(got_arguments) == 1
    assert got_argument_count == 1
    # Dump the MLIR for FileCheck.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:    %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           return
# CHECK:         }


def test_make_kernel_multiple_floats():
    """
    Test `cudaq.make_kernel` with multiple parameters.
    """
    kernel, parameter_1, parameter_2 = cudaq.make_kernel(float, float)
    # Kernel should have 2 arguments and parameters.
    got_arguments = kernel.arguments
    got_argument_count = kernel.argument_count
    assert len(got_arguments) == 2
    assert got_argument_count == 2
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:    %[[VAL_0:.*]]: f64,
# CHECK-SAME:    %[[VAL_1:.*]]: f64) {
# CHECK:           return
# CHECK:         }


def test_make_kernel_mixed_args():
    """
    Test `cudaq.make_kernel` with arguments of different types.
    """
    kernel, parameter_1, parameter_2 = cudaq.make_kernel(list, float)
    # Kernel should have 2 arguments and parameters.
    got_arguments = kernel.arguments
    got_argument_count = kernel.argument_count
    assert len(got_arguments) == 2
    assert got_argument_count == 2
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:    %[[VAL_0:.*]]: !cc.stdvec<f64>,
# CHECK-SAME:    %[[VAL_1:.*]]: f64) {
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_empty():
    """
    Test `cudaq.Kernel.qalloc()` when no arguments are provided.
    """
    kernel = cudaq.make_kernel()
    # Use `qalloc()` with no function arguments.
    qubit = kernel.qalloc()
    # Assert that only 1 qubit is allocated in the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_qubit():
    """
    Test `cudaq.Kernel.qalloc()` when a handle to a single qubit
    is provided.
    """
    kernel = cudaq.make_kernel()
    # Use `qalloc()` with 1 qubit allocated.
    qubit = kernel.qalloc(1)
    # Assert that only 1 qubit is allocated in the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_qubit_keyword():
    """
    Test `cudaq.Kernel.qalloc()` when a handle to a single qubit
    is provided with a keyword argument.
    """
    kernel = cudaq.make_kernel()
    # Use `qalloc()` with 1 qubit allocated and `qubit_count` keyword used.
    qubit = kernel.qalloc(qubit_count=1)
    # Assert that only 1 qubit is allocated in the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_qreg():
    """
    Test `cudaq.Kernel.qalloc()` when a handle to a register of
    qubits is provided.
    """
    kernel = cudaq.make_kernel()
    # Use `qalloc()` with 10 qubits allocated.
    qubit = kernel.qalloc(10)
    # Assert that 10 qubits have been allocated in the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qvec<10>
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_qreg_keyword():
    """
    Test `cudaq.Kernel.qalloc()` when a handle to a register of
    qubits is provided with a keyword argument.
    """
    kernel = cudaq.make_kernel()
    # Use `qalloc()` with 10 qubits allocated.
    qubit = kernel.qalloc(qubit_count=10)
    # Assert that 10 qubits have been allocated in the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qvec<10>
# CHECK:           return
# CHECK:         }


def test_kernel_qalloc_quake_val():
    """
    Test `cudaq.Kernel.qalloc()` when a `QuakeValue` is provided.
    """
    kernel, value = cudaq.make_kernel(int)
    qreg = kernel.qalloc(value)
    qubit_count = 10
    kernel(qubit_count)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca(%[[VAL_0]] : i32) : !quake.qvec<?>
# CHECK:           return
# CHECK:         }


def test_kernel_non_param_1q():
    """
    Test the `cudaq.Kernel` on each non-parameterized single qubit gate.
    Each gate is applied to a single qubit in a 1-qubit register.
    """
    # Empty constructor (no kernel type).
    kernel = cudaq.make_kernel()
    # Allocating a register of size 1 returns just a qubit.
    qubit = kernel.qalloc(1)
    # Apply each gate to the qubit.
    kernel.h(target=qubit)
    kernel.x(target=qubit)
    kernel.y(target=qubit)
    kernel.z(qubit)
    kernel.t(qubit)
    kernel.s(qubit)
    kernel()
    # Kernel arguments should still be an empty list.
    assert kernel.arguments == []
    # Kernel should still have 0 parameters.
    assert kernel.argument_count == 0
    # Check the conversion to Quake.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.h (%[[VAL_0]])
# CHECK:           quake.x (%[[VAL_0]])
# CHECK:           quake.y (%[[VAL_0]])
# CHECK:           quake.z (%[[VAL_0]])
# CHECK:           quake.t (%[[VAL_0]])
# CHECK:           quake.s (%[[VAL_0]])
# CHECK:           return
# CHECK:         }


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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 2 : index
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_3:.*]] = quake.alloca : !quake.qvec<2>
# CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_7:.*]]: index):
# CHECK:             %[[VAL_8:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_7]]] : !quake.qvec<2>[index] -> !quake.qref
# CHECK:             quake.h (%[[VAL_8]])
# CHECK:             cc.continue %[[VAL_7]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_9:.*]]: index):
# CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_10]] : index
# CHECK:           } {counted}
# CHECK:           %[[VAL_11:.*]] = cc.loop while ((%[[VAL_12:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_13:.*]] = arith.cmpi slt, %[[VAL_12]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_13]](%[[VAL_12]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_14:.*]]: index):
# CHECK:             %[[VAL_15:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_14]]] : !quake.qvec<2>[index] -> !quake.qref
# CHECK:             quake.x (%[[VAL_15]])
# CHECK:             cc.continue %[[VAL_14]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_16:.*]]: index):
# CHECK:             %[[VAL_17:.*]] = arith.addi %[[VAL_16]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_17]] : index
# CHECK:           } {counted}
# CHECK:           %[[VAL_18:.*]] = cc.loop while ((%[[VAL_19:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_20:.*]] = arith.cmpi slt, %[[VAL_19]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_20]](%[[VAL_19]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_21:.*]]: index):
# CHECK:             %[[VAL_22:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_21]]] : !quake.qvec<2>[index] -> !quake.qref
# CHECK:             quake.y (%[[VAL_22]])
# CHECK:             cc.continue %[[VAL_21]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_23:.*]]: index):
# CHECK:             %[[VAL_24:.*]] = arith.addi %[[VAL_23]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_24]] : index
# CHECK:           } {counted}
# CHECK:           %[[VAL_25:.*]] = cc.loop while ((%[[VAL_26:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_27:.*]] = arith.cmpi slt, %[[VAL_26]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_27]](%[[VAL_26]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_28:.*]]: index):
# CHECK:             %[[VAL_29:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_28]]] : !quake.qvec<2>[index] -> !quake.qref
# CHECK:             quake.z (%[[VAL_29]])
# CHECK:             cc.continue %[[VAL_28]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_30:.*]]: index):
# CHECK:             %[[VAL_31:.*]] = arith.addi %[[VAL_30]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_31]] : index
# CHECK:           } {counted}
# CHECK:           %[[VAL_32:.*]] = cc.loop while ((%[[VAL_33:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_34:.*]] = arith.cmpi slt, %[[VAL_33]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_34]](%[[VAL_33]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_35:.*]]: index):
# CHECK:             %[[VAL_36:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_35]]] : !quake.qvec<2>[index] -> !quake.qref
# CHECK:             quake.t (%[[VAL_36]])
# CHECK:             cc.continue %[[VAL_35]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_37:.*]]: index):
# CHECK:             %[[VAL_38:.*]] = arith.addi %[[VAL_37]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_38]] : index
# CHECK:           } {counted}
# CHECK:           %[[VAL_39:.*]] = cc.loop while ((%[[VAL_40:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_41:.*]] = arith.cmpi slt, %[[VAL_40]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_41]](%[[VAL_40]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_42:.*]]: index):
# CHECK:             %[[VAL_43:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_42]]] : !quake.qvec<2>[index] -> !quake.qref
# CHECK:             quake.s (%[[VAL_43]])
# CHECK:             cc.continue %[[VAL_42]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_44:.*]]: index):
# CHECK:             %[[VAL_45:.*]] = arith.addi %[[VAL_44]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_45]] : index
# CHECK:           } {counted}
# CHECK:           return
# CHECK:         }


def test_kernel_param_1q():
    """
    Test the `cudaq.Kernel` on each single-qubit, parameterized gate.
    Each gate is applied to a single qubit in a 1-qubit register. 
    Note: at this time, we can only apply rotation gates to one qubit at a time,
    not to an entire register.
    """
    kernel, parameter = cudaq.make_kernel(float)
    qubit = kernel.qalloc(1)
    # Apply each parameterized gate to the qubit.
    # Test both with and without keyword arguments.
    kernel.rx(parameter=parameter, target=qubit)
    kernel.ry(parameter, qubit)
    kernel.rz(parameter, qubit)
    kernel(3.14)
    # Should have 1 argument and parameter.
    got_arguments = kernel.arguments
    got_argument_count = kernel.argument_count
    assert len(got_arguments) == 1
    assert got_argument_count == 1
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:                                                                   %[[VAL_0:.*]]: f64) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.rx |%[[VAL_0]] : f64|(%[[VAL_1]])
# CHECK:           quake.ry |%[[VAL_0]] : f64|(%[[VAL_1]])
# CHECK:           quake.rz |%[[VAL_0]] : f64|(%[[VAL_1]])
# CHECK:           return
# CHECK:         }


def test_kernel_2q():
    """
    Test the `cudaq.Kernel` on each two-qubit gate (controlled 
    single qubit gates). We alternate the order of the control and target
    qubits between each successive gate.
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 2.
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # First three gates check the overload for providing a single control
    # qubit as a list of length 1.
    # Test both with and without keyword arguments.
    kernel.ch(controls=[qubit_0], target=qubit_1)
    kernel.cx([qubit_1], qubit_0)
    kernel.cy([qubit_0], qubit_1)
    # Check the overload for providing a single control qubit on its own.
    # Test both with and without keyword arguments.
    kernel.cz(control=qubit_1, target=qubit_0)
    kernel.ct(qubit_0, qubit_1)
    kernel.cs(qubit_1, qubit_0)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
# CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
# CHECK:           %[[VAL_2:.*]] = quake.alloca : !quake.qvec<2>
# CHECK:           %[[VAL_3:.*]] = quake.qextract %[[VAL_2]]{{\[}}%[[VAL_1]]] : !quake.qvec<2>[i32] -> !quake.qref
# CHECK:           %[[VAL_4:.*]] = quake.qextract %[[VAL_2]]{{\[}}%[[VAL_0]]] : !quake.qvec<2>[i32] -> !quake.qref
# CHECK:           quake.h {{\[}}%[[VAL_3]] : !quake.qref] (%[[VAL_4]])
# CHECK:           quake.x {{\[}}%[[VAL_4]] : !quake.qref] (%[[VAL_3]])
# CHECK:           quake.y {{\[}}%[[VAL_3]] : !quake.qref] (%[[VAL_4]])
# CHECK:           quake.z {{\[}}%[[VAL_4]] : !quake.qref] (%[[VAL_3]])
# CHECK:           quake.t {{\[}}%[[VAL_3]] : !quake.qref] (%[[VAL_4]])
# CHECK:           quake.s {{\[}}%[[VAL_4]] : !quake.qref] (%[[VAL_3]])
# CHECK:           return
# CHECK:         }


def test_kernel_3q():
    """
    Test the `cudaq.Kernel` on each multi-qubit gate (multi-controlled single
    qubit gates). We do this for the case of a 3-qubit kernel. 
    """
    kernel = cudaq.make_kernel()
    # Allocate a register of size 3.
    qreg = kernel.qalloc(3)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    qubit_2 = qreg[2]
    # Apply each gate to entire register.
    # Note: we alternate between orders to make the circuit less trivial.
    kernel.ch([qubit_0, qubit_1], qubit_2)
    kernel.cx([qubit_2, qubit_0], qubit_1)
    kernel.cy([qubit_1, qubit_2], qubit_0)
    kernel.cz([qubit_0, qubit_1], qubit_2)
    kernel.ct([qubit_2, qubit_0], qubit_1)
    kernel.cs([qubit_1, qubit_2], qubit_0)
    kernel()
    assert (kernel.arguments == [])
    assert (kernel.argument_count == 0)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i32
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : i32
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : i32
# CHECK:           %[[VAL_3:.*]] = quake.alloca : !quake.qvec<3>
# CHECK:           %[[VAL_4:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_2]]] : !quake.qvec<3>[i32] -> !quake.qref
# CHECK:           %[[VAL_5:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_1]]] : !quake.qvec<3>[i32] -> !quake.qref
# CHECK:           %[[VAL_6:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_0]]] : !quake.qvec<3>[i32] -> !quake.qref
# CHECK:           quake.h {{\[}}%[[VAL_4]], %[[VAL_5]] : !quake.qref, !quake.qref] (%[[VAL_6]])
# CHECK:           quake.x {{\[}}%[[VAL_6]], %[[VAL_4]] : !quake.qref, !quake.qref] (%[[VAL_5]])
# CHECK:           quake.y {{\[}}%[[VAL_5]], %[[VAL_6]] : !quake.qref, !quake.qref] (%[[VAL_4]])
# CHECK:           quake.z {{\[}}%[[VAL_4]], %[[VAL_5]] : !quake.qref, !quake.qref] (%[[VAL_6]])
# CHECK:           quake.t {{\[}}%[[VAL_6]], %[[VAL_4]] : !quake.qref, !quake.qref] (%[[VAL_5]])
# CHECK:           quake.s {{\[}}%[[VAL_5]], %[[VAL_6]] : !quake.qref, !quake.qref] (%[[VAL_4]])
# CHECK:           return
# CHECK:         }


def test_kernel_measure_1q():
    """
    Test the measurement instruction for `cudaq.Kernel` by applying
    measurements to qubits one at a time.
    """
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # Check that we can apply measurements to 1 qubit at a time.
    kernel.mx(qubit_0)
    kernel.mx(qubit_1)
    kernel.my(qubit_0)
    kernel.my(qubit_1)
    kernel.mz(qubit_0)
    kernel.mz(qubit_1)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 1 : i32
# CHECK:           %[[VAL_1:.*]] = arith.constant 0 : i32
# CHECK:           %[[VAL_2:.*]] = quake.alloca : !quake.qvec<2>
# CHECK:           %[[VAL_3:.*]] = quake.qextract %[[VAL_2]]{{\[}}%[[VAL_1]]] : !quake.qvec<2>[i32] -> !quake.qref
# CHECK:           %[[VAL_4:.*]] = quake.qextract %[[VAL_2]]{{\[}}%[[VAL_0]]] : !quake.qvec<2>[i32] -> !quake.qref
# CHECK:           %[[VAL_5:.*]] = quake.mx(%[[VAL_3]] : !quake.qref) {registerName = ""} : i1
# CHECK:           %[[VAL_6:.*]] = quake.mx(%[[VAL_4]] : !quake.qref) {registerName = ""} : i1
# CHECK:           %[[VAL_7:.*]] = quake.my(%[[VAL_3]] : !quake.qref) {registerName = ""} : i1
# CHECK:           %[[VAL_8:.*]] = quake.my(%[[VAL_4]] : !quake.qref) {registerName = ""} : i1
# CHECK:           %[[VAL_9:.*]] = quake.mz(%[[VAL_3]] : !quake.qref) {registerName = ""} : i1
# CHECK:           %[[VAL_10:.*]] = quake.mz(%[[VAL_4]] : !quake.qref) {registerName = ""} : i1
# CHECK:           return
# CHECK:         }


def test_kernel_measure_qreg():
    """
    Test the measurement instruciton for `cudaq.Kernel` by applying
    measurements to an entire qreg.
    """
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(3)
    # Check that we can apply measurements to an entire register.
    kernel.mx(qreg)
    kernel.my(qreg)
    kernel.mz(qreg)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 3 : index
# CHECK:           %[[VAL_1:.*]] = arith.constant 3 : i64
# CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_4:.*]] = quake.alloca : !quake.qvec<3>
# CHECK:           %[[VAL_5:.*]] = llvm.alloca %[[VAL_1]] x i1 : (i64) -> !llvm.ptr<i1>
# CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_3]]) -> (index)) {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_8]](%[[VAL_7]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_9:.*]]: index):
# CHECK:             %[[VAL_10:.*]] = quake.qextract %[[VAL_4]]{{\[}}%[[VAL_9]]] : !quake.qvec<3>[index] -> !quake.qref
# CHECK:             %[[VAL_11:.*]] = quake.mx(%[[VAL_10]] : !quake.qref) : i1
# CHECK:             %[[VAL_12:.*]] = arith.index_cast %[[VAL_9]] : index to i64
# CHECK:             %[[VAL_13:.*]] = llvm.getelementptr %[[VAL_5]]{{\[}}%[[VAL_12]]] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
# CHECK:             llvm.store %[[VAL_11]], %[[VAL_13]] : !llvm.ptr<i1>
# CHECK:             cc.continue %[[VAL_9]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_14:.*]]: index):
# CHECK:             %[[VAL_15:.*]] = arith.addi %[[VAL_14]], %[[VAL_2]] : index
# CHECK:             cc.continue %[[VAL_15]] : index
# CHECK:           } {counted}
# CHECK:           %[[VAL_16:.*]] = llvm.alloca %[[VAL_1]] x i1 : (i64) -> !llvm.ptr<i1>
# CHECK:           %[[VAL_17:.*]] = cc.loop while ((%[[VAL_18:.*]] = %[[VAL_3]]) -> (index)) {
# CHECK:             %[[VAL_19:.*]] = arith.cmpi slt, %[[VAL_18]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_19]](%[[VAL_18]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_20:.*]]: index):
# CHECK:             %[[VAL_21:.*]] = quake.qextract %[[VAL_4]]{{\[}}%[[VAL_20]]] : !quake.qvec<3>[index] -> !quake.qref
# CHECK:             %[[VAL_22:.*]] = quake.my(%[[VAL_21]] : !quake.qref) : i1
# CHECK:             %[[VAL_23:.*]] = arith.index_cast %[[VAL_20]] : index to i64
# CHECK:             %[[VAL_24:.*]] = llvm.getelementptr %[[VAL_16]]{{\[}}%[[VAL_23]]] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
# CHECK:             llvm.store %[[VAL_22]], %[[VAL_24]] : !llvm.ptr<i1>
# CHECK:             cc.continue %[[VAL_20]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_25:.*]]: index):
# CHECK:             %[[VAL_26:.*]] = arith.addi %[[VAL_25]], %[[VAL_2]] : index
# CHECK:             cc.continue %[[VAL_26]] : index
# CHECK:           } {counted}
# CHECK:           %[[VAL_27:.*]] = llvm.alloca %[[VAL_1]] x i1 : (i64) -> !llvm.ptr<i1>
# CHECK:           %[[VAL_28:.*]] = cc.loop while ((%[[VAL_29:.*]] = %[[VAL_3]]) -> (index)) {
# CHECK:             %[[VAL_30:.*]] = arith.cmpi slt, %[[VAL_29]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_30]](%[[VAL_29]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_31:.*]]: index):
# CHECK:             %[[VAL_32:.*]] = quake.qextract %[[VAL_4]]{{\[}}%[[VAL_31]]] : !quake.qvec<3>[index] -> !quake.qref
# CHECK:             %[[VAL_33:.*]] = quake.mz(%[[VAL_32]] : !quake.qref) : i1
# CHECK:             %[[VAL_34:.*]] = arith.index_cast %[[VAL_31]] : index to i64
# CHECK:             %[[VAL_35:.*]] = llvm.getelementptr %[[VAL_27]]{{\[}}%[[VAL_34]]] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
# CHECK:             llvm.store %[[VAL_33]], %[[VAL_35]] : !llvm.ptr<i1>
# CHECK:             cc.continue %[[VAL_31]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_36:.*]]: index):
# CHECK:             %[[VAL_37:.*]] = arith.addi %[[VAL_36]], %[[VAL_2]] : index
# CHECK:             cc.continue %[[VAL_37]] : index
# CHECK:           } {counted}
# CHECK:           return
# CHECK:         }


def test_kernel_conditional():
    """
    Test the conditional measurement functionality of `cudaq.Kernel`.
    """
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(2)
    qubit_0 = qreg[0]
    qubit_1 = qreg[1]
    # Rotate qubit 0 with an X-gate and measure.
    kernel.x(qubit_0)
    measurement_ = kernel.mz(qubit_0, "measurement_")

    # Check that we can use conditionals on a measurement
    def test_function():
        """Rotate and measure the first qubit."""
        kernel.x(qubit_1)
        kernel.mz(qubit_1)

    # If measurement is true, run the test function.
    kernel.c_if(measurement_, test_function)
    # Apply instructions to each qubit and repeat `c_if`
    # using keyword arguments.
    kernel.x(qreg)
    kernel.c_if(measurement=measurement_, function=test_function)
    kernel()
    assert kernel.arguments == []
    assert kernel.argument_count == 0
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 2 : index
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
# CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
# CHECK:           %[[VAL_5:.*]] = quake.alloca : !quake.qvec<2>
# CHECK:           %[[VAL_6:.*]] = quake.qextract %[[VAL_5]]{{\[}}%[[VAL_4]]] : !quake.qvec<2>[i32] -> !quake.qref
# CHECK:           %[[VAL_7:.*]] = quake.qextract %[[VAL_5]]{{\[}}%[[VAL_3]]] : !quake.qvec<2>[i32] -> !quake.qref
# CHECK:           quake.x (%[[VAL_6]])
# CHECK:           %[[VAL_8:.*]] = quake.mz(%[[VAL_6]] : !quake.qref) {registerName = "measurement_"} : i1
# CHECK:           cc.if(%[[VAL_8]]) {
# CHECK:             quake.x (%[[VAL_7]])
# CHECK:             %[[VAL_9:.*]] = quake.mz(%[[VAL_7]] : !quake.qref) {registerName = ""} : i1
# CHECK:           }
# CHECK:           %[[VAL_10:.*]] = cc.loop while ((%[[VAL_11:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_12:.*]] = arith.cmpi slt, %[[VAL_11]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_12]](%[[VAL_11]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_13:.*]]: index):
# CHECK:             %[[VAL_14:.*]] = quake.qextract %[[VAL_5]]{{\[}}%[[VAL_13]]] : !quake.qvec<2>[index] -> !quake.qref
# CHECK:             quake.x (%[[VAL_14]])
# CHECK:             cc.continue %[[VAL_13]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_15:.*]]: index):
# CHECK:             %[[VAL_16:.*]] = arith.addi %[[VAL_15]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_16]] : index
# CHECK:           } {counted}
# CHECK:           cc.if(%[[VAL_8]]) {
# CHECK:             quake.x (%[[VAL_7]])
# CHECK:             %[[VAL_17:.*]] = quake.mz(%[[VAL_7]] : !quake.qref) {registerName = ""} : i1
# CHECK:           }
# CHECK:           return
# CHECK:         }


def test_kernel_conditional_with_sample():
    """
    Test the conditional measurement functionality of `cudaq.Kernel`
    and assert that it runs as expected on the QPU.
    """
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()

    def then_function():
        kernel.x(qubit)

    kernel.x(qubit)

    # Measure the qubit.
    measurement_ = kernel.mz(qubit)
    # Apply `then_function` to the `kernel` if
    # the qubit was measured in the 1-state.
    kernel.c_if(measurement_, then_function)

    # Measure the qubit again.
    result = cudaq.sample(kernel)
    assert len(result) == 1
    # Qubit should be in the 0-state after undergoing
    # two X rotations.
    assert '0' in result

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.x (%[[VAL_0]])
# CHECK:           %[[VAL_1:.*]] = quake.mz(%[[VAL_0]] : !quake.qref) {registerName = ""} : i1
# CHECK:           cc.if(%[[VAL_1]]) {
# CHECK:             quake.x (%[[VAL_0]])
# CHECK:           }
# CHECK:           return
# CHECK:         }


def test_kernel_apply_call_no_args():
    """
    Tests that we can call a non-parameterized kernel (`other_kernel`), 
    from a :class:`Kernel`.
    """
    other_kernel = cudaq.make_kernel()
    other_qubit = other_kernel.qalloc()
    other_kernel.x(other_qubit)

    kernel = cudaq.make_kernel()
    kernel.apply_call(other_kernel)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() : () -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.x (%[[VAL_0]])
# CHECK:           return
# CHECK:         }


def test_kernel_apply_call_qubit_args():
    """
    Tests that we can call another kernel that's parameterized 
    by a qubit (`other_kernel`), from a :class:`Kernel`.
    """
    other_kernel, other_qubit = cudaq.make_kernel(cudaq.qubit)
    other_kernel.h(other_qubit)

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.apply_call(other_kernel, qubit)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_0]]) : (!quake.qref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:                                                                   %[[VAL_0:.*]]: !quake.qref) {
# CHECK:           quake.h (%[[VAL_0]])
# CHECK:           return
# CHECK:         }


def test_kernel_apply_call_qreg_args():
    """
    Tests that we can call another kernel that's parameterized 
    by a qubit (`other_kernel`), from a :class:`Kernel`.
    """
    other_kernel, other_qreg = cudaq.make_kernel(cudaq.qreg)
    other_kernel.h(other_qreg)

    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(5)
    kernel.apply_call(other_kernel, qreg)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qvec<5>
# CHECK:           %[[VAL_1:.*]] = quake.relax_size %[[VAL_0]] : (!quake.qvec<5>) -> !quake.qvec<?>
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_1]]) : (!quake.qvec<?>) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:                                                                   %[[VAL_0:.*]]: !quake.qvec<?>) {
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_3:.*]] = quake.qvec_size %[[VAL_0]] : (!quake.qvec<?>) -> i64
# CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
# CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_4]] : index
# CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_8:.*]]: index):
# CHECK:             %[[VAL_9:.*]] = quake.qextract %[[VAL_0]]{{\[}}%[[VAL_8]]] : !quake.qvec<?>[index] -> !quake.qref
# CHECK:             quake.h (%[[VAL_9]])
# CHECK:             cc.continue %[[VAL_8]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_10:.*]]: index):
# CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_11]] : index
# CHECK:           } {counted}
# CHECK:           return
# CHECK:         }


def test_kernel_apply_call_float_args():
    """
    Tests that we can call another kernel that's parameterized 
    by a float (`other_kernel`), from a :class:`Kernel`.
    """
    other_kernel, other_float = cudaq.make_kernel(float)
    other_qubit = other_kernel.qalloc()
    other_kernel.rx(other_float, other_qubit)

    kernel, _float = cudaq.make_kernel(float)
    kernel.apply_call(other_kernel, _float)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) {
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_0]]) : (f64) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.rx |%[[VAL_0]] : f64|(%[[VAL_1]])
# CHECK:           return
# CHECK:         }


def test_kernel_apply_call_int_args():
    """
    Tests that we can call another kernel that's parameterized 
    by an int (`other_kernel`), from a :class:`Kernel`.
    """
    other_kernel, other_int = cudaq.make_kernel(int)
    other_qubit = other_kernel.qalloc()
    # TODO:
    # Would like to be able to test kernel operations that
    # can accept an int.

    kernel, _int = cudaq.make_kernel(int)
    kernel.apply_call(other_kernel, _int)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_0]]) : (i32) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           return
# CHECK:         }


def test_kernel_apply_call_list_args():
    """
    Tests that we can call another kernel that's parameterized 
    by a list (`other_kernel`), from a :class:`Kernel`.
    """
    other_kernel, other_list = cudaq.make_kernel(list)
    other_qubit = other_kernel.qalloc()
    other_kernel.rx(other_list[0], other_qubit)

    kernel, _list = cudaq.make_kernel(list)
    kernel.apply_call(other_kernel, _list)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_0]]) : (!cc.stdvec<f64>) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           %[[VAL_2:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !llvm.ptr<f64>
# CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<f64>
# CHECK:           quake.rx |%[[VAL_3]] : f64|(%[[VAL_1]])
# CHECK:           return
# CHECK:         }


@pytest.mark.parametrize("qubit_count", [1, 5])
def test_kernel_control_no_args(qubit_count):
    """
    Tests `Kernel::control()` with another kernel that's
    not parameterized. Test for both a single qubit and a register
    of qubits as the controls.
    """
    other_kernel = cudaq.make_kernel()
    other_qubit = other_kernel.qalloc(qubit_count)
    other_kernel.x(other_qubit)

    kernel = cudaq.make_kernel()
    control_qubit = kernel.qalloc(qubit_count)
    # Call `kernel.control()`.
    kernel.control(target=other_kernel, control=control_qubit)

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_0]] : !quake.qref]  : () -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.x (%[[VAL_0]])
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qvec<5>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_0]] : !quake.qvec<5>]  : () -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 5 : index
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_3:.*]] = quake.alloca : !quake.qvec<5>
# CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_7:.*]]: index):
# CHECK:             %[[VAL_8:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_7]]] : !quake.qvec<5>[index] -> !quake.qref
# CHECK:             quake.x (%[[VAL_8]])
# CHECK:             cc.continue %[[VAL_7]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_9:.*]]: index):
# CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_10]] : index
# CHECK:           } {counted}
# CHECK:           return
# CHECK:         }


@pytest.mark.parametrize("qubit_count", [1, 5])
def test_kernel_control_float_args(qubit_count):
    """
    Tests `Kernel::control()` with another kernel that's
    parameterized by a float. Test for both a single qubit 
    and a register of qubits as the controls.
    """
    other_kernel, other_float = cudaq.make_kernel(float)
    other_qubit = other_kernel.qalloc()
    other_kernel.rx(other_float, other_qubit)

    kernel, float_ = cudaq.make_kernel(float)
    control_qubit = kernel.qalloc(qubit_count)
    # Call `kernel.control()`.
    kernel.control(other_kernel, control_qubit, float_)

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]] : !quake.qref] %[[VAL_0]] : (f64) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.rx |%[[VAL_0]] : f64|(%[[VAL_1]])
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qvec<5>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]] : !quake.qvec<5>] %[[VAL_0]] : (f64) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.rx |%[[VAL_0]] : f64|(%[[VAL_1]])
# CHECK:           return
# CHECK:         }


@pytest.mark.parametrize("qubit_count", [1, 5])
def test_kernel_control_int_args(qubit_count):
    """
    Tests `Kernel::control()` with another kernel that's
    parameterized by an int. Test for both a single qubit 
    and a register of qubits as the controls.
    """
    other_kernel, other_int = cudaq.make_kernel(int)
    other_qubit = other_kernel.qalloc(qubit_count)
    # TODO:
    # Would like to be able to test kernel operations that
    # can accept an int.

    kernel, _int = cudaq.make_kernel(int)
    control_qubit = kernel.qalloc(qubit_count)
    kernel.control(other_kernel, control_qubit, _int)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]] : !quake.qref] %[[VAL_0]] : (i32) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:                                                                   %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qvec<5>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]] : !quake.qvec<5>] %[[VAL_0]] : (i32) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qvec<5>
# CHECK:           return
# CHECK:         }


@pytest.mark.parametrize("qubit_count", [1, 5])
def test_kernel_control_list_args(qubit_count):
    """
    Tests `Kernel::control()` with another kernel that's
    parameterized by a list. Test for both a single qubit 
    and a register of qubits as the controls.
    """
    other_kernel, other_list = cudaq.make_kernel(list)
    other_qubit = other_kernel.qalloc()
    other_kernel.rx(other_list[0], other_qubit)

    kernel, _list = cudaq.make_kernel(list)
    control_qubit = kernel.qalloc(qubit_count)
    # Call `kernel.control()`.
    kernel.control(other_kernel, control_qubit, _list)

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]] : !quake.qref] %[[VAL_0]] : (!cc.stdvec<f64>) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           %[[VAL_2:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !llvm.ptr<f64>
# CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<f64>
# CHECK:           quake.rx |%[[VAL_3]] : f64|(%[[VAL_1]])
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qvec<5>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]] : !quake.qvec<5>] %[[VAL_0]] : (!cc.stdvec<f64>) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           %[[VAL_2:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !llvm.ptr<f64>
# CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<f64>
# CHECK:           quake.rx |%[[VAL_3]] : f64|(%[[VAL_1]])
# CHECK:           return
# CHECK:         }


def test_sample_control_qubit_args():
    """
    Tests `Kernel::control()` with another kernel that's
    parameterized by a `cudaq.qubit`. Uses a single qubit
    as the `control`. Checks for correctness on simulator.
    """
    # `other_kernel` applies an X-gate to the
    # parameterized qubit.
    other_kernel, other_qubit = cudaq.make_kernel(cudaq.qubit)
    other_kernel.x(other_qubit)

    kernel = cudaq.make_kernel()
    # Allocate control and target qubits to `kernel`
    target_qubit = kernel.qalloc()
    control_qubit = kernel.qalloc()
    # Apply `other_kernel` within `kernel` on the `target_qubit`.
    kernel.apply_call(other_kernel, target_qubit)
    kernel.h(control_qubit)
    # Apply `other_kernel` to `kernel` as a controlled operation.
    # `other_kernel` takes `target_qubit` as its argument, while `control_qubit`
    # serves as the control qubit for the operation.
    kernel.control(other_kernel, control_qubit, target_qubit)
    # Apply another hadamard to `control_qubit` and measure.
    kernel.h(control_qubit)
    kernel.mz(control_qubit)

    # Simulate `kernel` and check its expectation value.
    result = cudaq.sample(kernel)
    want_expectation = 0.0
    got_expectation = result.expectation_z()
    assert np.isclose(want_expectation, got_expectation, rtol=1e-12)

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_0]]) : (!quake.qref) -> ()
# CHECK:           quake.h (%[[VAL_1]])
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]] : !quake.qref] %[[VAL_0]] : (!quake.qref) -> ()
# CHECK:           quake.h (%[[VAL_1]])
# CHECK:           %[[VAL_2:.*]] = quake.mz(%[[VAL_1]] : !quake.qref) {registerName = ""} : i1
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.qref) {
# CHECK:           quake.x (%[[VAL_0]])
# CHECK:           return
# CHECK:         }


def test_sample_control_qreg_args():
    """
    Tests `Kernel::control()` with another kernel that's
    parameterized by a `cudaq.qubit`. Uses a register as 
    the `control`. Checks for correctness on the simulator.
    """
    # `other_kernel` applies an X-gate to the
    # parameterized qubit.
    other_kernel, other_qubit = cudaq.make_kernel(cudaq.qubit)
    other_kernel.x(other_qubit)

    kernel = cudaq.make_kernel()
    # Allocating a register to use as the `control`
    # in our call to `kernel.control()`.
    control_register = kernel.qalloc(2)
    target_qubit = kernel.qalloc()
    kernel.x(control_register[0])
    kernel.x(target_qubit)
    # Call `other_kernel` with the argument `target_qubit` from `kernel`.
    # Apply `other_kernel` (with the argument `target_qubit`), as a controlled
    # operation onto this `kernel`.
    kernel.control(other_kernel, control_register, target_qubit)

    # Measure.
    kernel.mz(control_register)
    kernel.mz(target_qubit)

    # Simulate and get results.
    result = cudaq.sample(kernel)
    assert len(result) == 1
    # Should be in the state `101`
    assert '101' in result

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 2 : index
# CHECK:           %[[VAL_1:.*]] = arith.constant 2 : i64
# CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_4:.*]] = arith.constant 0 : i32
# CHECK:           %[[VAL_5:.*]] = quake.alloca : !quake.qvec<2>
# CHECK:           %[[VAL_6:.*]] = quake.alloca : !quake.qref
# CHECK:           %[[VAL_7:.*]] = quake.qextract %[[VAL_5]]{{\[}}%[[VAL_4]]] : !quake.qvec<2>[i32] -> !quake.qref
# CHECK:           quake.x (%[[VAL_7]])
# CHECK:           quake.x (%[[VAL_6]])
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_5]] : !quake.qvec<2>] %[[VAL_6]] : (!quake.qref) -> ()
# CHECK:           %[[VAL_8:.*]] = llvm.alloca %[[VAL_1]] x i1 : (i64) -> !llvm.ptr<i1>
# CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_3]]) -> (index)) {
# CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_12:.*]]: index):
# CHECK:             %[[VAL_13:.*]] = quake.qextract %[[VAL_5]]{{\[}}%[[VAL_12]]] : !quake.qvec<2>[index] -> !quake.qref
# CHECK:             %[[VAL_14:.*]] = quake.mz(%[[VAL_13]] : !quake.qref) : i1
# CHECK:             %[[VAL_15:.*]] = arith.index_cast %[[VAL_12]] : index to i64
# CHECK:             %[[VAL_16:.*]] = llvm.getelementptr %[[VAL_8]]{{\[}}%[[VAL_15]]] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
# CHECK:             llvm.store %[[VAL_14]], %[[VAL_16]] : !llvm.ptr<i1>
# CHECK:             cc.continue %[[VAL_12]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_17:.*]]: index):
# CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_2]] : index
# CHECK:             cc.continue %[[VAL_18]] : index
# CHECK:           } {counted}
# CHECK:           %[[VAL_19:.*]] = quake.mz(%[[VAL_6]] : !quake.qref) {registerName = ""} : i1
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.qref) {
# CHECK:           quake.x (%[[VAL_0]])
# CHECK:           return
# CHECK:         }


def test_sample_apply_call_control():
    """
    More advanced integration that tests a kernel that:
        1. Calls `apply_call()` on another parameterized kernel (`x_kernel`).
        2. Calls `control()` on another parameterized kernel (`h_kernel`).
    """
    # Create an other kernel that applies an X-rotation
    # to a parameterized qubit.
    x_kernel, x_qubit = cudaq.make_kernel(cudaq.qubit)
    x_kernel.x(x_qubit)
    # Create an other kernel that applies a Hadamard to
    # a parameterized qubit.
    h_kernel, h_qubit = cudaq.make_kernel(cudaq.qubit)
    h_kernel.h(h_qubit)

    kernel = cudaq.make_kernel()
    target_qubit = kernel.qalloc()
    control_qubit = kernel.qalloc()
    # Call `x_kernel` from `kernel` with `target_qubit` as its argument.
    kernel.apply_call(x_kernel, target_qubit)
    kernel.h(control_qubit)
    # Apply `h_kernel` to `kernel` as a controlled operation.
    # `h_kernel` takes `target_qubit` as its argument, while `control_qubit`
    # serves as the control qubit for the operation.
    kernel.control(h_kernel, control_qubit, target_qubit)
    kernel.h(control_qubit)
    kernel.mz(control_qubit)

    # Simulate `kernel` and check its expectation value.
    result = cudaq.sample(kernel)
    want_expectation = -1. / np.sqrt(2.)
    got_expectation = result.expectation_z()
    assert np.isclose(want_expectation, got_expectation, rtol=1e-12)

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_0]]) : (!quake.qref) -> ()
# CHECK:           quake.h (%[[VAL_1]])
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]] : !quake.qref] %[[VAL_0]] : (!quake.qref) -> ()
# CHECK:           quake.h (%[[VAL_1]])
# CHECK:           %[[VAL_2:.*]] = quake.mz(%[[VAL_1]] : !quake.qref) {registerName = ""} : i1
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.qref) {
# CHECK:           quake.x (%[[VAL_0]])
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.qref) {
# CHECK:           quake.h (%[[VAL_0]])
# CHECK:           return
# CHECK:         }


def test_kernel_adjoint_no_args():
    """
    Tests the adjoint of a kernel that takes no arguments.
    """
    other_kernel = cudaq.make_kernel()
    other_qubit = other_kernel.qalloc()
    other_kernel.x(other_qubit)

    kernel = cudaq.make_kernel()
    kernel.adjoint(other_kernel)

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           quake.apply<adj> @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}  : () -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.x (%[[VAL_0]])
# CHECK:           return
# CHECK:         }


def test_kernel_adjoint_qubit_args():
    """
    Tests the adjoint of a kernel that takes a qubit as an argument.
    """
    other_kernel, other_qubit = cudaq.make_kernel(cudaq.qubit)
    other_kernel.h(other_qubit)

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.adjoint(other_kernel, qubit)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.apply<adj> @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}} %[[VAL_0]] : (!quake.qref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.qref) {
# CHECK:           quake.h (%[[VAL_0]])
# CHECK:           return
# CHECK:         }


def test_kernel_adjoint_qreg_args():
    """
    Tests the adjoint of a kernel that takes a qreg as an argument.
    """
    other_kernel, other_qreg = cudaq.make_kernel(cudaq.qreg)
    other_kernel.h(other_qreg)

    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(5)
    kernel.adjoint(other_kernel, qreg)
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qvec<5>
# CHECK:           quake.apply<adj> @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}} %[[VAL_0]] : (!quake.qvec<5>) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.qvec<?>) {
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_3:.*]] = quake.qvec_size %[[VAL_0]] : (!quake.qvec<?>) -> i64
# CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
# CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_4]] : index
# CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_8:.*]]: index):
# CHECK:             %[[VAL_9:.*]] = quake.qextract %[[VAL_0]]{{\[}}%[[VAL_8]]] : !quake.qvec<?>[index] -> !quake.qref
# CHECK:             quake.h (%[[VAL_9]])
# CHECK:             cc.continue %[[VAL_8]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_10:.*]]: index):
# CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_11]] : index
# CHECK:           } {counted}
# CHECK:           return
# CHECK:         }


def test_kernel_adjoint_float_args():
    """
    Tests the adjoint of a kernel that is parameterized
    by a float.
    """
    other_kernel, other_value = cudaq.make_kernel(float)
    other_qubit = other_kernel.qalloc()
    other_kernel.x(other_qubit)
    other_kernel.rx(other_value, other_qubit)

    kernel, _float = cudaq.make_kernel(float)
    kernel.adjoint(other_kernel, _float)

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) {
# CHECK:           quake.apply<adj> @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}} %[[VAL_0]] : (f64) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.x (%[[VAL_1]])
# CHECK:           quake.rx |%[[VAL_0]] : f64|(%[[VAL_1]])
# CHECK:           return
# CHECK:         }


def test_kernel_adjoint_int_args():
    """
    Tests the adjoint of a kernel that is parameterized
    by an int.

    Note: we can't currently do anything with `int` kernel
    parameters in `other_kernel`.
    """
    other_kernel, other_value = cudaq.make_kernel(int)
    other_qubit = other_kernel.qalloc()
    other_kernel.x(other_qubit)

    kernel, _int = cudaq.make_kernel(int)
    kernel.adjoint(other_kernel, _int)

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:     %[[VAL_0:.*]]: i32) {
# CHECK:           quake.apply<adj> @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}} %[[VAL_0]] : (i32) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.x (%[[VAL_1]])
# CHECK:           return
# CHECK:         }


def test_kernel_adjoint_list_args():
    """
    Tests the adjoint of a kernel that is parameterized
    by a list.
    """
    other_kernel, other_value = cudaq.make_kernel(list)
    other_qubit = other_kernel.qalloc()
    other_kernel.rx(other_value[0], other_qubit)

    kernel, _list = cudaq.make_kernel(list)
    kernel.adjoint(other_kernel, _list)

    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           quake.apply<adj> @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}} %[[VAL_0]] : (!cc.stdvec<f64>) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca : !quake.qref
# CHECK:           %[[VAL_2:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !llvm.ptr<f64>
# CHECK:           %[[VAL_3:.*]] = llvm.load %[[VAL_2]] : !llvm.ptr<f64>
# CHECK:           quake.rx |%[[VAL_3]] : f64|(%[[VAL_1]])
# CHECK:           return
# CHECK:         }


def test_sample_adjoint_qubit():
    """
    Tests the adjoint of a kernel that is parameterized
    by a qubit. Checks for correctness on simulator.
    """
    other_kernel, other_qubit = cudaq.make_kernel(cudaq.qubit)
    other_kernel.x(other_qubit)

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    kernel.x(qubit)
    # Call the other kernel on `qubit`.
    kernel.apply_call(other_kernel, qubit)
    # Apply adjoint of the other kernel to `qubit`.
    kernel.adjoint(other_kernel, qubit)
    # Measure `qubit`.
    kernel.mz(qubit)

    result = cudaq.sample(kernel)
    assert len(result) == 1
    # Qubit should be in the 1-state.
    assert '1' in result

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = quake.alloca : !quake.qref
# CHECK:           quake.x (%[[VAL_0]])
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_0]]) : (!quake.qref) -> ()
# CHECK:           quake.apply<adj> @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}} %[[VAL_0]] : (!quake.qref) -> ()
# CHECK:           %[[VAL_1:.*]] = quake.mz(%[[VAL_0]] : !quake.qref) {registerName = ""} : i1
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.qref) {
# CHECK:           quake.x (%[[VAL_0]])
# CHECK:           return
# CHECK:         }


def test_sample_adjoint_qreg():
    """
    Tests the adjoint of a kernel that is parameterized
    by a qreg. Checks for correctness on simulator.
    """
    other_kernel, other_qreg = cudaq.make_kernel(cudaq.qreg)
    other_kernel.x(other_qreg)

    kernel, qubit_variable = cudaq.make_kernel(int)
    qreg = kernel.qalloc(qubit_variable)
    kernel.x(qreg)
    # Call the other kernel on `qreg`.
    kernel.apply_call(other_kernel, qreg)
    # Apply adjoint of the other kernel to `qreg`.
    kernel.adjoint(other_kernel, qreg)
    # Measure `qreg`.
    kernel.mz(qreg)

    qubit_count = 5
    result = cudaq.sample(kernel, qubit_count)
    assert len(result) == 1
    # Qubits should be in the 1-state.
    assert '1' * qubit_count in result

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_3:.*]] = quake.alloca(%[[VAL_0]] : i32) : !quake.qvec<?>
# CHECK:           %[[VAL_4:.*]] = quake.qvec_size %[[VAL_3]] : (!quake.qvec<?>) -> i64
# CHECK:           %[[VAL_5:.*]] = arith.index_cast %[[VAL_4]] : i64 to index
# CHECK:           %[[VAL_6:.*]] = cc.loop while ((%[[VAL_7:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_8:.*]] = arith.cmpi slt, %[[VAL_7]], %[[VAL_5]] : index
# CHECK:             cc.condition %[[VAL_8]](%[[VAL_7]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_9:.*]]: index):
# CHECK:             %[[VAL_10:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_9]]] : !quake.qvec<?>[index] -> !quake.qref
# CHECK:             quake.x (%[[VAL_10]])
# CHECK:             cc.continue %[[VAL_9]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_11:.*]]: index):
# CHECK:             %[[VAL_12:.*]] = arith.addi %[[VAL_11]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_12]] : index
# CHECK:           } {counted}
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_3]]) : (!quake.qvec<?>) -> ()
# CHECK:           quake.apply<adj> @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}} %[[VAL_3]] : (!quake.qvec<?>) -> ()
# CHECK:           %[[VAL_13:.*]] = llvm.alloca %[[VAL_4]] x i1 : (i64) -> !llvm.ptr<i1>
# CHECK:           %[[VAL_14:.*]] = cc.loop while ((%[[VAL_15:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_16:.*]] = arith.cmpi slt, %[[VAL_15]], %[[VAL_5]] : index
# CHECK:             cc.condition %[[VAL_16]](%[[VAL_15]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_17:.*]]: index):
# CHECK:             %[[VAL_18:.*]] = quake.qextract %[[VAL_3]]{{\[}}%[[VAL_17]]] : !quake.qvec<?>[index] -> !quake.qref
# CHECK:             %[[VAL_19:.*]] = quake.mz(%[[VAL_18]] : !quake.qref) : i1
# CHECK:             %[[VAL_20:.*]] = arith.index_cast %[[VAL_17]] : index to i64
# CHECK:             %[[VAL_21:.*]] = llvm.getelementptr %[[VAL_13]]{{\[}}%[[VAL_20]]] : (!llvm.ptr<i1>, i64) -> !llvm.ptr<i1>
# CHECK:             llvm.store %[[VAL_19]], %[[VAL_21]] : !llvm.ptr<i1>
# CHECK:             cc.continue %[[VAL_17]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_22:.*]]: index):
# CHECK:             %[[VAL_23:.*]] = arith.addi %[[VAL_22]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_23]] : index
# CHECK:           } {counted}
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:                                                                   %[[VAL_0:.*]]: !quake.qvec<?>) {
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_3:.*]] = quake.qvec_size %[[VAL_0]] : (!quake.qvec<?>) -> i64
# CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_3]] : i64 to index
# CHECK:           %[[VAL_5:.*]] = cc.loop while ((%[[VAL_6:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_7:.*]] = arith.cmpi slt, %[[VAL_6]], %[[VAL_4]] : index
# CHECK:             cc.condition %[[VAL_7]](%[[VAL_6]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_8:.*]]: index):
# CHECK:             %[[VAL_9:.*]] = quake.qextract %[[VAL_0]]{{\[}}%[[VAL_8]]] : !quake.qvec<?>[index] -> !quake.qref
# CHECK:             quake.x (%[[VAL_9]])
# CHECK:             cc.continue %[[VAL_8]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_10:.*]]: index):
# CHECK:             %[[VAL_11:.*]] = arith.addi %[[VAL_10]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_11]] : index
# CHECK:           } {counted}
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
