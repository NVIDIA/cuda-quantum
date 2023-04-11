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