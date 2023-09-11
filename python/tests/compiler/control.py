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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<1>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_0]]]  : (!quake.veq<1>) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_1:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_2:.*]] = quake.alloca !quake.veq<1>
# CHECK:           %[[VAL_3:.*]] = cc.loop while ((%[[VAL_4:.*]] = %[[VAL_1]]) -> (index)) {
# CHECK:             %[[VAL_5:.*]] = arith.cmpi slt, %[[VAL_4]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_5]](%[[VAL_4]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_6:.*]]: index):
# CHECK:             %[[VAL_7:.*]] = quake.extract_ref %[[VAL_2]]{{\[}}%[[VAL_6]]] : (!quake.veq<1>, index) -> !quake.ref
# CHECK:             quake.x %[[VAL_7]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_6]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_8:.*]]: index):
# CHECK:             %[[VAL_9:.*]] = arith.addi %[[VAL_8]], %[[VAL_0]] : index
# CHECK:             cc.continue %[[VAL_9]] : index
# CHECK:           } {invariant}
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.veq<5>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_0]]] : (!quake.veq<5>) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() {
# CHECK:           %[[VAL_0:.*]] = arith.constant 5 : index
# CHECK:           %[[VAL_1:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_3:.*]] = quake.alloca !quake.veq<5>
# CHECK:           %[[VAL_4:.*]] = cc.loop while ((%[[VAL_5:.*]] = %[[VAL_2]]) -> (index)) {
# CHECK:             %[[VAL_6:.*]] = arith.cmpi slt, %[[VAL_5]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_6]](%[[VAL_5]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_7:.*]]: index):
# CHECK:             %[[VAL_8:.*]] = quake.extract_ref %[[VAL_3]][%[[VAL_7]]] : (!quake.veq<5>, index) -> !quake.ref
# CHECK:             quake.x %[[VAL_8]] : (!quake.ref) -> ()
# CHECK:             cc.continue %[[VAL_7]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_9:.*]]: index):
# CHECK:             %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[VAL_1]] : index
# CHECK:             cc.continue %[[VAL_10]] : index
# CHECK:           } {invariant}
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
# CHECK-SAME:      %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<1>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]]] %[[VAL_0]] : (!quake.veq<1>, f64) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64)  {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           quake.rx (%[[VAL_0]]) %[[VAL_1]] : (f64, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<5>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]]] %[[VAL_0]] : (!quake.veq<5>, f64) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: f64) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           quake.rx (%[[VAL_0]]) %[[VAL_1]] : (f64, !quake.ref) -> ()
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
# CHECK-SAME:      %[[VAL_0:.*]]: i32) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<1>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]]] %[[VAL_0]] : (!quake.veq<1>, i32) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<1>
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<5>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]]] %[[VAL_0]] : (!quake.veq<5>, i32) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: i32) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<5>
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
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<1>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]]] %[[VAL_0]] : (!quake.veq<1>, !cc.stdvec<f64>) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_2:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][0] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
# CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_4]] : !cc.ptr<f64>
# CHECK:           quake.rx (%[[VAL_3]]) %[[VAL_1]] : (f64, !quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.veq<5>
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]]] %[[VAL_0]] : (!quake.veq<5>, !cc.stdvec<f64>) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !cc.stdvec<f64>) {
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_2:.*]] = cc.stdvec_data %[[VAL_0]] : (!cc.stdvec<f64>) -> !cc.ptr<!cc.array<f64 x ?>>
# CHECK:           %[[VAL_4:.*]] = cc.compute_ptr %[[VAL_2]][0] : (!cc.ptr<!cc.array<f64 x ?>>) -> !cc.ptr<f64>
# CHECK:           %[[VAL_3:.*]] = cc.load %[[VAL_4]] : !cc.ptr<f64>
# CHECK:           quake.rx (%[[VAL_3]]) %[[VAL_1]] : (f64, !quake.ref) -> ()
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
    assert np.isclose(want_expectation, got_expectation, atol=1e-1)

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_0]]) : (!quake.ref) -> ()
# CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*\[}}%[[VAL_1]]] %[[VAL_0]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_1]] name "" : (!quake.ref) -> i1
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.ref) {
# CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
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


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = arith.constant 2 : index
# CHECK:           %[[VAL_2:.*]] = arith.constant 1 : index
# CHECK:           %[[VAL_3:.*]] = arith.constant 0 : index
# CHECK:           %[[VAL_5:.*]] = quake.alloca !quake.veq<2>
# CHECK:           %[[VAL_6:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_7:.*]] = quake.extract_ref %[[VAL_5]][0] : (!quake.veq<2>) -> !quake.ref
# CHECK:           quake.x %[[VAL_7]] : (!quake.ref) -> ()
# CHECK:           quake.x %[[VAL_6]] : (!quake.ref) -> ()
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_5]]] %[[VAL_6]] : (!quake.veq<2>, !quake.ref) -> ()
# CHECK:           %[[VAL_8:.*]] = cc.alloca !cc.array<i1 x 2>
# CHECK:           %[[VAL_9:.*]] = cc.loop while ((%[[VAL_10:.*]] = %[[VAL_3]]) -> (index)) {
# CHECK:             %[[VAL_11:.*]] = arith.cmpi slt, %[[VAL_10]], %[[VAL_0]] : index
# CHECK:             cc.condition %[[VAL_11]](%[[VAL_10]] : index)
# CHECK:           } do {
# CHECK:           ^bb0(%[[VAL_12:.*]]: index):
# CHECK:             %[[VAL_13:.*]] = quake.extract_ref %[[VAL_5]][%[[VAL_12]]] : (!quake.veq<2>, index) -> !quake.ref
# CHECK:             %[[VAL_14:.*]] = quake.mz %[[VAL_13]] : (!quake.ref) -> i1
# CHECK:             %[[VAL_15:.*]] = arith.index_cast %[[VAL_12]] : index to i64
# CHECK:             %[[VAL_16:.*]] = cc.compute_ptr %[[VAL_8]][%[[VAL_15]]] : (!cc.ptr<!cc.array<i1 x 2>>, i64) -> !cc.ptr<i1>
# CHECK:             cc.store %[[VAL_14]], %[[VAL_16]] : !cc.ptr<i1>
# CHECK:             cc.continue %[[VAL_12]] : index
# CHECK:           } step {
# CHECK:           ^bb0(%[[VAL_17:.*]]: index):
# CHECK:             %[[VAL_18:.*]] = arith.addi %[[VAL_17]], %[[VAL_2]] : index
# CHECK:             cc.continue %[[VAL_18]] : index
# CHECK:           } {invariant}
# CHECK:           %[[VAL_19:.*]] = quake.mz %[[VAL_6]] name "" : (!quake.ref) -> i1
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.ref) {
# CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
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
    assert np.isclose(want_expectation, got_expectation, atol=1e-1)

    # Check the MLIR.
    print(kernel)


# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}() attributes {"cudaq-entrypoint"} {
# CHECK:           %[[VAL_0:.*]] = quake.alloca !quake.ref
# CHECK:           %[[VAL_1:.*]] = quake.alloca !quake.ref
# CHECK:           call @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(%[[VAL_0]]) : (!quake.ref) -> ()
# CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           quake.apply @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}{{\[}}%[[VAL_1]]] %[[VAL_0]] : (!quake.ref, !quake.ref) -> ()
# CHECK:           quake.h %[[VAL_1]] : (!quake.ref) -> ()
# CHECK:           %[[VAL_2:.*]] = quake.mz %[[VAL_1]] name "" : (!quake.ref) -> i1
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.ref) {
# CHECK:           quake.x %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# CHECK-LABEL:   func.func @__nvqpp__mlirgen____nvqppBuilderKernel_{{.*}}(
# CHECK-SAME:      %[[VAL_0:.*]]: !quake.ref) {
# CHECK:           quake.h %[[VAL_0]] : (!quake.ref) -> ()
# CHECK:           return
# CHECK:         }

# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
