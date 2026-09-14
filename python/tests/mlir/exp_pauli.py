# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s
# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck --check-prefix=CTRL %s

import numpy as np
import cudaq


def test_exp_pauli():

    @cudaq.kernel
    def kernel_initial_state(angles: list[float]):
        qreg = cudaq.qvector(len(angles))
        for i in range(len(angles)):
            rx(angles[i], qreg[i])

    @cudaq.kernel
    def U_exp_pauli(qubits: cudaq.qview):
        exp_pauli(23.1, qubits, 'XIY')

    @cudaq.kernel
    def kernel_ancilla_exp_pauli(angles: list[float]):
        ancilla = cudaq.qubit()
        qreg = cudaq.qvector(len(angles))
        for i in range(len(angles)):
            rx(angles[i], qreg[i])
        h(ancilla)
        cudaq.control(U_exp_pauli, ancilla, qreg)

    @cudaq.kernel
    def rotate_y(qubit: cudaq.qview):
        ry(0.88, qubit)

    @cudaq.kernel
    def kernel_ancilla_rotation(angles: list[float]):
        ancilla = cudaq.qubit()
        qreg = cudaq.qvector(len(angles))
        for i in range(len(angles)):
            rx(angles[i], qreg[i])
        h(ancilla)
        cudaq.control(rotate_y, ancilla, qreg)

    @cudaq.kernel
    def kernel_noancilla_rotation(angles: list[float]):
        qreg = cudaq.qvector(len(angles))
        for i in range(len(angles)):
            rx(angles[i], qreg[i])
        rotate_y(qreg)

    cudaq.set_target('qpp-cpu')
    angles = [0.34, 1.2, 1.6]

    # create the initial state (using the initial state)
    initial = np.array(cudaq.get_state(kernel_initial_state, angles))

    # create the initial state + ancilla, hadamard, then perform a
    # controlled rotation on the |1> subspace of the ancilla
    full = np.array(cudaq.get_state(kernel_ancilla_rotation, angles))

    # create the initial state and perform a rotation (for comparison with full)
    rotation = np.array(cudaq.get_state(kernel_noancilla_rotation, angles))

    # create the initial state + ancilla, hadamard, then perform a
    # controlled exp_pauli on the |1> subspace of the ancilla
    epauli = np.array(cudaq.get_state(kernel_ancilla_exp_pauli, angles))

    print(cudaq.translate(kernel_ancilla_exp_pauli, angles, format='qir'))


# CHECK-LABEL: define void @__nvqpp__mlirgen__kernel_ancilla_exp_pauli..0x
# CHECK:         %[[VAL_0:.*]] = alloca [1 x { ptr, i64 }], align 8
# CHECK:         %[[VAL_1:.*]] = alloca [3 x double], align 8
# CHECK:         store double 3.400000e-01, ptr %[[VAL_1]], align 8
# CHECK:         %[[VAL_2:.*]] = getelementptr [3 x double], ptr %[[VAL_1]], i32 0, i32 1
# CHECK:         store double 1.200000e+00, ptr %[[VAL_2]], align 8
# CHECK:         %[[VAL_3:.*]] = getelementptr [3 x double], ptr %[[VAL_1]], i32 0, i32 2
# CHECK:         store double 1.600000e+00, ptr %[[VAL_3]], align 8
# CHECK:         %[[VAL_4:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_5:.*]] = call ptr @__quantum__rt__qubit_allocate_array(i64 3)
# CHECK:         br label %[[VAL_6:.*]]
# CHECK:       9:                                                ; preds = %[[VAL_7:.*]], %[[VAL_8:.*]]
# CHECK:         %[[VAL_9:.*]] = phi i64 [ %[[VAL_10:.*]], %[[VAL_7]] ], [ 0, %[[VAL_8]] ]
# CHECK:         %[[VAL_11:.*]] = icmp slt i64 %[[VAL_9]], 3
# CHECK:         br i1 %[[VAL_11]], label %[[VAL_7]], label %[[VAL_12:.*]]
# CHECK:       12:                                               ; preds = %[[VAL_6]]
# CHECK:         %[[VAL_13:.*]] = getelementptr [3 x double], ptr %[[VAL_1]], i32 0, i64 %[[VAL_9]]
# CHECK:         %[[VAL_14:.*]] = load double, ptr %[[VAL_13]], align 8
# CHECK:         %[[VAL_15:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_5]], i64 %[[VAL_9]])
# CHECK:         %[[VAL_16:.*]] = load ptr, ptr %[[VAL_15]], align 8
# CHECK:         call void @__quantum__qis__rx(double %[[VAL_14]], ptr %[[VAL_16]])
# CHECK:         %[[VAL_10]] = add i64 %[[VAL_9]], 1
# CHECK:         br label %[[VAL_6]]
# CHECK:       18:                                               ; preds = %[[VAL_6]]
# CHECK:         call void @__quantum__qis__h(ptr %[[VAL_4]])
# CHECK:         %[[VAL_17:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_18:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_17]], i64 0)
# CHECK:         store ptr %[[VAL_4]], ptr %[[VAL_18]], align 8
# CHECK:         store ptr @cstr.58495900, ptr %[[VAL_0]], align 8
# CHECK:         %[[VAL_19:.*]] = getelementptr [1 x { ptr, i64 }], ptr %[[VAL_0]], i32 0, i32 0, i32 1
# CHECK:         store i64 3, ptr %[[VAL_19]], align 8
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double 2.310000e+01, ptr %[[VAL_17]], ptr %[[VAL_5]], ptr %[[VAL_0]])
# CHECK-DAG:     call void @__quantum__rt__qubit_release(ptr %[[VAL_4]])
# CHECK-DAG:     call void @__quantum__rt__qubit_release_array(ptr %[[VAL_5]])
# CHECK:         ret void


def test_exp_pauli_loop_controlled():
    """Regression test for issue #2822: cudaq.control on a kernel that calls
    exp_pauli in a loop must compile without error."""

    @cudaq.kernel
    def exp_pauli_loop(qubits: cudaq.qview, coefficients: list[float],
                       words: list[cudaq.pauli_word], time: float):
        for i in range(len(coefficients)):
            exp_pauli(coefficients[i] * time, qubits, words[i])

    @cudaq.kernel
    def kernel_controlled_exp_pauli_loop(coefficients: list[float],
                                         words: list[cudaq.pauli_word]):
        ctrl = cudaq.qubit()
        qreg = cudaq.qvector(2)
        h(ctrl)
        cudaq.control(exp_pauli_loop, ctrl, qreg, coefficients, words, 1.0)

    cudaq.set_target('qpp-cpu')
    coefficients = [1.0, 0.5]
    words = [cudaq.pauli_word("ZZ"), cudaq.pauli_word("XX")]

    state = np.array(
        cudaq.get_state(kernel_controlled_exp_pauli_loop, coefficients, words))
    assert len(state) > 0

    # FileCheck below verifies the QIR contains the loop structure
    # (phi/icmp/br blocks) and calls __quantum__qis__exp_pauli__ctl per iteration.
    print(
        cudaq.translate(kernel_controlled_exp_pauli_loop,
                        coefficients,
                        words,
                        format='qir'))


# CTRL-LABEL: define void @__nvqpp__mlirgen__kernel_controlled_exp_pauli_loop..0x
# CTRL:         %[[VAL_0:.*]] = alloca [1 x { ptr, i64 }], align 8
# CTRL:         %[[VAL_1:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CTRL:         %[[VAL_2:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CTRL:         %[[VAL_3:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CTRL:         %[[VAL_4:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CTRL:         %[[VAL_5:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_4]], i64 0)
# CTRL:         store ptr %[[VAL_1]], ptr %[[VAL_5]], align 8
# CTRL:         %[[VAL_6:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CTRL:         %[[VAL_7:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_6]], i64 0)
# CTRL:         store ptr %[[VAL_2]], ptr %[[VAL_7]], align 8
# CTRL:         %[[VAL_8:.*]] = call ptr @__quantum__rt__array_concatenate(ptr %[[VAL_4]], ptr %[[VAL_6]])
# CTRL:         %[[VAL_9:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CTRL:         %[[VAL_10:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_9]], i64 0)
# CTRL:         store ptr %[[VAL_3]], ptr %[[VAL_10]], align 8
# CTRL:         %[[VAL_11:.*]] = call ptr @__quantum__rt__array_concatenate(ptr %[[VAL_8]], ptr %[[VAL_9]])
# CTRL:         %[[VAL_12:.*]] = alloca [2 x { ptr, i64 }], align 8
# CTRL:         store { ptr, i64 } { ptr @cstr.5A5A00, i64 3 }, ptr %[[VAL_12]], align 8
# CTRL:         %[[VAL_13:.*]] = getelementptr [2 x { ptr, i64 }], ptr %[[VAL_12]], i32 0, i32 1
# CTRL:         store { ptr, i64 } { ptr @cstr.585800, i64 3 }, ptr %[[VAL_13]], align 8
# CTRL:         %[[VAL_14:.*]] = alloca [2 x double], align 8
# CTRL:         store double 1.000000e+00, ptr %[[VAL_14]], align 8
# CTRL:         %[[VAL_15:.*]] = getelementptr [2 x double], ptr %[[VAL_14]], i32 0, i32 1
# CTRL:         store double 5.000000e-01, ptr %[[VAL_15]], align 8
# CTRL:         %[[VAL_16:.*]] = call ptr @__quantum__rt__array_slice(ptr %[[VAL_11]], i32 1, i64 1, i64 1, i64 2)
# CTRL:         call void @__quantum__qis__h(ptr %[[VAL_1]])
# CTRL:         %[[VAL_17:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CTRL:         %[[VAL_18:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_17]], i64 0)
# CTRL:         store ptr %[[VAL_1]], ptr %[[VAL_18]], align 8
# CTRL:         br label %[[VAL_19:.*]]
# CTRL:       23:                                               ; preds = %[[VAL_20:.*]], %[[VAL_21:.*]]
# CTRL:         %[[VAL_22:.*]] = phi i64 [ %[[VAL_23:.*]], %[[VAL_20]] ], [ 0, %[[VAL_21]] ]
# CTRL:         %[[VAL_24:.*]] = icmp slt i64 %[[VAL_22]], 2
# CTRL:         br i1 %[[VAL_24]], label %[[VAL_20]], label %[[VAL_25:.*]]
# CTRL:       26:                                               ; preds = %[[VAL_19]]
# CTRL:         %[[VAL_26:.*]] = getelementptr [2 x double], ptr %[[VAL_14]], i32 0, i64 %[[VAL_22]]
# CTRL:         %[[VAL_27:.*]] = load double, ptr %[[VAL_26]], align 8
# CTRL:         %[[VAL_28:.*]] = getelementptr [2 x { ptr, i64 }], ptr %[[VAL_12]], i32 0, i64 %[[VAL_22]]
# CTRL:         %[[VAL_29:.*]] = load { ptr, i64 }, ptr %[[VAL_28]], align 8
# CTRL:         store { ptr, i64 } %[[VAL_29]], ptr %[[VAL_0]], align 8
# CTRL:         call void @__quantum__qis__exp_pauli__ctl(double %[[VAL_27]], ptr %[[VAL_17]], ptr %[[VAL_16]], ptr %[[VAL_0]])
# CTRL:         %[[VAL_23]] = add i64 %[[VAL_22]], 1
# CTRL:         br label %[[VAL_19]]
# CTRL:       32:                                               ; preds = %[[VAL_19]]
# CTRL-DAG:     call void @__quantum__rt__qubit_release(ptr %[[VAL_1]])
# CTRL-DAG:     call void @__quantum__rt__qubit_release(ptr %[[VAL_2]])
# CTRL-DAG:     call void @__quantum__rt__qubit_release(ptr %[[VAL_3]])
# CTRL:         ret void
