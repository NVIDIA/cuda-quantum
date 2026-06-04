# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ pytest -rP  %s | FileCheck %s

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


# CHECK-LABEL: define void @__nvqpp__mlirgen__kernel_ancilla_exp_pauli..
# CHECK-SAME:  ({ ptr, i64 } %[[VAL_0:.*]], { ptr, ptr } %[[VAL_1:.*]]) {
# CHECK:         %[[VAL_2:.*]] = alloca [1 x { ptr, i64 }]
# CHECK:         %[[VAL_3:.*]] = alloca [3 x double]
# CHECK:         store double 3.400000e-01, ptr %[[VAL_3]]
# CHECK:         %[[VAL_4:.*]] = getelementptr [3 x double], ptr %[[VAL_3]], i32 0, i32 1
# CHECK:         store double 1.200000e+00, ptr %[[VAL_4]]
# CHECK:         %[[VAL_5:.*]] = getelementptr [3 x double], ptr %[[VAL_3]], i32 0, i32 2
# CHECK:         store double 1.600000e+00, ptr %[[VAL_5]]
# CHECK:         %[[VAL_6:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_7:.*]] = call ptr @__quantum__rt__qubit_allocate_array(i64 3)
# CHECK:         br label %[[VAL_8:.*]]
# CHECK:                                                    ; preds = %[[VAL_9:.*]], %[[VAL_10:.*]]
# CHECK:         %[[VAL_11:.*]] = phi i64 [ %[[VAL_12:.*]], %[[VAL_9]] ], [ 0, %[[VAL_10]] ]
# CHECK:         %[[VAL_13:.*]] = phi ptr [ %[[VAL_13]], %[[VAL_9]] ], [ %[[VAL_6]], %[[VAL_10]] ]
# CHECK:         %[[VAL_14:.*]] = icmp slt i64 %[[VAL_11]], 3
# CHECK:         br i1 %[[VAL_14]], label %[[VAL_9]], label %[[VAL_15:.*]]
# CHECK:                                                    ; preds = %[[VAL_8]]
# CHECK:         %[[VAL_16:.*]] = getelementptr [3 x double], ptr %[[VAL_3]], i32 0, i64 %[[VAL_11]]
# CHECK:         %[[VAL_17:.*]] = load double, ptr %[[VAL_16]]
# CHECK:         %[[VAL_18:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_7]], i64 %[[VAL_11]])
# CHECK:         %[[VAL_19:.*]] = load ptr, ptr %[[VAL_18]]
# CHECK:         call void @__quantum__qis__rx(double %[[VAL_17]], ptr %[[VAL_19]])
# CHECK:         %[[VAL_12]] = add i64 %[[VAL_11]], 1
# CHECK:         br label %[[VAL_8]]
# CHECK:                                                   ; preds = %[[VAL_8]]
# CHECK:         call void @__quantum__qis__h(ptr %[[VAL_13]])
# CHECK:         %[[VAL_20:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_21:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_20]], i64 0)
# CHECK:         store ptr %[[VAL_6]], ptr %[[VAL_21]]
# CHECK:         store ptr @cstr.58495900, ptr %[[VAL_2]]
# CHECK:         %[[VAL_22:.*]] = getelementptr [1 x { ptr, i64 }], ptr %[[VAL_2]], i32 0, i32 0, i32 1
# CHECK:         store i64 3, ptr %[[VAL_22]]
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double 2.310000e+01, ptr %[[VAL_20]], ptr %[[VAL_7]], ptr %[[VAL_2]])
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_6]])
# CHECK:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_7]])
# CHECK:         ret void
# CHECK:       }


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


# CHECK-LABEL: define void @__nvqpp__mlirgen__kernel_controlled_exp_pauli_loop..
# CHECK-SAME:   ({ ptr, i64 } %[[VAL_0:.*]], { ptr, i64 } %[[VAL_1:.*]], { ptr, ptr } %[[VAL_2:.*]]) {
# CHECK:         %[[VAL_3:.*]] = alloca [1 x { ptr, i64 }]
# CHECK:         %[[VAL_4:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_5:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_6:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_7:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_8:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_7]], i64 0)
# CHECK:         store ptr %[[VAL_4]], ptr %[[VAL_8]]
# CHECK:         %[[VAL_9:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_10:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_9]], i64 0)
# CHECK:         store ptr %[[VAL_5]], ptr %[[VAL_10]]
# CHECK:         %[[VAL_11:.*]] = call ptr @__quantum__rt__array_concatenate(ptr %[[VAL_7]], ptr %[[VAL_9]])
# CHECK:         %[[VAL_12:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_13:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_12]], i64 0)
# CHECK:         store ptr %[[VAL_6]], ptr %[[VAL_13]]
# CHECK:         %[[VAL_14:.*]] = call ptr @__quantum__rt__array_concatenate(ptr %[[VAL_11]], ptr %[[VAL_12]])
# CHECK:         %[[VAL_15:.*]] = alloca [2 x { ptr, i64 }]
# CHECK:         store { ptr, i64 } { ptr @cstr.5A5A00, i64 3 }, ptr %[[VAL_15]]
# CHECK:         %[[VAL_16:.*]] = getelementptr [2 x { ptr, i64 }], ptr %[[VAL_15]], i32 0, i32 1
# CHECK:         store { ptr, i64 } { ptr @cstr.585800, i64 3 }, ptr %[[VAL_16]]
# CHECK:         %[[VAL_17:.*]] = alloca [2 x double]
# CHECK:         store double 1.000000e+00, ptr %[[VAL_17]]
# CHECK:         %[[VAL_18:.*]] = getelementptr [2 x double], ptr %[[VAL_17]], i32 0, i32 1
# CHECK:         store double 5.000000e-01, ptr %[[VAL_18]]
# CHECK:         %[[VAL_19:.*]] = call ptr @__quantum__rt__array_slice(ptr %[[VAL_14]], i32 1, i64 1, i64 1, i64 2)
# CHECK:         call void @__quantum__qis__h(ptr %[[VAL_4]])
# CHECK:         %[[VAL_20:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_21:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_20]], i64 0)
# CHECK:         store ptr %[[VAL_4]], ptr %[[VAL_21]]
# CHECK:         br label %[[VAL_22:.*]]
# CHECK:                              ; preds = %[[VAL_23:.*]], %[[VAL_24:.*]]
# CHECK:         %[[VAL_25:.*]] = phi i64 [ %[[VAL_26:.*]], %[[VAL_23]] ], [ 0, %[[VAL_24]] ]
# CHECK:         %[[VAL_27:.*]] = phi ptr [ %[[VAL_27]], %[[VAL_23]] ], [ %[[VAL_4]], %[[VAL_24]] ]
# CHECK:         %[[VAL_28:.*]] = phi ptr [ %[[VAL_28]], %[[VAL_23]] ], [ %[[VAL_5]], %[[VAL_24]] ]
# CHECK:         %[[VAL_29:.*]] = phi ptr [ %[[VAL_29]], %[[VAL_23]] ], [ %[[VAL_6]], %[[VAL_24]] ]
# CHECK:         %[[VAL_30:.*]] = icmp slt i64 %[[VAL_25]], 2
# CHECK:         br i1 %[[VAL_30]], label %[[VAL_23]], label %[[VAL_31:.*]]
# CHECK:                          ; preds = %[[VAL_22]]
# CHECK:         %[[VAL_32:.*]] = getelementptr [2 x double], ptr %[[VAL_17]], i32 0, i64 %[[VAL_25]]
# CHECK:         %[[VAL_33:.*]] = load double, ptr %[[VAL_32]]
# CHECK:         %[[VAL_34:.*]] = getelementptr [2 x { ptr, i64 }], ptr %[[VAL_15]], i32 0, i64 %[[VAL_25]]
# CHECK:         %[[VAL_35:.*]] = load { ptr, i64 }, ptr %[[VAL_34]]
# CHECK:         store { ptr, i64 } %[[VAL_35]], ptr %[[VAL_3]]
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double %[[VAL_33]], ptr %[[VAL_20]], ptr %[[VAL_19]], ptr %[[VAL_3]])
# CHECK:         %[[VAL_26]] = add i64 %[[VAL_25]], 1
# CHECK:         br label %[[VAL_22]]
# CHECK:                                    ; preds = %[[VAL_22]]
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_27]])
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_28]])
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_29]])
# CHECK:         ret void
# CHECK:       }
