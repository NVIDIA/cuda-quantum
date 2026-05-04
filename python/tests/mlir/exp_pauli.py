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
# CHECK-SAME: ({ double*, i64 }
# CHECK-SAME: %[[VAL_0:.*]], { i8*, i8* }
# CHECK-SAME: %[[VAL_1:.*]]) {
# CHECK:         %[[VAL_2:.*]] = alloca [1 x { i8*, i64 }], align 8
# CHECK:         %[[VAL_3:.*]] = alloca [3 x double], align 8
# CHECK:         %[[VAL_4:.*]] = bitcast [3 x double]* %[[VAL_3]] to double*
# CHECK:         store double 3.400000e-01, double* %[[VAL_4]], align 8
# CHECK:         %[[VAL_5:.*]] = getelementptr [3 x double], [3 x double]* %[[VAL_3]], i32 0, i32 1
# CHECK:         store double 1.200000e+00, double* %[[VAL_5]], align 8
# CHECK:         %[[VAL_6:.*]] = getelementptr [3 x double], [3 x double]* %[[VAL_3]], i32 0, i32 2
# CHECK:         store double 1.600000e+00, double* %[[VAL_6]], align 8
# CHECK:         %[[VAL_7:.*]] = call %[[VAL_8:.*]]* @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_9:.*]] = call %[[VAL_10:.*]]* @__quantum__rt__qubit_allocate_array(i64 3)
# CHECK:         br label %[[VAL_11:.*]]
# CHECK:         %[[VAL_12:.*]] = phi i64 [ %[[VAL_13:.*]], %[[VAL_14:.*]] ], [ 0, %[[VAL_15:.*]] ]
# CHECK:         %[[VAL_16:.*]] = icmp slt i64 %[[VAL_12]], 3
# CHECK:         br i1 %[[VAL_16]], label %[[VAL_14]], label %[[VAL_17:.*]]
# CHECK:         %[[VAL_18:.*]] = phi i64 [ %[[VAL_12]], %[[VAL_11]] ]
# CHECK:         %[[VAL_19:.*]] = getelementptr [3 x double], [3 x double]* %[[VAL_3]], i32 0, i64 %[[VAL_18]]
# CHECK:         %[[VAL_20:.*]] = load double, double* %[[VAL_19]], align 8
# CHECK:         %[[VAL_21:.*]] = call %[[VAL_8]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_10]]* %[[VAL_9]], i64 %[[VAL_18]])
# CHECK:         %[[VAL_22:.*]] = load %[[VAL_8]]*, %[[VAL_8]]** %[[VAL_21]], align 8
# CHECK:         call void @__quantum__qis__rx(double %[[VAL_20]], %[[VAL_8]]* %[[VAL_22]])
# CHECK:         %[[VAL_13]] = add i64 %[[VAL_18]], 1
# CHECK:         br label %[[VAL_11]]
# CHECK:         call void @__quantum__qis__h(%[[VAL_8]]* %[[VAL_7]])
# CHECK:         %[[VAL_23:.*]] = call %[[VAL_10]]* @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_24:.*]] = call %[[VAL_8]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_10]]* %[[VAL_23]], i64 0)
# CHECK:         store %[[VAL_8]]* %[[VAL_7]], %[[VAL_8]]** %[[VAL_24]], align 8
# CHECK:         %[[VAL_25:.*]] = bitcast [1 x { i8*, i64 }]* %[[VAL_2]] to i8**
# CHECK:         store i8* getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.58495900, i32 0, i32 0), i8** %[[VAL_25]], align 8
# CHECK:         %[[VAL_26:.*]] = getelementptr [1 x { i8*, i64 }], [1 x { i8*, i64 }]* %[[VAL_2]], i32 0, i32 0, i32 1
# CHECK:         store i64 3, i64* %[[VAL_26]], align 8
# CHECK:         %[[VAL_27:.*]] = bitcast [1 x { i8*, i64 }]* %[[VAL_2]] to i8*
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double 2.310000e+01, %[[VAL_10]]* %[[VAL_23]], %[[VAL_10]]* %[[VAL_9]], i8* %[[VAL_27]])
# CHECK:         ret void

# CHECK-LABEL: define void @__nvqpp__mlirgen__U_exp_pauli..
# CHECK-SAME:    %[[VAL_0:.*]]* %[[VAL_1:.*]]) {
# CHECK:         %[[VAL_2:.*]] = alloca [1 x { i8*, i64 }], align 8
# CHECK:         %[[VAL_3:.*]] = bitcast [1 x { i8*, i64 }]* %[[VAL_2]] to i8**
# CHECK:         store i8* getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.58495900, i32 0, i32 0), i8** %[[VAL_3]], align 8
# CHECK:         %[[VAL_4:.*]] = getelementptr [1 x { i8*, i64 }], [1 x { i8*, i64 }]* %[[VAL_2]], i32 0, i32 0, i32 1
# CHECK:         store i64 3, i64* %[[VAL_4]], align 8
# CHECK:         %[[VAL_5:.*]] = bitcast [1 x { i8*, i64 }]* %[[VAL_2]] to i8*
# CHECK:         call void @__quantum__qis__exp_pauli(double 2.310000e+01, %[[VAL_0]]* %[[VAL_1]], i8* %[[VAL_5]])
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
# CHECK:         %[[VAL_0:.*]] = alloca [1 x { i8*, i64 }], align 8
# CHECK:         %[[VAL_1:.*]] = call %[[VAL_2:.*]]* @__quantum__rt__qubit_allocate_array(i64 3)
# CHECK:         %[[VAL_3:.*]] = alloca [2 x { i8*, i64 }], align 8
# CHECK:         %[[VAL_4:.*]] = bitcast [2 x { i8*, i64 }]* %[[VAL_3]] to { i8*, i64 }*
# CHECK:         store { i8*, i64 } { i8* getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.5A5A00, i32 0, i32 0), i64 3 }, { i8*, i64 }* %[[VAL_4]], align 8
# CHECK:         %[[VAL_5:.*]] = getelementptr [2 x { i8*, i64 }], [2 x { i8*, i64 }]* %[[VAL_3]], i32 0, i32 1
# CHECK:         store { i8*, i64 } { i8* getelementptr inbounds ([3 x i8], [3 x i8]* @cstr.585800, i32 0, i32 0), i64 3 }, { i8*, i64 }* %[[VAL_5]], align 8
# CHECK:         %[[VAL_6:.*]] = alloca [2 x double], align 8
# CHECK:         %[[VAL_7:.*]] = bitcast [2 x double]* %[[VAL_6]] to double*
# CHECK:         store double 1.000000e+00, double* %[[VAL_7]], align 8
# CHECK:         %[[VAL_8:.*]] = getelementptr [2 x double], [2 x double]* %[[VAL_6]], i32 0, i32 1
# CHECK:         store double 5.000000e-01, double* %[[VAL_8]], align 8
# CHECK:         %[[VAL_9:.*]] = call %[[VAL_10:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_1]], i64 0)
# CHECK:         %[[VAL_11:.*]] = load %[[VAL_10]]*, %[[VAL_10]]** %[[VAL_9]], align 8
# CHECK:         %[[VAL_12:.*]] = call %[[VAL_2]]* @__quantum__rt__array_slice(%[[VAL_2]]* %[[VAL_1]], i32 1, i64 1, i64 1, i64 2)
# CHECK:         call void @__quantum__qis__h(%[[VAL_10]]* %[[VAL_11]])
# CHECK:         %[[VAL_13:.*]] = call %[[VAL_2]]* @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_14:.*]] = call %[[VAL_10:.*]]** @__quantum__rt__array_get_element_ptr_1d(%[[VAL_2]]* %[[VAL_13]], i64 0)
# CHECK:         store %[[VAL_10]]* %[[VAL_11]], %[[VAL_10]]** %[[VAL_14]], align 8
# CHECK:         br label %[[VAL_15:.*]]
# CHECK:       17:                                               ; preds = %[[VAL_16:.*]], %[[VAL_17:.*]]
# CHECK:         %[[VAL_18:.*]] = phi i64 [ %[[VAL_19:.*]], %[[VAL_16]] ], [ 0, %[[VAL_17]] ]
# CHECK:         %[[VAL_20:.*]] = icmp slt i64 %[[VAL_18]], 2
# CHECK:         br i1 %[[VAL_20]], label %[[VAL_16]], label %[[VAL_21:.*]]
# CHECK:       20:                                               ; preds = %[[VAL_15]]
# CHECK:         %[[VAL_22:.*]] = phi i64 [ %[[VAL_18]], %[[VAL_15]] ]
# CHECK:         %[[VAL_23:.*]] = getelementptr [2 x double], [2 x double]* %[[VAL_6]], i32 0, i64 %[[VAL_22]]
# CHECK:         %[[VAL_24:.*]] = load double, double* %[[VAL_23]], align 8
# CHECK:         %[[VAL_25:.*]] = getelementptr [2 x { i8*, i64 }], [2 x { i8*, i64 }]* %[[VAL_3]], i32 0, i64 %[[VAL_22]]
# CHECK:         %[[VAL_26:.*]] = load { i8*, i64 }, { i8*, i64 }* %[[VAL_25]], align 8
# CHECK:         %[[VAL_27:.*]] = bitcast [1 x { i8*, i64 }]* %[[VAL_0]] to { i8*, i64 }*
# CHECK:         store { i8*, i64 } %[[VAL_26]], { i8*, i64 }* %[[VAL_27]], align 8
# CHECK:         %[[VAL_28:.*]] = bitcast [1 x { i8*, i64 }]* %[[VAL_0]] to i8*
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double %[[VAL_24]], %[[VAL_2]]* %[[VAL_13]], %[[VAL_2]]* %[[VAL_12]], i8* %[[VAL_28]])
# CHECK:         %[[VAL_19]] = add i64 %[[VAL_22]], 1
# CHECK:         br label %[[VAL_15]]
# CHECK:       29:                                               ; preds = %[[VAL_15]]
# CHECK:         call void @__quantum__rt__qubit_release_array(%[[VAL_2]]* %[[VAL_1]])
# CHECK:         ret void

# CHECK-LABEL: define void @__nvqpp__mlirgen__exp_pauli_loop..
# CHECK:         %[[VAL_29:.*]] = alloca [1 x { i8*, i64 }], align 8
# CHECK:         %[[VAL_30:.*]] = extractvalue { double*, i64 } %[[VAL_31:.*]], 1
# CHECK:         br label %[[VAL_32:.*]]
# CHECK:       7:                                                ; preds = %[[VAL_33:.*]], %[[VAL_34:.*]]
# CHECK:         %[[VAL_35:.*]] = phi i64 [ %[[VAL_36:.*]], %[[VAL_33]] ], [ 0, %[[VAL_34]] ]
# CHECK:         %[[VAL_37:.*]] = icmp slt i64 %[[VAL_35]], %[[VAL_30]]
# CHECK:         br i1 %[[VAL_37]], label %[[VAL_33]], label %[[VAL_38:.*]]
# CHECK:       10:                                               ; preds = %[[VAL_32]]
# CHECK:         %[[VAL_39:.*]] = phi i64 [ %[[VAL_35]], %[[VAL_32]] ]
# CHECK:         %[[VAL_40:.*]] = extractvalue { double*, i64 } %[[VAL_31]], 0
# CHECK:         %[[VAL_41:.*]] = getelementptr double, double* %[[VAL_40]], i64 %[[VAL_39]]
# CHECK:         %[[VAL_42:.*]] = load double, double* %[[VAL_41]], align 8
# CHECK:         %[[VAL_43:.*]] = fmul double %[[VAL_42]], %[[VAL_44:.*]]
# CHECK:         %[[VAL_45:.*]] = extractvalue { { i8*, i64 }*, i64 } %[[VAL_46:.*]], 0
# CHECK:         %[[VAL_47:.*]] = getelementptr { i8*, i64 }, { i8*, i64 }* %[[VAL_45]], i64 %[[VAL_39]]
# CHECK:         %[[VAL_48:.*]] = load { i8*, i64 }, { i8*, i64 }* %[[VAL_47]], align 8
# CHECK:         %[[VAL_49:.*]] = bitcast [1 x { i8*, i64 }]* %[[VAL_29]] to { i8*, i64 }*
# CHECK:         store { i8*, i64 } %[[VAL_48]], { i8*, i64 }* %[[VAL_49]], align 8
# CHECK:         %[[VAL_50:.*]] = bitcast [1 x { i8*, i64 }]* %[[VAL_29]] to i8*
# CHECK:         call void @__quantum__qis__exp_pauli(double %[[VAL_43]], %[[VAL_51:.*]]* %[[VAL_52:.*]], i8* %[[VAL_50]])
# CHECK:         %[[VAL_36]] = add i64 %[[VAL_39]], 1
# CHECK:         br label %[[VAL_32]]
# CHECK:       22:                                               ; preds = %[[VAL_32]]
# CHECK:         ret void
