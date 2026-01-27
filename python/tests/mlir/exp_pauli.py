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
    initial = np.array(
        cudaq.StateMemoryView(cudaq.get_state(kernel_initial_state, angles)))

    # create the initial state + ancilla, hadamard, then perform a
    # controlled rotation on the |1> subspace of the ancilla
    full = np.array(
        cudaq.StateMemoryView(cudaq.get_state(kernel_ancilla_rotation, angles)))

    # create the initial state and perform a rotation (for comparison with full)
    rotation = np.array(
        cudaq.StateMemoryView(cudaq.get_state(kernel_noancilla_rotation,
                                              angles)))

    # create the initial state + ancilla, hadamard, then perform a
    # controlled exp_pauli on the |1> subspace of the ancilla
    epauli = np.array(
        cudaq.StateMemoryView(cudaq.get_state(kernel_ancilla_exp_pauli,
                                              angles)))

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
