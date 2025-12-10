# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
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


# CHECK-LABEL: define void @__nvqpp__mlirgen__kernel_ancilla_exp_pauli
# CHECK-SAME: ({ ptr, i64 } %[[VAL_0:.*]], { ptr, ptr }
# CHECK-SAME:                  %[[VAL_1:.*]]) {
# CHECK:         %[[VAL_2:.*]] = alloca [1 x { ptr, i64 }], align 8
# CHECK:         %[[VAL_3:.*]] = alloca [3 x double], align 8
# CHECK:         store double 3.400000e-01, ptr %[[VAL_3]], align 8
# CHECK:         %[[VAL_4:.*]] = getelementptr [3 x double], ptr %[[VAL_3]], i32 0, i32 1
# CHECK:         store double 1.200000e+00, ptr %[[VAL_4]], align 8
# CHECK:         %[[VAL_5:.*]] = getelementptr [3 x double], ptr %[[VAL_3]], i32 0, i32 2
# CHECK:         store double 1.600000e+00, ptr %[[VAL_5]], align 8
# CHECK:         %[[VAL_6:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_7:.*]] = call ptr @__quantum__rt__qubit_allocate_array(i64 3)
# CHECK:         br label %[[VAL_8:.*]]
# CHECK:         %[[VAL_9:.*]] = phi i64 [ %[[VAL_10:.*]], %[[VAL_11:.*]] ], [ 0, %[[VAL_12:.*]] ]
# CHECK:         %[[VAL_13:.*]] = icmp slt i64 %[[VAL_9]], 3
# CHECK:         br i1 %[[VAL_13]], label %[[VAL_11]], label %[[VAL_14:.*]]
# CHECK:         %[[VAL_15:.*]] = phi i64 [ %[[VAL_9]], %[[VAL_8]] ]
# CHECK:         %[[VAL_16:.*]] = getelementptr [3 x double], ptr %[[VAL_3]], i32 0, i64 %[[VAL_15]]
# CHECK:         %[[VAL_17:.*]] = load double, ptr %[[VAL_16]], align 8
# CHECK:         %[[VAL_18:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_7]], i64 %[[VAL_15]])
# CHECK:         %[[VAL_19:.*]] = load ptr, ptr %[[VAL_18]], align 8
# CHECK:         call void @__quantum__qis__rx(double %[[VAL_17]], ptr %[[VAL_19]])
# CHECK:         %[[VAL_10]] = add i64 %[[VAL_15]], 1
# CHECK:         br label %[[VAL_8]]
# CHECK:         call void @__quantum__qis__h(ptr %[[VAL_6]])
# CHECK:         %[[VAL_20:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_21:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_20]], i64 0)
# CHECK:         store ptr %[[VAL_6]], ptr %[[VAL_21]], align 8
# CHECK:         store ptr @cstr.58495900, ptr %[[VAL_2]], align 8
# CHECK:         %[[VAL_22:.*]] = getelementptr [1 x { ptr, i64 }], ptr %[[VAL_2]], i32 0, i32 0, i32 1
# CHECK:         store i64 3, ptr %[[VAL_22]], align 8
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double 2.310000e+01, ptr %[[VAL_20]], ptr %[[VAL_7]], ptr %[[VAL_2]])
# CHECK:         call void @__quantum__rt__qubit_release(ptr %[[VAL_6]])
# CHECK:         ret void
# CHECK:       }

# CHECK-LABEL: define void @__nvqpp__mlirgen__U_exp_pauli
# CHECK-SAME:           %[[VAL_0:.*]]) {
# CHECK:         %[[VAL_1:.*]] = alloca [1 x { ptr, i64 }], align 8
# CHECK:         store ptr @cstr.58495900, ptr %[[VAL_1]], align 8
# CHECK:         %[[VAL_2:.*]] = getelementptr [1 x { ptr, i64 }], ptr %[[VAL_1]], i32 0, i32 0, i32 1
# CHECK:         store i64 3, ptr %[[VAL_2]], align 8
# CHECK:         call void @__quantum__qis__exp_pauli(double 2.310000e+01, ptr %[[VAL_0]], ptr %[[VAL_1]])
# CHECK:         ret void
# CHECK:       }
