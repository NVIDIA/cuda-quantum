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
# CHECK-SAME: ({ ptr, i64 }
# CHECK-SAME: %[[VAL_0:.*]], { ptr, ptr }
# CHECK-SAME: %[[VAL_1:.*]]) {
# CHECK:         %[[VAL_2:.*]] = alloca [1 x { ptr, i64 }], align 8
# CHECK:         %[[VAL_3:.*]] = alloca [3 x double], align 8
# CHECK:         store double 3.400000e-01, ptr %[[VAL_3]], align 8
# CHECK:         %[[VAL_5:.*]] = getelementptr [3 x double], ptr %[[VAL_3]], i32 0, i32 1
# CHECK:         store double 1.200000e+00, ptr %[[VAL_5]], align 8
# CHECK:         %[[VAL_6:.*]] = getelementptr [3 x double], ptr %[[VAL_3]], i32 0, i32 2
# CHECK:         store double 1.600000e+00, ptr %[[VAL_6]], align 8
# CHECK:         %[[VAL_7:.*]] = call ptr @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_9:.*]] = call ptr @__quantum__rt__qubit_allocate_array(i64 3)
# CHECK:         br label %[[HDR:[0-9]+]]
# CHECK: [[HDR]]:{{.*}}
# CHECK:         %[[VAL_12:.*]] = phi i64 [ %[[VAL_13:.*]], %[[VAL_14:[0-9]+]] ], [ 0, %[[VAL_15:[0-9]+]] ]
# CHECK:         %[[VAL_16:.*]] = icmp slt i64 %[[VAL_12]], 3
# CHECK:         br i1 %[[VAL_16]], label %[[VAL_14]], label %[[VAL_17:[0-9]+]]
# CHECK: [[VAL_14]]:{{.*}}
# CHECK:         %[[VAL_19:.*]] = getelementptr [3 x double], ptr %[[VAL_3]], i32 0, i64 %[[VAL_12]]
# CHECK:         %[[VAL_20:.*]] = load double, ptr %[[VAL_19]], align 8
# CHECK:         %[[VAL_21:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_9]], i64 %[[VAL_12]])
# CHECK:         %[[VAL_22:.*]] = load ptr, ptr %[[VAL_21]], align 8
# CHECK:         call void @__quantum__qis__rx(double %[[VAL_20]], ptr %[[VAL_22]])
# CHECK:         %[[VAL_13]] = add i64 %[[VAL_12]], 1
# CHECK:         br label %[[HDR]]
# CHECK: [[VAL_17]]:{{.*}}
# CHECK:         call void @__quantum__qis__h(ptr %[[VAL_7]])
# CHECK:         %[[VAL_23:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_24:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_23]], i64 0)
# CHECK:         store ptr %[[VAL_7]], ptr %[[VAL_24]], align 8
# CHECK:         store ptr @cstr.58495900, ptr %[[VAL_2]], align 8
# CHECK:         %[[VAL_26:.*]] = getelementptr [1 x { ptr, i64 }], ptr %[[VAL_2]], i32 0, i32 0, i32 1
# CHECK:         store i64 3, ptr %[[VAL_26]], align 8
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double 2.310000e+01, ptr %[[VAL_23]], ptr %[[VAL_9]], ptr %[[VAL_2]])
# CHECK-DAG:     call void @__quantum__rt__qubit_release_array(ptr %[[VAL_9]])
# CHECK-DAG:     call void @__quantum__rt__qubit_release(ptr %[[VAL_7]])
# CHECK:         ret void
# CHECK: }

# CHECK-LABEL: define void @__nvqpp__mlirgen__U_exp_pauli..
# CHECK-SAME: ptr %[[VAL_0:.*]]) {
# CHECK:         %[[VAL_2:.*]] = alloca [1 x { ptr, i64 }], align 8
# CHECK:         store ptr @cstr.58495900, ptr %[[VAL_2]], align 8
# CHECK:         %[[VAL_4:.*]] = getelementptr [1 x { ptr, i64 }], ptr %[[VAL_2]], i32 0, i32 0, i32 1
# CHECK:         store i64 3, ptr %[[VAL_4]], align 8
# CHECK:         call void @__quantum__qis__exp_pauli(double 2.310000e+01, ptr %[[VAL_0]], ptr %[[VAL_2]])
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
# CHECK:         %[[VAL_0:.*]] = alloca [1 x { ptr, i64 }], align 8
# CHECK:         %[[VAL_1:.*]] = call ptr @__quantum__rt__qubit_allocate_array(i64 3)
# CHECK:         %[[VAL_3:.*]] = alloca [2 x { ptr, i64 }], align 8
# CHECK:         store { ptr, i64 } { ptr @cstr.5A5A00, i64 3 }, ptr %[[VAL_3]], align 8
# CHECK:         %[[VAL_5:.*]] = getelementptr [2 x { ptr, i64 }], ptr %[[VAL_3]], i32 0, i32 1
# CHECK:         store { ptr, i64 } { ptr @cstr.585800, i64 3 }, ptr %[[VAL_5]], align 8
# CHECK:         %[[VAL_6:.*]] = alloca [2 x double], align 8
# CHECK:         store double 1.000000e+00, ptr %[[VAL_6]], align 8
# CHECK:         %[[VAL_8:.*]] = getelementptr [2 x double], ptr %[[VAL_6]], i32 0, i32 1
# CHECK:         store double 5.000000e-01, ptr %[[VAL_8]], align 8
# CHECK:         %[[VAL_9:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_1]], i64 0)
# CHECK:         %[[VAL_11:.*]] = load ptr, ptr %[[VAL_9]], align 8
# CHECK:         %[[VAL_12:.*]] = call ptr @__quantum__rt__array_slice(ptr %[[VAL_1]], i32 1, i64 1, i64 1, i64 2)
# CHECK:         call void @__quantum__qis__h(ptr %[[VAL_11]])
# CHECK:         %[[VAL_13:.*]] = call ptr @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_14:.*]] = call ptr @__quantum__rt__array_get_element_ptr_1d(ptr %[[VAL_13]], i64 0)
# CHECK:         store ptr %[[VAL_11]], ptr %[[VAL_14]], align 8
# CHECK:         br label %[[HDR:[0-9]+]]
# CHECK: [[HDR]]:{{.*}}
# CHECK:         %[[VAL_18:.*]] = phi i64 [ %[[VAL_19:.*]], %[[VAL_16:[0-9]+]] ], [ 0, %[[VAL_17:[0-9]+]] ]
# CHECK:         %[[VAL_20:.*]] = icmp slt i64 %[[VAL_18]], 2
# CHECK:         br i1 %[[VAL_20]], label %[[VAL_16]], label %[[VAL_21:[0-9]+]]
# CHECK: [[VAL_16]]:{{.*}}
# CHECK:         %[[VAL_23:.*]] = getelementptr [2 x double], ptr %[[VAL_6]], i32 0, i64 %[[VAL_18]]
# CHECK:         %[[VAL_24:.*]] = load double, ptr %[[VAL_23]], align 8
# CHECK:         %[[VAL_25:.*]] = getelementptr [2 x { ptr, i64 }], ptr %[[VAL_3]], i32 0, i64 %[[VAL_18]]
# CHECK:         %[[VAL_26:.*]] = load { ptr, i64 }, ptr %[[VAL_25]], align 8
# CHECK:         store { ptr, i64 } %[[VAL_26]], ptr %[[VAL_0]], align 8
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double %[[VAL_24]], ptr %[[VAL_13]], ptr %[[VAL_12]], ptr %[[VAL_0]])
# CHECK:         %[[VAL_19]] = add i64 %[[VAL_18]], 1
# CHECK:         br label %[[HDR]]
# CHECK: [[VAL_21]]:{{.*}}
# CHECK:         call void @__quantum__rt__qubit_release_array(ptr %[[VAL_1]])
# CHECK:         ret void

# CHECK-LABEL: define void @__nvqpp__mlirgen__exp_pauli_loop..
# CHECK:         %[[VAL_29:.*]] = alloca [1 x { ptr, i64 }], align 8
# CHECK:         %[[VAL_30:.*]] = extractvalue { ptr, i64 } %[[VAL_31:.*]], 1
# CHECK:         br label %[[HDR2:[0-9]+]]
# CHECK: [[HDR2]]:{{.*}}
# CHECK:         %[[VAL_35:.*]] = phi i64 [ %[[VAL_36:.*]], %[[VAL_33:[0-9]+]] ], [ 0, %[[VAL_34:[0-9]+]] ]
# CHECK:         %[[VAL_37:.*]] = icmp slt i64 %[[VAL_35]], %[[VAL_30]]
# CHECK:         br i1 %[[VAL_37]], label %[[VAL_33]], label %[[VAL_38:[0-9]+]]
# CHECK: [[VAL_33]]:{{.*}}
# CHECK:         %[[VAL_40:.*]] = extractvalue { ptr, i64 } %[[VAL_31]], 0
# CHECK:         %[[VAL_41:.*]] = getelementptr double, ptr %[[VAL_40]], i64 %[[VAL_35]]
# CHECK:         %[[VAL_42:.*]] = load double, ptr %[[VAL_41]], align 8
# CHECK:         %[[VAL_43:.*]] = fmul double %[[VAL_42]], %[[VAL_44:.*]]
# CHECK:         %[[VAL_45:.*]] = extractvalue { ptr, i64 } %[[VAL_46:.*]], 0
# CHECK:         %[[VAL_47:.*]] = getelementptr { ptr, i64 }, ptr %[[VAL_45]], i64 %[[VAL_35]]
# CHECK:         %[[VAL_48:.*]] = load { ptr, i64 }, ptr %[[VAL_47]], align 8
# CHECK:         store { ptr, i64 } %[[VAL_48]], ptr %[[VAL_29]], align 8
# CHECK:         call void @__quantum__qis__exp_pauli(double %[[VAL_43]], ptr %{{.*}}, ptr %[[VAL_29]])
# CHECK:         %[[VAL_36]] = add i64 %[[VAL_35]], 1
# CHECK:         br label %[[HDR2]]
# CHECK: [[VAL_38]]:{{.*}}
# CHECK:         ret void
