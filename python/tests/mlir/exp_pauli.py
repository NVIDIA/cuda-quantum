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
    initial = np.array(cudaq.get_state(kernel_initial_state, angles))

    # create the initial state + ancilla, hadamard, then perform a
    # controlled rotation on the |1> subspace of the ancilla
    full = np.array(cudaq.get_state(kernel_ancilla_rotation, angles))

    # create the initial state and perform a rotation (for comparison with full)
    rotation = np.array(cudaq.get_state(kernel_noancilla_rotation, angles))

    # create the initial state + ancilla, hadamard, then perform a
    # controlled exp_pauli on the |1> subspace of the ancilla
    epauli = np.array(cudaq.get_state(kernel_ancilla_exp_pauli, angles))

    print(cudaq.translate(kernel_ancilla_exp_pauli, format='qir'))


# CHECK-LABEL: define void @__nvqpp__mlirgen__U_exp_pauli.ctrl(
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double 2.310000e+01, %Array* %{{.*}}, %Array* %{{.*}}, i8* nonnull %{{.*}})

# CHECK-LABEL: define void @__nvqpp__mlirgen__U_exp_pauli(
# CHECK:         call void @__quantum__qis__exp_pauli(double 2.310000e+01, %Array* %{{.*}}, i8* nonnull {{.*}})

# CHECK-LABEL: define void @__nvqpp__mlirgen__kernel_ancilla_exp_pauli({ double*, i64 }
# CHECK-SAME:    %[[VAL_0:.*]])
# CHECK:         %[[VAL_1:.*]] = alloca [1 x { i8*, i64 }], align 8
# CHECK:         %[[VAL_2:.*]] = tail call %Qubit* @__quantum__rt__qubit_allocate()
# CHECK:         %[[VAL_4:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 1
# CHECK:         %[[VAL_5:.*]] = tail call %Array* @__quantum__rt__qubit_allocate_array(i64 %[[VAL_4]])
# CHECK:         %[[VAL_7:.*]] = icmp sgt i64 %[[VAL_4]], 0
# CHECK:         br i1 %[[VAL_7]], label %[[VAL_8:.*]], label %[[VAL_9:.*]]
# CHECK:       :                                           ; preds = %[[VAL_10:.*]]
# CHECK:         %[[VAL_11:.*]] = extractvalue { double*, i64 } %[[VAL_0]], 0
# CHECK:         br label %[[VAL_12:.*]]
# CHECK:       :                                                ; preds = %[[VAL_8]], %[[VAL_12]]
# CHECK:         %[[VAL_13:.*]] = phi i64 [ 0, %[[VAL_8]] ], [ %[[VAL_14:.*]], %[[VAL_12]] ]
# CHECK:         %[[VAL_15:.*]] = getelementptr double, double* %[[VAL_11]], i64 %[[VAL_13]]
# CHECK:         %[[VAL_16:.*]] = load double, double* %[[VAL_15]], align 8
# CHECK:         %[[VAL_17:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_5]], i64 %[[VAL_13]])
# CHECK:         %[[VAL_18:.*]] = load %Qubit*, %Qubit** %[[VAL_17]], align 8
# CHECK:         tail call void @__quantum__qis__rx(double %[[VAL_16]], %Qubit* %[[VAL_18]])
# CHECK:         %[[VAL_14]] = add nuw nsw i64 %[[VAL_13]], 1
# CHECK:         %[[VAL_19:.*]] = icmp slt i64 %[[VAL_14]], %[[VAL_4]]
# CHECK:         br i1 %[[VAL_19]], label %[[VAL_12]], label %[[VAL_9]]
# CHECK:       :                                      ; preds = %[[VAL_12]], %[[VAL_10]]
# CHECK:         tail call void @__quantum__qis__h(%Qubit* %[[VAL_2]])
# CHECK:         %[[VAL_20:.*]] = tail call %Array* @__quantum__rt__array_create_1d(i32 8, i64 1)
# CHECK:         %[[VAL_21:.*]] = tail call %Qubit** @__quantum__rt__array_get_element_ptr_1d(%Array* %[[VAL_20]], i64 0)
# CHECK:         store %Qubit* %[[VAL_2]], %Qubit** %[[VAL_21]], align 8
# CHECK:         %[[VAL_22:.*]] = bitcast [1 x { i8*, i64 }]* %[[VAL_1]] to i8*
# CHECK:         call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %[[VAL_22]])
# CHECK:         %[[VAL_23:.*]] = getelementptr inbounds [1 x { i8*, i64 }], [1 x { i8*, i64 }]* %[[VAL_1]], i64 0, i64 0, i32 0
# CHECK:         store i8* getelementptr inbounds ([4 x i8], [4 x i8]* @cstr.58495900, i64 0, i64 0), i8** %[[VAL_23]], align 8
# CHECK:         %[[VAL_24:.*]] = getelementptr inbounds [1 x { i8*, i64 }], [1 x { i8*, i64 }]* %[[VAL_1]], i64 0, i64 0, i32 1
# CHECK:         store i64 3, i64* %[[VAL_24]], align 8
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double 2.310000e+01, %Array* %[[VAL_20]], %Array* %[[VAL_5]], i8* nonnull %[[VAL_22]])
# CHECK:         call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %[[VAL_22]])
# CHECK-DAG:     call void @__quantum__rt__qubit_release(%Qubit* %[[VAL_2]])
# CHECK-DAG:     call void @__quantum__rt__qubit_release_array(%Array* %[[VAL_5]])
# CHECK:         ret void
# CHECK:       }
