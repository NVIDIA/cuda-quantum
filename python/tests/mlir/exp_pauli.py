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
# CHECK:         call void @__quantum__qis__exp_pauli__ctl(double 2.310000e+01,
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

    # FileCheck below verifies one __quantum__qis__exp_pauli__ctl per word,
    # since the simulator's stage unrolls the loop.
    print(
        cudaq.translate(kernel_controlled_exp_pauli_loop,
                        coefficients,
                        words,
                        format='qir'))


# CTRL-LABEL: define void @__nvqpp__mlirgen__kernel_controlled_exp_pauli_loop..0x
# CTRL:         call void @__quantum__qis__exp_pauli__ctl(double 1.000000e+00,
# CTRL:         call void @__quantum__qis__exp_pauli__ctl(double 5.000000e-01,
# CTRL:         ret void
