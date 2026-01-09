# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest
import numpy as np
from scipy.linalg import block_diag
import cudaq

H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)


def Rx(theta):
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * X


def general_single(n, op, qidx):
    M = 1
    for j in range(n):
        M = np.kron(M, op if j == qidx else I2)
    return M


def general_cnot(n, ctrl, tgt):
    N = 2**n
    M = np.zeros((N, N), dtype=np.complex128)
    for idx in range(N):
        bits = [(idx >> j) & 1 for j in range(n)][::-1]
        if bits[ctrl] == 1:
            bits[tgt] ^= 1
        j = sum(bit << jdx for jdx, bit in enumerate(bits[::-1]))
        M[j, idx] = 1
    return M


def test_single_hadamard():

    @cudaq.kernel
    def k():
        q = cudaq.qubit()
        h(q)

    U = cudaq.get_unitary(k)
    expected = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]],
                                           dtype=np.complex128)
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_two_x_gates_one_qubit():

    @cudaq.kernel
    def k():
        q = cudaq.qubit()
        x(q)
        x(q)

    U = cudaq.get_unitary(k)
    expected = np.eye(2, dtype=np.complex128)  # X*X = I
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_two_hadamards_two_qubits():

    @cudaq.kernel
    def k():
        q = cudaq.qvector(2)
        h(q[0])
        h(q[1])

    U = cudaq.get_unitary(k)
    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    expected = np.kron(H, H)
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_rotation_h_hadamard_rotation():
    theta1, theta2 = np.pi / 5, np.pi / 7

    @cudaq.kernel
    def k(a: float, b: float):
        q = cudaq.qubit()
        rx(a, q)
        h(q)
        rx(b, q)

    # pass the two angles into get_unitary:
    U = cudaq.get_unitary(k, theta1, theta2)
    expected = Rx(theta2) @ H @ Rx(theta1)
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_single_qubit_large():

    @cudaq.kernel
    def k():
        q = cudaq.qvector(3)
        x(q[2])

    U = cudaq.get_unitary(k)
    # U = I  ⊗ I ⊗ X
    expected = np.kron(np.kron(I2, I2), X)
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_two_sparse_qubits():

    @cudaq.kernel
    def k():
        q = cudaq.qvector(3)
        h(q[0])
        x(q[2])

    U = cudaq.get_unitary(k)
    # U = H ⊗ I ⊗ X
    expected = np.kron(np.kron(H, I2), X)
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_custom_single_qubit_gate():
    # Apply a user-defined 2×2 unitary M
    M = np.array([[0, 1j], [1, 0]], dtype=np.complex128)
    cudaq.register_operation("custom_gate", M)

    with pytest.raises(RuntimeError,
                       match='Invalid gate name provided: custom_gate'):

        @cudaq.kernel
        def k():
            q = cudaq.qubit()
            custom_gate(q)

    # currently, unsupported

        U = cudaq.get_unitary(k)
    # np.testing.assert_allclose(U, M, atol=1e-12)


def test_swap_two_qubits():

    @cudaq.kernel
    def k():
        q = cudaq.qvector(2)
        swap(q[0], q[1])

    U = cudaq.get_unitary(k)
    SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                    dtype=np.complex128)
    np.testing.assert_allclose(U, SWAP, atol=1e-12)


def test_swap_nonadjacent_qubits():
    qubit_cnt = 5
    ind1, ind2 = 1, 4

    @cudaq.kernel
    def k():
        q = cudaq.qvector(qubit_cnt)
        swap(q[1], q[4])

    U = cudaq.get_unitary(k)
    # build expected swap matrix
    N = 2**qubit_cnt
    EXPECTED = np.zeros((N, N), dtype=np.complex128)
    for idx in range(N):
        bits = [(idx >> j) & 1 for j in range(qubit_cnt)][::-1]
        bits[ind1], bits[ind2] = bits[ind2], bits[ind1]
        j = sum(bit << jdx for jdx, bit in enumerate(bits[::-1]))
        EXPECTED[j, idx] = 1
    np.testing.assert_allclose(U, EXPECTED, atol=1e-12)


def test_cnot_two_qubits_opposite():

    @cudaq.kernel
    def k():
        q = cudaq.qvector(2)
        x.ctrl(q[1], q[0])

    U = cudaq.get_unitary(k)
    CNOT = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                    dtype=np.complex128)
    np.testing.assert_allclose(U, CNOT, atol=1e-12)


def test_cnot_nonadjacent_qubits():

    @cudaq.kernel
    def k():
        q = cudaq.qvector(3)
        x.ctrl(q[2], q[0])

    U = cudaq.get_unitary(k)
    EXPECTED = general_cnot(3, ctrl=2, tgt=0)
    np.testing.assert_allclose(U, EXPECTED, atol=1e-12)


@pytest.mark.parametrize("num_qubits", [2, 3, 4, 5])
def test_parametrized_h_cnot_circuit(num_qubits):
    # H on odd qubits, then CNOT(i → (3*i)%num_qubits)
    @cudaq.kernel
    def circuit(n: int):
        q = cudaq.qvector(n)
        for i in range(n):
            if i % 2 == 1:
                h(q[i])
        for i in range(n):
            if i != (3 * i) % n:
                x.ctrl(q[i], q[(3 * i) % n])

    U = cudaq.get_unitary(circuit, num_qubits)
    # build expected

    expected = np.eye(2**num_qubits, dtype=np.complex128)
    for i in range(num_qubits):
        if i % 2 == 1:
            expected = general_single(num_qubits, H, i) @ expected
    for i in range(num_qubits):
        if i != (3 * i) % num_qubits:
            expected = general_cnot(
                num_qubits, ctrl=i, tgt=(3 * i) % num_qubits) @ expected
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_toffoli_three_qubits():

    @cudaq.kernel
    def k():
        q = cudaq.qvector(3)
        # two controls (q[0], q[1]) and target q[2]
        x.ctrl([q[0], q[1]], q[2])

    U = cudaq.get_unitary(k)
    # build expected 8×8 Toffoli matrix
    expected = block_diag(np.eye(6, dtype=np.complex128), X)
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_controlled_rotation():
    theta = np.pi / 6

    @cudaq.kernel
    def k(a: float):
        q = cudaq.qvector(3)
        rx.ctrl(a, [q[0], q[1]], q[2])

    # pass the two angles into get_unitary:
    U = cudaq.get_unitary(k, theta)
    expected = block_diag(np.eye(6, dtype=np.complex128), Rx(theta))
    np.testing.assert_allclose(U, expected, atol=1e-12)


def test_cy_to_cx():

    expected_cy = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]],
        dtype=np.complex128)

    @cudaq.kernel
    def kernel_with_cy():
        q = cudaq.qvector(2)
        y.ctrl(q[0], q[1])

    U = cudaq.get_unitary(kernel_with_cy)
    np.testing.assert_allclose(U, expected_cy, atol=1e-12)

    @cudaq.kernel
    def kernel_with_decomposed_cy():
        q = cudaq.qvector(2)
        s.adj(q[1])
        x.ctrl(q[0], q[1])
        s(q[1])

    U_decomposed = cudaq.get_unitary(kernel_with_decomposed_cy)
    np.testing.assert_allclose(U_decomposed, expected_cy, atol=1e-12)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
