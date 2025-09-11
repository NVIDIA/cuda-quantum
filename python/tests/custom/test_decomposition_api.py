# ============================================================================ #
# Copyright (c) 2025 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from unittest import result
import pytest
import numpy as np
import cudaq


def ry_matrix(angle):
    """Return a numpy.array for the RY gate."""
    cos = np.cos(angle / 2)
    sin = np.sin(angle / 2)
    return np.array([[cos, -sin], [sin, cos]], dtype='complex128')


def rz_matrix(angle):
    """Return a numpy.array for the RZ gate."""
    ilam2 = 0.5j * float(angle)
    return np.array([[np.exp(-ilam2), 0], [0, np.exp(ilam2)]],
                    dtype='complex128')


def test_zyz_identity():
    """Test ZYZ decomposition of identity gate."""
    identity = np.eye(2, dtype=complex)
    angles = cudaq.decompose.zyz_components(identity)
    print(angles.alpha, angles.beta, angles.gamma, angles.phase)
    # All angles should be approximately 0
    assert abs(angles.alpha) < 1e-7
    assert abs(angles.beta) < 1e-7
    assert abs(angles.gamma) < 1e-7
    assert abs(angles.phase) < 1e-7


def test_zyz_random():
    """Test ZYZ decomposition of random gate."""
    random_gate = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    random_gate = random_gate / np.linalg.norm(random_gate)
    angles = cudaq.decompose.zyz_components(random_gate)

    # reconstruct the gate from angles
    result = rz_matrix(angles.alpha) @ ry_matrix(angles.beta) @ rz_matrix(
        angles.gamma)
    reconstructed = result * np.exp(1.0j * angles.phase)

    np.allclose(random_gate, reconstructed, atol=1e-7)


def canonical_to_matrix(x, y, z):
    """Convert canonical parameters to a matrix."""
    x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
    y_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z_matrix = np.array([[1, 0], [0, -1]], dtype=complex)

    XX = np.kron(x_matrix, x_matrix)
    YY = np.kron(y_matrix, y_matrix)
    ZZ = np.kron(z_matrix, z_matrix)

    from scipy.linalg import expm
    return expm(1j * (x * XX + y * YY + z * ZZ))


def test_kak_random():
    """Test KAK decomposition of a random unitary matrix."""
    from scipy.stats import unitary_group
    unitary = unitary_group.rvs(4, random_state=42)
    kak = cudaq.decompose.kak_components(unitary)

    # Reconstruct the matrix
    interaction = canonical_to_matrix(kak.x, kak.y, kak.z)
    a1_tensor_a0 = np.kron(kak.a1, kak.a0)
    b1_tensor_b0 = np.kron(kak.b1, kak.b0)
    reconstructed = kak.phase * a1_tensor_a0 @ interaction @ b1_tensor_b0

    assert np.allclose(unitary, reconstructed, atol=1e-7)
