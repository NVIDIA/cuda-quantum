# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This file is responsible for testing the accuracy of gates within
# the kernel builder.

import numpy as np

import cudaq
from cudaq import spin


def test_sdg_0_state():
    """Tests the adjoint S-gate on a qubit starting in the 0-state."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in superposition state.
    kernel.h(qubit)
    # Rotate around Z by -pi/2, twice. Total rotation of -pi.
    kernel.sdg(qubit)
    kernel.sdg(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    counts.dump()

    # Since the qubit began in the 0-state, it should now be in the
    # 1-state.
    assert counts["1"] == 1000


def test_sdg_1_state():
    """Tests the adjoint S-gate on a qubit starting in the 1-state."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in 1-state.
    kernel.x(qubit)
    # Superpositoin.
    kernel.h(qubit)
    # Rotate around Z by -pi/2, twice. Total rotation of -pi.
    kernel.sdg(qubit)
    kernel.sdg(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    counts.dump()

    # Since the qubit began in the 1-state, it should now be in the
    # 0-state.
    assert counts["0"] == 1000


def test_sdg_0_state_negate():
    """Tests that the sdg and s gates cancel each other out."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in superposition state.
    kernel.h(qubit)
    # Rotate around Z by -pi/2, twice. Total rotation of -pi.
    kernel.sdg(qubit)
    kernel.sdg(qubit)
    # Rotate back around by pi. Will use two gates here, but will
    # also test with a plain Z-gate in the 1-state test.
    kernel.s(qubit)
    kernel.s(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    counts.dump()

    # Qubit should still be in 0 state.
    assert counts["0"] == 1000


def test_sdg_1_state_negate():
    """Tests that the sdg and s gates cancel each other out."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in 1-state.
    kernel.x(qubit)
    # Superpositoin.
    kernel.h(qubit)
    # Rotate around Z by -pi/2, twice. Total rotation of -pi.
    kernel.sdg(qubit)
    kernel.sdg(qubit)
    # Rotate back by pi.
    kernel.z(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    counts.dump()

    # Qubit should still be in 1 state.
    assert counts["1"] == 1000


def test_tdg_0_state():
    """Tests the adjoint T-gate on a qubit starting in the 0-state."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in superposition state.
    kernel.h(qubit)
    # Rotate around Z by -pi/4, four times. Total rotation of -pi.
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    counts.dump()

    # Since the qubit began in the 0-state, it should now be in the
    # 1-state.
    assert counts["1"] == 1000


def test_tdg_1_state():
    """Tests the adjoint T-gate on a qubit starting in the 1-state."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in 1-state.
    kernel.x(qubit)
    # Superposition.
    kernel.h(qubit)
    # Rotate around Z by -pi/4, four times. Total rotation of -pi.
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    counts.dump()

    # Since the qubit began in the 1-state, it should now be in the
    # 0-state.
    assert counts["0"] == 1000


def test_tdg_0_state_negate():
    """Tests that the adjoint T gate cancels with a T gate."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in superposition state.
    kernel.h(qubit)
    # Rotate around Z by -pi/4, four times. Total rotation of -pi.
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    # Rotate back by pi.
    kernel.z(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    counts.dump()

    # Qubit should remain in 0-state.
    assert counts["0"] == 1000


def test_tdg_1_state_negate():
    """Tests that the adjoint T gate cancels with a T gate."""
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc(1)

    # Place qubit in 1-state.
    kernel.x(qubit)
    # Superposition.
    kernel.h(qubit)
    # Rotate around Z by -pi/4, four times. Total rotation of -pi.
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    kernel.tdg(qubit)
    # Rotate back by pi.
    kernel.t(qubit)
    kernel.t(qubit)
    kernel.t(qubit)
    kernel.t(qubit)
    # Apply another hadamard.
    kernel.h(qubit)
    kernel.mz(qubit)

    counts = cudaq.sample(kernel)
    counts.dump()

    # Qubit should remain in 1-state.
    assert counts["1"] == 1000


def test_can_progressively_build():
    """Tests that a kernel can be build progressively."""
    cudaq.reset_target()
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    print(kernel)
    state = cudaq.get_state(kernel)
    assert np.isclose(1. / np.sqrt(2.), state[0].real)
    assert np.isclose(0., state[1].real)
    assert np.isclose(1. / np.sqrt(2.), state[2].real)
    assert np.isclose(0., state[3].real)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '10' in counts
    assert '00' in counts

    # Continue building the kernel
    kernel.cx(q[0], q[1])
    print(kernel)
    state = cudaq.get_state(kernel)
    assert np.isclose(1. / np.sqrt(2.), state[0].real)
    assert np.isclose(0., state[1].real)
    assert np.isclose(0., state[2].real)
    assert np.isclose(1. / np.sqrt(2.), state[3].real)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


def test_from_state():
    cudaq.reset_target()
    state = np.asarray([.70710678, 0., 0., 0.70710678])
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)

    cudaq.from_state(kernel, qubits, state)

    print(kernel)
    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    kernel = cudaq.from_state(state)
    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    state = np.asarray([0., .292786, .956178, 0.])
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    cudaq.from_state(kernel, qubits, state)
    energy = cudaq.observe(kernel, hamiltonian).expectation_z()
    assert np.isclose(-1.748, energy, 1e-3)

    ss = cudaq.get_state(kernel)
    for i in range(4):
        assert np.isclose(ss[i], state[i], 1e-3)
