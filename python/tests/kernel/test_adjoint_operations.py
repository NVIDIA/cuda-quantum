# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import cudaq
from cudaq import spin


def test_sdg_0_state():
    """Tests the adjoint S-gate on a qubit starting in the 0-state."""

    @cudaq.kernel
    def sdg_0_state():
        qubit = cudaq.qubit()

        # Place qubit in superposition state.
        h(qubit)
        # Rotate around Z by -pi/2, twice. Total rotation of -pi.
        sdg(qubit)
        sdg(qubit)
        # Apply another hadamard.
        h(qubit)
        mz(qubit)

    counts = cudaq.sample(sdg_0_state)
    print(counts)

    # Since the qubit began in the 0-state, it should now be in the
    # 1-state.
    assert counts["1"] == 1000


def test_sdg_1_state():
    """Tests the adjoint S-gate on a qubit starting in the 1-state."""

    @cudaq.kernel
    def sdg_1_state():
        qubit = cudaq.qubit()

        # Place qubit in 1-state.
        x(qubit)
        # Superposition.
        h(qubit)
        # Rotate around Z by -pi/2, twice. Total rotation of -pi.
        sdg(qubit)
        sdg(qubit)
        # Apply another hadamard.
        h(qubit)
        mz(qubit)

    counts = cudaq.sample(sdg_1_state)
    print(counts)

    # Since the qubit began in the 1-state, it should now be in the
    # 0-state.
    assert counts["0"] == 1000


def test_sdg_0_state_negate():
    """Tests that the sdg and s gates cancel each other out."""

    @cudaq.kernel
    def sdg_0_state_negate():
        qubit = cudaq.qubit()

        # Place qubit in superposition state.
        h(qubit)
        # Rotate around Z by -pi/2, twice. Total rotation of -pi.
        sdg(qubit)
        sdg(qubit)
        # Rotate back around by pi. Will use two gates here, but will
        # also test with a plain Z-gate in the 1-state test.
        s(qubit)
        s(qubit)
        # Apply another hadamard.
        h(qubit)
        mz(qubit)

    counts = cudaq.sample(sdg_0_state_negate)
    print(counts)

    # Qubit should still be in 0 state.
    assert counts["0"] == 1000


def test_sdg_1_state_negate():
    """Tests that the sdg and s gates cancel each other out."""

    @cudaq.kernel
    def sdg_1_state_negate():
        qubit = cudaq.qubit()

        # Place qubit in 1-state.
        x(qubit)
        # Superposition.
        h(qubit)
        # Rotate around Z by -pi/2, twice. Total rotation of -pi.
        sdg(qubit)
        sdg(qubit)
        # Rotate back by pi.
        z(qubit)
        # Apply another hadamard.
        h(qubit)
        mz(qubit)

    counts = cudaq.sample(sdg_1_state_negate)
    print(counts)

    # Qubit should still be in 1 state.
    assert counts["1"] == 1000


def test_tdg_0_state():
    """Tests the adjoint T-gate on a qubit starting in the 0-state."""

    @cudaq.kernel
    def tdg_0_state():
        qubit = cudaq.qubit()

        # Place qubit in superposition state.
        h(qubit)
        # Rotate around Z by -pi/4, four times. Total rotation of -pi.
        tdg(qubit)
        tdg(qubit)
        tdg(qubit)
        tdg(qubit)
        # Apply another hadamard.
        h(qubit)
        mz(qubit)

    counts = cudaq.sample(tdg_0_state)
    print(counts)

    # Since the qubit began in the 0-state, it should now be in the
    # 1-state.
    assert counts["1"] == 1000


def test_tdg_1_state():
    """Tests the adjoint T-gate on a qubit starting in the 1-state."""

    @cudaq.kernel
    def tdg_1_state():
        qubit = cudaq.qubit()

        # Place qubit in 1-state.
        x(qubit)
        # Superposition.
        h(qubit)
        # Rotate around Z by -pi/4, four times. Total rotation of -pi.
        tdg(qubit)
        tdg(qubit)
        tdg(qubit)
        tdg(qubit)
        # Apply another hadamard.
        h(qubit)
        mz(qubit)

    counts = cudaq.sample(tdg_1_state)
    print(counts)

    # Since the qubit began in the 1-state, it should now be in the
    # 0-state.
    assert counts["0"] == 1000


def test_tdg_0_state_negate():
    """Tests that the adjoint T gate cancels with a T gate."""

    @cudaq.kernel
    def tdg_0_state_negate():
        qubit = cudaq.qubit()

        # Place qubit in superposition state.
        h(qubit)
        # Rotate around Z by -pi/4, four times. Total rotation of -pi.
        tdg(qubit)
        tdg(qubit)
        tdg(qubit)
        tdg(qubit)
        # Rotate back by pi.
        z(qubit)
        # Apply another hadamard.
        h(qubit)
        mz(qubit)

    counts = cudaq.sample(tdg_0_state_negate)
    print(counts)

    # Qubit should remain in 0-state.
    assert counts["0"] == 1000


def test_tdg_1_state_negate():
    """Tests that the adjoint T gate cancels with a T gate."""

    @cudaq.kernel
    def tdg_1_state_negate():
        qubit = cudaq.qubit()

        # Place qubit in 1-state.
        x(qubit)
        # Superposition.
        h(qubit)
        # Rotate around Z by -pi/4, four times. Total rotation of -pi.
        tdg(qubit)
        tdg(qubit)
        tdg(qubit)
        tdg(qubit)
        # Rotate back by pi.
        t(qubit)
        t(qubit)
        t(qubit)
        t(qubit)
        # Apply another hadamard.
        h(qubit)
        mz(qubit)

    counts = cudaq.sample(tdg_1_state_negate)
    print(counts)

    # Qubit should remain in 1-state.
    assert counts["1"] == 1000
