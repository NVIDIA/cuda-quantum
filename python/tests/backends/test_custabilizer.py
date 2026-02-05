# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
from typing import List
import pytest

import cudaq
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def setTarget():
    try:
        cudaq.set_target('custabilizer')
    except RuntimeError:
        pytest.skip("target not available")
    yield
    cudaq.reset_target()


def test_custabilizer_non_clifford():
    """Custabilizer inherits from Stim, so non-Clifford gates should be rejected."""

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(10)
        rx(0.1, qubits[0])

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel)
    assert 'Gate not supported by Stim simulator' in repr(e)


def test_custabilizer_toffoli_gates():
    """Custabilizer inherits from Stim, so multi-control gates should be rejected."""

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(10)
        cx(qubits[0:9], qubits[9])

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel)
    assert 'Gates with >1 controls not supported by Stim simulator' in repr(e)


def test_custabilizer_sample():
    """Test basic sampling with large qubit count (custabilizer GPU backend)."""

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(250)
        h(qubits[0])
        for i in range(1, 250):
            cx(qubits[i - 1], qubits[i])
        mz(qubits)

    counts = cudaq.sample(kernel)
    assert len(counts) == 2
    assert '0' * 250 in counts
    assert '1' * 250 in counts


def test_custabilizer_all_mz_types():
    """Test different measurement basis types."""

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(10)
        mx(qubits)
        my(qubits)
        mz(qubits)

    counts = cudaq.sample(kernel)
    assert len(counts) > 1


def test_custabilizer_state_preparation():
    """Custabilizer does not support state initialization."""

    @cudaq.kernel
    def kernel(vec: List[complex]):
        qubits = cudaq.qvector(vec)

    with pytest.raises(RuntimeError) as e:
        state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
        cudaq.sample(kernel, state)
    assert 'does not support initialization of qubits from state data' in repr(e)


def test_custabilizer_state_preparation_builder():
    """Custabilizer does not support state initialization via builder."""
    kernel, state = cudaq.make_kernel(List[complex])
    qubits = kernel.qalloc(state)

    with pytest.raises(RuntimeError) as e:
        state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
        cudaq.sample(kernel, state)
    assert 'does not support initialization of qubits from state data' in repr(e)


def test_custabilizer_bell_state():
    """Test Bell state preparation and measurement."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])
        mz(q)

    cudaq.set_random_seed(13)
    counts = cudaq.sample(kernel, shots_count=1000)
    assert len(counts) == 2
    assert '00' in counts
    assert '11' in counts
    assert np.isclose(counts.probability('00'), 0.5, atol=0.2)
    assert np.isclose(counts.probability('11'), 0.5, atol=0.2)


def test_custabilizer_mid_circuit_measurement():
    """Test mid-circuit measurement with conditionals (exercises measureQubit)."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        h(q[0])
        # Mid-circuit measurement that triggers measureQubit call
        result = mz(q[0])
        if result:
            x(q[1])
        mz(q)

    cudaq.set_random_seed(13)
    counts = cudaq.sample(kernel, shots_count=1000)
    # Should see 00 and 11 (when q[0] is 0 or 1, q[1] follows)
    assert len(counts) == 2
    assert '00' in counts
    assert '11' in counts
    assert np.isclose(counts.probability('00'), 0.5, atol=0.2)
    assert np.isclose(counts.probability('11'), 0.5, atol=0.2)


def test_custabilizer_ghz_state():
    """Test GHZ state with varying qubit counts."""

    @cudaq.kernel
    def kernel(n: int):
        qubits = cudaq.qvector(n)
        h(qubits[0])
        for i in range(n - 1):
            cx(qubits[i], qubits[i + 1])
        mz(qubits)

    for n in [3, 5, 10]:
        counts = cudaq.sample(kernel, n, shots_count=100)
        assert len(counts) == 2
        assert '0' * n in counts
        assert '1' * n in counts


def test_custabilizer_noise_support():
    """Test that noise models work with custabilizer."""
    noise = cudaq.NoiseModel()
    # Add bit flip noise on X gates
    noise.add_channel('x', [0], cudaq.BitFlipChannel(0.1))

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(5)
        x(q)
        mz(q)

    # Without noise, should always get 11111
    counts_no_noise = cudaq.sample(kernel)
    assert counts_no_noise['11111'] == 1000

    # With noise, should see some bit flips
    counts_with_noise = cudaq.sample(kernel, noise_model=noise, shots_count=1000)
    # Should have multiple outcomes due to noise
    assert len(counts_with_noise) > 1


def test_custabilizer_explicit_measurements():
    """Test explicit measurements flag."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        x(q[0])
        mz(q[0])
        mz(q[1])
        mz(q[2])

    counts = cudaq.sample(kernel)
    assert counts['100'] == 1000
    counts = cudaq.sample(kernel, explicit_measurements=True)
    assert counts['100'] == 1000


def test_custabilizer_register_names():
    """Test named measurement registers."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        h(q[0])
        mz(q[0], register_name="first")
        cx(q[0], q[1])
        mz(q, register_name="both")

    result = cudaq.sample(kernel, shots_count=100)
    assert 'first' in result.register_names
    assert 'both' in result.register_names
    first_reg = result.get_register_counts('first')
    assert '0' in first_reg or '1' in first_reg


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
