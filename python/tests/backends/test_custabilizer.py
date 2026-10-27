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
    if cudaq.num_available_gpus() == 0:
        pytest.skip("cuStabilizer requires a CUDA-capable GPU")
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

    with pytest.raises(RuntimeError):
        state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
        cudaq.sample(kernel, state)


def test_custabilizer_state_preparation_builder():
    """Custabilizer does not support state initialization via builder."""
    kernel, state = cudaq.make_kernel(List[complex])
    qubits = kernel.qalloc(state)

    with pytest.raises(RuntimeError):
        state = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]
        cudaq.sample(kernel, state)


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
    def kernel() -> List[bool]:
        q = cudaq.qvector(2)
        h(q[0])
        # Mid-circuit measurement that triggers measureQubit call
        result = mz(q[0])
        if result:
            x(q[1])
        return mz(q)

    cudaq.set_random_seed(13)
    results = cudaq.run(kernel, shots_count=1000)
    assert len(results) == 1000

    # The conditional X makes q[1] match q[0], so only 00 and 11 are valid.
    count_00 = results.count([False, False])
    count_11 = results.count([True, True])
    assert count_00 + count_11 == len(results)

    # The initial Hadamard makes both outcomes equally likely.
    assert np.isclose(count_00 / len(results), 0.5, atol=0.2)
    assert np.isclose(count_11 / len(results), 0.5, atol=0.2)


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
    counts_with_noise = cudaq.sample(kernel,
                                     noise_model=noise,
                                     shots_count=1000)
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


@pytest.mark.parametrize(
    "shared_control", [True, False],
    ids=["shared-control-x-basis", "shared-target-z-basis"])
def test_custabilizer_overlapping_cnot_targets(shared_control):
    """cuStabilizer requires non-overlapping targets within each instruction.

    Stim can fuse adjacent CNOTs sharing a control or target. The backend must
    separate them to preserve the X-basis or Z-basis parity, respectively.
    https://docs.nvidia.com/cuda/cuquantum/26.09.0/custabilizer/overview.html#circuit
    """

    @cudaq.kernel
    def kernel(shared_control: bool):
        q = cudaq.qvector(3)
        h(q[0])
        if shared_control:
            cx(q[0], q[1])
            cx(q[0], q[2])
            h(q)  # Measure all three qubits in the X basis.
        else:
            h(q[1])
            cx(q[0], q[2])
            cx(q[1], q[2])
        mz(q)

    cudaq.set_random_seed(13)
    counts = cudaq.sample(kernel, shared_control, shots_count=1024)
    # In either basis, the three measurement bits have even parity.
    expected_outcomes = {'000', '011', '101', '110'}
    assert set(counts) == expected_outcomes
    for outcome in expected_outcomes:
        assert np.isclose(counts.probability(outcome), 0.25, atol=0.08)


@pytest.mark.parametrize("x_basis", [False, True], ids=["z-basis", "x-basis"])
def test_custabilizer_identical_cnots_cancel(x_basis):
    """Identical CNOTs reuse both targets and must execute sequentially."""

    @cudaq.kernel
    def kernel(x_basis: bool):
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])
        cx(q[0], q[1])
        if x_basis:
            h(q[0])
        mz(q)

    cudaq.set_random_seed(13)
    counts = cudaq.sample(kernel, x_basis, shots_count=1024)
    # Cancellation leaves |+0>: q[1] stays zero and q[0] retains its phase.
    if x_basis:
        assert len(counts) == 1
        assert counts['00'] == 1024
    else:
        assert set(counts) == {'00', '10'}
        assert np.isclose(counts.probability('00'), 0.5, atol=0.08)
        assert np.isclose(counts.probability('10'), 0.5, atol=0.08)


@pytest.mark.parametrize(
    "entangled,x_basis,expected_outcomes", [(False, False, {'000', '001'}),
                                            (False, True, {'000'}),
                                            (True, False, {'000', '101'}),
                                            (True, True, {'000', '101'})],
    ids=["superposition-z", "superposition-x", "entangled-z", "entangled-x"])
def test_custabilizer_swap_preserves_frames(entangled, x_basis,
                                            expected_outcomes):
    """SWAP's overlapping CNOTs must move randomness and phase correlations."""

    @cudaq.kernel
    def kernel(entangled: bool, x_basis: bool):
        q = cudaq.qvector(3)
        h(q[0])
        if entangled:
            cx(q[0], q[1])
            swap(q[1], q[2])
            if x_basis:
                h(q[0])
                h(q[2])
        else:
            swap(q[0], q[2])
            if x_basis:
                h(q[2])
        mz(q)

    cudaq.set_random_seed(13)
    counts = cudaq.sample(kernel, entangled, x_basis, shots_count=1024)
    # q[1] remains zero; the superposition or Bell correlation moves to q[2].
    assert set(counts) == expected_outcomes
    for outcome in expected_outcomes:
        assert np.isclose(counts.probability(outcome),
                          1.0 / len(expected_outcomes),
                          atol=0.08)


@pytest.mark.parametrize(
    "operation,expected_outcome", [(0, '0'), (1, '0'), (2, '1'), (3, '0')],
    ids=["two-hadamards", "two-resets", "one-bit-flip", "two-bit-flips"])
def test_custabilizer_repeated_single_qubit_operations(operation,
                                                       expected_outcome):
    """Split repeated targets for gates, resets, and noise instructions."""

    @cudaq.kernel
    def kernel(operation: int):
        q = cudaq.qubit()
        if operation == 0:
            h(q)
            h(q)
        elif operation == 1:
            x(q)
            reset(q)
            reset(q)
        else:
            cudaq.apply_noise(cudaq.XError, 1.0, q)
            if operation == 3:
                cudaq.apply_noise(cudaq.XError, 1.0, q)
        mz(q)

    counts = cudaq.sample(kernel,
                          operation,
                          shots_count=256,
                          noise_model=cudaq.NoiseModel())
    assert len(counts) == 1
    assert counts[expected_outcome] == 256


def test_custabilizer_repeated_measurements_and_resets():
    """Repeated measurement targets obey cuStabilizer's disjoint-target limit."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qubit()
        h(q)
        mz(q)
        mz(q)
        reset(q)
        reset(q)
        mz(q)
        mz(q)

    cudaq.set_random_seed(13)
    counts = cudaq.sample(kernel, shots_count=1024, explicit_measurements=True)
    # Repeated measurements agree; after reset both measurements are zero.
    assert set(counts) == {'0000', '1100'}
    assert np.isclose(counts.probability('0000'), 0.5, atol=0.08)
    assert np.isclose(counts.probability('1100'), 0.5, atol=0.08)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
