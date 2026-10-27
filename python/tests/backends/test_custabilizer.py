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


def test_custabilizer_noise_probability_precision():
    """Nearby noise probabilities must remain distinct during GPU sampling."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(256)
        x(q)
        mz(q)

    def sample_with_probability(probability):
        cudaq.set_random_seed(13)
        noise = cudaq.NoiseModel()
        noise.add_all_qubit_channel("x", cudaq.BitFlipChannel(probability))
        counts = cudaq.sample(kernel, noise_model=noise, shots_count=250_000)
        return dict(counts.items())

    # These probabilities serialize identically at six significant digits.
    # Use the same seed so that rounding makes the two samples identical.
    # 64 million noise applications make the small difference observable
    # when the original probability is preserved.
    nearly_certain = sample_with_probability(0.9999996)
    certain = sample_with_probability(1.0)
    assert nearly_certain != certain, "Noise probabilities below 1 were rounded to 1"


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


@pytest.mark.parametrize("shots", [1, 31, 32, 33, 65, 129])
def test_custabilizer_measurement_layout(shots):
    """Keep chronological measurements across resets and packed-shot boundaries."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        x(q[2])
        first = mz(q[2])
        reset(q[2])
        x(q[1])
        mz(q[1])
        mz(q[0])
        mz(q[2])

    cudaq.set_random_seed(13)
    result = cudaq.sample(kernel, shots_count=shots, explicit_measurements=True)
    measurements = result.get_sequential_data()
    assert len(measurements) == shots
    # Qubit-ordered or reversed layouts would give '0110' or '0011'.
    assert measurements == ['1100'] * shots


def test_custabilizer_named_register_correlations():
    """Named registers must preserve Bell correlations for each shot."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        h(q[0])
        cx(q[0], q[1])
        x(q[1])
        mz(q[0], register_name="first")
        mz(q[1], register_name="second")

    cudaq.set_random_seed(13)
    result = cudaq.sample(kernel, shots_count=1025)
    first = result.get_sequential_data('first')
    second = result.get_sequential_data('second')
    assert len(first) == len(second) == 1025
    # Anti-correlation also catches both names resolving to one register.
    assert all(a != b for a, b in zip(first, second))
    assert set(first) == {'0', '1'}


@pytest.mark.parametrize("late_allocation", [True, False],
                         ids=["allocate-later", "use-later"])
def test_custabilizer_measurement_correlations_after_qubit_growth(
        late_allocation):
    """Preserve earlier shots when allocation or use grows from 128 to 129 qubits."""

    @cudaq.kernel
    def allocate_later():
        q = cudaq.qvector(128)
        h(q[127])
        mz(q[127], register_name="early")
        x(q[127])  # Flush the measurement and invert its Z value.
        extra = cudaq.qubit()
        x(extra)
        mz(q[127], register_name="late")

    @cudaq.kernel
    def use_later():
        q = cudaq.qvector(129)
        h(q[127])
        mz(q[127], register_name="early")
        x(q[127])
        x(q[128])
        mz(q[127], register_name="late")

    kernel = allocate_later if late_allocation else use_later
    cudaq.set_random_seed(123)
    result = cudaq.sample(kernel, shots_count=32768)
    early = result.get_sequential_data("early")
    late = result.get_sequential_data("late")
    # Growing the state must preserve each shot's earlier random measurement.
    assert len(early) == len(late) == 32768
    assert all(a != b for a, b in zip(early, late))
    assert set(early) == {"0", "1"}


def test_custabilizer_seed_reproducibility_and_reallocation():
    """Reseeding remains reproducible while qubit and shot counts grow or shrink."""

    @cudaq.kernel
    def kernel(n: int):
        q = cudaq.qvector(n)
        h(q[0])
        for i in range(1, n):
            cx(q[i - 1], q[i])
        mz(q)

    prior_samples = {}
    for n, shots in [(3, 65), (7, 129), (2, 31), (3, 65)]:
        cudaq.set_random_seed(1234)
        first = cudaq.sample(kernel, n, shots_count=shots).get_sequential_data()
        cudaq.set_random_seed(1234)
        second = cudaq.sample(kernel, n,
                              shots_count=shots).get_sequential_data()
        assert len(first) == len(second) == shots
        assert first == second
        assert set(first) == {'0' * n, '1' * n}
        # Returning to the same dimensions must reproduce the original shots.
        if (n, shots) in prior_samples:
            assert first == prior_samples[n, shots]
        prior_samples[n, shots] = first
    # A different seed must change the shots, so the seed is not ignored.
    cudaq.set_random_seed(4321)
    reseeded = cudaq.sample(kernel, 3, shots_count=65).get_sequential_data()
    assert reseeded != prior_samples[3, 65]


def test_custabilizer_clifford_gate_phase_parity():
    """Preserve phases through S adjoint, controlled-Y, and SWAP conversion."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        h(q[0])
        s(q[0])
        s.adj(q[0])
        h(q[0])
        # CY on a |-i> target kicks back Z; CX or CZ would randomize q[1].
        h(q[1])
        h(q[2])
        s.adj(q[2])
        y.ctrl(q[1], q[2])
        s(q[2])
        h(q[2])
        h(q[1])
        swap(q[0], q[1])
        mz(q)

    cudaq.set_random_seed(13)
    result = cudaq.sample(kernel, shots_count=65)
    assert result.get_sequential_data() == ['100'] * 65


def test_custabilizer_deterministic_pauli_noise():
    """Broadcast probability-one bit flips cancel the initial X gates."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        x(q)
        mz(q)

    noise = cudaq.NoiseModel()
    noise.add_all_qubit_channel('x', cudaq.BitFlipChannel(1.0))
    cudaq.set_random_seed(13)
    result = cudaq.sample(kernel, shots_count=65, noise_model=noise)
    assert result.get_sequential_data() == ['00'] * 65


@pytest.mark.parametrize("x_basis", [False, True], ids=["Z", "X"])
@pytest.mark.parametrize("channel,expected", [
    (cudaq.XError(1.0), '00'),
    (cudaq.YError(1.0), '01'),
    (cudaq.ZError(1.0), '11'),
    (cudaq.PhaseFlipChannel(1.0), '11'),
    (cudaq.Pauli1([1.0, 0.0, 0.0]), '00'),
    (cudaq.Pauli1([0.0, 1.0, 0.0]), '01'),
    (cudaq.Pauli1([0.0, 0.0, 1.0]), '11'),
])
def test_custabilizer_single_qubit_pauli_channels(channel, expected, x_basis):
    """Check X- and Z-frame effects of each single-qubit Pauli channel."""

    @cudaq.kernel
    def kernel(x_basis: bool):
        q = cudaq.qubit()
        if x_basis:
            h(q)
        x(q)
        if x_basis:
            h(q)
        mz(q)

    noise = cudaq.NoiseModel()
    noise.add_channel('x', [0], channel)
    cudaq.set_random_seed(13)
    result = cudaq.sample(kernel, x_basis, noise_model=noise, shots_count=65)
    # `expected` holds the Z- and X-basis outcomes; without noise it is '10'.
    assert result.get_sequential_data() == [expected[x_basis]] * 65


@pytest.mark.parametrize("x_basis", [False, True], ids=["Z", "X"])
@pytest.mark.parametrize(
    "paulis", [a + b for a in 'IXYZ' for b in 'IXYZ' if a + b != 'II'])
def test_custabilizer_two_qubit_pauli_channels(paulis, x_basis):
    """Check the target order and outcome of all 15 two-qubit Pauli errors."""

    @cudaq.kernel
    def kernel(x_basis: bool):
        q = cudaq.qvector(2)
        if x_basis:
            h(q)
        cx(q[0], q[1])
        if x_basis:
            h(q)
        mz(q)

    probabilities = [0.0] * 15
    probabilities[4 * 'IXYZ'.index(paulis[0]) + 'IXYZ'.index(paulis[1]) -
                  1] = 1.0
    noise = cudaq.NoiseModel()
    noise.add_channel('x', [0, 1], cudaq.Pauli2(probabilities))
    flips = 'ZY' if x_basis else 'XY'
    expected = ''.join('1' if p in flips else '0' for p in paulis)
    cudaq.set_random_seed(13)
    result = cudaq.sample(kernel, x_basis, noise_model=noise, shots_count=65)
    assert result.get_sequential_data() == [expected] * 65


def test_custabilizer_depolarizing_statistics():
    """Two-qubit depolarization follows the specified Pauli probabilities."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        cx(q[0], q[1])
        mz(q)

    noise = cudaq.NoiseModel()
    noise.add_channel('x', [0, 1], cudaq.Depolarization2(0.3))
    cudaq.set_random_seed(42)
    result = cudaq.sample(kernel, noise_model=noise, shots_count=20001)
    for outcome, expected in [('00', 0.76), ('01', 0.08), ('10', 0.08),
                              ('11', 0.08)]:
        assert abs(result.probability(outcome) - expected) < 0.02


def test_custabilizer_qec_measurement_contract():
    """QEC annotations preserve the raw measurements from repeated checks."""

    @cudaq.kernel
    def memory(rounds: int):
        q = cudaq.qvector(2)
        x(q[0])
        previous = mz(q[1])
        for _ in range(rounds):
            cx(q[0], q[1])
            current = mz(q[1])
            cudaq.detector(current, previous)
            previous = current
            reset(q[1])
        final = mz(q[0])
        cudaq.logical_observable(final)

    cudaq.set_random_seed(13)
    samples = cudaq.sample(memory,
                           3,
                           shots_count=65,
                           explicit_measurements=True)
    measurements = samples.get_sequential_data()
    # Without the reset, the second check would read '0'.
    assert measurements == ['01111'] * 65


@cudaq.kernel
def custabilizer_parity_checks(data: cudaq.qview, ancillas: cudaq.qview,
                               x_basis: bool) -> List[cudaq.measure_handle]:
    if x_basis:
        h(ancillas)
        for i in range(2):
            cx(ancillas[i], data[i])
            cx(ancillas[i], data[i + 1])
        h(ancillas)
    else:
        for i in range(2):
            cx(data[i], ancillas[i])
            cx(data[i + 1], ancillas[i])
    results = mz(ancillas)
    reset(ancillas)
    return results


@cudaq.kernel
def custabilizer_repeated_checks(x_basis: bool, logical_one: bool,
                                 fault_qubit: int, probability: float):
    data = cudaq.qvector(3)
    ancillas = cudaq.qvector(2)
    if logical_one:
        x(data)
    if x_basis:
        h(data)
    previous = custabilizer_parity_checks(data, ancillas, x_basis)
    cudaq.detector(previous[0])
    cudaq.detector(previous[1])
    if fault_qubit >= 0:
        if x_basis:
            cudaq.apply_noise(cudaq.ZError, probability, data[fault_qubit])
        else:
            cudaq.apply_noise(cudaq.XError, probability, data[fault_qubit])
    for _ in range(2):
        current = custabilizer_parity_checks(data, ancillas, x_basis)
        cudaq.detectors(previous, current)
        previous = current
    if x_basis:
        h(data)
    final = mz(data)
    cudaq.detector(previous[0], final[0], final[1])
    cudaq.detector(previous[1], final[1], final[2])
    cudaq.logical_observable(final[0])


@pytest.mark.parametrize("x_basis", [False, True], ids=["Z", "X"])
@pytest.mark.parametrize("logical_one", [False, True])
@pytest.mark.parametrize("shots", [1, 31, 32, 33, 129])
def test_custabilizer_qec_repeated_checks_raw_measurement_layout(
        x_basis, logical_one, shots):
    """Reset/reuse and QEC annotations preserve chronological measurements."""
    expected_bits = "000000" + ("111" if logical_one else "000")
    cudaq.set_random_seed(13)
    result = cudaq.sample(custabilizer_repeated_checks,
                          x_basis,
                          logical_one,
                          -1,
                          0.0,
                          shots_count=shots,
                          explicit_measurements=True)
    assert result.get_sequential_data() == [expected_bits] * shots


@pytest.mark.parametrize("x_basis", [False, True], ids=["Z", "X"])
@pytest.mark.parametrize("fault_qubit", [0, 1, 2])
def test_custabilizer_qec_faults_preserve_measurement_history(
        x_basis, fault_qubit):
    """A persistent fault affects subsequent checks and final data bits."""
    parities = [int(fault_qubit in (0, 1)), int(fault_qubit in (1, 2))]
    data = [int(i == fault_qubit) for i in range(3)]
    expected_bits = "00" + "".join(map(str, parities * 2 + data))
    cudaq.set_random_seed(13)
    result = cudaq.sample(custabilizer_repeated_checks,
                          x_basis,
                          False,
                          fault_qubit,
                          1.0,
                          noise_model=cudaq.NoiseModel(),
                          shots_count=65,
                          explicit_measurements=True)
    assert result.get_sequential_data() == [expected_bits] * 65


@pytest.mark.parametrize("x_basis", [False, True], ids=["Z", "X"])
def test_custabilizer_qec_noisy_samples_preserve_fault_correlations(x_basis):
    """The check history and final logical bit share the same sampled fault."""
    probability, shots = 0.125, 4097
    cudaq.set_random_seed(987)
    result = cudaq.sample(custabilizer_repeated_checks,
                          x_basis,
                          False,
                          0,
                          probability,
                          noise_model=cudaq.NoiseModel(),
                          shots_count=shots,
                          explicit_measurements=True)
    assert set(result) == {"000000000", "001010100"}
    measurements = np.array(
        [list(map(int, bits)) for bits in result.get_sequential_data()],
        dtype=np.uint8)
    assert measurements.shape == (shots, 9)
    first_check_change = measurements[:, 0] ^ measurements[:, 2]
    logical_bits = measurements[:, 6]
    np.testing.assert_array_equal(first_check_change, logical_bits)
    assert np.mean(logical_bits) == pytest.approx(
        probability, abs=6 * np.sqrt(probability * (1 - probability) / shots))


def test_custabilizer_qec_observable_annotations_preserve_raw_measurements():
    """Detector and indexed observable annotations must not add measurement bits."""

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        x(q[0])
        x(q[2])
        m = mz(q)
        cudaq.detector(m[0], m[0])
        cudaq.detector(m[0], m[1], m[1])
        cudaq.logical_observable(m[0], observable_index=2)
        cudaq.logical_observable(m[1], m[2], observable_index=2)

    cudaq.set_random_seed(13)
    result = cudaq.sample(kernel, shots_count=33, explicit_measurements=True)
    assert result.get_sequential_data() == ["101"] * 33


def test_custabilizer_qec_noisy_conditional_feedback():
    """A measured fault drives feedback while reset clears the measured qubit."""

    @cudaq.kernel
    def feedback() -> List[bool]:
        q = cudaq.qvector(2)
        cudaq.apply_noise(cudaq.XError, 1.0, q[0])
        outcome = mz(q[0])
        reset(q[0])
        if outcome:
            x(q[1])
        return mz(q)

    cudaq.set_random_seed(13)
    results = cudaq.run(feedback,
                        noise_model=cudaq.NoiseModel(),
                        shots_count=33)
    assert len(results) == 33
    assert all(list(bits) == [False, True] for bits in results)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
