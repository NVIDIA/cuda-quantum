# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np

import cudaq


def test_depolarization_channel():
    """Tests the depolarization channel in the case of a non-zero probability."""
    cudaq.set_random_seed(13)
    cudaq.set_target('density-matrix-cpu')
    circuit = cudaq.make_kernel()
    q = circuit.qalloc()
    circuit.x(q)

    depol = cudaq.DepolarizationChannel(.1)
    noise = cudaq.NoiseModel()
    noise.add_channel("x", [0], depol)

    counts = cudaq.sample(circuit, noise_model=noise, shots_count=100)
    assert (len(counts) == 2)
    assert ('0' in counts)
    assert ('1' in counts)
    assert (counts.count('0') + counts.count('1') == 100)

    counts = cudaq.sample(circuit)
    assert (len(counts) == 1)
    assert ('1' in counts)

    cudaq.reset_target()
    counts = cudaq.sample(circuit)
    assert (len(counts) == 1)
    assert ('1' in counts)


def test_depolarization_channel_simple():
    """Tests the depolarization channel in the case of `probability = 1.0`"""
    cudaq.set_random_seed(13)
    cudaq.set_target('density-matrix-cpu')
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    noise = cudaq.NoiseModel()

    # Depolarization channel with `1.0` probability of the qubit state
    # being scrambled.
    depolarization_channel = cudaq.DepolarizationChannel(1.0)
    # Channel applied to any Y-gate on the depolarization channel.
    noise.add_channel('y', [0], depolarization_channel)

    # Bring the qubit to the |1> state, where it will remain
    # with a probability of `1 - p = 0.0`.
    kernel.y(qubit)
    kernel.mz(qubit)

    # Without noise, the qubit should still be in the |1> state.
    counts = cudaq.sample(kernel)
    want_counts = 1000
    got_counts = counts["1"]
    assert got_counts == want_counts

    # With noise, the measurements should be a roughly 50/50
    # mix between the |0> and |1> states.
    noisy_counts = cudaq.sample(kernel, noise_model=noise)
    want_probability = 0.5
    got_zero_probability = noisy_counts.probability("0")
    got_one_probability = noisy_counts.probability("1")
    assert np.isclose(got_zero_probability, want_probability, atol=.2)
    assert np.isclose(got_one_probability, want_probability, atol=.2)


def test_amplitude_damping_simple():
    """Tests the amplitude damping channel in the case of `probability = 1.0`"""
    cudaq.set_random_seed(13)
    cudaq.set_target('density-matrix-cpu')
    noise = cudaq.NoiseModel()
    # Amplitude damping channel with `1.0` probability of the qubit
    # decaying to the ground state.
    amplitude_damping = cudaq.AmplitudeDampingChannel(1.0)
    # Applied to any Hadamard gate on the qubit.
    noise.add_channel('h', [0], amplitude_damping)

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    # This will bring qubit to `1/sqrt(2) (|0> + |1>)`, where it will remain
    # with a probability of `1 - p = 0.0`.
    kernel.h(qubit)
    kernel.mz(qubit)

    # Without noise, the qubit will now have a 50/50 mix of measurements
    # between |0> and |1>.
    counts = cudaq.sample(kernel)
    want_probability = 0.5
    got_zero_probability = counts.probability("0")
    got_one_probability = counts.probability("1")
    assert np.isclose(got_zero_probability, want_probability, atol=.1)
    assert np.isclose(got_one_probability, want_probability, atol=.1)

    # With noise, all measurements should be in the |0> state,
    noisy_counts = cudaq.sample(kernel, noise_model=noise)
    want_counts = 1000
    got_counts = noisy_counts["0"]
    assert (got_counts == want_counts)


def test_phase_flip_simple():
    """Tests the phase flip channel in the case of `probability = 1.0`"""
    cudaq.set_random_seed(13)
    cudaq.set_target('density-matrix-cpu')
    noise = cudaq.NoiseModel()
    # Phase flip channel with `1.0` probability of the qubit
    # undergoing a phase rotation of 180 degrees (π).
    phase_flip = cudaq.PhaseFlipChannel(1.0)
    noise.add_channel('z', [0], phase_flip)

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()

    # Place qubit in superposition state.
    kernel.h(qubit)
    # Rotate the phase around Z by 180 degrees (π).
    kernel.z(qubit)
    # Apply another hadamard and measure.
    kernel.h(qubit)
    kernel.mz(qubit)

    # Without noise, we'd expect the qubit to end in the |1>
    # state due to the phase rotation between the two hadamard
    # gates.
    counts = cudaq.sample(kernel)
    want_counts = 1000
    got_one_counts = counts["1"]
    assert got_one_counts == want_counts

    # With noise, should be in the |0> state.
    noisy_counts = cudaq.sample(kernel, noise_model=noise)
    got_zero_counts = noisy_counts["0"]
    assert got_zero_counts == want_counts


def test_bit_flip_simple():
    """
    Tests the bit flip channel with the probability at `0.0` on qubit 0, 
    and `1.0` on qubit 1.
    """
    cudaq.set_random_seed(13)
    cudaq.set_target('density-matrix-cpu')
    noise = cudaq.NoiseModel()
    # Bit flip channel with `0.0` probability of the qubit flipping 180 degrees.
    bit_flip_zero = cudaq.BitFlipChannel(0.0)
    noise.add_channel('x', [0], bit_flip_zero)
    # Bit flip channel with `1.0` probability of the qubit flipping 180 degrees.
    bit_flip_one = cudaq.BitFlipChannel(1.0)
    noise.add_channel('x', [1], bit_flip_one)

    # Now we may define our simple kernel function and allocate a register
    # of qubits to it.
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    # This will bring the qubit to the |1> state.
    # Remains with a probability of `1 - p = 1.0`.
    kernel.x(qubits[0])
    # Now we apply an X-gate to qubit 1.
    # Remains in the |1> state with a probability of `1 - p = 0.0`.
    kernel.x(qubits[1])
    kernel.mz(qubits)

    # Without noise, both qubits in the |1> state.
    counts = cudaq.sample(kernel)
    counts.dump()
    want_counts = 1000
    got_one_one_counts = counts["11"]
    assert got_one_one_counts == want_counts

    # With noise, the state should be |1>|0> == |10>
    noisy_counts = cudaq.sample(kernel, noise_model=noise)
    noisy_counts.dump()
    got_one_zero_counts = noisy_counts["10"]
    assert got_one_zero_counts == want_counts


def test_kraus_channel():
    """Tests the Kraus Channel with a series of custom Kraus Operators."""
    cudaq.set_random_seed(13)
    cudaq.set_target('density-matrix-cpu')
    k0 = np.array([[0.05773502691896258, 0.0], [0., -0.05773502691896258]],
                  dtype=np.complex128)
    k1 = np.array([[0., 0.05773502691896258], [0.05773502691896258, 0.]],
                  dtype=np.complex128)
    k2 = np.array([[0., -0.05773502691896258j], [0.05773502691896258j, 0.]],
                  dtype=np.complex128)
    k3 = np.array([[0.99498743710662, 0.0], [0., 0.99498743710662]],
                  dtype=np.complex128)

    depolarization = cudaq.KrausChannel([k0, k1, k2, k3])

    assert ((depolarization[0] == k0).all())
    assert ((depolarization[1] == k1).all())
    assert ((depolarization[2] == k2).all())
    assert ((depolarization[3] == k3).all())

    noise = cudaq.NoiseModel()
    noise.add_channel('x', [0], depolarization)
    cudaq.set_noise(noise)
    circuit = cudaq.make_kernel()
    q = circuit.qalloc()
    circuit.x(q)

    counts = cudaq.sample(circuit)
    want_count_length = 2
    got_count_length = len(counts)
    assert (got_count_length == want_count_length)
    assert ('0' in counts)
    assert ('1' in counts)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
