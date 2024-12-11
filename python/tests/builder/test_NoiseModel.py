# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest
import numpy as np

import cudaq
import random


@pytest.mark.parametrize('target', ['density-matrix-cpu', 'stim'])
def test_depolarization_channel(target: str):
    """Tests the depolarization channel in the case of a non-zero probability."""
    cudaq.set_target(target)
    cudaq.set_random_seed(13)
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


@pytest.mark.parametrize('target', ['density-matrix-cpu', 'stim'])
def test_depolarization_channel_simple(target: str):
    """Tests the depolarization channel in the case of `probability = 1.0`"""
    cudaq.set_target(target)
    cudaq.set_random_seed(13)
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
    cudaq.reset_target()


def test_amplitude_damping_simple():
    """Tests the amplitude damping channel in the case of `probability = 1.0`"""
    cudaq.set_target('density-matrix-cpu')
    cudaq.set_random_seed(13)
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
    cudaq.reset_target()


@pytest.mark.parametrize('target', ['density-matrix-cpu', 'stim'])
def test_phase_flip_simple(target: str):
    """Tests the phase flip channel in the case of `probability = 1.0`"""
    cudaq.set_target(target)
    cudaq.set_random_seed(13)
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
    cudaq.reset_target()


@pytest.mark.parametrize('target', ['density-matrix-cpu', 'stim'])
def test_bit_flip_simple(target: str):
    """
    Tests the bit flip channel with the probability at `0.0` on qubit 0, 
    and `1.0` on qubit 1.
    """
    cudaq.set_target(target)
    cudaq.set_random_seed(13)
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
    cudaq.reset_target()


def test_kraus_channel():
    """Tests the Kraus Channel with a series of custom Kraus Operators."""
    cudaq.set_target('density-matrix-cpu')
    cudaq.set_random_seed(13)
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
    cudaq.reset_target()


def test_row_major():
    cudaq.set_target('density-matrix-cpu')
    cudaq.set_random_seed(13)
    # Amplitude damping
    error_prob = 0.2
    shots = 10000
    # Default numpy array is row major
    kraus_0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - error_prob)]],
                       dtype=np.complex128)
    kraus_1 = np.array([[0.0, np.sqrt(error_prob)], [0.0, 0.0]],
                       dtype=np.complex128)
    # This will throw if the row-column major convention is mixed up
    t1_channel = cudaq.KrausChannel([kraus_0, kraus_1])
    noise = cudaq.NoiseModel()
    noise.add_channel('x', [0], t1_channel)
    cudaq.set_noise(noise)
    circuit = cudaq.make_kernel()
    q = circuit.qalloc()
    circuit.x(q)
    noisy_counts = cudaq.sample(circuit, shots_count=shots)
    noisy_counts.dump()
    # Decay to |0> ~ error_prob
    assert np.isclose(noisy_counts.probability("0"), error_prob, atol=.2)
    cudaq.reset_target()


def test_column_major():
    cudaq.set_target('density-matrix-cpu')
    cudaq.set_random_seed(13)
    # Amplitude damping
    error_prob = 0.2
    shots = 10000
    # Input data in column major
    # Note: same data but with order = 'F' => the buffer storage will be in column major
    kraus_0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - error_prob)]],
                       dtype=np.complex128,
                       order='F')
    kraus_1 = np.array([[0.0, np.sqrt(error_prob)], [0.0, 0.0]],
                       dtype=np.complex128,
                       order='F')
    # This will throw if the row-column major convention is mixed up
    t1_channel = cudaq.KrausChannel([kraus_0, kraus_1])
    noise = cudaq.NoiseModel()
    noise.add_channel('x', [0], t1_channel)
    cudaq.set_noise(noise)
    circuit = cudaq.make_kernel()
    q = circuit.qalloc()
    circuit.x(q)
    noisy_counts = cudaq.sample(circuit, shots_count=shots)
    noisy_counts.dump()
    # Decay to |0> ~ error_prob
    assert np.isclose(noisy_counts.probability("0"), error_prob, atol=.2)
    cudaq.reset_target()


def test_noise_u3():
    cudaq.set_target('density-matrix-cpu')
    cudaq.set_random_seed(13)
    # Amplitude damping
    error_prob = 0.2
    shots = 10000
    kraus_0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1 - error_prob)]],
                       dtype=np.complex128)
    kraus_1 = np.array([[0.0, np.sqrt(error_prob)], [0.0, 0.0]],
                       dtype=np.complex128)
    # This will throw if the row-column major convention is mixed up
    t1_channel = cudaq.KrausChannel([kraus_0, kraus_1])
    noise = cudaq.NoiseModel()
    noise.add_channel('u3', [0], t1_channel)
    cudaq.set_noise(noise)
    circuit = cudaq.make_kernel()
    q = circuit.qalloc()
    # U3(pi,−pi/2,pi/2) == X
    circuit.u3(np.pi, -np.pi / 2, np.pi / 2, q)
    noisy_counts = cudaq.sample(circuit, shots_count=shots)
    noisy_counts.dump()
    # Decay to |0> ~ error_prob
    assert np.isclose(noisy_counts.probability("0"), error_prob, atol=.1)
    cudaq.reset_target()


@pytest.mark.parametrize('target', ['density-matrix-cpu', 'stim'])
def test_all_qubit_channel(target: str):
    cudaq.set_target(target)
    cudaq.set_random_seed(13)
    noise = cudaq.NoiseModel()
    bf = cudaq.BitFlipChannel(1.0)
    noise.add_all_qubit_channel('x', bf)
    kernel = cudaq.make_kernel()
    num_qubits = 3
    qubits = kernel.qalloc(num_qubits)
    kernel.x(qubits)
    kernel.mz(qubits)
    shots = 252
    noisy_counts = cudaq.sample(kernel, shots_count=shots, noise_model=noise)
    noisy_counts.dump()
    # Decay to |000>
    assert np.isclose(noisy_counts.probability("0" * num_qubits), 1.0)
    cudaq.reset_target()


def test_all_qubit_channel_with_control():
    cudaq.set_target('density-matrix-cpu')
    cudaq.set_random_seed(13)
    noise = cudaq.NoiseModel()
    k0 = np.array(
        [[0.99498743710662, 0., 0., 0.], [0., 0.99498743710662, 0., 0.],
         [0., 0., 0.99498743710662, 0.], [0., 0., 0., 0.99498743710662]],
        dtype=np.complex128)
    k1 = np.array(
        [[0., 0., 0.05773502691896258, 0.], [0., 0., 0., 0.05773502691896258],
         [0.05773502691896258, 0., 0., 0.], [0., 0.05773502691896258, 0., 0.]],
        dtype=np.complex128)
    k2 = np.array([[0., 0., -1j * 0.05773502691896258, 0.],
                   [0., 0., 0., -1j * 0.05773502691896258],
                   [1j * 0.05773502691896258, 0., 0., 0.],
                   [0., 1j * 0.05773502691896258, 0., 0.]],
                  dtype=np.complex128)
    k3 = np.array(
        [[0.05773502691896258, 0., 0., 0.], [0., 0.05773502691896258, 0., 0.],
         [0., 0., -0.05773502691896258, 0.], [0., 0., 0., -0.05773502691896258]
        ],
        dtype=np.complex128)
    kraus_channel = cudaq.KrausChannel([k0, k1, k2, k3])
    noise.add_all_qubit_channel('x', kraus_channel, num_controls=1)
    num_qubits = 5
    num_tests = 4
    for i in range(num_tests):
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(num_qubits)
        # Pick a qubit pair
        qubit_pair = random.sample(range(num_qubits), 2)
        print(f"qubit pair: {qubit_pair}")
        q = qubits[qubit_pair[0]]
        r = qubits[qubit_pair[1]]
        kernel.h(q)
        kernel.cx(q, r)
        kernel.mz(qubits)
        shots = 1024
        noisy_counts = cudaq.sample(kernel,
                                    shots_count=shots,
                                    noise_model=noise)
        noisy_counts.dump()
        # All tests have some noisy states beside the bell pair.
        assert (len(noisy_counts) > 2)
    cudaq.reset_target()


def test_all_qubit_channel_with_control_prefix():
    cudaq.set_target('density-matrix-cpu')
    cudaq.set_random_seed(13)
    noise = cudaq.NoiseModel()
    k0 = np.array(
        [[0.99498743710662, 0., 0., 0.], [0., 0.99498743710662, 0., 0.],
         [0., 0., 0.99498743710662, 0.], [0., 0., 0., 0.99498743710662]],
        dtype=np.complex128)
    k1 = np.array(
        [[0., 0., 0.05773502691896258, 0.], [0., 0., 0., 0.05773502691896258],
         [0.05773502691896258, 0., 0., 0.], [0., 0.05773502691896258, 0., 0.]],
        dtype=np.complex128)
    k2 = np.array([[0., 0., -1j * 0.05773502691896258, 0.],
                   [0., 0., 0., -1j * 0.05773502691896258],
                   [1j * 0.05773502691896258, 0., 0., 0.],
                   [0., 1j * 0.05773502691896258, 0., 0.]],
                  dtype=np.complex128)
    k3 = np.array(
        [[0.05773502691896258, 0., 0., 0.], [0., 0.05773502691896258, 0., 0.],
         [0., 0., -0.05773502691896258, 0.], [0., 0., 0., -0.05773502691896258]
        ],
        dtype=np.complex128)
    kraus_channel = cudaq.KrausChannel([k0, k1, k2, k3])
    noise.add_all_qubit_channel('cx', kraus_channel)
    num_qubits = 5
    num_tests = 4
    for i in range(num_tests):
        kernel = cudaq.make_kernel()
        qubits = kernel.qalloc(num_qubits)
        # Pick a qubit pair
        qubit_pair = random.sample(range(num_qubits), 2)
        print(f"qubit pair: {qubit_pair}")
        q = qubits[qubit_pair[0]]
        r = qubits[qubit_pair[1]]
        kernel.h(q)
        kernel.cx(q, r)
        kernel.mz(qubits)
        shots = 1024
        noisy_counts = cudaq.sample(kernel,
                                    shots_count=shots,
                                    noise_model=noise)
        noisy_counts.dump()
        # All tests have some noisy states beside the bell pair.
        assert (len(noisy_counts) > 2)
    cudaq.reset_target()


def test_callback_channel():
    cudaq.set_target('density-matrix-cpu')
    cudaq.set_random_seed(13)

    def noise_cb(qubits, params):
        if qubits[0] != 2:
            return cudaq.BitFlipChannel(1.0)
        return cudaq.KrausChannel()

    noise = cudaq.NoiseModel()
    noise.add_channel('x', noise_cb)
    kernel = cudaq.make_kernel()
    num_qubits = 5
    qubits = kernel.qalloc(num_qubits)
    kernel.x(qubits)
    kernel.mz(qubits)
    shots = 252
    noisy_counts = cudaq.sample(kernel, shots_count=shots, noise_model=noise)
    noisy_counts.dump()
    # All qubits, except q[2], are flipped.
    assert np.isclose(noisy_counts.probability("00100"), 1.0)
    cudaq.reset_target()


def test_callback_channel_with_params():
    cudaq.set_target('density-matrix-cpu')
    cudaq.set_random_seed(13)

    def noise_cb(qubits, params):
        assert len(params) == 1
        # For testing: only add noise if the angle is positive.
        if params[0] > 0:
            return cudaq.BitFlipChannel(1.0)
        return cudaq.KrausChannel()

    noise = cudaq.NoiseModel()
    noise.add_channel('rx', noise_cb)
    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    # Rx(pi) == X
    kernel.rx(np.pi, qubit)
    kernel.mz(qubit)
    shots = 252
    noisy_counts = cudaq.sample(kernel, shots_count=shots, noise_model=noise)
    noisy_counts.dump()
    # Due to 100% bit-flip, it becomes "0".
    assert np.isclose(noisy_counts.probability("0"), 1.0)

    kernel = cudaq.make_kernel()
    qubit = kernel.qalloc()
    # Rx(-pi) == X
    kernel.rx(-np.pi, qubit)
    kernel.mz(qubit)
    shots = 252
    noisy_counts = cudaq.sample(kernel, shots_count=shots, noise_model=noise)
    noisy_counts.dump()
    # Due to our custom setup, a negative angle will have no noise.
    assert np.isclose(noisy_counts.probability("1"), 1.0)
    cudaq.reset_target()


def check_custom_op_noise(noise_model):
    cudaq.set_target('density-matrix-cpu')
    cudaq.set_random_seed(13)

    @cudaq.kernel
    def basic():
        q = cudaq.qubit()
        custom_x(q)

    shots = 100
    counts = cudaq.sample(basic, shots_count=shots, noise_model=noise_model)
    counts.dump()
    assert np.isclose(counts.probability("0"), 1.0)
    cudaq.reset_target()


def test_custom_op():
    cudaq.register_operation("custom_x", np.array([0, 1, 1, 0]))

    # (Gate name + Operand)
    noise = cudaq.NoiseModel()
    # Bit flip channel with `1.0` probability of the qubit flipping 180 degrees.
    bit_flip_one = cudaq.BitFlipChannel(1.0)
    noise.add_channel('custom_x', [0], bit_flip_one)
    check_custom_op_noise(noise)

    # All-qubit
    noise = cudaq.NoiseModel()
    # Bit flip channel with `1.0` probability of the qubit flipping 180 degrees.
    noise.add_all_qubit_channel('custom_x', bit_flip_one)
    check_custom_op_noise(noise)

    # Callback
    def noise_cb(qubits, params):
        return bit_flip_one

    noise = cudaq.NoiseModel()
    noise.add_channel('custom_x', noise_cb)
    check_custom_op_noise(noise)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
