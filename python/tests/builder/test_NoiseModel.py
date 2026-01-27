# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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


def test_apply_noise_custom():
    cudaq.set_target('density-matrix-cpu')

    class CustomNoiseChannelBad(cudaq.KrausChannel):
        # NEEDS num_parameters member, but it is missing, so this is Bad
        def __init__(self, params: list[float]):
            cudaq.KrausChannel.__init__(self)
            # Example: Create Kraus ops based on params
            p = params[0]
            k0 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]],
                          dtype=np.complex128)
            k1 = np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]],
                          dtype=np.complex128)

            # Create KrausOperators and add to channel
            self.append(cudaq.KrausOperator(k0))
            self.append(cudaq.KrausOperator(k1))

            # Set noise type for Stim integration
            self.noise_type = cudaq.NoiseModelType.Unknown

    class CustomNoiseChannel(cudaq.KrausChannel):
        num_parameters = 1
        num_targets = 1

        def __init__(self, params: list[float]):
            cudaq.KrausChannel.__init__(self)
            # Example: Create Kraus ops based on params
            p = params[0]
            k0 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]],
                          dtype=np.complex128)
            k1 = np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]],
                          dtype=np.complex128)

            # Create KrausOperators and add to channel
            self.append(cudaq.KrausOperator(k0))
            self.append(cudaq.KrausOperator(k1))

            # Set noise type for Stim integration
            self.noise_type = cudaq.NoiseModelType.Unknown

    class CustomNoiseChannelTwoParams(cudaq.KrausChannel):
        num_parameters = 2
        num_targets = 1

        def __init__(self, params: list[float]):
            cudaq.KrausChannel.__init__(self)
            # Example: Create Kraus ops based on params
            p = params[0]
            q = params[1]
            k0 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]],
                          dtype=np.complex128)
            k1 = np.array([[0, np.sqrt(q)], [np.sqrt(q), 0]],
                          dtype=np.complex128)

            # Create KrausOperators and add to channel
            self.append(cudaq.KrausOperator(k0))
            self.append(cudaq.KrausOperator(k1))

            # Set noise type for Stim integration
            self.noise_type = cudaq.NoiseModelType.Unknown

    noise = cudaq.NoiseModel()
    noise.register_channel(CustomNoiseChannel)
    noise.register_channel(CustomNoiseChannelBad)

    @cudaq.kernel
    def test():
        q = cudaq.qubit()
        x(q)
        # can pass as vector of params
        cudaq.apply_noise(CustomNoiseChannel, [0.1], q)

    counts = cudaq.sample(test)
    assert len(counts) == 1 and '1' in counts

    counts = cudaq.sample(test, noise_model=noise)
    assert len(counts) == 2 and '0' in counts and '1' in counts

    @cudaq.kernel
    def test():
        q = cudaq.qubit()
        x(q)
        # can pass as standard arguments
        cudaq.apply_noise(CustomNoiseChannel, 0.1, q)

    counts = cudaq.sample(test)
    assert len(counts) == 1 and '1' in counts

    counts = cudaq.sample(test, noise_model=noise)
    assert len(counts) == 2 and '0' in counts and '1' in counts

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def testbad():
            q = cudaq.qubit()
            x(q)
            # can pass as standard arguments
            cudaq.apply_noise(CustomNoiseChannelBad, 0.1, q)

        cudaq.sample(testbad)

    @cudaq.kernel
    def test():
        q = cudaq.qubit()
        x(q)
        # can pass as standard arguments
        cudaq.apply_noise(CustomNoiseChannelTwoParams, 0.1, 0.2, q)

    noise.register_channel(CustomNoiseChannelTwoParams)

    counts = cudaq.sample(test, noise_model=noise)
    counts.dump()
    assert len(counts) == 2 and '0' in counts and '1' in counts

    cudaq.reset_target()


@pytest.mark.parametrize('target', ['density-matrix-cpu', 'stim'])
def test_apply_noise_builtin(target: str):
    cudaq.set_target(target)

    noise = cudaq.NoiseModel()

    # Test builtin channels
    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(3)
        cudaq.apply_noise(cudaq.DepolarizationChannel, 0.1, q[0])
        mz(q)

    counts = cudaq.sample(kernel, noise_model=noise)
    print(counts)
    assert len(counts) == 2 and '000' in counts and '100' in counts

    @cudaq.kernel
    def bell_depol2(d: float, flag: bool):
        q, r = cudaq.qubit(), cudaq.qubit()
        h(q)
        x.ctrl(q, r)
        if flag:
            cudaq.apply_noise(cudaq.Depolarization2, d, q, r)
        else:
            cudaq.apply_noise(cudaq.Depolarization2, [d], q, r)

    counts = cudaq.sample(bell_depol2, 0.2, True, noise_model=noise)
    assert len(counts) == 4
    print(counts)

    counts = cudaq.sample(bell_depol2, 0.2, False, noise_model=noise)
    assert len(counts) == 4
    print(counts)

    @cudaq.kernel
    def bell_x():
        q, r = cudaq.qubit(), cudaq.qubit()
        h(q)
        x.ctrl(q, r)
        cudaq.apply_noise(cudaq.XError, 0.1, q)
        cudaq.apply_noise(cudaq.XError, 0.1, r)

    @cudaq.kernel
    def bell_y():
        q, r = cudaq.qubit(), cudaq.qubit()
        h(q)
        x.ctrl(q, r)
        cudaq.apply_noise(cudaq.YError, 0.1, q)
        cudaq.apply_noise(cudaq.YError, 0.1, r)

    @cudaq.kernel
    def test_z():
        q = cudaq.qvector(2)
        h(q)
        cudaq.apply_noise(cudaq.ZError, 0.1, q[0])
        cudaq.apply_noise(cudaq.ZError, 0.1, q[1])
        h(q)
        mz(q)

    counts = cudaq.sample(bell_x, noise_model=noise)
    assert len(counts) == 4
    print(counts)
    counts = cudaq.sample(bell_y, noise_model=noise)
    assert len(counts) == 4
    print(counts)
    counts = cudaq.sample(test_z, noise_model=noise)
    assert len(counts) == 4
    print(counts)

    @cudaq.kernel
    def pauli1_test():
        q, r = cudaq.qubit(), cudaq.qubit()
        h(q)
        x.ctrl(q, r)
        cudaq.apply_noise(cudaq.Pauli1, 0.1, 0.1, 0.1, q)
        cudaq.apply_noise(cudaq.Pauli1, 0.1, 0.1, 0.1, r)

    counts = cudaq.sample(pauli1_test, noise_model=noise)
    assert len(counts) == 4
    print(counts)

    @cudaq.kernel
    def pauli2_test():
        q, r = cudaq.qubit(), cudaq.qubit()
        h(q)
        x.ctrl(q, r)
        cudaq.apply_noise(cudaq.Pauli2, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                          0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                          q, r)

    counts = cudaq.sample(pauli2_test, noise_model=noise)
    assert len(counts) == 4
    print(counts)

    cudaq.reset_target()


@pytest.mark.parametrize('target', ['density-matrix-cpu'])
def test_noise_observe_reset(target: str):
    cudaq.set_target(target)
    noise_model = cudaq.NoiseModel()
    # Amplitude damping channel with `1.0` probability of the qubit
    # decaying to the ground state.
    amplitude_damping = cudaq.AmplitudeDampingChannel(1.0)

    noise_model.add_all_qubit_channel('x', amplitude_damping)

    test_x = cudaq.make_kernel()
    qubit = test_x.qalloc(1)
    test_x.x(qubit)

    observable = cudaq.spin.z(0)
    for i in range(10):
        # Run this in a loop to check that noise model argument is isolated to each observe call.
        result_noiseless = cudaq.observe(test_x, observable)
        result_noisy = cudaq.observe(test_x,
                                     observable,
                                     noise_model=noise_model)
        assert np.isclose(result_noiseless.expectation(), -1.)
        assert np.isclose(result_noisy.expectation(), 1.)


@pytest.mark.parametrize('target', ['density-matrix-cpu'])
def test_get_channel(target: str):
    cudaq.set_target(target)
    noise_model = cudaq.NoiseModel()
    # Amplitude damping channel with `1.0` probability of the qubit
    # decaying to the ground state.
    amplitude_damping = cudaq.AmplitudeDampingChannel(1.0)

    noise_model.add_all_qubit_channel('x', amplitude_damping)

    # Get the channel from the noise model for a specific gate and qubit
    for iq in range(5):
        channels = noise_model.get_channels('x', [iq])
        assert len(channels) == 1
        channel = channels[0]
        assert channel.noise_type == cudaq.NoiseModelType.AmplitudeDampingChannel
        assert len(channel.parameters) == 1
        assert channel.parameters[0] == 1.0

    noise_model.add_all_qubit_channel('x', amplitude_damping)
    for iq in range(5):
        channels = noise_model.get_channels('x', [iq])
        assert len(channels) == 2
        for channel in channels:
            assert channel.noise_type == cudaq.NoiseModelType.AmplitudeDampingChannel
            assert len(channel.parameters) == 1
            assert channel.parameters[0] == 1.0


@pytest.mark.parametrize('target', ['density-matrix-cpu'])
def test_get_channel_with_control(target: str):
    cudaq.set_target(target)
    noise_model = cudaq.NoiseModel()
    # Amplitude damping channel with `1.0` probability of the qubit
    # decaying to the ground state.
    depol2 = cudaq.Depolarization2(0.2)

    noise_model.add_all_qubit_channel('x', depol2, num_controls=1)

    # Get the channel from the noise model for a specific gate and adjacent qubit pairs
    for iq in range(5):
        channels = noise_model.get_channels('x', [iq], [iq + 1])
        assert len(channels) == 1
        channel = channels[0]
        assert channel.noise_type == cudaq.NoiseModelType.Depolarization2
        assert len(channel.parameters) == 1
        assert channel.parameters[0] == 0.2

    # Check syntactic sugar for all-qubit channel with control
    noise_model.add_all_qubit_channel('cx', depol2)
    for iq in range(5):
        channels = noise_model.get_channels('x', [iq], [iq + 1])
        assert len(channels) == 2
        for channel in channels:
            assert channel.noise_type == cudaq.NoiseModelType.Depolarization2
            assert len(channel.parameters) == 1
            assert channel.parameters[0] == 0.2


def test_builder_apply_noise_custom():
    cudaq.set_target('density-matrix-cpu')

    class CustomNoiseChannel(cudaq.KrausChannel):
        num_parameters = 1
        num_targets = 1

        def __init__(self, params: list[float]):
            cudaq.KrausChannel.__init__(self)
            # Example: Create Kraus ops based on params
            p = params[0]
            k0 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]],
                          dtype=np.complex128)
            k1 = np.array([[0, np.sqrt(p)], [np.sqrt(p), 0]],
                          dtype=np.complex128)

            # Create KrausOperators and add to channel
            self.append(cudaq.KrausOperator(k0))
            self.append(cudaq.KrausOperator(k1))

            # Set noise type for Stim integration
            self.noise_type = cudaq.NoiseModelType.Unknown

    class CustomNoiseChannelTwoParams(cudaq.KrausChannel):
        num_parameters = 2
        num_targets = 1

        def __init__(self, params: list[float]):
            cudaq.KrausChannel.__init__(self)
            # Example: Create Kraus ops based on params
            p = params[0]
            q = params[1]
            k0 = np.array([[np.sqrt(1 - p), 0], [0, np.sqrt(1 - p)]],
                          dtype=np.complex128)
            k1 = np.array([[0, np.sqrt(q)], [np.sqrt(q), 0]],
                          dtype=np.complex128)

            # Create KrausOperators and add to channel
            self.append(cudaq.KrausOperator(k0))
            self.append(cudaq.KrausOperator(k1))

            # Set noise type for Stim integration
            self.noise_type = cudaq.NoiseModelType.Unknown

    noise = cudaq.NoiseModel()
    noise.register_channel(CustomNoiseChannel)

    test = cudaq.make_kernel()
    q = test.qalloc()
    test.x(q)
    # can pass as vector of params
    test.apply_noise(CustomNoiseChannel, [0.1], q)

    counts = cudaq.sample(test)
    counts.dump()
    assert len(counts) == 1 and '1' in counts

    counts = cudaq.sample(test, noise_model=noise)
    counts.dump()
    assert len(counts) == 2 and '0' in counts and '1' in counts

    test = cudaq.make_kernel()
    q = test.qalloc()
    test.x(q)
    # can pass as standard arguments
    test.apply_noise(CustomNoiseChannel, 0.1, q)

    counts = cudaq.sample(test)
    counts.dump()
    assert len(counts) == 1 and '1' in counts

    counts = cudaq.sample(test, noise_model=noise)
    counts.dump()
    assert len(counts) == 2 and '0' in counts and '1' in counts

    test = cudaq.make_kernel()
    q = test.qalloc()
    test.x(q)
    # can pass as standard arguments
    test.apply_noise(CustomNoiseChannelTwoParams, 0.1, 0.2, q)

    noise.register_channel(CustomNoiseChannelTwoParams)

    counts = cudaq.sample(test, noise_model=noise)
    counts.dump()
    assert len(counts) == 2 and '0' in counts and '1' in counts

    cudaq.reset_target()


@pytest.mark.parametrize('target', ['density-matrix-cpu', 'stim'])
def test_builder_apply_noise_builtin(target: str):
    cudaq.set_target(target)

    noise = cudaq.NoiseModel()

    # Test builtin channels
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    kernel.apply_noise(cudaq.DepolarizationChannel, 0.1, q[0])
    kernel.mz(q)

    counts = cudaq.sample(kernel, noise_model=noise)
    print(counts)
    assert len(counts) == 2 and '000' in counts and '100' in counts

    bell_depol2, d = cudaq.make_kernel(float)
    q, r = bell_depol2.qalloc(), bell_depol2.qalloc()
    bell_depol2.h(q)
    bell_depol2.cx(q, r)
    bell_depol2.apply_noise(cudaq.Depolarization2, d, q, r)

    counts = cudaq.sample(bell_depol2, 0.2, noise_model=noise)
    print(counts)
    assert len(counts) == 4

    bell_x = cudaq.make_kernel()
    q, r = bell_x.qalloc(), bell_x.qalloc()
    bell_x.h(q)
    bell_x.cx(q, r)
    bell_x.apply_noise(cudaq.XError, 0.1, q)
    bell_x.apply_noise(cudaq.XError, 0.1, r)

    bell_y = cudaq.make_kernel()
    q, r = bell_y.qalloc(), bell_y.qalloc()
    bell_y.h(q)
    bell_y.cx(q, r)
    bell_y.apply_noise(cudaq.YError, 0.1, q)
    bell_y.apply_noise(cudaq.YError, 0.1, r)

    test_z = cudaq.make_kernel()
    q = test_z.qalloc(2)
    test_z.h(q)
    test_z.apply_noise(cudaq.ZError, 0.1, q[0])
    test_z.apply_noise(cudaq.ZError, 0.1, q[1])
    test_z.h(q)
    test_z.mz(q)

    counts = cudaq.sample(bell_x, noise_model=noise)
    assert len(counts) == 4
    print(counts)
    counts = cudaq.sample(bell_y, noise_model=noise)
    assert len(counts) == 4
    print(counts)
    counts = cudaq.sample(test_z, noise_model=noise)
    assert len(counts) == 4
    print(counts)

    pauli1_test = cudaq.make_kernel()
    q, r = pauli1_test.qalloc(), pauli1_test.qalloc()
    pauli1_test.h(q)
    pauli1_test.cx(q, r)
    pauli1_test.apply_noise(cudaq.Pauli1, 0.1, 0.1, 0.1, q)
    pauli1_test.apply_noise(cudaq.Pauli1, 0.1, 0.1, 0.1, r)

    counts = cudaq.sample(pauli1_test, noise_model=noise)
    assert len(counts) == 4
    print(counts)

    pauli2_test = cudaq.make_kernel()
    q, r = pauli2_test.qalloc(), pauli2_test.qalloc()
    pauli2_test.h(q)
    pauli2_test.cx(q, r)
    pauli2_test.apply_noise(cudaq.Pauli2, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                            0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                            0.02, q, r)

    counts = cudaq.sample(pauli2_test, noise_model=noise)
    assert len(counts) == 4
    print(counts)

    cudaq.reset_target()


def test_builder_apply_noise_inplace():
    cudaq.set_target("density-matrix-cpu")
    cudaq.set_random_seed(13)

    def kraus_mats(error_probability):

        kraus_0 = np.sqrt(1 - error_probability) * np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

        kraus_1 = np.sqrt(error_probability) * np.array(
            [[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        return [kraus_0, kraus_1]

    test = cudaq.make_kernel()
    q, r = test.qalloc(), test.qalloc()
    test.x(q)
    test.x(r)
    test.apply_noise(cudaq.KrausChannel(kraus_mats(0.5)), q)
    test.apply_noise(cudaq.KrausChannel(kraus_mats(0.25)), r)
    counts = cudaq.sample(test,
                          noise_model=cudaq.NoiseModel(),
                          shots_count=10000)
    counts.dump()
    assert np.isclose(counts.probability("00"), 0.5 * 0.25,
                      atol=1e-2)  # both decay
    assert np.isclose(counts.probability("11"), 0.5 * 0.75,
                      atol=1e-2)  # both stay
    assert np.isclose(counts.probability("10"), 0.5 * 0.25,
                      atol=1e-2)  # q stays, r decays
    assert np.isclose(counts.probability("01"), 0.5 * 0.75,
                      atol=1e-2)  # q decays, r stays
    cudaq.reset_target()


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
