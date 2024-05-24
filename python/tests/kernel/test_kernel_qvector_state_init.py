# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import pytest

import cudaq
import numpy as np

skipIfNvidiaFP64NotInstalled = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia-fp64')),
    reason='Could not find nvidia-fp64 in installation')

skipIfNvidiaNotInstalled = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia')),
    reason='Could not find nvidia in installation')

# float


@skipIfNvidiaFP64NotInstalled
def test_kernel_float_params_f64():

    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    f = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)], dtype=float)

    with pytest.raises(RuntimeError) as e:
        state = cudaq.State.from_data(f)
    assert 'A numpy array with only floating point elements passed to state.from_data.' in repr(
        e)


# works
@skipIfNvidiaFP64NotInstalled
def test_kernel_float_params_f32():

    cudaq.reset_target()
    cudaq.set_target('nvidia')

    f = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)], dtype=np.float32)

    with pytest.raises(RuntimeError) as e:
        state = cudaq.State.from_data(f)
    assert 'A numpy array with only floating point elements passed to state.from_data.' in repr(
        e)


# complex


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=complex)
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel(vec: cudaq.State):
        q = cudaq.qvector(vec)

    counts = cudaq.sample(kernel, state)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_complex_params_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=np.complex64)
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel(vec: cudaq.State):
        q = cudaq.qvector(vec)

    counts = cudaq.sample(kernel, state)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=complex)
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_complex_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=np.complex64)
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel2():
        q = cudaq.qvector(state)

    counts = cudaq.sample(kernel2)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# simulation dtype


@skipIfNvidiaFP64NotInstalled
def test_kernel_simulation_dtype_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=cudaq.complex())
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel(vec: cudaq.State):
        q = cudaq.qvector(vec)

    counts = cudaq.sample(kernel, state)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_simulation_dtype_complex_params_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=cudaq.complex())
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel(vec: cudaq.State):
        q = cudaq.qvector(vec)

    counts = cudaq.sample(kernel, state)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_init_from_other_kernel_state_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        cx(qubits[0], qubits[1])

    state = cudaq.get_state(bell)
    state.dump()

    @cudaq.kernel
    def kernel(initialState: cudaq.State):
        qubits = cudaq.qvector(initialState)

    state2 = cudaq.get_state(kernel, state)
    state2.dump()

    counts = cudaq.sample(kernel, state)
    print(counts)
    assert '11' in counts
    assert '00' in counts
    assert not '10' in counts
    assert not '01' in counts


@skipIfNvidiaFP64NotInstalled
def test_init_from_other_kernel_state_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        cx(qubits[0], qubits[1])

    state = cudaq.get_state(bell)
    state.dump()

    @cudaq.kernel
    def kernel(initialState: cudaq.State):
        qubits = cudaq.qvector(initialState)

    state2 = cudaq.get_state(kernel, state)
    state2.dump()

    counts = cudaq.sample(kernel, state)
    print(counts)
    assert '11' in counts
    assert '00' in counts
    assert not '10' in counts
    assert not '01' in counts
