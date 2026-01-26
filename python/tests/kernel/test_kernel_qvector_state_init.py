# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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

# synthesis


def test_kernel_synthesis_complex():
    cudaq.reset_target()

    c = np.array([1. / np.sqrt(2.) + 0j, 1. / np.sqrt(2.), 0., 0.],
                 dtype=cudaq.complex())
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel(vec: cudaq.State):
        q = cudaq.qvector(vec)

    counts = cudaq.sample(kernel, state)
    assert '00' in counts
    assert '10' in counts
    assert len(counts) == 2

    synthesized = cudaq.synthesize(kernel, state)
    counts = cudaq.sample(synthesized)
    assert '00' in counts
    assert '10' in counts
    assert len(counts) == 2


# float


@skipIfNvidiaFP64NotInstalled
def test_kernel_float_params_f64():

    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

    f = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)], dtype=float)

    with pytest.raises(RuntimeError) as e:
        state = cudaq.State.from_data(f)
    assert 'A numpy array with only floating point elements passed to `state.from_data`.' in repr(
        e)


@skipIfNvidiaFP64NotInstalled
def test_kernel_float_params_f32():

    cudaq.reset_target()
    cudaq.set_target('nvidia')

    f = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)], dtype=np.float32)

    with pytest.raises(RuntimeError) as e:
        state = cudaq.State.from_data(f)
    assert 'A numpy array with only floating point elements passed to `state.from_data`.' in repr(
        e)


# complex


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

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


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex128_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=np.complex128)
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel(vec: cudaq.State):
        q = cudaq.qvector(vec)

    counts = cudaq.sample(kernel, state)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex64_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=np.complex64)

    with pytest.raises(RuntimeError) as e:
        state = cudaq.State.from_data(c)
    assert '[sim-state] invalid data precision.' in repr(e)


@skipIfNvidiaNotInstalled
def test_kernel_complex64_params_f32():
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


@skipIfNvidiaNotInstalled
def test_kernel_complex128_params_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=np.complex128)

    with pytest.raises(RuntimeError) as e:
        state = cudaq.State.from_data(c)
    assert '[sim-state] invalid data precision.' in repr(e)


@skipIfNvidiaNotInstalled
def test_kernel_complex_params_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=complex)

    with pytest.raises(RuntimeError) as e:
        state = cudaq.State.from_data(c)
    assert '[sim-state] invalid data precision.' in repr(e)


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

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


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex128_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=np.complex128)
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex128_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=np.complex64)

    with pytest.raises(RuntimeError) as e:
        state = cudaq.State.from_data(c)
    assert '[sim-state] invalid data precision.' in repr(e)


@skipIfNvidiaNotInstalled
def test_kernel_complex64_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=np.complex64)
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_complex128_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=np.complex128)

    with pytest.raises(RuntimeError) as e:
        state = cudaq.State.from_data(c)
    assert '[sim-state] invalid data precision.' in repr(e)


@skipIfNvidiaNotInstalled
def test_kernel_complex_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=complex)

    with pytest.raises(RuntimeError) as e:
        state = cudaq.State.from_data(c)
    assert '[sim-state] invalid data precision.' in repr(e)


# simulation dtype


@skipIfNvidiaFP64NotInstalled
def test_kernel_simulation_dtype_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

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
def test_kernel_simulation_dtype_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=cudaq.complex())
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_simulation_dtype_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=cudaq.complex())
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# Initializing from state from another kernel


@skipIfNvidiaFP64NotInstalled
def test_init_from_other_kernel_state_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

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
    cudaq.StateMemoryView(state2).dump()

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


def test_inner_kernels_state():
    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=cudaq.complex())
    state = cudaq.State.from_data(c)

    @cudaq.kernel
    def kernel0():

        @cudaq.kernel
        def kernel1():
            q1 = cudaq.qvector(state)

        kernel1()

        @cudaq.kernel
        def kernel2():
            q2 = cudaq.qvector(state)

        kernel2()

    counts = cudaq.sample(kernel0)
    print(counts)
    assert '1111' in counts
    assert '1100' in counts
    assert '0011' in counts
    assert '0000' in counts


def test_invalid_arg_error_msg():
    cudaq.reset_target()

    c = np.array([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)],
                 dtype=complex)

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel(vec: cudaq.State):
            q = cudaq.qvector(vec)

        counts = cudaq.sample(kernel, c)
    assert 'Invalid runtime argument type.' in repr(e)
