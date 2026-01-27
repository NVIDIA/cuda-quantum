# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys
import pytest

import cudaq
import numpy as np

skipIfNvidiaFP64NotInstalled = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia-fp64')),
    reason='Could not find nvidia-fp64 in installation')

skipIfNvidiaNotInstalled = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia')),
    reason='Could not find nvidia in installation')


def swap(arr, index1, index2):
    t = arr[index2]
    arr[index2] = arr[index1]
    arr[index1] = t


# state preparation and synthesis


def test_kernel_complex_synthesize():
    cudaq.reset_target()

    c = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(vec)

    synthesized = cudaq.synthesize(kernel, c)
    counts = cudaq.sample(synthesized)
    assert '00' in counts
    assert '10' in counts


def test_kernel_float_synthesize():
    cudaq.reset_target()

    c = [1. / np.sqrt(2.), 1. / np.sqrt(2.), 0., 0.]

    @cudaq.kernel
    def kernel(vec: list[float]):
        q = cudaq.qvector(vec)

    synthesized = cudaq.synthesize(kernel, c)
    counts = cudaq.sample(synthesized)
    assert '00' in counts
    assert '10' in counts


# float


def test_kernel_float_params():
    cudaq.reset_target()

    f = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel(vec: list[float]):
        q = cudaq.qvector(vec)

    counts = cudaq.sample(kernel, f)
    print(counts)
    assert '11' in counts
    assert '00' in counts


def test_kernel_float_capture():
    cudaq.reset_target()

    f = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(f)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    swap(f, 1, 3)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '10' in counts
    assert '00' in counts


def test_kernel_float_np_array_from_capture():
    cudaq.reset_target()

    f = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(f)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    swap(f, 1, 3)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '10' in counts
    assert '00' in counts


def test_kernel_float_definition():
    cudaq.reset_target()

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)])

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# complex


def test_kernel_complex_params_rotate():
    cudaq.reset_target()

    c = [0. + 0j, 0., 0., 1.]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(vec)
        x(q.front())
        y(q.back())
        h(q)
        mz(q)

    counts = cudaq.sample(kernel, c)
    print(f'rotate: {counts}')
    assert '11' in counts
    assert '00' in counts
    assert '01' in counts
    assert '10' in counts


def test_kernel_complex_params():
    cudaq.reset_target()

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(vec)

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


def test_kernel_complex_capture():
    cudaq.reset_target()

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(c)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    swap(c, 1, 3)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '10' in counts
    assert '00' in counts


def test_kernel_complex_np_array_from_capture():
    cudaq.reset_target()

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(c)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    swap(c, 1, 3)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '10' in counts
    assert '00' in counts


def test_kernel_complex_definition():
    cudaq.reset_target()

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)])

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# np arrays


def test_kernel_dtype_complex_params():
    cudaq.reset_target()

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=complex))

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


def test_kernel_dtype_complex128_params():
    cudaq.reset_target()

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=np.complex128))

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_dtype_complex64_params_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=np.complex64))

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# simulation dtype


@skipIfNvidiaFP64NotInstalled
def test_kernel_simulation_dtype_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_simulation_dtype_complex_params_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


def test_kernel_amplitudes_complex_params():
    cudaq.reset_target()

    c = cudaq.amplitudes([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)])

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(vec)

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


def test_kernel_amplitudes_complex_from_capture():
    cudaq.reset_target()

    c = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(cudaq.amplitudes(c))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    swap(c, 1, 3)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '10' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_simulation_dtype_np_array_from_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia', option='fp64')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(np.array(c, dtype=cudaq.complex()))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    swap(c, 1, 3)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '10' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_simulation_dtype_np_array_from_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(np.array(c, dtype=cudaq.complex()))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    swap(c, 1, 3)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '10' in counts
    assert '00' in counts


def test_kernel_simulation_dtype_np_array_capture():
    cudaq.reset_target()

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]
    state = np.array(c, dtype=cudaq.complex())

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    swap(c, 1, 3)
    state = np.array(c, dtype=cudaq.complex())

    counts = cudaq.sample(kernel)
    print(counts)
    assert '10' in counts
    assert '00' in counts


# test errors


def test_kernel_error_invalid_array_size():
    cudaq.reset_target()

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel():
            qubits = cudaq.qvector(np.array([1., 0., 0.], dtype=complex))

        counts = cudaq.sample(kernel)
    assert 'Invalid input state size for qvector init (not a power of 2)' in repr(
        e)


def test_kernel_error_invalid_list_size():
    cudaq.reset_target()

    with pytest.raises(RuntimeError) as e:

        @cudaq.kernel
        def kernel():
            qubits = cudaq.qvector([1., 0., 0.])

        counts = cudaq.sample(kernel)
    assert 'Invalid input state size for qvector init (not a power of 2)' in repr(
        e)


def test_kernel_qvector_init_from_param_int():
    cudaq.reset_target()

    @cudaq.kernel
    def kernel(n: int):
        q = cudaq.qvector(n)

    counts = cudaq.sample(kernel, 2)
    print(counts)
    assert not '11' in counts
    assert not '10' in counts
    assert not '01' in counts
    assert '00' in counts


def test_kernel_qvector_init_from_capture_int():
    cudaq.reset_target()

    n = 2

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(n)

    counts = cudaq.sample(kernel)
    print(counts)
    assert not '11' in counts
    assert not '10' in counts
    assert not '01' in counts
    assert '00' in counts

    n = 1

    counts = cudaq.sample(kernel)
    print(counts)
    assert not '1' in counts
    assert '0' in counts


def test_kernel_qvector_init_from_int():
    cudaq.reset_target()

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)

    counts = cudaq.sample(kernel)
    print(counts)
    assert not '11' in counts
    assert not '10' in counts
    assert not '01' in counts
    assert '00' in counts
