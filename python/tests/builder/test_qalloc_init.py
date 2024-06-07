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

    f = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]

    kernel, vec = cudaq.make_kernel(list[float])
    qubits = kernel.qalloc(vec)

    counts = cudaq.sample(kernel, f)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_float_params_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    f = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]

    kernel, vec = cudaq.make_kernel(list[float])
    qubits = kernel.qalloc(vec)

    counts = cudaq.sample(kernel, f)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_float_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    f = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(f)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_float_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    f = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(f)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_float_np_array_from_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    f = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(np.array(f))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_float_np_array_from_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    f = [1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)]

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(np.array(f))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_float_definition_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)])

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_float_definition_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)])

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# complex


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex_params_rotate_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [0. + 0j, 0., 0., 1.]

    kernel, vec = cudaq.make_kernel(list[complex])
    q = kernel.qalloc(vec)
    kernel.x(q[0])
    kernel.y(q[1])
    kernel.h(q)
    kernel.mz(q)

    counts = cudaq.sample(kernel, c)
    print(f'rotate: {counts}')
    assert '11' in counts
    assert '00' in counts
    assert '01' in counts
    assert '10' in counts


@skipIfNvidiaNotInstalled
def test_kernel_complex_params_rotate_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = [0. + 0j, 0., 0., 1.]

    kernel, vec = cudaq.make_kernel(list[np.complex64])
    q = kernel.qalloc(vec)
    kernel.x(q[0])
    kernel.y(q[1])
    kernel.h(q)
    kernel.mz(q)

    counts = cudaq.sample(kernel, c)
    print(f'rotate: {counts}')
    assert '11' in counts
    assert '00' in counts
    assert '01' in counts
    assert '10' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    kernel, vec = cudaq.make_kernel(list[complex])
    q = kernel.qalloc(vec)

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_complex_params_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    kernel, vec = cudaq.make_kernel(list[np.complex64])
    q = kernel.qalloc(vec)

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(c)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_complex_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(c)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex_np_array_from_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(np.array(c))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_complex_np_array_from_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(np.array(c))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_complex_definition_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    kernel = cudaq.make_kernel()
    q = kernel.qalloc([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)])

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_complex_definition_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    kernel = cudaq.make_kernel()
    q = kernel.qalloc([1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)])

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# np arrays


@skipIfNvidiaFP64NotInstalled
def test_kernel_dtype_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]
    a = np.array(c, dtype=complex)

    kernel, vec = cudaq.make_kernel(list[complex])
    q = kernel.qalloc(vec)

    counts = cudaq.sample(kernel, a)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_dtype_complex128_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]
    a = np.array(c, dtype=np.complex128)

    kernel, vec = cudaq.make_kernel(list[np.complex128])
    q = kernel.qalloc(vec)

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# simulation dtype


@skipIfNvidiaNotInstalled
def test_kernel_amplitudes_complex_params_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = cudaq.amplitudes([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)])

    kernel, vec = cudaq.make_kernel(list[np.complex64])
    q = kernel.qalloc(vec)

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_simulation_dtype_np_array_from_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(np.array(c, dtype=cudaq.complex()))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_simulation_dtype_np_array_from_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(np.array(c, dtype=cudaq.complex()))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaFP64NotInstalled
def test_kernel_simulation_dtype_np_array_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    state = np.array(c, dtype=cudaq.complex())

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


@skipIfNvidiaNotInstalled
def test_kernel_simulation_dtype_np_array_capture_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    c = [1. / np.sqrt(2.) + 0j, 0., 0., 1. / np.sqrt(2.)]

    state = np.array(c, dtype=cudaq.complex())

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# test errors

# invalid initializer size


@skipIfNvidiaFP64NotInstalled
def test_kernel_error_invalid_array_size_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    kernel = cudaq.make_kernel()

    with pytest.raises(RuntimeError) as e:
        q = kernel.qalloc(np.array([1., 0., 0.], dtype=complex))
    assert 'invalid input state size for qalloc (not a power of 2)' in repr(e)


@skipIfNvidiaFP64NotInstalled
def test_kernel_error_invalid_list_size_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    kernel = cudaq.make_kernel()

    with pytest.raises(RuntimeError) as e:
        qubits = kernel.qalloc([1.j, 0., 0.])
    assert 'invalid input state size for qalloc (not a power of 2)' in repr(e)


@skipIfNvidiaNotInstalled
def test_kernel_error_invalid_array_size_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    kernel = cudaq.make_kernel()

    with pytest.raises(RuntimeError) as e:
        qubits = kernel.qalloc(np.array([1.j, 0., 0.], dtype=complex))
    assert 'invalid input state size for qalloc (not a power of 2)' in repr(e)


@skipIfNvidiaNotInstalled
def test_kernel_error_invalid_list_size_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    kernel = cudaq.make_kernel()

    with pytest.raises(RuntimeError) as e:
        qubits = kernel.qalloc([1.j, 0., 0.])
    assert 'invalid input state size for qalloc (not a power of 2)' in repr(e)


# initializer is not normalized


@skipIfNvidiaFP64NotInstalled
def test_kernel_error_np_array_not_normalized_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    kernel = cudaq.make_kernel()

    with pytest.raises(RuntimeError) as e:
        q = kernel.qalloc(np.array([1., 0., 0., 1.], dtype=complex))
    assert 'invalid input state for qalloc (not normalized)' in repr(e)


@skipIfNvidiaFP64NotInstalled
def test_kernel_error_list_not_normalized_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    kernel = cudaq.make_kernel()

    with pytest.raises(RuntimeError) as e:
        qubits = kernel.qalloc([1.j, 0., 0., 1.])
    assert 'invalid input state for qalloc (not normalized)' in repr(e)


@skipIfNvidiaNotInstalled
def test_kernel_error_np_array_not_normalized_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    kernel = cudaq.make_kernel()

    with pytest.raises(RuntimeError) as e:
        qubits = kernel.qalloc(np.array([1.j, 0., 0., 1.], dtype=complex))
    assert 'invalid input state for qalloc (not normalized)' in repr(e)


@skipIfNvidiaNotInstalled
def test_kernel_error_list_not_normalized_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    kernel = cudaq.make_kernel()

    with pytest.raises(RuntimeError) as e:
        qubits = kernel.qalloc([1.j, 0., 0., 1.])
    assert 'invalid input state for qalloc (not normalized)' in repr(e)


# invalid initializer


@skipIfNvidiaFP64NotInstalled
def test_kernel_error_invalid_initializer_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    kernel = cudaq.make_kernel()

    with pytest.raises(RuntimeError) as e:
        qubits = kernel.qalloc("hello")
    assert 'invalid initializer argument for qalloc: hello' in repr(e)


@skipIfNvidiaNotInstalled
def test_kernel_error_invalid_initializer_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    kernel = cudaq.make_kernel()

    with pytest.raises(RuntimeError) as e:
        qubits = kernel.qalloc("hello")
    assert 'invalid initializer argument for qalloc: hello' in repr(e)


# qalloc(int)


def test_kernel_qvector_init_from_param_int():

    kernel, n = cudaq.make_kernel(int)
    q = kernel.qalloc(n)

    counts = cudaq.sample(kernel, 2)
    print(counts)
    assert not '11' in counts
    assert not '10' in counts
    assert not '01' in counts
    assert '00' in counts


def test_kernel_qvector_init_from_capture_int():
    n = 2

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(n)

    counts = cudaq.sample(kernel)
    print(counts)
    assert not '11' in counts
    assert not '10' in counts
    assert not '01' in counts
    assert '00' in counts


def test_kernel_qvector_init_from_int():

    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)

    counts = cudaq.sample(kernel)
    print(counts)
    assert not '11' in counts
    assert not '10' in counts
    assert not '01' in counts
    assert '00' in counts
