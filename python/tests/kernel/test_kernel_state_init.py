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

# vector of float

# "qvector initialization data must be of complex dtype."
# def test_kernel_float_params():

#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     f = [0., 1., 1., 0.]

#     @cudaq.kernel
#     def kernel(vec : list[float]):
#         q = cudaq.qvector(vec)

#     counts = cudaq.sample(kernel, f)
#     print(counts)
#     assert '11' in counts
#     assert '00' in counts

# RuntimeError: qvector initialization data must be of complex dtype.
# def test_kernel_float_capture():
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     f = [0., 1., 1., 0.]

#     @cudaq.kernel
#     def kernel():
#         q = cudaq.qvector(f)

#     cudaq.sample(kernel).dump()

# # RuntimeError: qvector initialization data must be of complex dtype.
# def test_kernel_float_np_array_from_capture():
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     f = [0., 1., 1., 0.]

#     @cudaq.kernel
#     def kernel():
#         q = cudaq.qvector(np.array(f))

#     cudaq.sample(kernel).dump()

# vector of complex


def test_kernel_complex_params_rotate():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [0. + 0j, 0., 0., 1.]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(vec)
        # Can now operate on the qvector as usual:
        # Rotate state of the front qubit 180 degrees along X.
        x(q.front())
        # Rotate state of the back qubit 180 degrees along Y.
        y(q.back())
        # Put qubits into superposition state.
        h(q)

        # Measure.
        mz(q)

    counts = cudaq.sample(kernel, c)
    print(f'rotate: {counts}')
    assert '11' in counts
    assert '00' in counts
    assert '01' in counts
    assert '10' in counts


# no-copy
def test_kernel_complex_params():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(vec)

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# no-copy
def test_kernel_complex_capture():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(c)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# no-copy
def test_kernel_complex_np_array_from_capture():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel(verbose=True)
    def kernel():
        q = cudaq.qvector(np.array(c))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# no-copy
def test_kernel_complex_definition():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector([1.0 + 0j, 0., 0., 1.])

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# np arrays with various dtypes


# no-copy
def test_kernel_dtype_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel(verbose=True)
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=complex))

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# no-copy
def test_kernel_dtype_complex128_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=np.complex128))

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# copy
def test_kernel_dtype_complex64_params_f32():
    cudaq.reset_target()

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=np.complex64))

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# Needs https://github.com/NVIDIA/cuda-quantum/pull/1610
# def test_kernel_dtype_complex_complex128_params_f64():
#     cudaq.reset_target()
#     cudaq.set_target('nvidia-fp64')

#     c = [.70710678 + 0j, 0., 0., 0.70710678]
#     arr = np.array(c, dtype=complex)
#     print(arr.data)

#     @cudaq.kernel
#     def kernel(vec : np.ndarray):
#         q = cudaq.qvector(np.array(vec, dtype=np.complex128))

#     # RuntimeError: error: Invalid runtime argument type. Argument of type list[complex] was provided, but list[float] was expected.
#     cudaq.sample(kernel, arr).dump()

# simulation scalar


# no-copy
def test_kernel_simulation_dtype_complex_params_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# copy
def test_kernel_simulation_dtype_complex_params_f32():
    cudaq.reset_target()

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel(vec: list[complex]):
        q = cudaq.qvector(np.array(vec, dtype=cudaq.complex()))

    counts = cudaq.sample(kernel, c)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# needs https://github.com/NVIDIA/cuda-quantum/pull/1610
# def test_kernel_simulation_dtype_np_array_params():
#     cudaq.reset_target()

#     c = [.70710678 + 0j, 0., 0., 0.70710678]

#     @cudaq.kernel(verbose=True)
#     def kernel(vec : np.array):
#         q = cudaq.qvector(vec)

#     cudaq.sample(kernel, np.array(c, dtype=cudaq.complex())).dump()


# no-copy
def test_kernel_simulation_dtype_np_array_from_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel(verbose=True)
    def kernel():
        q = cudaq.qvector(np.array(c, dtype=cudaq.complex()))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# copy
def test_kernel_simulation_dtype_np_array_from_capture_f32():
    cudaq.reset_target()

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel(verbose=True)
    def kernel():
        q = cudaq.qvector(np.array(c, dtype=cudaq.complex()))

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# no-copy
def test_kernel_simulation_dtype_np_array_capture_f64():
    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    state = np.array(c, dtype=cudaq.complex())

    @cudaq.kernel(verbose=True)
    def kernel():
        q = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# no-copy
def test_kernel_simulation_dtype_np_array_capture_f32():
    cudaq.reset_target()

    c = [.70710678 + 0j, 0., 0., 0.70710678]

    state = np.array(c, dtype=cudaq.complex())

    @cudaq.kernel(verbose=True)
    def kernel():
        q = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts


# tests from builder, updated for kernel defs

skipIfNvidiaFP64NotInstalled = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia-fp64')),
    reason='Could not find nvidia-fp64 in installation')


@skipIfNvidiaFP64NotInstalled
def test_from_state_fp64():
    cudaq.set_target('nvidia-fp64')

    @cudaq.kernel
    def kernel(initState: list[complex]):
        qubits = cudaq.qvector(initState)

    # needs https://github.com/NVIDIA/cuda-quantum/pull/1610
    # # Test float64 list, casts to complex
    # state = [.70710678, 0., 0., 0.70710678]
    # counts = cudaq.sample(kernel, state)
    # print(counts)
    # assert '11' in counts
    # assert '00' in counts

    # Test complex list
    state = [.70710678j, 0., 0., 0.70710678]
    counts = cudaq.sample(kernel, state)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    # needs https://github.com/NVIDIA/cuda-quantum/pull/1610
    # # Test Numpy array of floats
    # state = np.asarray([.70710678, 0., 0., 0.70710678])
    # counts = cudaq.sample(kernel, state)
    # print(counts)
    # assert '11' in counts
    # assert '00' in counts

    # Test Numpy array of complex
    state = np.asarray([.70710678j, 0., 0., 0.70710678])
    counts = cudaq.sample(kernel, state)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    # Now test constant array data, not kernel input
    state = np.array([.70710678, 0., 0., 0.70710678], dtype=complex)

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    state = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    state = np.array([.70710678, 0., 0., 0.70710678])

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(state)

    #with pytest.raises(RuntimeError) as e:
    # float data and not complex data
    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    # state = np.array([.70710678, 0., 0., 0.70710678], dtype=np.complex64)

    # TODO: wrong simulator precision
    # @cudaq.kernel
    # def kernel():
    #     qubits = cudaq.qvector(state)

    # #with pytest.raises(RuntimeError) as e:
    #     # Wrong precision for fp64 simulator
    # counts = cudaq.sample(kernel)
    # print(counts)
    # assert '11' in counts
    # assert '00' in counts

    # error: 'quake.alloca' op size operand required
    # @cudaq.kernel
    # def kernel():
    #     qubits = cudaq.qvector(np.array([1., 0., 0.], dtype=complex))

    # #with pytest.raises(RuntimeError) as e:
    # counts = cudaq.sample(kernel)
    # print(counts)
    # assert '11' in counts
    # assert '00' in counts


skipIfNvidiaNotInstalled = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia')),
    reason='Could not find nvidia in installation')


@skipIfNvidiaNotInstalled
def test_from_state_f32():
    cudaq.reset_target()
    cudaq.set_target('nvidia')

    state = np.array([.70710678, 0., 0., 0.70710678], dtype=np.complex128)

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(state)

    with pytest.raises(RuntimeError) as e:
        counts = cudaq.sample(kernel)

    state = np.array([.70710678, 0., 0., 0.70710678], dtype=np.complex64)

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    cudaq.reset_target()

    # Regardless of the target precision, use
    # cudaq.complex() or cudaq.amplitudes()
    state = np.array([.70710678, 0., 0., 0.70710678], dtype=cudaq.complex())

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    state = cudaq.amplitudes([.70710678, 0., 0., 0.70710678])

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts

    state = cudaq.amplitudes(np.array([.5] * 4))

    @cudaq.kernel
    def kernel():
        qubits = cudaq.qvector(state)

    counts = cudaq.sample(kernel)
    print(counts)
    assert '11' in counts
    assert '00' in counts
    assert '01' in counts
    assert '10' in counts

    # needs https://github.com/NVIDIA/cuda-quantum/pull/1610
    # @cudaq.kernel
    # def kernel(initState: list[np.complex64]):
    #     qubits = cudaq.qvector(initState)

    # state = cudaq.amplitudes([.70710678, 0., 0., 0.70710678])
    # counts = cudaq.sample(kernel, state)
    # print(counts)
    # assert '11' in counts
    # assert '00' in counts
    # cudaq.reset_target()
