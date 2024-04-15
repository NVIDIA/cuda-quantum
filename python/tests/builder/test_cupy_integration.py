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

cp = pytest.importorskip('cupy')


def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance


def test_state_vector_simple_py_float():
    # Test overlap with device state vector
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])

    # State is on the GPU, this is nvidia target
    state = cudaq.get_state(kernel)
    # Create a state on GPU
    expected = cp.array([.707107, 0, 0, .707107])

    # We are using the nvidia-fp32 target by default, it requires
    # cupy overlaps to also have complex f32 data types, this
    # should throw since we are using float data types only
    with pytest.raises(RuntimeError) as error:
        result = state.overlap(expected)


def test_state_vector_simple_cfp32():
    # Test overlap with device state vector
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])

    # State is on the GPU, this is nvidia target
    state = cudaq.get_state(kernel)
    state.dump()
    # Create a state on GPU
    expected = cp.array([.707107, 0, 0, .707107], dtype=cp.complex64)
    # Compute the overlap
    result = state.overlap(expected)
    assert np.isclose(result, 1.0, atol=1e-3)


def test_state_vector_simple_cfp64():
    cudaq.set_target('nvidia-fp64')
    # Test overlap with device state vector
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])

    # State is on the GPU, this is nvidia target
    state = cudaq.get_state(kernel)
    state.dump()
    # Create a state on GPU
    expected = cp.array([.707107, 0, 0, .707107], dtype=cp.complex128)
    # Compute the overlap
    result = state.overlap(expected)
    assert np.isclose(result, 1.0, atol=1e-3)
    cudaq.reset_target()


def test_state_vector_to_cupy():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])

    # State is on the GPU, this is nvidia target
    state = cudaq.get_state(kernel)
    state.dump()

    # Convert our cudaq.State to a list of CuPy Tensors
    stateInCuPy = cudaq.to_cupy(state)[0]
    print(stateInCuPy)
    assert np.isclose(stateInCuPy[0], 1. / np.sqrt(2.), atol=1e-3)
    assert np.isclose(stateInCuPy[3], 1. / np.sqrt(2.), atol=1e-3)
    assert np.isclose(stateInCuPy[1], 0., atol=1e-3)
    assert np.isclose(stateInCuPy[2], 0., atol=1e-3)


def test_cupy_to_state():
    cp_data = cp.array([.707107, 0, 0, .707107], dtype=cp.complex64)
    state_from_cupy = cudaq.State.from_data(cp_data)
    state_from_cupy.dump()
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    kernel.h(q[0])
    kernel.cx(q[0], q[1])
    # State is on the GPU, this is nvidia target
    state = cudaq.get_state(kernel)
    result = state.overlap(state_from_cupy)
    assert np.isclose(result, 1.0, atol=1e-3)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
