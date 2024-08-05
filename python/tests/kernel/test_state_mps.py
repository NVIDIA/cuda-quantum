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
import sys

import cudaq

cp = pytest.importorskip('cupy')

## [PYTHON_VERSION_FIX]
skipIfPythonLessThan39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="built-in collection types such as `list` not supported")

skipIfNoGPU = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('tensornet-mps')),
    reason="tensornet-mps backend not available")


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


@skipIfNoGPU
def test_state_from_mps_simple():
    cudaq.set_target('tensornet-mps')
    tensor_q1 = np.array([1., 0.], dtype=np.complex128).reshape((2, 1))
    tensor_q2 = np.array([1., 0.], dtype=np.complex128).reshape((1, 2))
    state = cudaq.State.from_data([tensor_q1, tensor_q2])
    assert np.isclose(state[0], 1.0)
    assert np.isclose(state[1], 0.0)
    assert np.isclose(state[2], 0.0)
    assert np.isclose(state[3], 0.0)


@skipIfNoGPU
def test_state_from_mps_cupy():
    cudaq.set_target('tensornet-mps')
    tensor_q1 = cp.array([1., 0.], dtype=np.complex128).reshape((2, 1))
    tensor_q2 = cp.array([1., 0.], dtype=np.complex128).reshape((1, 2))
    state = cudaq.State.from_data([tensor_q1, tensor_q2])
    assert np.isclose(state[0], 1.0)
    assert np.isclose(state[1], 0.0)
    assert np.isclose(state[2], 0.0)
    assert np.isclose(state[3], 0.0)


@skipIfNoGPU
def test_state_from_mps_numpy():
    cudaq.set_target('tensornet-mps')

    def random_state_vector(dim, seed=None):
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.complex128)
        vec += 1j * rng.standard_normal(dim)
        vec /= np.linalg.norm(vec)
        return vec

    state_vec = random_state_vector(4, 1)
    print("random state: ", state_vec)
    stacked_state_vec = state_vec.reshape(2, 2).transpose()
    # Do SVD
    U, S, Vh = np.linalg.svd(stacked_state_vec)
    # Important: Tensor data must be in column major.
    left_tensor = np.asfortranarray(U)
    right_tensor = np.asfortranarray(np.dot(np.diag(S), Vh))
    state = cudaq.State.from_data([left_tensor, right_tensor])
    state.dump()
    assert np.isclose(state[0], state_vec[0])
    assert np.isclose(state[2], state_vec[1])
    assert np.isclose(state[1], state_vec[2])
    assert np.isclose(state[3], state_vec[3])


@skipIfNoGPU
def test_state_from_mps_cupy():
    cudaq.set_target('tensornet-mps')

    def random_state_vector(dim, seed=None):
        rng = cp.random.default_rng(seed)
        vec = rng.standard_normal(dim).astype(np.complex128)
        vec += 1j * rng.standard_normal(dim)
        vec /= cp.linalg.norm(vec)
        return vec

    state_vec = random_state_vector(4, 1)
    print("Cupy random state: ", state_vec)
    stacked_state_vec = state_vec.reshape(2, 2).transpose()
    # Do SVD
    U, S, Vh = cp.linalg.svd(stacked_state_vec)
    # Important: Tensor data must be in column major.
    left_tensor = cp.asfortranarray(U)
    right_tensor = cp.asfortranarray(cp.dot(cp.diag(S), Vh))
    state = cudaq.State.from_data([left_tensor, right_tensor])
    state.dump()
    assert cp.isclose(state[0], state_vec[0])
    assert cp.isclose(state[2], state_vec[1])
    assert cp.isclose(state[1], state_vec[2])
    assert cp.isclose(state[3], state_vec[3])


@skipIfNoGPU
def test_state_from_mps_tensors():
    cudaq.set_target('tensornet-mps')

    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(10)
        h(qubits[0])
        for q in range(len(qubits) - 1):
            x.ctrl(qubits[q], qubits[q + 1])

    state = cudaq.get_state(bell)
    reconstructed = cudaq.State.from_data(state.getTensors())
    assert np.isclose(reconstructed[0], 1. / np.sqrt(2.))
    assert np.isclose(reconstructed[2**10 - 1], 1. / np.sqrt(2.))


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
