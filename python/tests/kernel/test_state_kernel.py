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
from typing import List

import cudaq

## [PYTHON_VERSION_FIX]
skipIfPythonLessThan39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="built-in collection types such as `list` not supported")


skipIfNoGQPU = pytest.mark.skipif(
    not (cudaq.num_available_gpus() > 0 and cudaq.has_target('nvidia')),
    reason="nvidia-mqpu backend not available")

@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


@skipIfNoGQPU
def test_state_vector_simple():
    """
    A simple end-to-end test of the state class on a state vector
    backend. Begins with a kernel, converts to state, then checks
    its member functions.
    """
    cudaq.set_target('nvidia-fp64')
    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])

    # Get the quantum state, which should be a vector.
    got_state = cudaq.get_state(bell)

    want_state = cudaq.State.from_data(np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)],
                          dtype=np.complex128))
    
    assert len(want_state) == 4 

    # Check the indexing operators on the State class
    # while also checking their values
    np.isclose(want_state[0], got_state[0].real)
    np.isclose(want_state[1], got_state[1].real)
    np.isclose(want_state[2], got_state[2].real)
    np.isclose(want_state[3], got_state[3].real)

    # Check the entire vector with numpy.
    got_vector = np.array(got_state, copy=False)
    for i in range(len(want_state)):
        assert np.isclose(want_state[i], got_vector[i])
        # if not np.isclose(got_vector[i], got_vector_b[i]):
        print(f"want = {want_state[i]}")
        print(f"got = {got_vector[i]}")
    assert np.allclose(want_state, np.array(got_state))
    cudaq.reset_target()


def check_state_vector_integration(entity):
    want_state = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)],
                          dtype=np.complex128)

    def objective(x):
        got_state = cudaq.get_state(entity, x)
        return 1. - np.real(np.dot(want_state.transpose(), got_state))

    # Compute the parameters that make this kernel produce the
    # Bell state.
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 100
    optimal_infidelity, optimal_parameters = optimizer.optimize(6, objective)

    # Did we maximize the overlap (i.e, minimize the infidelity)?
    assert np.isclose(optimal_infidelity, 0.0, atol=1e-3)

    # Check the state from the kernel at the fixed parameters.
    bell_state = cudaq.get_state(entity, optimal_parameters)
    print(bell_state)
    assert np.allclose(want_state, bell_state, atol=1e-3)


@skipIfPythonLessThan39
def test_state_vector_integration():
    """
    An integration test on the state vector class. Uses a CUDA Quantum
    optimizer to find the correct kernel parameters for a Bell state.
    """
    # Make a general 2 qubit SO4 rotation.
    @cudaq.kernel
    def kernel(parameters: list[float]):
        qubits = cudaq.qvector(2)
        ry(parameters[0], qubits[0])
        ry(parameters[1], qubits[1])
        z.ctrl(qubits[0], qubits[1])
        ry(parameters[2], qubits[0])
        ry(parameters[3], qubits[1])
        z.ctrl(qubits[0], qubits[1])
        ry(parameters[4], qubits[0])
        ry(parameters[5], qubits[1])
        z.ctrl(qubits[0], qubits[1])

    check_state_vector_integration(kernel)


def test_state_vector_integration_with_List():
    """
    An integration test on the state vector class. Uses a CUDA Quantum
    optimizer to find the correct kernel parameters for a Bell state.
    """
    # Make a general 2 qubit SO4 rotation.
    @cudaq.kernel
    def kernel_with_List(parameters: List[float]):
        qubits = cudaq.qvector(2)
        ry(parameters[0], qubits[0])
        ry(parameters[1], qubits[1])
        z.ctrl(qubits[0], qubits[1])
        ry(parameters[2], qubits[0])
        ry(parameters[3], qubits[1])
        z.ctrl(qubits[0], qubits[1])
        ry(parameters[4], qubits[0])
        ry(parameters[5], qubits[1])
        z.ctrl(qubits[0], qubits[1])

    check_state_vector_integration(kernel_with_List)


def test_state_density_matrix_simple():
    """
    A simple end-to-end test of the state class on a density matrix
    backend. Begins with a kernel, converts to state, then checks
    its member functions.
    """
    cudaq.set_target('density-matrix-cpu')

    # Create the bell state
    @cudaq.kernel
    def bell():
        qubits = cudaq.qvector(2)
        h(qubits[0])
        x.ctrl(qubits[0], qubits[1])

    got_state = cudaq.get_state(bell)
    print(got_state)

    want_state = np.array([[0.5, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0],
                           [0.0, 0.0, 0.0, 0.0], [0.5, 0.0, 0.0, 0.5]],
                          dtype=np.complex128)

    # Check the indexing operators on the State class
    # while also checking their values
    np.isclose(.5, got_state[0, 0].real)
    np.isclose(.5, got_state[0, 3].real)
    np.isclose(.5, got_state[3, 0].real)
    np.isclose(.5, got_state[3, 3].real)

    # Check the entire matrix with numpy.
    assert np.allclose(want_state, np.array(got_state))

    cudaq.reset_target()


def test_state_vector_async():
    """Tests `cudaq.get_state_async` on a simple kernel."""

    @cudaq.kernel
    def kernel(theta: float, phi: float):
        qubits = cudaq.qvector(2)
        ry(phi, qubits[0])
        rx(theta, qubits[0])
        x.ctrl(qubits[0], qubits[1])

    # Creating the bell state with rx and ry instead of hadamard
    # need a pi rotation and a pi/2 rotation
    # Note: rx(pi)ry(pi/2) == -i*H (with a global phase)
    future = cudaq.get_state_async(kernel, np.pi, np.pi / 2.)
    want_state = np.array([-1j / np.sqrt(2.), 0., 0., -1j / np.sqrt(2.)],
                          dtype=np.complex128)
    state = future.get()
    state.dump()
    assert np.allclose(state, want_state, atol=1e-3)
    # Check invalid qpu_id
    with pytest.raises(Exception) as error:
        # Invalid qpu_id type.
        result = cudaq.get_state_async(kernel, 0.0, 0.0, qpu_id=12)
