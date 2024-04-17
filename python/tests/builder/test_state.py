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


def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance

def test_state_vector_simple():
    """
    A simple end-to-end test of the state class on a state vector
    backend. Begins with a kernel, converts to state, then checks
    its member functions.
    """
    cudaq.reset_target()
    target = cudaq.get_target()
    print(target)
    dtype = np.complex64 
    wrongDtype = np.complex128
    if target.name == 'qpp-cpu' or target.name == 'custatevec_fp64':
        dtype = np.complex128 
        wrongDtype = np.complex64


    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])

    # Get the quantum state, which should be a vector.
    got_state = cudaq.get_state(kernel)

    # Data type needs to be the same as the internal state vector
    want_state = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)],
                          dtype=dtype)

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

    # Check overlaps.
    # Check the overlap overload with want numpy array.
    assert np.isclose(got_state.overlap(want_state), 1.0)
    # Check the overlap overload with itself.
    assert np.isclose(got_state.overlap(got_state), 1.0)

    # Can't use FP64 with FP32 data
    want_state_bad_datatype = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)],
                          dtype=wrongDtype)
    with pytest.raises(Exception) as error:
        got_state.overlap(want_state_bad_datatype)


def test_state_vector_integration():
    """
    An integration test on the state vector class. Uses a CUDA Quantum
    optimizer to find the correct kernel parameters for a Bell state.
    """
    cudaq.reset_target()
    target = cudaq.get_target()
    print(target)
    dtype = np.complex64 
    if target.name == 'qpp-cpu' or target.name == 'custatevec_fp64':
        dtype = np.complex128 

    # Make a general 2 qubit SO4 rotation.
    kernel, parameters = cudaq.make_kernel(list)
    qubits = kernel.qalloc(2)
    kernel.ry(parameters[0], qubits[0])
    kernel.ry(parameters[1], qubits[1])
    kernel.cz(qubits[0], qubits[1])
    kernel.ry(parameters[2], qubits[0])
    kernel.ry(parameters[3], qubits[1])
    kernel.cz(qubits[0], qubits[1])
    kernel.ry(parameters[4], qubits[0])
    kernel.ry(parameters[5], qubits[1])
    kernel.cz(qubits[0], qubits[1])

    want_state = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)],
                 dtype=dtype)

    def objective(x):
        got_state = cudaq.get_state(kernel, x)
        return np.real(1. - got_state.overlap(want_state))

    # Compute the parameters that make this kernel produce the
    # Bell state.
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 100
    optimal_infidelity, optimal_parameters = optimizer.optimize(6, objective)

    # Did we maximize the overlap (i.e, minimize the infidelity)?
    assert np.isclose(optimal_infidelity, 0.0, atol=1e-3)

    # Check the state from the kernel at the fixed parameters.
    bell_state = cudaq.get_state(kernel, optimal_parameters)
    print(bell_state)
    assert np.allclose(want_state, bell_state, atol=1e-3)


def test_state_density_matrix_simple():
    """
    A simple end-to-end test of the state class on a density matrix
    backend. Begins with a kernel, converts to state, then checks
    its member functions.
    """
    cudaq.set_target('density-matrix-cpu')

    # Create the bell state
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])

    got_state = cudaq.get_state(kernel)
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

    # Check overlaps.
    # Check the overlap overload with want numpy array.
    assert np.isclose(got_state.overlap(want_state), 1.0)
    # Check the overlap overload with itself.
    assert np.isclose(got_state.overlap(got_state), 1.0)

    cudaq.reset_target()


def test_state_density_matrix_integration():
    """
    An integration test on the state density matrix class. Uses a CUDA Quantum
    optimizer to find the correct kernel parameters for a Bell state.
    """
    cudaq.set_target('density-matrix-cpu')

    # Make a general 2 qubit SO4 rotation.
    kernel, parameters = cudaq.make_kernel(list)
    qubits = kernel.qalloc(2)
    kernel.ry(parameters[0], qubits[0])
    kernel.ry(parameters[1], qubits[1])
    kernel.cz(qubits[0], qubits[1])
    kernel.ry(parameters[2], qubits[0])
    kernel.ry(parameters[3], qubits[1])
    kernel.cz(qubits[0], qubits[1])
    kernel.ry(parameters[4], qubits[0])
    kernel.ry(parameters[5], qubits[1])
    kernel.cz(qubits[0], qubits[1])

    want_state = np.array([[.5, 0., 0., .5], [0., 0., 0., 0.], [0., 0., 0., 0.],
                  [.5, 0., 0., .5]],
                 dtype=np.complex128)

    def objective(x):
        got_state = cudaq.get_state(kernel, x)
        return np.real(1. - got_state.overlap(want_state))

    # Compute the parameters that make this kernel produce the
    # Bell state.
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 100
    optimal_infidelity, optimal_parameters = optimizer.optimize(6, objective)

    # Did we maximize the overlap (i.e, minimize the infidelity)?
    assert np.isclose(optimal_infidelity, 0.0, atol=1e-3)

    # Check the state from the kernel at the fixed parameters.
    bell_state = cudaq.get_state(kernel, optimal_parameters)
    assert np.allclose(want_state, bell_state, atol=1e-3)

    cudaq.reset_target()


def test_state_vector_async():
    """Tests `cudaq.get_state_async` on a simple kernel."""
    target = cudaq.get_target()
    print(target)
    dtype = np.complex64 
    if target.name == 'qpp-cpu' or target.name == 'custatevec_fp64':
        dtype = np.complex128 

    kernel, theta, phi = cudaq.make_kernel(float, float)
    qubits = kernel.qalloc(2)
    kernel.ry(phi, qubits[0])
    kernel.rx(theta, qubits[0])
    kernel.cx(qubits[0], qubits[1])

    # Creating the bell state with rx and ry instead of hadamard
    # need a pi rotation and a pi/2 rotation
    # Note: rx(pi)ry(pi/2) == -i*H (with a global phase)
    future = cudaq.get_state_async(kernel, np.pi, np.pi / 2.)
    want_state = np.array([-1j / np.sqrt(2.), 0., 0., -1j / np.sqrt(2.)],
                          dtype=dtype)
    state = future.get()
    state.dump()
    assert np.allclose(state, want_state, atol=1e-3)
    # Check invalid qpu_id
    with pytest.raises(Exception) as error:
        # Invalid qpu_id type.
        result = cudaq.get_state_async(kernel, 0.0, 0.0, qpu_id=12)


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
