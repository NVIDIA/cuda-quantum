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


def assert_close(want, got, tolerance=1.e-5) -> bool:
    return abs(want - got) < tolerance


@pytest.mark.parametrize("want_vector", [
    np.array([0.0, 1.0], dtype=np.complex128),
    np.array([1.0, 0.0], dtype=np.complex128),
    np.array([1.0 / np.sqrt(2), 1.0 / np.sqrt(2)], dtype=np.complex128),
    np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128),
    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
    np.array([1.0 / np.sqrt(2), 0.0, 0.0, 1.0 / np.sqrt(2)],
             dtype=np.complex128),
])
def test_state_buffer_vector(want_vector):
    """
    Tests writing to and returning from the :class:`State` buffer
    on different state vectors.
    """
    got_state_a = cudaq.State(want_vector)
    got_state_b = cudaq.State(want_vector)

    print("got_state_a = ", got_state_a)
    print("got_state_b = ", got_state_b)

    # Check all of the `overlap` overloads.
    assert np.isclose(got_state_a.overlap(want_vector), 1.0)
    assert np.isclose(got_state_b.overlap(want_vector), 1.0)
    assert np.isclose(got_state_a.overlap(got_state_b), 1.0)

    # Should be identical vectors.
    got_vector_a = np.array(got_state_a, copy=False)
    got_vector_b = np.array(got_state_b, copy=False)

    print(f"want_vector = {want_vector}\n")
    print(f"got_vector_a = {got_vector_a}\n")
    print(f"got_vector_b = {got_vector_b}\n")
    for i in range(len(got_vector_a)):
        a_same = np.isclose(want_vector[i], got_vector_a[i])
        b_same = np.isclose(want_vector[i], got_vector_b[i])
        print(f"{i}-th element. Passes? {a_same and b_same}:\n")
        print(f"want = {want_vector[i]}")
        print(f"got_a = {got_vector_a[i]}")
        print(f"got_b = {got_vector_b[i]}\n")
        assert a_same
        assert b_same
    # assert np.allclose(want_vector, got_vector_a)
    # assert np.allclose(want_vector, got_vector_b)


@pytest.mark.parametrize("want_matrix", [
    np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
    np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128),
    np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0]],
             dtype=np.complex128),
    np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0]],
             dtype=np.complex128),
    np.array([[0.5, 0.0, 0.0, 0.5], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
              [0.5, 0.0, 0.0, 0.5]],
             dtype=np.complex128),
])
def test_state_buffer_density_matrix(want_matrix):
    """
    Tests writing to and returning from the :class:`State` buffer
    on different density matrices.
    """
    got_state_a = cudaq.State(want_matrix)
    got_state_b = cudaq.State(want_matrix)

    # Check all of the `overlap` overloads.
    assert np.isclose(got_state_a.overlap(want_matrix), 1.0)
    assert np.isclose(got_state_b.overlap(want_matrix), 1.0)
    assert np.isclose(got_state_a.overlap(got_state_b), 1.0)

    # Should be identical matrices.
    got_matrix_a = np.array(got_state_a, copy=False)
    got_matrix_b = np.array(got_state_b, copy=False)
    assert np.allclose(want_matrix, got_matrix_a)
    assert np.allclose(want_matrix, got_matrix_b)


def test_state_vector_simple():
    """
    A simple end-to-end test of the state class on a state vector
    backend. Begins with a kernel, converts to state, then checks
    its member functions.
    """
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])

    # Get the quantum state, which should be a vector.
    got_state = cudaq.get_state(kernel)

    want_state = np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)],
                          dtype=np.complex128)

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
    want_state_object = cudaq.State(want_state)
    # Check the overlap overload with want numpy array.
    assert np.isclose(got_state.overlap(want_state), 1.0)
    # Check the overlap overload with want state object.
    assert np.isclose(got_state.overlap(want_state_object), 1.0)
    # Check the overlap overload with itself.
    assert np.isclose(got_state.overlap(got_state), 1.0)


def test_state_vector_integration():
    """
    An integration test on the state vector class. Uses a CUDA Quantum
    optimizer to find the correct kernel parameters for a Bell state.
    """
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

    want_state = cudaq.State(
        np.array([1. / np.sqrt(2.), 0., 0., 1. / np.sqrt(2.)],
                 dtype=np.complex128))

    def objective(x):
        got_state = cudaq.get_state(kernel, x)
        return 1. - want_state.overlap(got_state)

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
    want_state_object = cudaq.State(want_state)
    # Check the overlap overload with want numpy array.
    assert np.isclose(got_state.overlap(want_state), 1.0)
    # Check the overlap overload with want state object.
    assert np.isclose(got_state.overlap(want_state_object), 1.0)
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

    want_state = cudaq.State(
        np.array([[.5, 0., 0., .5], [0., 0., 0., 0.], [0., 0., 0., 0.],
                  [.5, 0., 0., .5]],
                 dtype=np.complex128))

    def objective(x):
        got_state = cudaq.get_state(kernel, x)
        return 1. - want_state.overlap(got_state)

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


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-rP"])
