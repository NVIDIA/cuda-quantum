# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import sys
from typing import List

import cudaq
from cudaq import spin
import numpy as np

## [PYTHON_VERSION_FIX]
skipIfPythonLessThan39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="built-in collection types such as `list` not supported")


# Helper function for asserting two values are within a
# certain tolerance. If we make numpy a dependency,
# this may be replaced in the future with `np.allclose`.
def assert_close(want, got, tolerance=1.e-4) -> bool:
    return abs(want - got) < tolerance


@pytest.fixture(autouse=True)
def do_something():
    yield
    cudaq.__clearKernelRegistries()


def test_two_qubit_vqe_float():
    """A 2-qubit VQE ansatz used to benchmark `cudaq.vqe`."""

    @cudaq.kernel()
    def kernel_float(theta: float):
        qubits = cudaq.qvector(2)
        x(qubits[0])
        ry(theta, qubits[1])
        x.ctrl(qubits[1], qubits[0])

    optimizer = cudaq.optimizers.COBYLA()
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    got_expectation, got_parameters = cudaq.vqe(
        kernel_float,
        hamiltonian,
        optimizer,
        argument_mapper=lambda parameter_vector: parameter_vector[0],
        parameter_count=1)

    assert np.isclose(got_expectation, -1.74, atol=1e-2)


@skipIfPythonLessThan39
def test_two_qubit_vqe_vecfloat():
    """A 2-qubit VQE ansatz used to benchmark `cudaq.vqe`."""

    @cudaq.kernel()
    def kernel_vecfloat(thetas: list[float]):
        qubits = cudaq.qvector(2)
        x(qubits[0])
        ry(thetas[0], qubits[1])
        x.ctrl(qubits[1], qubits[0])

    optimizer = cudaq.optimizers.COBYLA()
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    got_expectation, got_parameters = cudaq.vqe(kernel_vecfloat,
                                                hamiltonian,
                                                optimizer,
                                                parameter_count=1)

    assert np.isclose(got_expectation, -1.74, atol=1e-2)


def test_two_qubit_vqe_with_List():
    """A 2-qubit VQE ansatz used to benchmark `cudaq.vqe`."""

    @cudaq.kernel()
    def kernel_vecfloat(thetas: List[float]):
        qubits = cudaq.qvector(2)
        x(qubits[0])
        ry(thetas[0], qubits[1])
        x.ctrl(qubits[1], qubits[0])

    optimizer = cudaq.optimizers.COBYLA()
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    got_expectation, got_parameters = cudaq.vqe(kernel_vecfloat,
                                                hamiltonian,
                                                optimizer,
                                                parameter_count=1)

    assert np.isclose(got_expectation, -1.74, atol=1e-2)


def test_vqe_two_qubit_float_gradients():
    """
    Test `cudaq.vqe` on a 2-qubit benchmark for each gradient based optimizer,
    with each cudaq supported gradient strategy. Also checks for the different
    `Kernel` and `Callable` overloads.
    """

    def argument_map(parameter_vector):
        """Takes the `parameter_vector` from optimizer as input and returns
        its single element indexed out as as float."""
        return parameter_vector[0]

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    @cudaq.kernel()
    def kernel_vecfloat(theta: float):
        qubits = cudaq.qvector(2)
        x(qubits[0])
        ry(theta, qubits[1])
        x.ctrl(qubits[1], qubits[0])

    optimizer = cudaq.optimizers.LBFGS()
    gradient = cudaq.gradients.CentralDifference()
    optimizer.max_iterations = 100
    # Should be able to call this by passing a function that returns a kernel:
    got_expectation, got_parameters = cudaq.vqe(kernel=kernel_vecfloat,
                                                gradient_strategy=gradient,
                                                spin_operator=hamiltonian,
                                                optimizer=optimizer,
                                                parameter_count=1,
                                                argument_mapper=argument_map,
                                                shots=-1)

    # Known minimal expectation value for this system:
    want_expectation_value = -1.7487948611472093
    want_optimal_parameters = [0.59]
    assert assert_close(want_expectation_value, got_expectation, tolerance=1e-2)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-2)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 got_parameters))


@skipIfPythonLessThan39
def test_vqe_two_qubit_list_gradients():
    """
    Test `cudaq.vqe` on a 2-qubit benchmark for each gradient based optimizer,
    with each cudaq supported gradient strategy. Also checks for the different
    `Kernel` and `Callable` overloads.
    """

    def argument_map(parameter_vector):
        """Takes the `parameter_vector` from optimizer as input and returns
        its single element indexed out as as float."""
        return parameter_vector[0]

    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)

    @cudaq.kernel()
    def kernel_vecfloat(thetas: list[float]):
        qubits = cudaq.qvector(2)
        x(qubits[0])
        ry(thetas[0], qubits[1])
        x.ctrl(qubits[1], qubits[0])

    optimizer = cudaq.optimizers.LBFGS()
    gradient = cudaq.gradients.CentralDifference()
    optimizer.max_iterations = 100
    # Should be able to call this by passing a function that returns a kernel:
    got_expectation, got_parameters = cudaq.vqe(kernel=kernel_vecfloat,
                                                gradient_strategy=gradient,
                                                spin_operator=hamiltonian,
                                                optimizer=optimizer,
                                                parameter_count=1,
                                                shots=-1)

    # Known minimal expectation value for this system:
    want_expectation_value = -1.7487948611472093
    want_optimal_parameters = [0.59]
    assert assert_close(want_expectation_value, got_expectation, tolerance=1e-2)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-2)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 got_parameters))
