# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os

import pytest

import cudaq
from cudaq import spin


# Helper function for asserting two values are within a
# certain tolerance. If we make numpy a dependency,
# this may be replaced in the future with `np.allclose`.
def assert_close(want, got, tolerance=1.e-4) -> bool:
    return abs(want - got) < tolerance


@pytest.fixture
def hamiltonian_2q():
    """Spin operator for 2-qubit VQE benchmark used in this test suite."""
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(1)
    return hamiltonian


@pytest.fixture
def kernel_two_qubit_vqe_float() -> cudaq.Kernel:
    """A 2-qubit VQE ansatz used to benchmark `cudaq.vqe`."""
    kernel, theta = cudaq.make_kernel(float)
    qubits = kernel.qalloc(2)
    kernel.x(qubits[0])
    kernel.ry(theta, qubits[1])
    kernel.cx(qubits[1], qubits[0])
    return kernel


@pytest.fixture
def kernel_two_qubit_vqe_list() -> cudaq.Kernel:
    """A 2-qubit VQE ansatz used to benchmark `cudaq.vqe`."""
    kernel, thetas = cudaq.make_kernel(list)
    qubits = kernel.qalloc(2)
    kernel.x(qubits[0])
    kernel.ry(thetas[0], qubits[1])
    kernel.cx(qubits[1], qubits[0])
    return kernel


@pytest.fixture
def hamiltonian_3q():
    """Spin operator for 3-qubit VQE benchmark used in this test suite."""
    hamiltonian = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(
        0) * spin.y(1) + .21829 * spin.z(0) - 6.125 * spin.z(
            1) + 9.625 - 9.625 * spin.z(2) - 3.913119 * spin.x(1) * spin.x(
                2) - 3.913119 * spin.y(1) * spin.y(2)
    return hamiltonian


@pytest.fixture
def kernel_three_qubit_vqe() -> cudaq.Kernel:
    """
    A 3-qubit VQE ansatz used to benchmark `cudaq.vqe`.
    Note: the parameters are stored in the kernel as 
    individual float values.
    """
    kernel, theta, phi = cudaq.make_kernel(float, float)
    qubits = kernel.qalloc(3)
    kernel.x(qubits[0])
    kernel.ry(theta, qubits[1])
    kernel.ry(phi, qubits[2])
    kernel.cx(qubits[2], qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.ry(theta * -1., qubits[1])
    kernel.cx(qubits[0], qubits[1])
    kernel.cx(qubits[1], qubits[0])
    return kernel


@pytest.fixture
def kernel_three_qubit_vqe_list() -> cudaq.Kernel:
    """
    A 3-qubit VQE ansatz used to benchmark `cudaq.vqe`.
    Note: the parameters are all stored in the kernel
    in a single `list`.

    FIXME: List arguments are currently incompatible with
    `cudaq.vqe`.
    """
    kernel, angles = cudaq.make_kernel(list)
    theta = angles[0]
    phi = angles[1]
    qubits = kernel.qalloc(3)
    kernel.x(qubits[0])
    kernel.ry(theta, qubits[1])
    kernel.ry(phi, qubits[2])
    kernel.cx(qubits[2], qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.ry(theta * -1., qubits[1])
    kernel.cx(qubits[0], qubits[1])
    kernel.cx(qubits[1], qubits[0])
    return kernel


@pytest.mark.parametrize(
    "optimizer", [cudaq.optimizers.COBYLA(),
                  cudaq.optimizers.NelderMead()])
def test_vqe_two_qubit_float(optimizer, kernel_two_qubit_vqe_float,
                             hamiltonian_2q):
    """
    Test `cudaq.vqe` on a 2-qubit benchmark for each gradient-free optimizer, and
    for both the `Kernel` and `Callable` overloads.
    """
    # Should be able to call this by passing a function that returns a kernel
    # along with a lambda (or function) for the `argument_wrapper`:
    got_expectation, got_parameters = cudaq.vqe(
        kernel_two_qubit_vqe_float,
        hamiltonian_2q,
        optimizer,
        parameter_count=1,
        argument_mapper=lambda parameter_vector: parameter_vector[0],
        shots=-1)

    # Known minimal expectation value for this system:
    want_expectation_value = -1.7487948611472093
    want_optimal_parameters = [0.59]
    assert assert_close(want_expectation_value, got_expectation, tolerance=1e-2)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-2)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 got_parameters))


@pytest.mark.parametrize(
    "optimizer", [cudaq.optimizers.COBYLA(),
                  cudaq.optimizers.NelderMead()])
def test_vqe_two_qubit_list(optimizer, kernel_two_qubit_vqe_list,
                            hamiltonian_2q):
    """
    Test `cudaq.vqe` on a 2-qubit benchmark for each gradient-free optimizer, and
    for both the `Kernel` and `Callable` overloads.
    """
    # Should be able to call this by passing a function that returns a kernel:
    got_expectation, got_parameters = cudaq.vqe(kernel_two_qubit_vqe_list,
                                                hamiltonian_2q,
                                                optimizer,
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


@pytest.mark.parametrize(
    "optimizer",
    [
        # FIXME: cudaq.optimizers.SPSA(),
        cudaq.optimizers.COBYLA(),
        cudaq.optimizers.LBFGS(),
        cudaq.optimizers.Adam(),
        cudaq.optimizers.GradientDescent(),
        cudaq.optimizers.SGD(),
    ])
@pytest.mark.parametrize("gradient", [
    cudaq.gradients.CentralDifference(),
    cudaq.gradients.ParameterShift(),
    cudaq.gradients.ForwardDifference()
])
def test_vqe_two_qubit_float_gradients(optimizer, gradient,
                                       kernel_two_qubit_vqe_float,
                                       hamiltonian_2q):
    """
    Test `cudaq.vqe` on a 2-qubit benchmark for each gradient based optimizer,
    with each cudaq supported gradient strategy. Also checks for the different
    `Kernel` and `Callable` overloads.
    """

    def argument_map(parameter_vector):
        """Takes the `parameter_vector` from optimizer as input and returns
        its single element indexed out as as float."""
        return parameter_vector[0]

    # Should be able to call this by passing a function that returns a kernel:
    got_expectation, got_parameters = cudaq.vqe(
        kernel=kernel_two_qubit_vqe_float,
        gradient_strategy=gradient,
        spin_operator=hamiltonian_2q,
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


@pytest.mark.parametrize(
    "optimizer",
    [
        # FIXME: cudaq.optimizers.SPSA(),
        cudaq.optimizers.COBYLA(),
        cudaq.optimizers.LBFGS(),
        cudaq.optimizers.Adam(),
        cudaq.optimizers.GradientDescent(),
        cudaq.optimizers.SGD(),
    ])
@pytest.mark.parametrize("gradient", [
    cudaq.gradients.CentralDifference(),
    cudaq.gradients.ParameterShift(),
    cudaq.gradients.ForwardDifference()
])
def test_vqe_two_qubit_list_gradients(optimizer, gradient,
                                      kernel_two_qubit_vqe_list,
                                      hamiltonian_2q):
    """
    Test `cudaq.vqe` on a 2-qubit benchmark for each gradient based optimizer,
    with each cudaq supported gradient strategy. Also checks for the different
    `Kernel` and `Callable` overloads.
    """
    # Should be able to call this by passing a function that returns a kernel:
    got_expectation, got_parameters = cudaq.vqe(
        kernel=kernel_two_qubit_vqe_list,
        gradient_strategy=gradient,
        spin_operator=hamiltonian_2q,
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


@pytest.mark.parametrize(
    "optimizer", [cudaq.optimizers.COBYLA(),
                  cudaq.optimizers.NelderMead()])
def test_vqe_three_qubit_float(optimizer, kernel_three_qubit_vqe,
                               hamiltonian_3q):
    # Should be able to call this by passing a function that returns a kernel:
    got_expectation, got_parameters = cudaq.vqe(
        kernel_three_qubit_vqe,
        hamiltonian_3q,
        optimizer,
        parameter_count=2,
        argument_mapper=lambda parameter_vector: tuple(parameter_vector))

    # Known minimal expectation value for this system:
    want_expectation_value = -2.045375
    want_optimal_parameters = [0.359, 0.257]
    assert assert_close(want_expectation_value, got_expectation, tolerance=1e-3)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-2)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 got_parameters))


@pytest.mark.parametrize(
    "optimizer", [cudaq.optimizers.COBYLA(),
                  cudaq.optimizers.NelderMead()])
def test_vqe_three_qubit_list(optimizer, kernel_three_qubit_vqe_list,
                              hamiltonian_3q):
    # Should be able to call this by passing a function that returns a kernel:
    got_expectation, got_parameters = cudaq.vqe(kernel_three_qubit_vqe_list,
                                                hamiltonian_3q,
                                                optimizer,
                                                parameter_count=2)

    # Known minimal expectation value for this system:
    want_expectation_value = -2.045375
    want_optimal_parameters = [0.359, 0.257]
    assert assert_close(want_expectation_value, got_expectation, tolerance=1e-3)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-2)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 got_parameters))


@pytest.mark.parametrize(
    "optimizer",
    [
        # FIXME: cudaq.optimizers.SPSA(),
        cudaq.optimizers.COBYLA(),
        cudaq.optimizers.LBFGS(),
        cudaq.optimizers.Adam(),
        cudaq.optimizers.GradientDescent(),
        cudaq.optimizers.SGD(),
    ])
@pytest.mark.parametrize(
    "gradient",
    [
        cudaq.gradients.CentralDifference(),
        cudaq.gradients.ForwardDifference(),
        # FIXME: cudaq.gradients.ParameterShift()
    ])
def test_vqe_three_qubit_float_gradients(optimizer, gradient,
                                         kernel_three_qubit_vqe,
                                         hamiltonian_3q):

    def argument_map(parameter_vector):
        """Takes the `parameter_vector` from optimizer as input and returns
        both of its elements as a tuple."""
        return tuple(parameter_vector)

    optimizer.max_iterations = 100
    # Should be able to call this by passing a function that returns a kernel:
    got_expectation, got_parameters = cudaq.vqe(kernel=kernel_three_qubit_vqe,
                                                gradient_strategy=gradient,
                                                spin_operator=hamiltonian_3q,
                                                optimizer=optimizer,
                                                parameter_count=2,
                                                argument_mapper=argument_map)

    # Known minimal expectation value for this system:
    want_expectation_value = -2.045375
    want_optimal_parameters = [0.359, 0.257]
    assert assert_close(want_expectation_value, got_expectation, tolerance=1e-2)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-1)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 got_parameters))


@pytest.mark.parametrize(
    "optimizer",
    [
        # FIXME: cudaq.optimizers.SPSA(),
        cudaq.optimizers.COBYLA(),
        cudaq.optimizers.LBFGS(),
        cudaq.optimizers.Adam(),
        cudaq.optimizers.GradientDescent(),
        cudaq.optimizers.SGD(),
    ])
@pytest.mark.parametrize(
    "gradient",
    [
        cudaq.gradients.CentralDifference(),
        cudaq.gradients.ForwardDifference(),
        # FIXME: cudaq.gradients.ParameterShift()
    ])
def test_vqe_three_qubit_list_gradients(optimizer, gradient,
                                        kernel_three_qubit_vqe_list,
                                        hamiltonian_3q):
    optimizer.max_iterations = 100
    # Should be able to call this by passing a function that returns a kernel:
    got_expectation, got_parameters = cudaq.vqe(
        kernel=kernel_three_qubit_vqe_list,
        gradient_strategy=gradient,
        spin_operator=hamiltonian_3q,
        optimizer=optimizer,
        parameter_count=2)

    # Known minimal expectation value for this system:
    want_expectation_value = -2.045375
    want_optimal_parameters = [0.359, 0.257]
    assert assert_close(want_expectation_value, got_expectation, tolerance=1e-2)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-1)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 got_parameters))


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
