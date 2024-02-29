# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import os
import math

import pytest
from typing import List, Tuple, Callable

import cudaq


# Helper function for asserting two values are within a
# certain tolerance. If we make numpy a cudaq dependency,
# this may be replaced in the future with `np.allclose`.
def assert_close(want, got, tolerance=1.e-4) -> bool:
    return abs(want - got) < tolerance


# Define a custom gradient strategy that can be used on arbitrary
# python functions.
def gradient_central_difference(objective_function: Callable,
                                parameter_vector: List[float],
                                dx=1e-4) -> float:
    gradient_vector = [0.0] * len(parameter_vector)
    parameters_copy = parameter_vector.copy()
    for index, parameter in enumerate(parameter_vector):
        parameter_minus_dx = parameter - dx
        parameter_plus_dx = parameter + dx
        # Change the value in this entry to `x_i - dx_i` while
        # leaving all other values fixed.
        parameters_copy[index] = parameter_minus_dx
        low_value = objective_function(parameters_copy)
        # Change the value to `x_i + dx_i`.
        parameters_copy[index] = parameter_plus_dx
        high_value = objective_function(parameters_copy)
        # Store the gradient df/dx = (f(x+dx) - f(x-dx)) / 2dx
        gradient_vector[index] = (high_value - low_value) / (2 * dx)
        # Reset the value back in `parameters_copy`
        parameters_copy[index] = parameter
    return gradient_vector


# Define functions for benchmarking our optimization suite:
def ackley_function_no_grad(parameter_vector: List[float]) -> float:
    """
    Using the Ackley function as a simple test case. This function 
    was pulled from the following optimization benchmarking resource:
    https://www.sfu.ca/~ssurjano/optimization.html

    Also see:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    We expect to find a global minimum of `f(parameter_vector) = 0.0`
    when `parameter_vector = [0, ..., 0]`.
    """
    # Define magic numbers. These are the recommended values
    # from the above resource.
    a_coefficient = 20
    b_coefficient = 0.2
    c_coefficient = 2 * math.pi

    # term 1 = (1/d) * \sum_{i=1}^{d}{ x_{i}^{2} }
    subterm_1 = sum(
        (x_i * x_i for x_i in parameter_vector)) / len(parameter_vector)
    # term 2 = (1/d) * \sum_{i=1}^{d}{ \cos(2 * pi * x_{i}) }
    subterm_2 = sum((math.cos(c_coefficient * x_i)
                     for x_i in parameter_vector)) / len(parameter_vector)

    term_1 = -1 * a_coefficient * math.exp(-1 * b_coefficient * subterm_1)
    term_2 = math.exp(subterm_2)
    # f(x) = term_1 - term_2 + a_coefficient + exp(1)
    return term_1 - term_2 + a_coefficient + math.exp(1)


def ackley_gradient(parameter_vector: List[float],
                    delta_x=1e-4) -> Tuple[float, List[float]]:
    function_value = ackley_function_no_grad(parameter_vector)
    gradient_vector = gradient_central_difference(ackley_function_no_grad,
                                                  parameter_vector,
                                                  dx=delta_x)
    # Check the equivalent gradient strategy each time
    # this function is called:
    cudaq_gradient = cudaq.gradients.CentralDifference()
    cudaq_gradient_vector = cudaq_gradient.compute(parameter_vector,
                                                   ackley_function_no_grad,
                                                   function_value)
    assert (gradient_vector == cudaq_gradient_vector)
    return function_value, gradient_vector


def schwefel_N_dimension(parameter_vector: List[float]) -> float:
    """
    Using the Schwefel function as a more difficult, n-dimensional
    test case. This function was pulled from the following 
    optimization benchmarking resource:
    https://www.sfu.ca/~ssurjano/optimization.html

    Also see:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization

    Since we expect to find a global minimum of `f(parameter_vector) = 0`,
    we add a fixed constant to the return value (`function_offset`) to have
    a non-zero optimal value. This minima is at 
    `parameter_vector = [420.9687,...,420.9687]`.  
    """
    function_offset = 0.5
    return 418.9829 * len(parameter_vector) - sum(
        (x_i * math.sin(math.sqrt(abs(x_i)))
         for x_i in parameter_vector)) + function_offset


def schwefel_gradient(parameter_vector: List[float], delta_x=1e-4) -> float:
    """
    Using the Schwefel function as a more difficult, n-dimensional
    test case. This function was pulled from the following 
    optimization benchmarking resource:
    https://www.sfu.ca/~ssurjano/optimization.html

    Also see:
    https://en.wikipedia.org/wiki/Test_functions_for_optimization  
    """
    function_value = schwefel_N_dimension(parameter_vector)
    gradient_vector = gradient_central_difference(schwefel_N_dimension,
                                                  parameter_vector,
                                                  dx=delta_x)
    # Check the equivalent CUDA Quantum gradient strategy each time
    # this function is called:
    # cudaq_gradient = cudaq.gradients.CentralDifference()
    # cudaq_gradient_vector = cudaq_gradient.compute(parameter_vector,
    #                                              schwefel_N_dimension,
    #                                              delta_x)
    # assert (gradient_vector == cudaq_gradient_vector)
    return function_value, gradient_vector


@pytest.mark.parametrize("optimizer", [
    cudaq.optimizers.COBYLA(),
    cudaq.optimizers.NelderMead(),
    cudaq.optimizers.SPSA()
])
def test_ackley_no_gradients(optimizer):
    """Test gradient free optimizers on the 2-dimension Ackley function."""
    dimension = 2
    optimizer.max_iterations = 10
    want_optimal_value = 0.0
    want_optimal_parameters = [0.0, 0.0]
    got_optimal_value, got_optimal_parameters = optimizer.optimize(
        dimension, ackley_function_no_grad)
    assert assert_close(want_optimal_value, got_optimal_value)
    assert all(
        assert_close(want_parameter, got_parameter) for want_parameter,
        got_parameter in zip(want_optimal_parameters, got_optimal_parameters))


@pytest.mark.parametrize("optimizer", [
    cudaq.optimizers.COBYLA(),
    cudaq.optimizers.NelderMead(),
    cudaq.optimizers.SPSA(),
    cudaq.optimizers.LBFGS(),
    cudaq.optimizers.GradientDescent(),
    cudaq.optimizers.Adam(),
    cudaq.optimizers.SGD()
])
def test_ackley_gradients(optimizer):
    """Test gradient based optimizers on the 2-dimension Ackley function."""
    dimension = 2
    optimizer.max_iterations = 10
    want_optimal_value = 0.0
    want_optimal_parameters = [0.0, 0.0]
    got_optimal_value, got_optimal_parameters = optimizer.optimize(
        dimension, ackley_gradient)
    assert assert_close(want_optimal_value, got_optimal_value)
    assert all(
        assert_close(want_parameter, got_parameter) for want_parameter,
        got_parameter in zip(want_optimal_parameters, got_optimal_parameters))


@pytest.mark.parametrize("optimizer", [
    cudaq.optimizers.COBYLA(),
    cudaq.optimizers.NelderMead(),
])
@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_schwefel_no_gradients(optimizer, dimension):
    """Test gradient free optimizers on the Schwefel function."""
    print(optimizer)
    optimizer.max_iterations = 500
    # This is a difficult function, so most optimizers require
    # a good head start on the parameters. This has been referenced
    # against optimizers in the scipy optimization suite.
    optimizer.lower_bounds = [300] * dimension
    optimizer.upper_bounds = [500] * dimension
    optimizer.initial_parameters = [400] * dimension
    # The following parameters and expected function values
    # were pulled from the resource listed in the source function.
    want_optimal_value = 0.5
    want_optimal_parameters = [420.9687] * dimension

    got_optimal_value, got_optimal_parameters = optimizer.optimize(
        dimension, schwefel_N_dimension)
    assert assert_close(want_optimal_value, got_optimal_value)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-2)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 got_optimal_parameters))


@pytest.mark.parametrize("optimizer", [
    cudaq.optimizers.COBYLA(),
    cudaq.optimizers.NelderMead(),
    cudaq.optimizers.SPSA(),
    cudaq.optimizers.LBFGS(),
    cudaq.optimizers.GradientDescent(),
    cudaq.optimizers.Adam(),
    cudaq.optimizers.SGD()
])
@pytest.mark.parametrize("dimension", [2, 3, 4, 5])
def test_schwefel_gradients(optimizer, dimension):
    """Test each of the `cudaq.optimizers` on the Schwefel function with gradients."""
    optimizer = cudaq.optimizers.COBYLA()
    optimizer.max_iterations = 100
    # This is a difficult function, so most optimizers require
    # a good head start on the parameters. This has been referenced
    # against optimizers in the scipy optimization suite.
    optimizer.lower_bounds = [300] * dimension
    optimizer.upper_bounds = [500] * dimension
    optimizer.initial_parameters = [400] * dimension
    want_optimal_value = 0.5
    want_optimal_parameters = [420.9687] * dimension

    got_optimal_value, got_optimal_parameters = optimizer.optimize(
        dimension, schwefel_gradient)
    assert assert_close(want_optimal_value, got_optimal_value)
    assert all(
        assert_close(want_parameter, got_parameter, tolerance=1e-2)
        for want_parameter, got_parameter in zip(want_optimal_parameters,
                                                 got_optimal_parameters))


# leave for gdb debugging
if __name__ == "__main__":
    loc = os.path.abspath(__file__)
    pytest.main([loc, "-s"])
