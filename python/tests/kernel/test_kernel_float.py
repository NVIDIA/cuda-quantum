# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys

import pytest
import numpy as np

import cudaq

## [PYTHON_VERSION_FIX]
skipIfPythonLessThan39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="built-in collection types such as `list` not supported")


def is_close(expected, actual) -> bool:
    return np.isclose(expected, actual, atol=1e-6)


@skipIfPythonLessThan39
def test_float_params():
    """Test that we can pass float lists to kernel functions."""

    f = [1., 2]

    # Pass a list of float as a parameter
    @cudaq.kernel
    def float_vec_param(vec: list[float], i: int) -> float:
        return vec[i]

    for i in range(len(f)):
        assert is_close(f[i], float_vec_param(f, i))


def test_float_capture():
    """Test that we can capture float lists inside kernel functions."""

    f = [1., 2]

    # Capture a list of float
    @cudaq.kernel
    def float_vec_capture(i: int) -> float:
        return f[i]

    for i in range(len(f)):
        assert is_close(f[i], float_vec_capture(i))


def test_float_definition():
    """Test that we can define float lists inside kernel functions."""

    f = [1., 2]

    # Define a list of float inside a kernel
    @cudaq.kernel
    def float_vec_definition(i: int) -> float:
        return [1., 2][i]

    for i in range(len(f)):
        assert is_close(f[i], float_vec_definition(i))


def test_float_use():
    """Test that we can use floats inside kernel functions."""

    # Use a float inside np in a kernel
    @cudaq.kernel
    def float_np_use() -> float:
        return np.sin(np.pi / 2 + 1)

    assert is_close(np.sin(np.pi / 2 + 1), float_np_use())


# np.float64


@skipIfPythonLessThan39
def test_float64_params():
    """Test that we can pass float lists to kernel functions."""

    f = [np.float64(1.), np.float64(2)]

    # Pass a list of float as a parameter
    @cudaq.kernel
    def float_vec_param(vec: list[np.float64], i: int) -> np.float64:
        return vec[i]

    for i in range(len(f)):
        assert is_close(f[i], float_vec_param(f, i))


def test_float64_capture():
    """Test that we can capture float lists inside kernel functions."""

    f = [np.float64(1.), np.float64(2)]

    # Capture a list of float
    @cudaq.kernel
    def float_vec_capture(i: int) -> np.float64:
        return f[i]

    for i in range(len(f)):
        assert is_close(f[i], float_vec_capture(i))


def test_float64_definition():
    """Test that we can define float lists inside kernel functions."""

    f = [np.float64(1.), np.float64(2)]

    # Define a list of float inside a kernel
    @cudaq.kernel
    def float_vec_definition(i: int) -> np.float64:
        return [np.float64(1.), np.float64(2)][i]

    for i in range(len(f)):
        assert is_close(f[i], float_vec_definition(i))


def test_float64_use():
    """Test that we can use floats inside kernel functions."""

    # Use a float inside np in a kernel (sin)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.sin(np.float64(np.pi / 2 + 1))

    t = np.sin(np.float64(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (cos)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.cos(np.float64(np.pi / 2 + 1))

    t = np.cos(np.float64(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (sqrt)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.sqrt(np.float64(np.pi / 2 + 1))

    t = np.sqrt(np.float64(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (ceil)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.ceil(np.float64(np.pi / 2 + 1))

    t = np.ceil(np.float64(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (exp)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.exp(np.float64(np.pi / 2 + 1))

    t = np.exp(np.float64(np.pi / 2 + 1))
    assert is_close(t, float_np_use())


# np.float32


@skipIfPythonLessThan39
def test_float32_params():
    """Test that we can pass float lists to kernel functions."""

    f = [np.float32(1.), np.float32(2)]

    # Pass a list of float as a parameter
    @cudaq.kernel
    def float_vec_param(vec: list[np.float32], i: int) -> np.float32:
        return vec[i]

    for i in range(len(f)):
        assert is_close(f[i], float_vec_param(f, i))


def test_float32_capture():
    """Test that we can capture float lists inside kernel functions."""

    f = [np.float32(1.), np.float32(2)]

    # Capture a list of float
    @cudaq.kernel
    def float_vec_capture(i: int) -> np.float32:
        return f[i]

    for i in range(len(f)):
        assert is_close(f[i], float_vec_capture(i))


def test_float32_definition():
    """Test that we can define float lists inside kernel functions."""

    f = [np.float32(1.), np.float32(2)]

    # Define a list of float inside a kernel
    @cudaq.kernel
    def float_vec_definition(i: int) -> np.float32:
        return [np.float32(1.), np.float32(2)][i]

    for i in range(len(f)):
        assert is_close(f[i], float_vec_definition(i))


def test_float32_use():
    """Test that we can use floats inside kernel functions."""

    # Use a float inside np in a kernel (sin)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.sin(np.float32(np.pi / 2 + 1))

    t = np.sin(np.float32(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (cos)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.cos(np.float32(np.pi / 2 + 1))

    t = np.cos(np.float32(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (sqrt)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.sqrt(np.float32(np.pi / 2 + 1))

    t = np.sqrt(np.float32(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (ceil)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.ceil(np.float32(np.pi / 2 + 1))

    t = np.ceil(np.float32(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (exp)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.exp(np.float32(np.pi / 2 + 1))

    t = np.exp(np.float32(np.pi / 2 + 1))
    assert is_close(t, float_np_use())


@skipIfPythonLessThan39
def test_float_list_parameter_promotion():

    @cudaq.kernel
    def kernel(c: list[float], i: int) -> float:
        return c[i]

    def non_kernel(c: list[float], i: int) -> float:
        return c[i]

    def check(c: any):
        for i in range(len(c)):
            is_close(kernel(c, i), non_kernel(c, i))

    check([np.pi / 2, 0])
    check([0, np.pi / 2])
    check([np.float64(np.pi / 2), 0])
    check([np.float32(np.pi / 2), 0])
    check([1, 0])
    check([np.float32(np.pi / 2), True])


@skipIfPythonLessThan39
def test_float64_list_parameter_promotion():

    @cudaq.kernel
    def kernel(c: list[np.float64], i: int) -> np.float64:
        return c[i]

    def non_kernel(c: list[np.float64], i: int) -> np.float64:
        return c[i]

    def check(c: any):
        for i in range(len(c)):
            is_close(kernel(c, i), non_kernel(c, i))

    check([np.pi / 2, 0])
    check([0, np.pi / 2])
    check([np.float64(np.pi / 2), 0])
    check([np.float32(np.pi / 2), 0])
    check([1, 0])
    check([np.float32(np.pi / 2), 0, True])


@skipIfPythonLessThan39
def test_float32_list_parameter_promotion():

    @cudaq.kernel
    def kernel(c: list[np.float32], i: int) -> np.float32:
        return c[i]

    def non_kernel(c: list[np.float32], i: int) -> np.float32:
        return c[i]

    def check(c: any):
        for i in range(len(c)):
            is_close(kernel(c, i), non_kernel(c, i))

    check([np.pi / 2, 0])
    check([0, np.pi / 2])
    check([np.float64(np.pi / 2), 0])
    check([np.float32(np.pi / 2), 0])
    check([1, 0])
    check([np.pi / 2, 0, True])
