# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import math
import sys

import pytest
import numpy as np

import cudaq


def is_close(expected, actual) -> bool:
    return np.isclose(expected, actual, atol=1e-6)


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


def test_math_functions_match_python():

    def python_math(function: int, value: float) -> float:
        if function == 0:
            return math.sin(value)
        if function == 1:
            return math.cos(value)
        if function == 2:
            return math.tan(value)
        if function == 3:
            return math.asin(value)
        if function == 4:
            return math.acos(value)
        if function == 5:
            return math.atan(value)
        if function == 6:
            return math.sqrt(value)
        if function == 7:
            return math.exp(value)
        return math.log(value)

    @cudaq.kernel
    def kernel_math(function: int, value: float) -> float:
        if function == 0:
            return math.sin(value)
        if function == 1:
            return math.cos(value)
        if function == 2:
            return math.tan(value)
        if function == 3:
            return math.asin(value)
        if function == 4:
            return math.acos(value)
        if function == 5:
            return math.atan(value)
        if function == 6:
            return math.sqrt(value)
        if function == 7:
            return math.exp(value)
        return math.log(value)

    for function in range(9):
        for value in [0.25, 0.5, 0.75]:
            assert is_close(python_math(function, value),
                            kernel_math(function, value))


def test_math_floor_and_ceil_match_python():

    @cudaq.kernel
    def kernel_floor(value: float) -> int:
        return math.floor(value)

    @cudaq.kernel
    def kernel_ceil(value: float) -> int:
        return math.ceil(value)

    for value in [-1.75, -0.25, 0.25, 1.75]:
        expected_floor = math.floor(value)
        expected_ceil = math.ceil(value)
        actual_floor = kernel_floor(value)
        actual_ceil = kernel_ceil(value)
        assert type(actual_floor) is type(expected_floor) is int
        assert type(actual_ceil) is type(expected_ceil) is int
        assert actual_floor == expected_floor
        assert actual_ceil == expected_ceil


def test_math_and_numpy_floor_ceil_result_types():

    def python_math_floor_div(value: float) -> int:
        return math.floor(value) // 2

    def python_math_ceil_div(value: float) -> int:
        return math.ceil(value) // 2

    @cudaq.kernel
    def kernel_math_floor_div(value: float) -> int:
        return math.floor(value) // 2

    @cudaq.kernel
    def kernel_math_ceil_div(value: float) -> int:
        return math.ceil(value) // 2

    for value in [-1.75, -0.25, 0.25, 1.75]:
        assert python_math_floor_div(value) == kernel_math_floor_div(value)
        assert python_math_ceil_div(value) == kernel_math_ceil_div(value)

    @cudaq.kernel
    def kernel_numpy_floor_div(value: float) -> float:
        return np.floor(value) // 2.0

    @cudaq.kernel
    def kernel_numpy_ceil_div(value: float) -> float:
        return np.ceil(value) // 2.0

    assert isinstance(np.floor(1.75), np.float64)
    assert isinstance(np.ceil(1.75), np.float64)
    with pytest.raises(RuntimeError,
                       match=r"floor division with floating-point operands"):
        kernel_numpy_floor_div(1.75)
    with pytest.raises(RuntimeError,
                       match=r"floor division with floating-point operands"):
        kernel_numpy_ceil_div(1.75)


def test_math_rejects_numpy_only_names():
    assert not hasattr(math, 'arcsin')
    assert not hasattr(math, 'float64')

    @cudaq.kernel
    def kernel_arcsin(value: float) -> float:
        return math.arcsin(value)

    @cudaq.kernel
    def kernel_float64(value: float) -> float:
        return math.float64(value)

    with pytest.raises(RuntimeError, match=r"unsupported math call \(arcsin\)"):
        kernel_arcsin(0.5)
    with pytest.raises(RuntimeError,
                       match=r"unsupported math call \(float64\)"):
        kernel_float64(0.5)


def test_math_rejects_numpy_only_attributes():
    for name in ['arcsin', 'array', 'euler_gamma', 'float64']:
        assert not hasattr(math, name)

    @cudaq.kernel
    def kernel_arcsin_attribute():
        math.arcsin

    @cudaq.kernel
    def kernel_array_attribute():
        math.array

    @cudaq.kernel
    def kernel_euler_gamma_attribute():
        math.euler_gamma

    @cudaq.kernel
    def kernel_float64_attribute():
        math.float64

    with pytest.raises(RuntimeError, match=r"math\.arcsin is not supported"):
        kernel_arcsin_attribute()
    with pytest.raises(RuntimeError, match=r"math\.array is not supported"):
        kernel_array_attribute()
    with pytest.raises(RuntimeError,
                       match=r"math\.euler_gamma is not supported"):
        kernel_euler_gamma_attribute()
    with pytest.raises(RuntimeError, match=r"math\.float64 is not supported"):
        kernel_float64_attribute()


def test_math_attributes():

    @cudaq.kernel
    def kernel_math_constants() -> float:
        return math.pi + math.e

    assert is_close(kernel_math_constants(), math.pi + math.e)


def test_math_rejects_complex_arguments():
    value = 1.0 + 2.0j
    with pytest.raises(TypeError):
        math.sin(value)

    @cudaq.kernel
    def kernel_sin(value: complex) -> complex:
        return math.sin(value)

    with pytest.raises(RuntimeError,
                       match=r"math\.sin does not accept complex arguments"):
        kernel_sin(value)


def test_integer_floor_division_matches_python():

    def python_floor_div(left: int, right: int) -> int:
        return left // right

    @cudaq.kernel
    def kernel_floor_div(left: int, right: int) -> int:
        return left // right

    for left, right in [(9, 2), (-9, 2), (9, -2), (-9, -2)]:
        assert python_floor_div(left, right) == kernel_floor_div(left, right)


def test_float_floor_division_error():
    error = ("floor division with floating-point operands is not supported; "
             "use integer operands or math.floor(...), numpy.floor(...), or "
             "np.floor(...) instead")

    @cudaq.kernel
    def float64_floor_div(left: float, right: float) -> float:
        return left // right

    with pytest.raises(RuntimeError,
                       match=r"floor division with floating") as e:
        float64_floor_div(9.0, 2.0)
    assert error in str(e.value)

    @cudaq.kernel
    def float32_floor_div(left: np.float32, right: np.float32) -> np.float32:
        return left // right

    with pytest.raises(RuntimeError,
                       match=r"floor division with floating") as e:
        float32_floor_div(np.float32(9.0), np.float32(2.0))
    assert error in str(e.value)


# np.float64


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

    # Use a float inside np in a kernel (tan)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.tan(np.float64(0.25))

    t = np.tan(np.float64(0.25))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (arcsin)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.arcsin(np.float64(0.25))

    t = np.arcsin(np.float64(0.25))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (arccos)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.arccos(np.float64(0.25))

    t = np.arccos(np.float64(0.25))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (arctan)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.arctan(np.float64(0.25))

    t = np.arctan(np.float64(0.25))
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

    # Use a float inside np in a kernel (floor)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.floor(np.float64(np.pi / 2 + 1))

    t = np.floor(np.float64(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (exp)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.exp(np.float64(np.pi / 2 + 1))

    t = np.exp(np.float64(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (log)
    @cudaq.kernel
    def float_np_use() -> np.float64:
        return np.log(np.float64(np.pi / 2 + 1))

    t = np.log(np.float64(np.pi / 2 + 1))
    assert is_close(t, float_np_use())


# np.float32


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

    # Use a float inside np in a kernel (tan)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.tan(np.float32(0.25))

    t = np.tan(np.float32(0.25))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (arcsin)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.arcsin(np.float32(0.25))

    t = np.arcsin(np.float32(0.25))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (arccos)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.arccos(np.float32(0.25))

    t = np.arccos(np.float32(0.25))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (arctan)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.arctan(np.float32(0.25))

    t = np.arctan(np.float32(0.25))
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

    # Use a float inside np in a kernel (floor)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.floor(np.float32(np.pi / 2 + 1))

    t = np.floor(np.float32(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (exp)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.exp(np.float32(np.pi / 2 + 1))

    t = np.exp(np.float32(np.pi / 2 + 1))
    assert is_close(t, float_np_use())

    # Use a float inside np in a kernel (log)
    @cudaq.kernel
    def float_np_use() -> np.float32:
        return np.log(np.float32(np.pi / 2 + 1))

    t = np.log(np.float32(np.pi / 2 + 1))
    assert is_close(t, float_np_use())


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
