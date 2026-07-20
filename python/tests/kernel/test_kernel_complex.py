# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys

import pytest
import numpy as np

import cudaq


def is_close(expected, actual):
    return np.isclose(expected, actual, atol=1e-6)


def test_complex_params():
    """Test that we can pass complex lists to kernel functions."""

    # Pass a list of complex as a parameter
    c = [.70710678 + 1j, 2j, 0., 0.70710678]

    @cudaq.kernel
    def complex_vec_param(vec: list[complex], i: int) -> complex:
        return vec[i]

    for i in range(len(c)):
        is_close(c[i].real, complex_vec_param(c, i).real)
    for i in range(len(c)):
        is_close(c[i].imag, complex_vec_param(c, i).imag)

    @cudaq.kernel
    def complex_vec_param_real(vec: list[complex], i: int) -> float:
        v = vec[i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_param_real(c, i))

    @cudaq.kernel
    def complex_vec_param_imag(vec: list[complex], i: int) -> float:
        v = vec[i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_param_imag(c, i))


def test_complex_capture():
    """Test that we can pass complex lists to kernel functions."""

    # Capture a list of complex
    c = [.70710678 + 1j, 2j, 0., 0.70710678]

    @cudaq.kernel
    def complex_vec_capture_real(i: int) -> float:
        v = c[i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_capture_real(i))

    @cudaq.kernel
    def complex_vec_capture_imag(i: int) -> float:
        v = c[i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_capture_imag(i))


def test_complex_definition():
    """Test that we can define complex lists inside kernel functions."""

    # Define a list of complex inside a kernel
    c = [.70710678 + 1j, 2j, 0., 0.70710678]

    @cudaq.kernel
    def complex_vec_definition_real(i: int) -> float:
        v = [.70710678 + 1j, 2j, 0., 0.70710678][i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_definition_real(i))

    @cudaq.kernel
    def complex_vec_definition_imag(i: int) -> float:
        v = [.70710678 + 1j, 2j, 0., 0.70710678][i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_definition_imag(i))


def test_complex_definition_with_constructor():
    """Test that we can define complex lists inside kernel functions."""

    # Define a list of complex inside a kernel
    c = [complex(.70710678, 1), complex(0, 2), 0., 0.70710678]

    @cudaq.kernel
    def complex_vec_definition_real(i: int) -> float:
        v = [complex(.70710678, 1), complex(0, 2), 0., 0.70710678][i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_definition_real(i))

    @cudaq.kernel
    def complex_vec_definition_imag(i: int) -> float:
        v = [complex(.70710678, 1), complex(0, 2), 0., 0.70710678][i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_definition_imag(i))


def test_complex_definition_with_constructor_named_params():
    """Test that we can define complex lists inside kernel functions."""

    # Define a list of complex inside a kernel
    c = [
        complex(real=.70710678, imag=1),
        complex(imag=0., real=2), 0., 0.70710678
    ]

    @cudaq.kernel
    def complex_vec_definition_real(i: int) -> float:
        v = [
            complex(real=.70710678, imag=1),
            complex(imag=0., real=2), 0., 0.70710678
        ][i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_definition_real(i))

    @cudaq.kernel
    def complex_vec_definition_imag(i: int) -> float:
        v = [
            complex(real=.70710678, imag=1),
            complex(imag=0., real=2), 0., 0.70710678
        ][i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_definition_imag(i))


def test_complex_use():
    """Test that we can use complex numbers inside kernel functions."""

    # Use a complex inside np in a kernel (sin)
    @cudaq.kernel
    def complex_np_use_real() -> float:
        v = np.sin(np.pi / 2 + 0j)
        return v.real

    t = np.sin(np.pi / 2 + 0j).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> float:
        v = np.sin(np.pi / 2 + 0j)
        return v.imag

    t = np.sin(np.pi / 2 + 0j).imag
    assert is_close(t, complex_np_use_imag())

    # Use a complex inside np in a kernel (cos)
    @cudaq.kernel
    def complex_np_use_real() -> float:
        v = np.cos(np.pi / 2 + 0j)
        return v.real

    t = np.cos(np.pi / 2 + 0j).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> float:
        v = np.cos(np.pi / 2 + 0j)
        return v.imag

    t = np.cos(np.pi / 2 + 0j).imag
    assert is_close(t, complex_np_use_imag())

    # Use a complex inside np in a kernel (sqrt)
    @cudaq.kernel
    def complex_np_use_real() -> float:
        v = np.sqrt(np.pi / 2 + 0j)
        return v.real

    t = np.sqrt(np.pi / 2 + 0j).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> float:
        v = np.sqrt(np.pi / 2 + 0j)
        return v.imag

    t = np.sqrt(np.pi / 2 + 0j).imag
    assert is_close(t, complex_np_use_imag())

    # Use a complex inside np in a kernel (exp)
    @cudaq.kernel
    def complex_np_use_real() -> float:
        v = np.exp(np.pi / 2 + 0j)
        return v.real

    t = np.exp(np.pi / 2 + 0j).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> float:
        v = np.exp(np.pi / 2 + 0j)
        return v.imag

    t = np.exp(np.pi / 2 + 0j).imag
    assert is_close(t, complex_np_use_imag())


# np.complex128


def test_np_complex128_params():
    """Test that we can pass complex lists to kernel functions."""

    # Pass a list of complex as a parameter
    c = [
        np.complex128(.70710678 + 1j),
        np.complex128(0. + 2j),
        np.complex128(0.),
        np.complex128(0.70710678)
    ]

    @cudaq.kernel
    def complex_vec_param(vec: list[np.complex128], i: int) -> complex:
        return vec[i]

    for i in range(len(c)):
        is_close(c[i].real, complex_vec_param(c, i).real)
    for i in range(len(c)):
        is_close(c[i].imag, complex_vec_param(c, i).imag)

    @cudaq.kernel
    def complex_vec_param_real(vec: list[np.complex128], i: int) -> float:
        v = vec[i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_param_real(c, i))

    @cudaq.kernel
    def complex_vec_param_imag(vec: list[np.complex128], i: int) -> float:
        v = vec[i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_param_imag(c, i))


def test_np_complex128_capture():
    """Test that we can pass complex lists to kernel functions."""

    # Capture a list of complex
    c = [
        np.complex128(.70710678 + 1j),
        np.complex128(0. + 2j),
        np.complex128(0.),
        np.complex128(0.70710678)
    ]

    @cudaq.kernel
    def complex_vec_capture_real(i: int) -> float:
        v = c[i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_capture_real(i))

    @cudaq.kernel
    def complex_vec_capture_imag(i: int) -> float:
        v = c[i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_capture_imag(i))


def test_np_complex128_definition():
    """Test that we can define complex lists inside kernel functions."""

    # Define a list of complex inside a kernel
    c = [
        np.complex128(.70710678 + 1j),
        np.complex128(0. + 2j),
        np.complex128(0.),
        np.complex128(0.70710678)
    ]

    @cudaq.kernel
    def complex_vec_definition_real(i: int) -> float:
        v = [
            np.complex128(.70710678 + 1j),
            np.complex128(0. + 2j),
            np.complex128(0.),
            np.complex128(0.70710678)
        ][i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_definition_real(i))

    @cudaq.kernel
    def complex_vec_definition_imag(i: int) -> float:
        v = [
            np.complex128(.70710678 + 1j),
            np.complex128(0. + 2j),
            np.complex128(0.),
            np.complex128(0.70710678)
        ][i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_definition_imag(i))


def test_np_complex128_use():
    """Test that we can use complex numbers inside kernel functions."""

    # Use a complex inside np in a kernel (sin)
    @cudaq.kernel
    def complex_np_use_real() -> float:
        v = np.sin(np.complex128(np.pi / 2 + 1 + 1j))
        return v.real

    t = np.sin(np.complex128(np.pi / 2 + 1 + 1j)).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> float:
        v = np.sin(np.complex128(np.pi / 2 + 1 + 1j))
        return v.imag

    t = np.sin(np.complex128(np.pi / 2 + 1 + 1j)).imag
    assert is_close(t, complex_np_use_imag())

    # Use a complex inside np in a kernel (cos)
    @cudaq.kernel
    def complex_np_use_real() -> float:
        v = np.cos(np.complex128(np.pi / 2 + 1 + 1j))
        return v.real

    t = np.cos(np.complex128(np.pi / 2 + 1 + 1j)).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> float:
        v = np.cos(np.complex128(np.pi / 2 + 1 + 1j))
        return v.imag

    t = np.cos(np.complex128(np.pi / 2 + 1 + 1j)).imag
    assert is_close(t, complex_np_use_imag())

    # Use a complex inside np in a kernel (sqrt)
    @cudaq.kernel
    def complex_np_use_real() -> float:
        v = np.sqrt(np.complex128(np.pi / 2 + 1 + 1j))
        return v.real

    t = np.sqrt(np.complex128(np.pi / 2 + 1 + 1j)).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> float:
        v = np.sqrt(np.complex128(np.pi / 2 + 1 + 1j))
        return v.imag

    t = np.sqrt(np.complex128(np.pi / 2 + 1 + 1j)).imag
    assert is_close(t, complex_np_use_imag())

    # Use a complex inside np in a kernel (exp)
    @cudaq.kernel
    def complex_np_use_real() -> float:
        v = np.exp(np.complex128(np.pi / 2 + 1 + 1j))
        return v.real

    t = np.exp(np.complex128(np.pi / 2 + 1 + 1j)).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> float:
        v = np.exp(np.complex128(np.pi / 2 + 1 + 1j))
        return v.imag

    t = np.exp(np.complex128(np.pi / 2 + 1 + 1j)).imag
    assert is_close(t, complex_np_use_imag())


# Complex64


def test_np_complex64_params():
    """Test that we can pass complex lists to kernel functions."""

    # Pass a list of complex as a parameter
    c = [
        np.complex64(.70710678 + 1j),
        np.complex64(0. + 2j),
        np.complex64(0.),
        np.complex64(0.70710678)
    ]

    @cudaq.kernel
    def complex_vec_param(vec: list[np.complex64], i: int) -> np.complex64:
        return vec[i]

    for i in range(len(c)):
        is_close(c[i].real, complex_vec_param(c, i).real)
    for i in range(len(c)):
        is_close(c[i].imag, complex_vec_param(c, i).imag)

    @cudaq.kernel
    def complex_vec_param_real(vec: list[np.complex64], i: int) -> np.float32:
        v = vec[i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_param_real(c, i))

    @cudaq.kernel
    def complex_vec_param_imag(vec: list[np.complex64], i: int) -> np.float32:
        v = vec[i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_param_imag(c, i))


def test_np_complex64_capture():
    """Test that we can pass complex lists to kernel functions."""

    # Capture a list of complex
    c = [
        np.complex64(.70710678 + 1j),
        np.complex64(0. + 2j),
        np.complex64(0.),
        np.complex64(0.70710678)
    ]

    @cudaq.kernel
    def complex_vec_capture_real(i: int) -> np.float32:
        v = c[i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_capture_real(i))

    @cudaq.kernel
    def complex_vec_capture_imag(i: int) -> np.float32:
        v = c[i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_capture_imag(i))


def test_np_complex64_definition():
    """Test that we can define complex lists inside kernel functions."""

    # Define a list of complex inside a kernel
    c = [
        np.complex64(.70710678 + 1j),
        np.complex64(0. + 2j),
        np.complex64(0.),
        np.complex64(0.70710678)
    ]

    @cudaq.kernel
    def complex_vec_definition_real(i: int) -> np.float32:
        v = [
            np.complex64(.70710678 + 1j),
            np.complex64(0. + 2j),
            np.complex64(0.),
            np.complex64(0.70710678)
        ][i]
        return v.real

    for i in range(len(c)):
        assert is_close(c[i].real, complex_vec_definition_real(i))

    @cudaq.kernel
    def complex_vec_definition_imag(i: int) -> np.float32:
        v = [
            np.complex64(.70710678 + 1j),
            np.complex64(0. + 2j),
            np.complex64(0.),
            np.complex64(0.70710678)
        ][i]
        return v.imag

    for i in range(len(c)):
        assert is_close(c[i].imag, complex_vec_definition_imag(i))


def test_np_complex64_use():
    """Test that we can use complex numbers inside kernel functions."""

    # Use a complex inside np in a kernel (sin)
    @cudaq.kernel
    def complex_np_use_real() -> np.float32:
        v = np.sin(np.complex64((np.pi / 2. + 1) + 1j))
        return v.real

    t = np.sin(np.complex64((np.pi / 2. + 1) + 1j)).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> np.float32:
        v = np.sin(np.complex64((np.pi / 2. + 1) + 1j))
        return v.imag

    t = np.sin(np.complex64((np.pi / 2. + 1) + 1j)).imag
    assert is_close(t, complex_np_use_imag())

    # Use a complex inside np in a kernel (cos)
    @cudaq.kernel
    def complex_np_use_real() -> np.float32:
        v = np.cos(np.complex64((np.pi / 2. + 1) + 1j))
        return v.real

    t = np.cos(np.complex64((np.pi / 2. + 1) + 1j)).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> np.float32:
        v = np.cos(np.complex64((np.pi / 2. + 1) + 1j))
        return v.imag

    t = np.cos(np.complex64((np.pi / 2. + 1) + 1j)).imag
    assert is_close(t, complex_np_use_imag())

    # Use a complex inside np in a kernel (sqrt)
    @cudaq.kernel
    def complex_np_use_real() -> np.float32:
        v = np.sqrt(np.complex64((np.pi / 2. + 1) + 1j))
        return v.real

    t = np.sqrt(np.complex64((np.pi / 2. + 1) + 1j)).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> np.float32:
        v = np.sqrt(np.complex64((np.pi / 2. + 1) + 1j))
        return v.imag

    t = np.sqrt(np.complex64((np.pi / 2. + 1) + 1j)).imag
    assert is_close(t, complex_np_use_imag())

    # Use a complex inside np in a kernel (exp)
    @cudaq.kernel
    def complex_np_use_real() -> np.float32:
        v = np.exp(np.complex64((np.pi / 2. + 1) + 1j))
        return v.real

    t = np.exp(np.complex64((np.pi / 2. + 1) + 1j)).real
    assert is_close(t, complex_np_use_real())

    @cudaq.kernel
    def complex_np_use_imag() -> np.float32:
        v = np.exp(np.complex64((np.pi / 2. + 1) + 1j))
        return v.imag

    t = np.exp(np.complex64((np.pi / 2. + 1) + 1j)).imag
    assert is_close(t, complex_np_use_imag())


def test_complex_list_parameter_promotion():

    @cudaq.kernel
    def kernel(c: list[complex], i: int) -> complex:
        return c[i]

    def non_kernel(c: list[complex], i: int) -> complex:
        return c[i]

    def check(c: any):
        for i in range(len(c)):
            is_close(kernel(c, i).real, non_kernel(c, i).real)
        for i in range(len(c)):
            is_close(kernel(c, i).imag, non_kernel(c, i).imag)

    check([0 + 2j])
    check([0.70710678, 0.70710678 + 2j])
    check([0.70710678, np.complex128(0.70710678 + 2j)])
    check([0.70710678, np.complex64(0.70710678 + 2j)])
    check([0.70710678])
    check([1])
    check([0.70710678, 1 + 2j])
    check([0, 0.70710678 + 2j])
    check([0, 1.0])
    check([0, 1])
    check([0, 0.70710678 + 2j, True])


def test_complex128_list_parameter_promotion():

    @cudaq.kernel
    def kernel(c: list[np.complex128], i: int) -> np.complex128:
        return c[i]

    def non_kernel(c: list[np.complex128], i: int) -> np.complex128:
        return c[i]

    def check(c: any):
        for i in range(len(c)):
            is_close(kernel(c, i).real, non_kernel(c, i).real)
        for i in range(len(c)):
            is_close(kernel(c, i).imag, non_kernel(c, i).imag)

    check([0 + 2j])
    check([0.70710678, 0.70710678 + 2j])
    check([0.70710678, np.complex128(0.70710678 + 2j)])
    check([0.70710678, np.complex64(0.70710678 + 2j)])
    check([0.70710678])
    check([1])
    check([0.70710678, 1 + 2j])
    check([0, 0.70710678 + 2j])
    check([0, 1.0])
    check([0, 1])
    check([0, 0.70710678 + 2j, True])


def test_complex64_list_parameter_promotion():

    @cudaq.kernel
    def kernel(c: list[np.complex64], i: int) -> np.complex64:
        return c[i]

    def non_kernel(c: list[np.complex64], i: int) -> np.complex64:
        return c[i]

    def check(c: any):
        for i in range(len(c)):
            is_close(kernel(c, i).real, non_kernel(c, i).real)
        for i in range(len(c)):
            is_close(kernel(c, i).imag, non_kernel(c, i).imag)

    check([0 + 2j])
    check([0.70710678, 0.70710678 + 2j])
    check([0.70710678, np.complex128(0.70710678 + 2j)])
    check([0.70710678, np.complex64(0.70710678 + 2j)])
    check([0.70710678])
    check([1])
    check([0.70710678, 1 + 2j])
    check([0, 0.70710678 + 2j])
    check([0, 1.0])
    check([0, 1])
    check([0, 0.70710678 + 2j, True])
