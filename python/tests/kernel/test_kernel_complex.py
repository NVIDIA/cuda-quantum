# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os, sys

import pytest
import numpy as np
import math
from typing import List

import cudaq
from cudaq import spin

## [PYTHON_VERSION_FIX]
skipIfPythonLessThan39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="built-in collection types such as `list` not supported")

def is_equal(expected, actual):
    return np.isclose(expected, actual, atol=1e-6)

def test_complex_params():
    """Test that we can pass complex lists to kernel functions."""

    # Pass a list of complex as a parameter
    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def complex_vec_param(vec : list[complex], i: int) -> complex:
        return vec[i]

    # Returning complex is not supported yet
    # for i in range(len(c)):
    #    is_equal(c[i].real, complex_vec_param(c, i).real)
    # for i in range(len(c)):
    #    is_equal(c[i].imag, complex_vec_param(c, i).imag)

    @cudaq.kernel
    def complex_vec_param_real(vec : list[complex], i: int) -> float:
        v = vec[i]
        return v.real
    # Fails due to https://github.com/NVIDIA/cuda-quantum/issues/1529
    # for i in range(len(c)):
    #    assert is_equal(c[i].real, complex_vec_param_real(c, i))

    @cudaq.kernel
    def complex_vec_param_real_temp(vec : list[complex]) -> float:
        v = vec[1]
        return v.real
    assert is_equal(c[1], complex_vec_param_real_temp(c))

    @cudaq.kernel
    def complex_vec_param_imag(vec : list[complex], i: int) -> float:
        v = vec[i]
        return v.imag
    # Fails due to https://github.com/NVIDIA/cuda-quantum/issues/1529
    # for i in range(len(c)):
    #    assert is_equal(c[i].imag, complex_vec_param_imag(c, i))

    @cudaq.kernel
    def complex_vec_param_imag_temp(vec : list[complex]) -> float:
        v = vec[1]
        return v.imag
    assert is_equal(c[1], complex_vec_param_imag_temp(c))


def test_complex_capture():
    """Test that we can pass complex lists to kernel functions."""

    # Capture a list of complex
    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def complex_vec_capture_real(i: int) -> float:
        v = c[i]
        return v.real
    for i in range(len(c)):
        assert is_equal(c[i].real, complex_vec_capture_real(i))

    @cudaq.kernel
    def complex_vec_capture_imag(i: int) -> float:
        v = c[i]
        return v.imag
    for i in range(len(c)):
        assert is_equal(c[i].imag, complex_vec_capture_imag(i))

def test_complex_definition():
    """Test that we can define complex lists inside kernel functions."""

    # Define a list of complex inside a kernel
    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def complex_vec_definition_real(i:int) -> float:
        v = [.70710678 + 0j, 0., 0., 0.70710678][i]
        return v.real
    for i in range(len(c)):
        assert is_equal(c[i].real, complex_vec_definition_real(i))

    @cudaq.kernel
    def complex_vec_definition_imag(i:int) -> float:
        v = [.70710678 + 0j, 0., 0., 0.70710678][i]
        return v.imag
    for i in range(len(c)):
        assert is_equal(c[i].imag, complex_vec_definition_imag(i))

def test_complex_definition_with_constructor():
    """Test that we can define complex lists inside kernel functions."""

    # Define a list of complex inside a kernel
    c = [complex(.70710678, 0), 0., 0., 0.70710678]

    @cudaq.kernel
    def complex_vec_definition_real(i:int) -> float:
        v = [complex(.70710678, 0.), 0., 0., 0.70710678][i]
        return v.real
    for i in range(len(c)):
        assert is_equal(c[i].real, complex_vec_definition_real(i))

    @cudaq.kernel
    def complex_vec_definition_imag(i:int) -> float:
        v = [complex(.70710678, 0.), 0., 0., 0.70710678][i]
        return v.imag
    for i in range(len(c)):
        assert is_equal(c[i].imag, complex_vec_definition_imag(i))

def test_complex_definition_with_constructor_named():
    """Test that we can define complex lists inside kernel functions."""

    # Define a list of complex inside a kernel
    c = [complex(real=.70710678, imag=0), 0., 0., 0.70710678]

    # Use float arguments to complex constructor
    @cudaq.kernel
    def complex_vec_definition_real(i:int) -> float:
        v = [complex(real=.70710678, imag=0.), 0., 0., 0.70710678][i]
        return v.real
    for i in range(len(c)):
        assert is_equal(c[i].real, complex_vec_definition_real(i))

    @cudaq.kernel
    def complex_vec_definition_imag(i:int) -> float:
       v = [complex(real=.70710678, imag=0.), 0., 0., 0.70710678][i]
       return v.imag
    for i in range(len(c)):
        assert is_equal(c[i].imag, complex_vec_definition_imag(i))

    # Use int arguments to complex constructor
    @cudaq.kernel
    def complex_vec_definition_real_i(i:int) -> float:
        v = [complex(real=.70710678, imag=0.), 0., 0., 0.70710678][i]
        return v.real
    for i in range(len(c)):
        assert is_equal(c[i].real, complex_vec_definition_real_i(i))

    @cudaq.kernel
    def complex_vec_definition_imag_i(i:int) -> float:
        v = [complex(real=.70710678, imag=0.), 0., 0., 0.70710678][i]
        return v.imag
    for i in range(len(c)):
        assert is_equal(c[i].imag, complex_vec_definition_imag_i(i))

    # Use int in list of complex
    @cudaq.kernel
    def complex_vec_definition_real_ii(i:int) -> float:
        v = [complex(real=.70710678, imag=0.), 0., 0., 0.70710678][i]
        return v.real
    for i in range(len(c)):
        assert is_equal(c[i].real, complex_vec_definition_real_ii(i))

    @cudaq.kernel
    def complex_vec_definition_imag_ii(i:int) -> float:
        v = [complex(real=.70710678, imag=0.), 0., 0., 0.70710678][i]
        return v.imag
    for i in range(len(c)):
        assert is_equal(c[i].imag, complex_vec_definition_imag_ii(i))

def test_complex_use():
    """Test that we can use complex numbers inside kernel functions."""

    # Use a complex inside a kernel
    @cudaq.kernel
    def complex_use_real() -> float:
        v =  complex.sin(0j)
        return v.real
    assert is_equal(0., complex_use_real())

    # Use a complex inside np in a kernel
    @cudaq.kernel
    def complex_np_use_real() -> float:
        v = np.sin(0j)
        return v.real
    assert is_equal(0., complex_np_use_real())


# np.complex128

def test_np_complex128_params():
    """Test that we can pass complex lists to kernel functions."""

    # Pass a list of complex as a parameter
    c = [np.complex128(.70710678+0j),np.complex128(0.), np.complex128(0.), np.complex128(0.70710678)]

    @cudaq.kernel
    def complex_vec_param(vec : list[np.complex128], i: int) -> complex:
        return vec[i]

    # Returning complex is not supported yet
    # for i in range(len(c)):
    #    is_equal(c[i].real, complex_vec_param(c, i).real)
    # for i in range(len(c)):
    #    is_equal(c[i].imag, complex_vec_param(c, i).imag)

    @cudaq.kernel
    def complex_vec_param_real(vec : list[np.complex128], i: int) -> float:
        v = vec[i]
        return v.real
    # Fails due to https://github.com/NVIDIA/cuda-quantum/issues/1529
    # for i in range(len(c)):
    #    assert is_equal(c[i].real, complex_vec_param_real(c, i))

    @cudaq.kernel
    def complex_vec_param_real_temp(vec : list[np.complex128]) -> float:
        v = vec[1]
        return v.real
    assert is_equal(c[1], complex_vec_param_real_temp(c))

    @cudaq.kernel
    def complex_vec_param_imag(vec : list[np.complex128], i: int) -> float:
        v = vec[i]
        return v.imag
    # Fails due to https://github.com/NVIDIA/cuda-quantum/issues/1529
    # for i in range(len(c)):
    #    assert is_equal(c[i].imag, complex_vec_param_imag(c, i))

    @cudaq.kernel
    def complex_vec_param_imag_temp(vec : list[np.complex128]) -> float:
        v = vec[1]
        return v.imag
    assert is_equal(c[1], complex_vec_param_imag_temp(c))


def test_np_complex128_capture():
    """Test that we can pass complex lists to kernel functions."""

    # Capture a list of complex
    c = [np.complex128(.70710678 + 0j),np.complex128(0.), np.complex128(0.), np.complex128(0.70710678)]

    @cudaq.kernel
    def complex_vec_capture_real(i: int) -> float:
        v = c[i]
        return v.real
    for i in range(len(c)):
        assert is_equal(c[i].real, complex_vec_capture_real(i))

    @cudaq.kernel
    def complex_vec_capture_imag(i: int) -> float:
        v = c[i]
        return v.imag
    for i in range(len(c)):
        assert is_equal(c[i].imag, complex_vec_capture_imag(i))

def test_np_complex128_definition():
    """Test that we can define complex lists inside kernel functions."""

    # Define a list of complex inside a kernel
    c = [np.complex128(.70710678 + 0j),np.complex128(0.), np.complex128(0.), np.complex128(0.70710678)]

    @cudaq.kernel
    def complex_vec_definition_real(i:int) -> float:
        v = [np.complex128(.70710678 + 0j),np.complex128(0.), np.complex128(0.), np.complex128(0.70710678)][i]
        return v.real
    for i in range(len(c)):
        assert is_equal(c[i].real, complex_vec_definition_real(i))

    @cudaq.kernel
    def complex_vec_definition_imag(i:int) -> float:
        v = [np.complex128(.70710678 + 0j),np.complex128(0.), np.complex128(0.), np.complex128(0.70710678)][i]
        return v.imag
    for i in range(len(c)):
        assert is_equal(c[i].imag, complex_vec_definition_imag(i))

def test_np_complex128_use():
    """Test that we can use complex numbers inside kernel functions."""

    # Use a complex inside a kernel
    @cudaq.kernel
    def complex_use_real() -> float:
        v =  complex.sin(np.complex128(0. + 0j))
        return v.real
    assert is_equal(0., complex_use_real())

    # Use a complex inside np in a kernel
    @cudaq.kernel
    def complex_np_use_real() -> float:
        v = np.sin(np.complex128(0. + 0j))
        return v.real
    assert is_equal(0., complex_np_use_real())








