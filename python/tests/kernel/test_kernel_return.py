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
from typing import Callable, List
import sys

import cudaq
from cudaq import spin

## [PYTHON_VERSION_FIX]
skipIfPythonLessThan39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="built-in collection types such as `list` not supported")


def test_return_complex_capture():
    c = 1 + 2j

    @cudaq.kernel
    def kernel() -> complex:
        return c

    assert kernel() == c


def test_return_complex_param_copy():

    @cudaq.kernel(verbose=True)
    def kernel(c: complex) -> complex:
        c1 = c
        return c1

    assert kernel(1 + 2j) == 1 + 2j


def test_return_float_param():

    @cudaq.kernel(verbose=True)
    def kernel(c: float) -> float:
        return c

    assert kernel(1.1234567) == 1.1234567


def test_return_float_param_cond():

    @cudaq.kernel(verbose=True)
    def kernel(c: float, b: bool) -> float:
        if b:
            return c
        return 5.0

    assert kernel(1.1234567, True) == 1.1234567


def test_return_float_2bool_param_cond():

    @cudaq.kernel(verbose=True)
    def kernel(c: float, b: bool, b2: bool) -> float:
        if b:
            return c
        return 5.0

    assert kernel(1.1234567, True, True) == 1.1234567


def test_return_bool_param_cond():

    @cudaq.kernel(verbose=True)
    def kernel(b: bool, b2: bool) -> bool:
        return b

    assert kernel(True, False) == True


def test_return_int_int_param_cond():

    @cudaq.kernel(verbose=True)
    def kernel(b: int, b2: int) -> int:
        return b

    assert kernel(42, 53) == 42


def test_return_int_param_cond():

    @cudaq.kernel(verbose=True)
    def kernel(c: int, b: bool) -> int:
        if b:
            return c
        return 2

    assert kernel(1, True) == 1
    assert kernel(1, False) == 2


# WORKS
def test_return_complex_param():

    @cudaq.kernel
    def kernel(c: complex) -> complex:
        return c

    assert kernel(1 + 2j) == 1 + 2j


def test_return_complex_param_cond():

    @cudaq.kernel(verbose=True)
    def kernel(c: complex, b: bool) -> complex:
        if b:
            return c
        return 3

    assert kernel(1 + 2j, True) == 1 + 2j


def test_return_complex_definition():

    @cudaq.kernel
    def kernel() -> complex:
        return 1 + 2j

    assert kernel() == 1 + 2j


def test_return_np_complex128():

    @cudaq.kernel
    def kernel(c: np.complex128) -> np.complex128:
        return c

    assert kernel(np.complex128(1 + 2j)) == 1 + 2j


def test_return_np_complex64():

    @cudaq.kernel(verbose=True)
    def kernel(c: np.float32, i: np.float32) -> np.complex64:
        return np.complex64(complex(c, i))

    assert kernel(np.float32(1.0), np.float32(2.0)) == np.complex64(1.0 + 2.0j)
