# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np
import sys

import cudaq


def test_return_float_param():

    @cudaq.kernel
    def kernel(c: float) -> float:
        return c

    assert kernel(1.1234567) == 1.1234567


def test_return_float_param_cond():

    @cudaq.kernel
    def kernel(c: float, b: bool) -> float:
        if b:
            return c
        return 5.0

    assert kernel(1.1234567, True) == 1.1234567


def test_return_bool_param_cond():

    @cudaq.kernel
    def kernel(b: bool, b2: bool) -> bool:
        if b2:
            return b
        return False

    assert kernel(True, True) == True
    assert kernel(True, False) == False


def test_return_int_param_cond():

    @cudaq.kernel
    def kernel(c: int, b: bool) -> int:
        if b:
            return c
        return 2

    assert kernel(1, True) == 1
    assert kernel(1, False) == 2


def test_return_complex_capture():
    c = 1 + 2j

    @cudaq.kernel
    def kernel() -> complex:
        return c

    assert kernel() == c


def test_return_complex_param():

    @cudaq.kernel
    def kernel(c: complex) -> complex:
        return c

    assert kernel(1 + 2j) == 1 + 2j


def test_return_complex_param_copy():

    @cudaq.kernel
    def kernel(c: complex) -> complex:
        c1 = c
        return c1

    assert kernel(1 + 2j) == 1 + 2j


def test_return_complex_param_cond():

    @cudaq.kernel
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


def test_return_np_complex128_param():

    @cudaq.kernel
    def kernel(c: np.complex128) -> np.complex128:
        return c

    assert kernel(np.complex128(1 + 2j)) == 1 + 2j


def test_return_np_complex128_definition():

    @cudaq.kernel
    def kernel(c: np.float64, i: np.float64) -> np.complex64:
        return np.complex128(complex(c, i))

    assert kernel(np.float64(1.0), np.float64(2.0)) == np.complex128(1.0 + 2.0j)


def test_return_np_complex64_param():

    @cudaq.kernel
    def kernel(c: np.complex64) -> np.complex64:
        return c

    assert kernel(np.complex64(1 + 2j)) == np.complex64(1 + 2j)


def test_return_np_complex64_definition():

    @cudaq.kernel
    def kernel(c: np.float32, i: np.float32) -> np.complex64:
        return np.complex64(complex(c, i))

    assert kernel(np.float32(1.0), np.float32(2.0)) == np.complex64(1.0 + 2.0j)


def test_return_int():

    @cudaq.kernel
    def kernel() -> int:
        return 1

    assert kernel() == 1


def test_return_negative_int():

    @cudaq.kernel
    def kernel() -> int:
        return -1

    assert kernel() == -1


def test_param_negative_int():

    @cudaq.kernel
    def kernel(i: int) -> int:
        return i

    assert kernel(1) == 1
    assert kernel(-1) == -1


def test_param_negative_int_list():

    @cudaq.kernel
    def kernel(l: list[int], i: int) -> int:
        return l[i]

    lst = [0, 1, -1]
    for i in range(len(lst)):
        assert kernel(lst, i) == lst[i]
