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

def test_float_params():
    """Test that we can pass float lists to kernel functions."""

    f = [0., 1., 1., 0.]

    # Pass a list of float as a parameter
    @cudaq.kernel
    def float_vec_param(vec : list[float], i : int) -> float:
        return vec[i]

    # Fails due to https://github.com/NVIDIA/cuda-quantum/issues/1529
    # for i in range(len(f)):
    #    assert np.isclose(f[i], float_vec_param(f, i), atol=1e-6)
    #
    # Using below as a temporary replacement test until the issue is fixed
    def float_vec_param_temp(vec : list[float]) -> float:
        return vec[1]

    assert np.isclose(f[1], float_vec_param_temp(f), atol=1e-6)

def test_float_capture():
    """Test that we can capture float lists inside kernel functions."""

    f = [0., 1., 1., 0.]

    # Capture a list of float
    @cudaq.kernel
    def float_vec_capture(i: int) -> float:
        return f[i]

    for i in range(len(f)):
        assert np.isclose(f[i], float_vec_capture(i), atol=1e-6)


def test_float_definition():
    """Test that we can define float lists inside kernel functions."""

    f = [0., 1., 1., 0.]

    # Define a list of float inside a kernel
    @cudaq.kernel
    def float_vec_definition(i: int) -> float:
        return [0., 1., 1., 0.][i]

    for i in range(len(f)):
        assert np.isclose(f[i], float_vec_definition(i), atol=1e-6)


def test_float_use():
    """Test that we can use floats inside kernel functions."""

    # Use a float inside a kernel
    @cudaq.kernel
    def float_use() -> float:
        return math.sin(0.)

    assert np.isclose(0., float_use(), atol=1e-6)

    # Use a float inside np in a kernel
    @cudaq.kernel
    def float_np_use() -> float:
        return np.sin(0.)

    assert np.isclose(0., float_np_use(), atol=1e-6)
