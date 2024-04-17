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

def test_float():
    """Test that we can use float numbers inside kernel functions."""

    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    f = [0., 1., 1., 0.]

    # Pass a list of float as a parameter
    @cudaq.kernel
    def test_float_vec_param(vec : list[float], i : int) -> float: 
        return vec[i]

    # for i in range(len(f)): 
    print(f)
    # assert np.isclose(f[0], test_float_vec_param(f, 0), atol=1e-6)
    # assert np.isclose(f[2], test_float_vec_param(f, 2), atol=1e-6) 


    # Capture a list of float
    @cudaq.kernel
    def test_float_vec_capture():
        f1 = f

    counts = cudaq.sample(test_float_vec_capture)
    assert len(counts) == 0


    # Define a list of float inside a kernel
    @cudaq.kernel
    def test_float_vec_definition():
        f1 = [1.0, 0., 0., 1.]

    counts = cudaq.sample(test_float_vec_definition)
    assert len(counts) == 0

    # Use a float inside a kernel
    @cudaq.kernel
    def test_float_use():
        f1 = math.sin(0)

    counts = cudaq.sample(test_float_use)
    assert len(counts) == 0

    # Use a float inside np in a kernel
    @cudaq.kernel
    def test_float_np_use():
        f1 = np.sin(0)

    counts = cudaq.sample(test_float_np_use)
    assert len(counts) == 0

def test_complex():
    """Test that we can use complex numbers inside kernel functions."""

    # Pass a list of complex as a parameter
    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def test_complex_vec_param(vec : list[complex]):
        f1 = vec

    counts = cudaq.sample(test_complex_vec_param, c)
    assert len(counts) == 0


    # Capture a list of complex
    @cudaq.kernel
    def test_complex_vec_capture():
        f1 = c

    counts = cudaq.sample(test_complex_vec_capture)
    assert len(counts) == 0

    # Define a list of complex inside a kernel
    @cudaq.kernel
    def test_complex_vec_definition():
        f1 = [1.0 + 0j, 0., 0., 1.]

    counts = cudaq.sample(test_complex_vec_definition)
    assert len(counts) == 0

    # Use a complex inside a kernel
    @cudaq.kernel
    def test_complex_use():
        f1 = math.sin(0j)

    counts = cudaq.sample(test_complex_use)
    assert len(counts) == 0

    # Use a complex inside np in a kernel
    @cudaq.kernel
    def test_complex_np_use():
        f1 = np.sin(0j)

    counts = cudaq.sample(test_complex_np_use)
    assert len(counts) == 0