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
from typing import List 

import cudaq
from cudaq import spin

## [PYTHON_VERSION_FIX]
skipIfPythonLessThan39 = pytest.mark.skipif(
    sys.version_info < (3, 9),
    reason="built-in collection types such as `list` not supported")

def test_float_lists():
    """Test that we can use float numbers inside kernel functions."""

    cudaq.reset_target()
    cudaq.set_target('nvidia-fp64')

    f = [0., 1., 1., 0.]

    # Pass list of float as a parameter
    @cudaq.kernel
    def test_float_vec_param(vec : list[float]):
        f1 = vec

    counts = cudaq.sample(test_float_vec_param, f)
    assert len(counts) == 0


    # Capture list of float
    @cudaq.kernel
    def test_float_vec_capture():
        f1 = f

    counts = cudaq.sample(test_float_vec_capture)
    assert len(counts) == 0


    # Define list of float inside kernel
    @cudaq.kernel
    def test_float_vec_definition():
        f1 = [1.0, 0., 0., 1.]

    counts = cudaq.sample(test_float_vec_definition)
    assert len(counts) == 0


def test_float_lists():
    """Test that we can use complex numbers inside kernel functions."""

    # Pass list of complex as a parameter
    c = [.70710678 + 0j, 0., 0., 0.70710678]

    @cudaq.kernel
    def test_complex_vec_param(vec : list[complex]):
        f1 = vec

    counts = cudaq.sample(test_complex_vec_param, c)
    assert len(counts) == 0


    # Capture list of complex
    @cudaq.kernel
    def test_complex_vec_capture():
        f1 = c

    counts = cudaq.sample(test_complex_vec_capture)
    assert len(counts) == 0

    # Define list of complex inside kernel
    @cudaq.kernel
    def test_complex_vec_definition():
        f1 = [1.0 + 0j, 0., 0., 1.]


    counts = cudaq.sample(test_complex_vec_definition)
    assert len(counts) == 0
