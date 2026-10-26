# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest

import cudaq


@pytest.fixture(autouse=True)
def qpp_cpu_target():
    cudaq.reset_target()
    cudaq.set_target('qpp-cpu')


def test_amplitude_encode_builder_qalloc():
    state = cudaq.contrib.amplitude_encode([0.5, 0.5, 0.5], pad=0)
    kernel = cudaq.make_kernel()
    kernel.qalloc(state)
    counts = cudaq.sample(kernel, shots_count=1000)
    assert '00' in counts
    assert '01' in counts
    assert '10' in counts
    assert '11' not in counts


def test_amplitude_encode_builder_parameterized_state():
    encoded = cudaq.contrib.amplitude_encode([1.0, 0.0], pad=0)
    kernel, state_arg = cudaq.make_kernel(cudaq.State)
    kernel.qalloc(state_arg)
    counts = cudaq.sample(kernel, encoded, shots_count=1000)
    assert counts['0'] > 900


def test_angular_encode_builder_ry():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(3)
    cudaq.contrib.angular_encode(kernel, q, [0.1, 0.2, 0.3], rotation='Y')
    drawn = cudaq.draw(kernel)
    assert 'ry(0.1)' in drawn
    assert 'ry(0.2)' in drawn
    assert 'ry(0.3)' in drawn


def test_angular_encode_builder_rx():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    cudaq.contrib.angular_encode(kernel, q, [0.5, 1.0], rotation='X')
    drawn = cudaq.draw(kernel)
    assert 'rx(0.5)' in drawn
    assert 'rx(1)' in drawn or 'rx(1.0)' in drawn


def test_angular_encode_builder_rz():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    cudaq.contrib.angular_encode(kernel, q, [0.7, 0.8], rotation='Z')
    drawn = cudaq.draw(kernel)
    assert 'rz(0.7)' in drawn
    assert 'rz(0.8)' in drawn


def test_angular_encode_builder_default_rotation():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    cudaq.contrib.angular_encode(kernel, q, [0.4, 0.5])
    drawn = cudaq.draw(kernel)
    assert 'ry(0.4)' in drawn
    assert 'ry(0.5)' in drawn


def test_angular_encode_builder_case_insensitive_rotation():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(1)
    cudaq.contrib.angular_encode(kernel, q, [0.1], rotation='y')
    assert 'ry(0.1)' in cudaq.draw(kernel)


def test_angular_encode_builder_parameterized_angles():
    kernel, angles = cudaq.make_kernel(list[float])
    q = kernel.qalloc(3)
    cudaq.contrib.angular_encode(kernel, q, angles, rotation='Y')
    drawn = cudaq.draw(kernel, [0.1, 0.2, 0.3])
    assert 'ry(0.1)' in drawn
    assert 'ry(0.2)' in drawn
    assert 'ry(0.3)' in drawn


def test_angular_encode_builder_invalid_rotation():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(1)
    with pytest.raises(ValueError, match="unsupported rotation"):
        cudaq.contrib.angular_encode(kernel, q, [0.1], rotation='W')


def test_angular_encode_builder_angle_qubit_mismatch():
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(2)
    with pytest.raises(ValueError, match="number of angles must match"):
        cudaq.contrib.angular_encode(kernel, q, [0.1, 0.2, 0.3], rotation='Y')
