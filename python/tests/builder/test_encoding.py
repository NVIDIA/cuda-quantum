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


@pytest.mark.parametrize("rotation, angles, expected", [
    ('Y', [0.1, 0.2, 0.3], ['ry(0.1)', 'ry(0.2)', 'ry(0.3)']),
    ('X', [0.5, 1.0], ['rx(0.5)', 'rx(1']),
    ('Z', [0.7, 0.8], ['rz(0.7)', 'rz(0.8)']),
    (None, [0.4, 0.5], ['ry(0.4)', 'ry(0.5)']),
    ('y', [0.1], ['ry(0.1)']),
])
def test_angular_encode_builder_draw(rotation, angles, expected):
    kernel = cudaq.make_kernel()
    q = kernel.qalloc(len(angles))
    if rotation is None:
        cudaq.contrib.angular_encode(kernel, q, angles)
    else:
        cudaq.contrib.angular_encode(kernel, q, angles, rotation=rotation)
    drawn = cudaq.draw(kernel)
    for operation in expected:
        assert operation in drawn


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
