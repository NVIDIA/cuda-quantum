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


def test_amplitude_encode_issue_example():
    state = cudaq.amplitude_encode([0.5, 0.5, 0.5], pad=0)
    want = np.array([0.5773, 0.5773, 0.5773, 0.0], dtype=cudaq.complex())
    got = np.asarray(state)
    assert np.allclose(got, want, rtol=1e-3)
    assert np.isclose(np.linalg.norm(got), 1.0)


def test_amplitude_encode_already_power_of_two():
    state = cudaq.amplitude_encode([1.0, 0.0, 0.0, 0.0], pad=0)
    got = np.asarray(state)
    assert np.allclose(got, [1.0, 0.0, 0.0, 0.0])


def test_amplitude_encode_list_and_ndarray():
    from_list = cudaq.amplitude_encode([0.5, 0.5, 0.5], pad=0)
    from_array = cudaq.amplitude_encode(np.array([0.5, 0.5, 0.5], dtype=float),
                                        pad=0)
    assert np.allclose(np.asarray(from_list), np.asarray(from_array))


def test_amplitude_encode_state_round_trip():
    state = cudaq.amplitude_encode([0.5, 0.5, 0.5], pad=0)
    again = cudaq.amplitude_encode(state, pad=0)
    assert np.allclose(np.asarray(state), np.asarray(again))


def test_amplitude_encode_empty():
    with pytest.raises(ValueError, match="non-empty"):
        cudaq.amplitude_encode([], pad=0)


def test_amplitude_encode_zero_vector():
    with pytest.raises(ValueError, match="zero vector"):
        cudaq.amplitude_encode([0.0, 0.0, 0.0], pad=0)


def test_amplitude_encode_2d_rejected():
    with pytest.raises(ValueError, match="1D vector"):
        cudaq.amplitude_encode(np.eye(2), pad=0)
