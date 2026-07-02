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


def test_amplitude_encode_basic_power_of_two():
    state = cudaq.contrib.amplitude_encode([0.5, 0.5, 0.5, 0.5])
    assert state.num_qubits() == 2
    got = np.asarray(state)
    assert np.allclose(got, [0.5, 0.5, 0.5, 0.5])


def test_amplitude_encode_normalization():
    state = cudaq.contrib.amplitude_encode([3.0, 4.0])
    got = np.asarray(state)
    assert np.allclose(got, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(got), 1.0)


def test_amplitude_encode_padding():
    state = cudaq.contrib.amplitude_encode([0.5, 0.5, 0.5], pad=0)
    assert state.num_qubits() == 2
    expected_norm = np.sqrt(0.75)
    got = np.asarray(state)
    want = np.array([0.5, 0.5, 0.5, 0.0], dtype=cudaq.complex()) / expected_norm
    assert np.allclose(got, want)


def test_amplitude_encode_issue_example():
    state = cudaq.contrib.amplitude_encode([0.5, 0.5, 0.5], pad=0)
    want = np.array([0.5773, 0.5773, 0.5773, 0.0], dtype=cudaq.complex())
    got = np.asarray(state)
    assert np.allclose(got, want, rtol=1e-3)
    assert np.isclose(np.linalg.norm(got), 1.0)


def test_amplitude_encode_already_power_of_two():
    state = cudaq.contrib.amplitude_encode([1.0, 0.0, 0.0, 0.0], pad=0)
    got = np.asarray(state)
    assert np.allclose(got, [1.0, 0.0, 0.0, 0.0])


def test_amplitude_encode_single_element():
    state = cudaq.contrib.amplitude_encode([1.0])
    assert state.num_qubits() == 0
    got = np.asarray(state)
    assert np.allclose(got, [1.0])


def test_amplitude_encode_complex_input():
    state = cudaq.contrib.amplitude_encode([1.0 + 0j, 0.0 + 1j])
    assert state.num_qubits() == 1
    got = np.asarray(state)
    assert np.isclose(np.linalg.norm(got), 1.0)
    assert np.allclose(got, [1.0 / np.sqrt(2), 1.0j / np.sqrt(2)])


def test_amplitude_encode_large_padding():
    state = cudaq.contrib.amplitude_encode([1.0, 1.0, 1.0, 1.0, 1.0], pad=0)
    assert state.num_qubits() == 3
    got = np.asarray(state)
    assert got.size == 8
    assert np.isclose(np.linalg.norm(got), 1.0)


def test_amplitude_encode_list_and_ndarray():
    from_list = cudaq.contrib.amplitude_encode([0.5, 0.5, 0.5], pad=0)
    from_array = cudaq.contrib.amplitude_encode(np.array([0.5, 0.5, 0.5],
                                                         dtype=float),
                                                pad=0)
    assert np.allclose(np.asarray(from_list), np.asarray(from_array))


def test_amplitude_encode_state_round_trip():
    state = cudaq.contrib.amplitude_encode([0.5, 0.5, 0.5], pad=0)
    again = cudaq.contrib.amplitude_encode(state, pad=0)
    assert np.allclose(np.asarray(state), np.asarray(again))


def test_amplitude_encode_empty():
    with pytest.raises(ValueError, match="non-empty"):
        cudaq.contrib.amplitude_encode([], pad=0)


def test_amplitude_encode_zero_vector():
    with pytest.raises(ValueError, match="zero vector"):
        cudaq.contrib.amplitude_encode([0.0, 0.0, 0.0], pad=0)


def test_amplitude_encode_2d_rejected():
    with pytest.raises(ValueError, match="1D vector"):
        cudaq.contrib.amplitude_encode(np.eye(2), pad=0)


def test_amplitude_encode_2d_list_rejected():
    # A nested list/tuple must be rejected like the equivalent 2-D array,
    # rather than being silently flattened into a 1-D vector.
    with pytest.raises(ValueError, match="1D vector"):
        cudaq.contrib.amplitude_encode([[0.5, 0.5], [0.5, 0.5]], pad=0)
    with pytest.raises(ValueError, match="1D vector"):
        cudaq.contrib.amplitude_encode(((0.5, 0.5), (0.5, 0.5)), pad=0)
