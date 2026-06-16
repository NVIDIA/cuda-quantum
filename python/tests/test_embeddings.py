# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import pytest

import cudaq


# ============================================================================ #
# amplitude_encode tests
# ============================================================================ #

class TestAmplitudeEncode:
    """Tests for cudaq.amplitude_encode."""

    def test_basic_normalization(self):
        """A real-valued vector is L2-normalized."""
        data = [1.0, 1.0]
        state = cudaq.amplitude_encode(data)
        expected = 1.0 / np.sqrt(2)
        assert np.isclose(abs(state[0]), expected, atol=1e-6)
        assert np.isclose(abs(state[1]), expected, atol=1e-6)

    def test_three_element_pad_zero(self):
        """A 3-element vector is zero-padded to length 4 and normalized.

        [0.5, 0.5, 0.5] with pad=0 -> [0.5, 0.5, 0.5, 0.0]
        norm = sqrt(0.5^2 * 3) = sqrt(0.75) ≈ 0.8660
        each element = 0.5 / 0.8660 ≈ 0.57735
        """
        data = [0.5, 0.5, 0.5]
        state = cudaq.amplitude_encode(data, pad=0.0)
        expected_val = 0.5 / np.sqrt(0.75)
        for i in range(3):
            assert np.isclose(abs(state[i]), expected_val, atol=1e-4)
        assert np.isclose(abs(state[3]), 0.0, atol=1e-6)

    def test_power_of_two_no_padding(self):
        """Input that is already a power of 2 requires no pad argument."""
        data = [1.0, 0.0, 0.0, 0.0]
        state = cudaq.amplitude_encode(data)
        assert np.isclose(abs(state[0]), 1.0, atol=1e-6)

    def test_power_of_two_with_pad_ignored(self):
        """pad is silently unused when length is already a power of 2."""
        data = [0.6, 0.8]
        state_with_pad = cudaq.amplitude_encode(data, pad=0.0)
        state_no_pad = cudaq.amplitude_encode(data)
        assert np.isclose(abs(state_with_pad[0]), abs(state_no_pad[0]),
                          atol=1e-6)

    def test_numpy_input(self):
        """numpy.ndarray input is accepted."""
        arr = np.array([3.0, 4.0])
        state = cudaq.amplitude_encode(arr)
        norm = 5.0
        assert np.isclose(abs(state[0]), 3.0 / norm, atol=1e-6)
        assert np.isclose(abs(state[1]), 4.0 / norm, atol=1e-6)

    def test_empty_data_raises(self):
        """Empty input raises ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            cudaq.amplitude_encode([])

    def test_all_zeros_raises(self):
        """All-zero data raises ValueError."""
        with pytest.raises(ValueError, match="all zeros"):
            cudaq.amplitude_encode([0.0, 0.0, 0.0, 0.0])

    def test_not_power_of_two_no_pad_raises(self):
        """Non-power-of-2 without pad raises ValueError."""
        with pytest.raises(ValueError, match="not a power of 2"):
            cudaq.amplitude_encode([0.1, 0.2, 0.3])

    def test_return_type_is_cudaq_state(self):
        """Return type is cudaq.State."""
        state = cudaq.amplitude_encode([1.0, 0.0])
        assert isinstance(state, cudaq.State)

    def test_state_is_normalized(self):
        """Returned state has L2-norm = 1."""
        for data in [[0.5, 0.5, 0.5],
                     [1.0, 2.0, 3.0, 4.0],
                     [0.1, 0.2]]:
            state = cudaq.amplitude_encode(data, pad=0.0 if len(data) < 4 else None)
            n = 1 << state.get_num_qubits()
            vals = np.array([abs(state[i]) for i in range(n)])
            l2_norm = np.sqrt(np.sum(vals**2))
            assert np.isclose(l2_norm, 1.0, atol=1e-5)

    def test_pad_with_nonzero_value(self):
        """Pad uses the specified constant value."""
        data = [1.0, 0.0, 0.0]
        state = cudaq.amplitude_encode(data, pad=1.0)
        # [1, 0, 0] -> [1, 0, 0, 1], norm = sqrt(2)
        # last element should be 1/sqrt(2)
        assert np.isclose(abs(state[3]), 1.0 / np.sqrt(2), atol=1e-6)


# ============================================================================ #
# angular_encode tests
# ============================================================================ #

class TestAngularEncode:
    """Tests for cudaq.kernels.angular_encode."""

    @staticmethod
    def _check_ry_rotation(angle, tolerance=1e-4):
        """Helper: Verify that a kernel with a single Ry(angle) produces the
        expected 0/1 probabilities."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(1)
        kernel.ry(angle, q)
        counts = cudaq.sample(kernel, shots_count=100000)
        p0 = counts['0'] / 100000
        p1 = counts['1'] / 100000
        expected_p1 = np.sin(angle / 2.0)**2
        assert np.isclose(p1, expected_p1, atol=tolerance)

    def test_ry_rotation(self):
        """Ry rotation matches expected probability modulation."""
        self._check_ry_rotation(np.pi / 3)

    def test_angular_encode_y(self):
        """Angular encoding with rotation='Y' applies Ry gates."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(2)
        cudaq.kernels.angular_encode(kernel, q, [0.5, 1.0], rotation='Y')
        counts = cudaq.sample(kernel, shots_count=10000)
        assert isinstance(counts, cudaq.SampleResult)

    def test_angular_encode_x(self):
        """Angular encoding with rotation='X' works."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(1)
        cudaq.kernels.angular_encode(kernel, q, [np.pi], rotation='X')
        counts = cudaq.sample(kernel, shots_count=10000)
        # rx(pi) flips |0> to -i|1>; sampling should give mostly 1
        assert counts['1'] > counts['0']

    def test_angular_encode_z(self):
        """Angular encoding with rotation='Z' works (phase rotation)."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(2)
        cudaq.kernels.angular_encode(kernel, q, [0.3, 0.7], rotation='Z')
        counts = cudaq.sample(kernel, shots_count=10000)
        assert isinstance(counts, cudaq.SampleResult)

    def test_default_rotation_is_y(self):
        """Default rotation axis is 'Y'."""
        kernel1 = cudaq.make_kernel()
        q1 = kernel1.qalloc(1)
        cudaq.kernels.angular_encode(kernel1, q1, [0.5])

        kernel2 = cudaq.make_kernel()
        q2 = kernel2.qalloc(1)
        cudaq.kernels.angular_encode(kernel2, q2, [0.5], rotation='Y')

        # Both should produce identical results
        counts1 = cudaq.sample(kernel1, shots_count=10000)
        counts2 = cudaq.sample(kernel2, shots_count=10000)
        assert np.isclose(counts1['0'] / 10000, counts2['0'] / 10000,
                          atol=0.05)

    def test_case_insensitive_rotation(self):
        """Rotation axis is case-insensitive."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(1)
        cudaq.kernels.angular_encode(kernel, q, [0.5], rotation='y')
        counts = cudaq.sample(kernel, shots_count=1000)
        assert isinstance(counts, cudaq.SampleResult)

    def test_three_qubit_encoding(self):
        """Encoding works with 3 qubits as shown in the issue."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(3)
        cudaq.kernels.angular_encode(kernel, q, [0.1, 0.2, 0.3],
                                     rotation='Y')
        counts = cudaq.sample(kernel, shots_count=10000)
        assert isinstance(counts, cudaq.SampleResult)

    def test_empty_data_raises(self):
        """Empty data raises ValueError."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(1)
        with pytest.raises(ValueError, match="non-empty"):
            cudaq.kernels.angular_encode(kernel, q, [])

    def test_mismatched_length_raises(self):
        """Data length mismatch raises ValueError."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(2)
        with pytest.raises(ValueError, match="must equal"):
            cudaq.kernels.angular_encode(kernel, q, [0.1, 0.2, 0.3])

    def test_invalid_rotation_raises(self):
        """Invalid rotation axis raises ValueError."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(1)
        with pytest.raises(ValueError, match="Unsupported rotation"):
            cudaq.kernels.angular_encode(kernel, q, [0.5], rotation='W')

    def test_numpy_data_accepted(self):
        """numpy.ndarray data is accepted for angular encoding."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(2)
        cudaq.kernels.angular_encode(kernel, q,
                                     np.array([0.3, 0.6]),
                                     rotation='Y')
        counts = cudaq.sample(kernel, shots_count=10000)
        assert isinstance(counts, cudaq.SampleResult)

    def test_multidimensional_numpy_flattened(self):
        """Multi-dimensional numpy arrays are flattened properly."""
        kernel = cudaq.make_kernel()
        q = kernel.qalloc(3)
        data = np.array([[0.1, 0.2, 0.3]])
        cudaq.kernels.angular_encode(kernel, q, data, rotation='Y')
        counts = cudaq.sample(kernel, shots_count=10000)
        assert isinstance(counts, cudaq.SampleResult)
