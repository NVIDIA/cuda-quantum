"""Tests for cudaq.contrib.embeddings (amplitude_encode & angular_encode)."""

import numpy as np
import math
import sys
import os

# ---------------------------------------------------------------------------
# Ensure we can import from the installed cudaq package
# ---------------------------------------------------------------------------
import cudaq
from cudaq.contrib.embeddings import amplitude_encode, angular_encode

# ===================================================================== #
#                       amplitude_encode tests                          #
# ===================================================================== #


def test_amplitude_encode_basic_power_of_2():
    """4-element vector (already 2^2) should encode without padding."""
    data = [0.5, 0.5, 0.5, 0.5]
    state = amplitude_encode(data)
    assert state.num_qubits() == 2
    # Each amplitude should be 0.5 / norm(0.5,0.5,0.5,0.5) = 0.5/1.0 = 0.5
    print("[PASS] test_amplitude_encode_basic_power_of_2")


def test_amplitude_encode_normalization():
    """Output state must have L2 norm == 1."""
    data = [3.0, 4.0]
    state = amplitude_encode(data)
    # Amplitudes should be [0.6, 0.8]
    amp0 = abs(state.amplitude([0]))
    amp1 = abs(state.amplitude([1]))
    norm = math.sqrt(amp0**2 + amp1**2)
    assert abs(norm - 1.0) < 1e-10, f"Norm was {norm}, expected 1.0"
    assert abs(amp0 - 0.6) < 1e-10, f"amp0 was {amp0}, expected 0.6"
    assert abs(amp1 - 0.8) < 1e-10, f"amp1 was {amp1}, expected 0.8"
    print("[PASS] test_amplitude_encode_normalization")


def test_amplitude_encode_padding():
    """3-element vector with pad=0 should be padded to length 4."""
    data = [0.5, 0.5, 0.5]
    state = amplitude_encode(data, pad=0)
    assert state.num_qubits() == 2  # 4 amplitudes = 2 qubits

    # Expected: [0.5, 0.5, 0.5, 0.0] normalized
    # norm = sqrt(0.25 + 0.25 + 0.25) = sqrt(0.75)
    expected_norm = math.sqrt(0.75)
    amp0 = abs(state.amplitude([0, 0]))
    amp3 = abs(state.amplitude([1, 1]))
    assert abs(amp0 - 0.5 / expected_norm) < 1e-6, \
        f"amp[00] was {amp0}, expected {0.5/expected_norm}"
    assert abs(amp3) < 1e-10, f"amp[11] was {amp3}, expected 0.0"
    print("[PASS] test_amplitude_encode_padding")


def test_amplitude_encode_issue_example():
    """Reproduce the exact example from GitHub issue #2982."""
    state = amplitude_encode([0.5, 0.5, 0.5], pad=0)
    amp0 = abs(state.amplitude([0, 0]))
    amp1 = abs(state.amplitude([0, 1]))
    amp2 = abs(state.amplitude([1, 0]))
    amp3 = abs(state.amplitude([1, 1]))

    # Expected: ~0.5773 for first three, 0 for last
    assert abs(amp0 - 0.5773) < 1e-3, f"got {amp0}"
    assert abs(amp1 - 0.5773) < 1e-3, f"got {amp1}"
    assert abs(amp2 - 0.5773) < 1e-3, f"got {amp2}"
    assert abs(amp3) < 1e-10, f"got {amp3}"
    print("[PASS] test_amplitude_encode_issue_example")


def test_amplitude_encode_single_element():
    """Single element [1.0] gives a 0-qubit state (2^0 = 1 amplitude)."""
    state = amplitude_encode([1.0])
    assert state.num_qubits() == 0  # 2^0 = 1 amplitude
    print("[PASS] test_amplitude_encode_single_element")


def test_amplitude_encode_numpy_input():
    """Should accept numpy arrays."""
    data = np.array([1.0, 0.0, 0.0, 0.0])
    state = amplitude_encode(data)
    amp0 = abs(state.amplitude([0, 0]))
    assert abs(amp0 - 1.0) < 1e-10
    print("[PASS] test_amplitude_encode_numpy_input")


def test_amplitude_encode_complex_input():
    """Should handle complex-valued inputs."""
    data = [1.0 + 0j, 0.0 + 1j]
    state = amplitude_encode(data)
    # norm = sqrt(1 + 1) = sqrt(2)
    assert state.num_qubits() == 1
    print("[PASS] test_amplitude_encode_complex_input")


def test_amplitude_encode_large_padding():
    """5 elements should pad to 8 (2^3)."""
    data = [1.0, 1.0, 1.0, 1.0, 1.0]
    state = amplitude_encode(data, pad=0)
    assert state.num_qubits() == 3  # 8 amplitudes = 3 qubits
    print("[PASS] test_amplitude_encode_large_padding")


def test_amplitude_encode_error_empty():
    """Empty input should raise ValueError."""
    try:
        amplitude_encode([])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "empty" in str(e).lower()
    print("[PASS] test_amplitude_encode_error_empty")


def test_amplitude_encode_error_no_pad():
    """Non-power-of-2 without pad should raise ValueError."""
    try:
        amplitude_encode([1.0, 2.0, 3.0])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "power of 2" in str(e).lower()
    print("[PASS] test_amplitude_encode_error_no_pad")


def test_amplitude_encode_error_zero_vector():
    """All-zero vector should raise ValueError."""
    try:
        amplitude_encode([0.0, 0.0, 0.0, 0.0])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "zero" in str(e).lower()
    print("[PASS] test_amplitude_encode_error_zero_vector")


# ===================================================================== #
#                        angular_encode tests                           #
# ===================================================================== #


def test_angular_encode_ry():
    """Basic Ry angular encoding should produce correct circuit."""
    data = [0.1, 0.2, 0.3]
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(3)
    angular_encode(kernel, qubits, data, 3, rotation='Y')

    circuit_str = cudaq.draw(kernel)
    assert "ry(0.1)" in circuit_str
    assert "ry(0.2)" in circuit_str
    assert "ry(0.3)" in circuit_str
    print("[PASS] test_angular_encode_ry")


def test_angular_encode_rx():
    """Rx angular encoding."""
    data = [0.5, 1.0]
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    angular_encode(kernel, qubits, data, 2, rotation='X')

    circuit_str = cudaq.draw(kernel)
    assert "rx(0.5)" in circuit_str
    assert "rx(1)" in circuit_str
    print("[PASS] test_angular_encode_rx")


def test_angular_encode_rz():
    """Rz angular encoding."""
    data = [0.7, 0.8]
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    angular_encode(kernel, qubits, data, 2, rotation='Z')

    circuit_str = cudaq.draw(kernel)
    assert "rz(0.7)" in circuit_str
    assert "rz(0.8)" in circuit_str
    print("[PASS] test_angular_encode_rz")


def test_angular_encode_case_insensitive():
    """Rotation parameter should be case-insensitive."""
    data = [0.1]
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(1)
    angular_encode(kernel, qubits, data, 1, rotation='y')

    circuit_str = cudaq.draw(kernel)
    assert "ry(0.1)" in circuit_str
    print("[PASS] test_angular_encode_case_insensitive")


def test_angular_encode_default_rotation():
    """Default rotation should be 'Y'."""
    data = [0.4, 0.5]
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    angular_encode(kernel, qubits, data, 2)  # no rotation arg

    circuit_str = cudaq.draw(kernel)
    assert "ry(0.4)" in circuit_str
    assert "ry(0.5)" in circuit_str
    print("[PASS] test_angular_encode_default_rotation")


def test_angular_encode_parameterized():
    """Should work with parameterized kernels (QuakeValue list)."""
    kernel, params = cudaq.make_kernel(list)
    qubits = kernel.qalloc(3)
    angular_encode(kernel, qubits, params, 3, rotation='Y')

    circuit_str = cudaq.draw(kernel, [0.1, 0.2, 0.3])
    assert "ry(0.1)" in circuit_str
    assert "ry(0.2)" in circuit_str
    assert "ry(0.3)" in circuit_str
    print("[PASS] test_angular_encode_parameterized")


def test_angular_encode_issue_example():
    """Reproduce the exact circuit from GitHub issue #2982."""
    data = [0.1, 0.2, 0.3]
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(3)
    angular_encode(kernel, qubits, data, 3, rotation='Y')

    circuit_str = cudaq.draw(kernel)
    # The issue shows this exact circuit:
    #      ╭─────────╮
    # q0 : ┤ ry(0.1) ├
    #      ├─────────┤
    # q1 : ┤ ry(0.2) ├
    #      ├─────────┤
    # q2 : ┤ ry(0.3) ├
    #      ╰─────────╯
    assert "ry(0.1)" in circuit_str
    assert "ry(0.2)" in circuit_str
    assert "ry(0.3)" in circuit_str
    assert "q0" in circuit_str
    assert "q1" in circuit_str
    assert "q2" in circuit_str
    print("[PASS] test_angular_encode_issue_example")


def test_angular_encode_error_invalid_rotation():
    """Invalid rotation axis should raise ValueError."""
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(1)
    try:
        angular_encode(kernel, qubits, [0.1], 1, rotation='W')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid rotation" in str(e)
    print("[PASS] test_angular_encode_error_invalid_rotation")


def test_angular_encode_error_too_few_qubits():
    """num_qubits < len(data) should raise ValueError."""
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    try:
        angular_encode(kernel, qubits, [0.1, 0.2, 0.3], 2, rotation='Y')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_qubits" in str(e)
    print("[PASS] test_angular_encode_error_too_few_qubits")


# ===================================================================== #
#                              Run all                                  #
# ===================================================================== #

if __name__ == "__main__":
    print("=" * 60)
    print("Running amplitude_encode tests")
    print("=" * 60)
    test_amplitude_encode_basic_power_of_2()
    test_amplitude_encode_normalization()
    test_amplitude_encode_padding()
    test_amplitude_encode_issue_example()
    test_amplitude_encode_single_element()
    test_amplitude_encode_numpy_input()
    test_amplitude_encode_complex_input()
    test_amplitude_encode_large_padding()
    test_amplitude_encode_error_empty()
    test_amplitude_encode_error_no_pad()
    test_amplitude_encode_error_zero_vector()

    print()
    print("=" * 60)
    print("Running angular_encode tests")
    print("=" * 60)
    test_angular_encode_ry()
    test_angular_encode_rx()
    test_angular_encode_rz()
    test_angular_encode_case_insensitive()
    test_angular_encode_default_rotation()
    test_angular_encode_parameterized()
    test_angular_encode_issue_example()
    test_angular_encode_error_invalid_rotation()
    test_angular_encode_error_too_few_qubits()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)