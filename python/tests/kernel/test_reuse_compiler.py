# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest
import numpy as np

import os

import cudaq


def test_reuse():
    """Test that we can reuse a compiled jit across launches"""

    @cudaq.kernel
    def simple(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        x(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits - 1)):
            x.ctrl(qubit, qubits[i + 1])

    @cudaq.kernel
    def nop(numQubits: int):
        qubits = cudaq.qvector(numQubits)

    res = cudaq.sample(simple, 2, shots_count=1)
    assert (res.count("11") == 1)
    res = cudaq.sample(simple, 3, shots_count=1)
    assert (res.count("111") == 1)
    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        res = cudaq.sample(simple, 4, shots_count=1)
        assert (res.count("1111") == 1)

        res = cudaq.sample(simple, 4, shots_count=1)
        assert (res.count("1111") == 1)

        res = cudaq.sample(simple, 5, shots_count=1)
        assert (res.count("11111") == 1)

        @cudaq.kernel
        def simple(numQubits: int):
            qubits = cudaq.qvector(numQubits)

        with pytest.raises(RuntimeError):
            res = cudaq.sample(simple, 4, shots_count=1)

        simple = nop
        with pytest.raises(RuntimeError):
            res = cudaq.sample(simple, 4, shots_count=1)
    res = cudaq.sample(simple, 6, shots_count=1)
    assert (res.count("000000") == 1)


def test_reuse_different_kernel_raises():
    """Calling a different kernel inside reuse_compiler_artifacts should raise."""

    @cudaq.kernel
    def kernel_a(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        x(qubits.front())

    @cudaq.kernel
    def kernel_b(numQubits: int):
        qubits = cudaq.qvector(numQubits)

    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        cudaq.sample(kernel_a, 2, shots_count=1)
        with pytest.raises(RuntimeError):
            cudaq.sample(kernel_b, 2, shots_count=1)


def test_reuse_no_arguments():
    """A no-arg kernel should be reusable in artifact-reuse mode."""

    @cudaq.kernel
    def no_arg_kernel():
        qubits = cudaq.qvector(2)
        x(qubits[0])
        x(qubits[1])

    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        res = cudaq.sample(no_arg_kernel, shots_count=1)
        assert (res.count("11") == 1)
        res = cudaq.sample(no_arg_kernel, shots_count=1)
        assert (res.count("11") == 1)


def test_reuse_reset_after_context_exit():
    """Disabling reuse should clear saved artifacts for a fresh session."""

    @cudaq.kernel
    def ones(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        for qubit in qubits:
            x(qubit)

    @cudaq.kernel
    def zeros(numQubits: int):
        qubits = cudaq.qvector(numQubits)

    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        res = cudaq.sample(ones, 4, shots_count=1)
        assert (res.count("1111") == 1)

    # A new reuse context should start from an empty saved artifact.
    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        res = cudaq.sample(zeros, 4, shots_count=1)
        assert (res.count("0000") == 1)
        res = cudaq.sample(zeros, 4, shots_count=1)
        assert (res.count("0000") == 1)


def test_reuse_complex_arguments():
    """Reuse validation should handle list arguments properly."""

    @cudaq.kernel
    def apply_complex_angles(angles: list[complex]):
        qubits = cudaq.qvector(2)
        rx(angles[0].real, qubits[0])
        rx(angles[1].real, qubits[1])

    angles = [complex(np.pi, 0.125), complex(np.pi, -0.25)]
    same_angles_different_value = [complex(np.pi, 0.125), complex(np.pi, -0.25)]
    different_angles = [complex(np.pi, 0.125), complex(0.0, -0.25)]

    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        res = cudaq.sample(apply_complex_angles, angles, shots_count=1)
        assert (res.count("11") == 1)
        res = cudaq.sample(apply_complex_angles, angles, shots_count=1)
        assert (res.count("11") == 1)
        res = cudaq.sample(apply_complex_angles,
                           same_angles_different_value,
                           shots_count=1)
        assert (res.count("11") == 1)
        cudaq.sample(apply_complex_angles, different_angles, shots_count=1)


def test_different_launch_mode():
    """Reuse validation should reject different launch modes"""

    @cudaq.kernel
    def simple(numQubits: int):
        qubits = cudaq.qvector(numQubits)
        x(qubits.front())
        for i, qubit in enumerate(qubits.front(numQubits - 1)):
            x.ctrl(qubit, qubits[i + 1])

    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        res = cudaq.sample(simple, 4, shots_count=1)
        assert (res.count("1111") == 1)

        with pytest.raises(RuntimeError):
            res = cudaq.get_state(simple, 4)


def test_reuse_of_builder():
    kernel = cudaq.make_kernel()
    qreg = kernel.qalloc(5)
    kernel.x(qreg)
    kernel.mz(qreg)

    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        cudaq.sample(kernel, shots_count=1)
        cudaq.sample(kernel, shots_count=1)

    kernel, nQubits = cudaq.make_kernel(int)
    qreg = kernel.qalloc(nQubits)
    kernel.x(qreg)
    kernel.mz(qreg)

    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        cudaq.sample(kernel, 5, shots_count=1)
        cudaq.sample(kernel, 5, shots_count=1)


def test_reuse_with_result_no_args():

    @cudaq.kernel
    def flip() -> bool:
        q = cudaq.qubit()
        x(q)
        return mz(q)

    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        result = flip()
        assert result == True
        result = flip()  # cached kernel
        assert result == True


def test_reuse_with_result_and_args():

    @cudaq.kernel
    def count_ones(n: int) -> int:
        qubits = cudaq.qvector(n)
        for qubit in qubits:
            x(qubit)
        result = 0
        for i in range(n):
            if mz(qubits[i]):
                result += 1
        return result

    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        result = count_ones(3)
        assert result == 3
        result = count_ones(3)  # cached kernel, same arg
        assert result == 3
        result = count_ones(4)  # cached kernel, different arg
        assert result == 4


def test_reuse_via_run_with_result():

    @cudaq.kernel
    def count_ones(n: int) -> int:
        qubits = cudaq.qvector(n)
        for qubit in qubits:
            x(qubit)
        result = 0
        for i in range(n):
            if mz(qubits[i]):
                result += 1
        return result

    with cudaq.cudaq_runtime.reuse_compiler_artifacts():
        results = cudaq.run(count_ones, 3, shots_count=2)
        assert len(results) == 2
        assert all(r == 3 for r in results)
        results = cudaq.run(count_ones, 4, shots_count=2)  # cached kernel
        assert len(results) == 2
        assert all(r == 4 for r in results)
