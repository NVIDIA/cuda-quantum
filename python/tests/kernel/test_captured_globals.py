# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Regression tests for https://github.com/NVIDIA/cuda-quantum/issues/2608.
# A module-level Python scalar used as a rotation parameter must be lifted into
# the kernel signature as an f64 value, not as `!cc.ptr<f64>`.

import numpy as np
import pytest

import cudaq

# Module-scope globals — the exact shape reported in the issue.
ANGLE = 3.14
ANGLE_INT = 2
SCALE = 2.0


@pytest.fixture(autouse=True)
def run_and_clear_registries():
    yield
    cudaq.__clearKernelRegistries()


def test_issue_2608_reproducer():
    """The exact reproducer from issue #2608 must compile and run."""

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        rz(ANGLE, qubit)
        mz(qubit)

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


def test_rx_with_captured_float():
    """`rx` must accept a captured float global."""

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        rx(ANGLE, qubit)
        mz(qubit)

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


def test_ry_with_captured_float():
    """`ry` must accept a captured float global."""

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        ry(ANGLE, qubit)
        mz(qubit)

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


def test_rz_with_captured_float():
    """`rz` must accept a captured float global (issue #2608 reproducer)."""

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        rz(ANGLE, qubit)
        mz(qubit)

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


def test_r1_with_captured_float():
    """`r1` must accept a captured float global."""

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        r1(ANGLE, qubit)
        mz(qubit)

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


def test_rotation_with_captured_int():
    """Captured int globals must promote to f64 for rotation parameters."""

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        rz(ANGLE_INT, qubit)
        mz(qubit)

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


def test_rotation_with_expression_over_captured_global():
    """Captured globals must flow through arithmetic into rotation params."""

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        rz(SCALE * ANGLE, qubit)
        mz(qubit)

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


def test_controlled_rotation_with_captured_global():
    """Controlled rotations must accept captured globals as the angle."""

    @cudaq.kernel
    def kernel():
        ctrl = cudaq.qubit()
        tgt = cudaq.qubit()
        x(ctrl)
        crz(ANGLE, ctrl, tgt)
        mz(ctrl)
        mz(tgt)

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


def test_captured_global_ir_is_f64_not_pointer():
    """The lifted captured argument must be f64, never !cc.ptr<f64>.

    This is the direct assertion of the bug described in issue #2608.
    """

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        rz(ANGLE, qubit)
        mz(qubit)

    ir = str(kernel)
    assert "f64" in ir
    assert "!cc.ptr<f64>" not in ir.split("func.func @__nvqpp")[1].split(
        "return")[0], (
            f"Captured global was lowered as a pointer slot. IR:\n{ir}")


def test_multiple_captured_globals_in_one_kernel():
    """Several captured globals in one kernel must all lift as values."""

    @cudaq.kernel
    def kernel():
        qubit = cudaq.qubit()
        rz(ANGLE, qubit)
        rx(ANGLE, qubit)
        ry(SCALE, qubit)
        r1(ANGLE, qubit)
        mz(qubit)

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


_CAPTURED_EMPTY: list[int] = []
_CAPTURED_EMPTY_NO_ANNOTATION = []


def test_captured_empty_list_with_annotation():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        if len(_CAPTURED_EMPTY) > 0:
            x(q[0])

    counts = cudaq.sample(kernel)
    assert '0' in counts


def test_captured_empty_list_without_annotation():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        if len(_CAPTURED_EMPTY_NO_ANNOTATION) > 0:
            x(q[0])

    with pytest.raises(RuntimeError) as e:
        cudaq.sample(kernel)
    assert '_CAPTURED_EMPTY_NO_ANNOTATION' in repr(e)


_CAPTURED_PAULI_STRINGS = ['XI', 'ZZ']


def test_captured_list_of_strings_as_pauli_words():

    @cudaq.kernel
    def with_arg(words: list[cudaq.pauli_word]):
        q = cudaq.qvector(2)
        for i in range(len(words)):
            exp_pauli(0.1, q, words[i])

    @cudaq.kernel
    def with_capture():
        q = cudaq.qvector(2)
        for i in range(len(_CAPTURED_PAULI_STRINGS)):
            exp_pauli(0.1, q, _CAPTURED_PAULI_STRINGS[i])

    counts_arg = cudaq.sample(with_arg,
                              [cudaq.pauli_word(w) for w in ['XI', 'ZZ']])
    counts_cap = cudaq.sample(with_capture)
    assert len(counts_arg) >= 1
    assert len(counts_cap) >= 1


_CAPTURED_NESTED = [[0.5, -0.5], [1.0]]


def test_captured_nested_list():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        for i in range(len(_CAPTURED_NESTED)):
            for j in range(len(_CAPTURED_NESTED[i])):
                ry(_CAPTURED_NESTED[i][j], q[0])

    counts = cudaq.sample(kernel)
    assert len(counts) >= 1


def test_nested_list_arg_sample_equals_get_state():

    @cudaq.kernel
    def kernel(ws: list[list[cudaq.pauli_word]], cs: list[list[float]]):
        q = cudaq.qvector(2)
        for i in range(len(ws)):
            g = ws[i]
            gc = cs[i]
            for j in range(len(g)):
                exp_pauli(gc[j], q, g[j])

    nw = [[cudaq.pauli_word('XY'),
           cudaq.pauli_word('YX')], [cudaq.pauli_word('ZI')]]
    nc = [[0.5, -0.5], [1.0]]

    cudaq.get_state(kernel, nw, nc)
    counts = cudaq.sample(kernel, nw, nc)
    assert len(counts) >= 1


_CAPTURED_EMPTY_NDARRAY = np.array([], dtype=np.float64)
_CAPTURED_EMPTY_NDARRAY_INT = np.array([], dtype=np.int64)
_CAPTURED_NDARRAY = np.array([0.5, -0.5])


def test_captured_empty_ndarray():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        for i in range(len(_CAPTURED_EMPTY_NDARRAY)):
            rx(_CAPTURED_EMPTY_NDARRAY[i], q[0])

    counts = cudaq.sample(kernel)
    assert '0' in counts


def test_captured_empty_ndarray_int():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(2)
        for i in range(len(_CAPTURED_EMPTY_NDARRAY_INT)):
            x(q[_CAPTURED_EMPTY_NDARRAY_INT[i]])
        x(q[1])

    counts = cudaq.sample(kernel)
    assert '01' in counts


def test_captured_nonempty_ndarray():

    @cudaq.kernel
    def kernel():
        q = cudaq.qvector(1)
        for i in range(len(_CAPTURED_NDARRAY)):
            ry(_CAPTURED_NDARRAY[i], q[0])

    counts = cudaq.sample(kernel)
    assert '0' in counts
