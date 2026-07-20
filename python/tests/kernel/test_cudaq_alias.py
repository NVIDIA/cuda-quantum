# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import pytest

import cudaq as cq

# Also keep a canonical import so we can verify both work in the same file.
import cudaq


@pytest.fixture(autouse=True)
def run_and_clear_registries():
    yield
    cudaq.__clearKernelRegistries()


# --------------------------------------------------------------------------- #
# Basic kernel definition and sampling with aliased import
# --------------------------------------------------------------------------- #


def test_alias_basic_kernel():
    """The exact reproducer from issue #2341."""

    @cq.kernel
    def simple():
        q = cq.qubit()

    counts = cq.sample(simple)
    assert len(counts) == 1
    assert '0' in counts


def test_alias_qvector():
    """cudaq.qvector must work through an alias."""

    @cq.kernel
    def bell():
        qubits = cq.qvector(2)
        h(qubits[0])
        cx(qubits[0], qubits[1])

    counts = cq.sample(bell)
    assert len(counts) == 2
    assert '00' in counts
    assert '11' in counts


def test_alias_kernel_with_int_arg():
    """Kernel with integer argument using alias."""

    @cq.kernel
    def kernel(n: int):
        qubits = cq.qvector(n)

    counts = cq.sample(kernel, 3)
    assert len(counts) == 1
    assert '000' in counts


def test_alias_kernel_with_float_arg():
    """Kernel with float argument using alias."""

    @cq.kernel
    def kernel(angle: float):
        q = cq.qubit()
        rx(angle, q)

    counts = cq.sample(kernel, 0.0)
    assert len(counts) == 1
    assert '0' in counts


# --------------------------------------------------------------------------- #
# Type annotations using aliased module name
# --------------------------------------------------------------------------- #


def test_alias_pauli_word_annotation():
    """Type annotation cudaq.pauli_word must resolve through alias."""

    @cq.kernel
    def kernel(pw: cq.pauli_word):
        q = cq.qvector(2)

    # Just verify it compiles without error
    kernel(cq.pauli_word("XX"))


# --------------------------------------------------------------------------- #
# Mixed usage: alias and canonical name in the same process
# --------------------------------------------------------------------------- #


def test_canonical_still_works():
    """Ensure standard `import cudaq` usage still works."""

    @cudaq.kernel
    def simple():
        q = cudaq.qubit()

    counts = cudaq.sample(simple)
    assert len(counts) == 1
    assert '0' in counts


def test_alias_and_canonical_separate_kernels():
    """Both aliased and canonical kernels can coexist."""

    @cq.kernel
    def kernel_alias():
        q = cq.qubit()
        x(q)

    @cudaq.kernel
    def kernel_canonical():
        q = cudaq.qubit()
        x(q)

    counts_alias = cq.sample(kernel_alias)
    counts_canonical = cudaq.sample(kernel_canonical)
    assert '1' in counts_alias
    assert '1' in counts_canonical


# --------------------------------------------------------------------------- #
# Gate operations through alias (gates are called without module prefix inside
# kernel bodies; the alias is used for cudaq.qvector/qubit/sample etc.)
# --------------------------------------------------------------------------- #


def test_alias_common_gates():
    """Common gate operations work in kernel defined via alias."""

    @cq.kernel
    def kernel():
        q = cq.qvector(2)
        h(q[0])
        x(q[1])
        cx(q[0], q[1])
        y(q[0])
        z(q[1])
        t(q[0])
        s(q[1])

    counts = cq.sample(kernel)
    assert len(counts) > 0


def test_alias_rotation_gates():
    """Rotation gates work in kernel defined via alias."""

    @cq.kernel
    def kernel():
        q = cq.qubit()
        rx(1.5708, q)
        ry(1.5708, q)
        rz(1.5708, q)

    counts = cq.sample(kernel)
    assert len(counts) > 0


# --------------------------------------------------------------------------- #
# Adjoint and control modifiers
# --------------------------------------------------------------------------- #


def test_alias_adjoint():
    """Adjoint modifier works in kernel defined via alias."""

    @cq.kernel
    def kernel():
        q = cq.qubit()
        t(q)
        t.adj(q)

    counts = cq.sample(kernel)
    assert '0' in counts


def test_alias_control():
    """Control modifier works in kernel defined via alias."""

    @cq.kernel
    def kernel():
        q = cq.qvector(2)
        x(q[0])
        x.ctrl(q[0], q[1])

    counts = cq.sample(kernel)
    assert '11' in counts


# --------------------------------------------------------------------------- #
# Return values
# --------------------------------------------------------------------------- #


def test_alias_kernel_return_bool():
    """Kernel returning a boolean value, defined via alias."""

    @cq.kernel
    def kernel() -> bool:
        q = cq.qubit()
        return mz(q)

    result = kernel()
    assert isinstance(result, bool)


# --------------------------------------------------------------------------- #
# Multiple qubit allocations
# --------------------------------------------------------------------------- #


def test_alias_multiple_allocations():
    """Multiple qvector allocations via alias."""

    @cq.kernel
    def kernel():
        a = cq.qvector(2)
        b = cq.qvector(2)
        h(a[0])
        cx(a[0], b[0])

    counts = cq.sample(kernel)
    assert len(counts) > 0
