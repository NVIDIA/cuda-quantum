# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Tests for `import cudaq as <alias>` support (GitHub issue #2341).

import cudaq as cq
import pytest


def test_alias_qubit():
    """Kernel using aliased module for qubit allocation."""

    @cq.kernel
    def kernel():
        q = cq.qubit()
        cq.h(q)
        cq.mz(q)

    counts = cq.sample(kernel)
    assert len(counts) > 0


def test_alias_qvector():
    """Kernel using aliased module for qvector allocation."""

    @cq.kernel
    def kernel():
        qv = cq.qvector(2)
        cq.h(qv[0])
        cq.cx(qv[0], qv[1])
        cq.mz(qv)

    counts = cq.sample(kernel)
    assert len(counts) > 0


def test_alias_adjoint():
    """Kernel using aliased module for adjoint."""

    @cq.kernel
    def kernel():
        q = cq.qubit()
        cq.t(q)
        cq.adjoint(cq.t, q)
        cq.mz(q)

    counts = cq.sample(kernel)
    assert '0' in counts


def test_alias_control():
    """Kernel using aliased module for control."""

    @cq.kernel
    def inner(q: cq.qubit):
        cq.x(q)

    @cq.kernel
    def kernel():
        qv = cq.qvector(2)
        cq.h(qv[0])
        cq.control(inner, qv[0], qv[1])
        cq.mz(qv)

    counts = cq.sample(kernel)
    assert len(counts) > 0


def test_alias_type_annotation():
    """Type annotations with aliased module name."""

    @cq.kernel
    def kernel(q: cq.qubit):
        cq.h(q)

    # If this compiles without error, the annotation resolved correctly.
    assert kernel is not None


def test_different_alias():
    """Different alias name works too."""
    import cudaq as quda

    @quda.kernel
    def kernel():
        q = quda.qubit()
        quda.h(q)
        quda.mz(q)

    counts = quda.sample(kernel)
    assert len(counts) > 0
