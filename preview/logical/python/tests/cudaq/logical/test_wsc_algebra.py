# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Regression coverage for WSC stabilizer conversion."""

from types import SimpleNamespace

import pytest

from cudaq.logical.algebra.gf2 import GF2Matrix
from cudaq.logical.architectures._wsc import _base_stabilizers, _LabeledPauli


@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 63, 64, 65, 729, 1458])
def test_base_stabilizers_preserves_xz_masks_and_labels(n):
    full = (1 << n) - 1
    high = (1 << (n - 1)) if n else 0
    low = int(n > 0)
    masks = [(0, 0), (full, 0), (0, full), (low, high), (high, low),
             (full, full)]
    rows = [
        tuple((x >> qubit) & 1
              for qubit in range(n)) + tuple((z >> qubit) & 1
                                             for qubit in range(n))
        for x, z in masks
    ]
    code = SimpleNamespace(n=n, stabilizer_basis=GF2Matrix(rows, ncols=2 * n))

    assert _base_stabilizers(code) == tuple(
        _LabeledPauli.from_masks(x, z, 0, f"base_{index}")
        for index, (x, z) in enumerate(masks))


def test_base_stabilizers_empty_basis():
    code = SimpleNamespace(n=9, stabilizer_basis=GF2Matrix(ncols=18))
    assert _base_stabilizers(code) == ()
