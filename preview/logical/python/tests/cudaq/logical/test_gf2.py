# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Regression coverage for GF(2) arithmetic and packed MLIR matrices."""

import itertools
import random

import pytest
import cudaq.mlir.ir as mlir_ir

from cudaq.logical.algebra.gf2 import GF2Matrix, _row_bits
from cudaq.logical.compiler.qec import _gf2_matrix


@pytest.mark.parametrize("width", [0, 1, 7, 8, 9, 63, 64, 65, 1458, 5000])
def test_row_bits_preserves_column_order(width):
    rng = random.Random(width)
    rows = [
        tuple(rng.randrange(2) for _ in range(width)), (0,) * width,
        (1,) * width
    ]
    if width:
        rows.extend([(1,) + (0,) * (width - 1), (0,) * (width - 1) + (1,)])
    for row in rows:
        expected = sum(bit << column for column, bit in enumerate(row))
        assert _row_bits(row) == expected
        assert _row_bits(iter(row)) == expected


@pytest.mark.parametrize("shape", [(0, 0), (0, 9), (3, 0), (1, 1), (1, 7),
                                   (1, 8), (1, 9), (2, 3), (3, 3), (5, 9),
                                   (17, 65)])
@pytest.mark.parametrize("pattern", ["zeros", "ones", "random"])
def test_gf2_matrix_matches_text_attribute(shape, pattern):
    nrows, ncols = shape
    rng = random.Random(42)
    rows = [[
        rng.randrange(2) if pattern == "random" else int(pattern == "ones")
        for _ in range(ncols)
    ]
            for _ in range(nrows)]
    matrix = GF2Matrix(rows, ncols=ncols)
    context = mlir_ir.Context()
    literal = str(rows) if nrows and ncols else ""
    expected = mlir_ir.Attribute.parse(
        f"dense<{literal}> : tensor<{nrows}x{ncols}xi1>", context=context)
    assert _gf2_matrix(context, matrix) == expected


def test_gf2_matrix_rank_and_product():
    # Every small square matrix, including singular and non-symmetric cases.
    for entries in itertools.product((0, 1), repeat=4):
        rows = (entries[:2], entries[2:])
        matrix = GF2Matrix(rows)
        span = {
            tuple((a * rows[0][i]) ^ (b * rows[1][i])
                  for i in range(2))
            for a, b in itertools.product((0, 1), repeat=2)
        }
        assert matrix.rank == len(span).bit_length() - 1
        assert (matrix @ GF2Matrix([[1, 0], [0, 1]])) == matrix
        expected = [[
            sum(rows[i][k] * rows[k][j] for k in range(2)) % 2 for j in range(2)
        ] for i in range(2)]
        assert matrix @ matrix == GF2Matrix(expected)
    assert GF2Matrix([], ncols=9).rank == 0
    assert GF2Matrix([[], []]).rank == 0
    assert GF2Matrix([[], []]) @ GF2Matrix([], ncols=3) == GF2Matrix([[0] * 3] *
                                                                     2)


@pytest.mark.parametrize("entry, error", [(2, ValueError), (-1, ValueError),
                                          (0.5, TypeError), ("1", TypeError)])
def test_gf2_matrix_rejects_nonbinary_entries(entry, error):
    with pytest.raises(error):
        GF2Matrix([[entry]])
