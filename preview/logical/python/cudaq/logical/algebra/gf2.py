# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pure immutable GF(2) values shared across QEC model families."""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as integer_index


def _normalize_binary_value(value, *, what: str) -> int:
    """Return one exact authored GF(2) bit without lossy coercion."""

    try:
        normalized = integer_index(value)
    except TypeError as exc:
        raise TypeError(f"{what} must be binary integer values") from exc
    if normalized not in (0, 1):
        raise ValueError(f"{what} must be binary")
    return int(normalized)


def _normalize_binary_values(values, *, what: str) -> tuple[int, ...]:
    return tuple(_normalize_binary_value(value, what=what) for value in values)


def _row_bits(row) -> int:
    return sum(int(bit) << index for index, bit in enumerate(row))


@dataclass(frozen=True, slots=True, init=False)
class GF2Matrix:
    """Small immutable binary matrix used by QEC algebra declarations."""

    rows: tuple[tuple[int, ...], ...]
    ncols: int

    def __init__(self, rows=(), *, ncols: int | None = None) -> None:
        normalized = tuple(
            _normalize_binary_values(row, what="GF2Matrix entries")
            for row in rows)
        widths = {len(row) for row in normalized}
        if len(widths) > 1:
            raise ValueError("GF2Matrix rows must have equal width")
        inferred = next(iter(widths), 0 if ncols is None else ncols)
        if ncols is not None and ncols != inferred:
            raise ValueError("GF2Matrix ncols does not match its row width")
        if not isinstance(inferred, int) or isinstance(inferred,
                                                       bool) or inferred < 0:
            raise TypeError("GF2Matrix ncols must be a nonnegative int")
        object.__setattr__(self, "rows", normalized)
        object.__setattr__(self, "ncols", inferred)

    @classmethod
    def from_rows(cls, rows, *, ncols: int | None = None) -> "GF2Matrix":
        return cls(rows, ncols=ncols)

    @classmethod
    def _from_normalized_rows(cls, rows: tuple[tuple[int, ...], ...], *,
                              ncols: int) -> "GF2Matrix":
        """Construct from exact internal binary rows without rescanning cells.

        Public declarations must use ``GF2Matrix(...)`` or ``from_rows`` and
        retain their fail-closed integer/binary validation.  This private path
        is only for immutable rows already produced or validated by CUDA-Q Logical in the
        same call chain.
        """

        assert isinstance(ncols, int) and not isinstance(ncols, bool)
        assert ncols >= 0 and isinstance(rows, tuple)
        assert all(isinstance(row, tuple) and len(row) == ncols for row in rows)
        value = object.__new__(cls)
        object.__setattr__(value, "rows", rows)
        object.__setattr__(value, "ncols", ncols)
        return value

    @property
    def nrows(self) -> int:
        return len(self.rows)

    @property
    def rank(self) -> int:
        basis: dict[int, int] = {}
        for row in self.rows:
            value = sum(bit << index for index, bit in enumerate(row))
            while value:
                pivot = value.bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = value
                    break
                value ^= basis[pivot]
        return len(basis)

    def __matmul__(self, other: "GF2Matrix") -> "GF2Matrix":
        if not isinstance(other, GF2Matrix):
            return NotImplemented
        if self.ncols != other.nrows:
            raise ValueError("GF(2) matrix dimensions do not align")
        encoded = tuple(_row_bits(row) for row in other.rows)
        rows = []
        for row in self.rows:
            value = 0
            for selected, encoded_row in zip(row, encoded):
                if selected:
                    value ^= encoded_row
            rows.append(
                tuple((value >> column) & 1 for column in range(other.ncols)))
        return GF2Matrix(rows, ncols=other.ncols)


# Preserve the established pickle and durable type identity while the
# implementation lives in this neutral leaf module.  ``cudaq.logical.model.qec`` keeps
# an exact alias, so historical payloads continue to resolve the same class.
GF2Matrix.__module__ = "cudaq.logical.model.qec"

__all__ = ["GF2Matrix"]
