# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from inspect import signature
import json
import math
import re
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from ..errors import InvalidCodeAlgebra
from cudaq.logical._core.immutable import ImmutableValue
from cudaq.logical.algebra.clifford import CliffordAction
from cudaq.logical.algebra.gf2 import (
    GF2Matrix,
    _normalize_binary_value,
    _normalize_binary_values,
    _row_bits,
)
from cudaq.logical.architecture.logical import (
    LogicalValueGroup,
    LogicalValueRef,
)

from .selection import _deep_freeze
from .structure import CarrierRoleMap


class Schedule(str, Enum):
    """Typed entangling-schedule tags for CSS extraction circuits.

    ``HX``/``HZ`` order the data-ancilla CX layers by the corresponding
    check-matrix incidence.
    """

    HX = "hx"
    HZ = "hz"


class DistanceScope(str, Enum):
    """Which notion of distance a piece of evidence concerns.

    Subsystem codes distinguish bare-logical, gauge-dressed, and
    circuit-level distance; evidence must say which one it proves.
    """

    BARE = "bare"
    DRESSED = "dressed"
    CIRCUIT = "circuit"


def _scope_name(scope) -> str | None:
    if scope is None:
        return None
    if isinstance(scope, DistanceScope):
        return scope.value
    text = str(scope).lower()
    if text not in ("bare", "dressed", "circuit"):
        raise ValueError("distance scope must be a DistanceScope or one of "
                         "'bare'/'dressed'/'circuit'")
    return text


_DISTANCE_STATUSES = (
    "claimed",
    "exact",
    "lower_bound",
    "upper_bound",
    "circuit",
    "asymmetric",
    "unknown",
)

_EVIDENCE_STATUSES = ("exact", "lower_bound", "upper_bound", "circuit")


def _positive_distance(value, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{what} must be a positive int")
    return value


@dataclass(frozen=True, slots=True)
class Distance:
    """Typed distance evidence.

    A bare integer in concise source always normalizes to ``claimed`` — a
    recorded assertion, never a proof. The evidence-bearing constructors
    (``exact``, ``lower_bound``, ``upper_bound``, ``circuit``) require a
    method and provenance; a machine-checkable certificate stays optional.
    ``asymmetric`` carries independent X- and Z-basis evidence rather than
    promoting two integers to one proof status.
    """

    value: int | None
    status: str = "claimed"
    reason: str | None = None
    method: str | None = None
    provenance: Any = None
    certificate: Any = None
    scope: str | None = None
    x: "Distance | None" = None
    z: "Distance | None" = None

    def __post_init__(self) -> None:
        if self.status not in _DISTANCE_STATUSES:
            raise ValueError(
                f"unknown distance status {self.status!r}; expected one of "
                f"{_DISTANCE_STATUSES}")
        if self.status in _EVIDENCE_STATUSES:
            if not self.method or self.provenance is None:
                raise TypeError(
                    f"Distance.{self.status} evidence requires method= and "
                    "provenance=; use Distance.claimed(...) to record an "
                    "unproved assertion")
            _positive_distance(self.value, f"{self.status} distance")
        if self.status == "asymmetric":
            if not isinstance(self.x, Distance) or not isinstance(
                    self.z, Distance):
                raise TypeError(
                    "Distance.asymmetric requires x= and z= Distance evidence")
            if self.x.status == "asymmetric" or self.z.status == "asymmetric":
                raise ValueError(
                    "asymmetric distance evidence cannot nest asymmetric "
                    "components")
        object.__setattr__(
            self,
            "provenance",
            _deep_freeze(self.provenance, what="distance provenance"),
        )
        object.__setattr__(
            self,
            "certificate",
            _deep_freeze(self.certificate, what="distance certificate"),
        )

    @classmethod
    def unknown(cls, reason: str = "not established") -> "Distance":
        return cls(None, "unknown", reason)

    @classmethod
    def claimed(cls,
                value: int,
                *,
                provenance: Any = None,
                scope=None) -> "Distance":
        return cls(
            _positive_distance(value, "code distance claim"),
            "claimed",
            provenance=provenance,
            scope=_scope_name(scope),
        )

    # Historical spelling; ``claimed`` is canonical.
    @classmethod
    def claim(cls, value: int) -> "Distance":
        return cls.claimed(value)

    @classmethod
    def exact(
        cls,
        value: int,
        *,
        method: str | None = None,
        provenance: Any = None,
        certificate: Any = None,
        scope=None,
    ) -> "Distance":
        return cls(
            value,
            "exact",
            method=method,
            provenance=provenance,
            certificate=certificate,
            scope=_scope_name(scope),
        )

    @classmethod
    def lower_bound(
        cls,
        value: int,
        *,
        method: str | None = None,
        provenance: Any = None,
        certificate: Any = None,
        scope=None,
    ) -> "Distance":
        return cls(
            value,
            "lower_bound",
            method=method,
            provenance=provenance,
            certificate=certificate,
            scope=_scope_name(scope),
        )

    @classmethod
    def upper_bound(
        cls,
        value: int,
        *,
        method: str | None = None,
        provenance: Any = None,
        certificate: Any = None,
        scope=None,
    ) -> "Distance":
        return cls(
            value,
            "upper_bound",
            method=method,
            provenance=provenance,
            certificate=certificate,
            scope=_scope_name(scope),
        )

    @classmethod
    def circuit(
        cls,
        value: int,
        *,
        method: str | None = None,
        provenance: Any = None,
        certificate: Any = None,
        gadget_scope: str | None = None,
        fault_scope: str | None = None,
    ) -> "Distance":
        reason = "; ".join(part for part in (
            f"gadget_scope={gadget_scope}" if gadget_scope else None,
            f"fault_scope={fault_scope}" if fault_scope else None,
        ) if part)
        return cls(
            value,
            "circuit",
            reason=reason or None,
            method=method,
            provenance=provenance,
            certificate=certificate,
            scope="circuit",
        )

    @classmethod
    def asymmetric(cls, x, z) -> "Distance":

        def normalize(component, basis: str) -> "Distance":
            if isinstance(component, Distance):
                return component
            return cls.claimed(
                _positive_distance(component, f"asymmetric {basis} distance"))

        return cls(None, "asymmetric", x=normalize(x, "x"), z=normalize(z, "z"))

    @property
    def is_proved(self) -> bool:
        return self.status in _EVIDENCE_STATUSES

    @property
    def conservative_value(self) -> int | None:
        """Scalar full-code distance implied by this evidence, when known.

        A legacy scalar consumer of asymmetric evidence must use the weaker
        basis component, never promote either component to a symmetric claim.
        The typed ``x``/``z`` fields remain the authoritative evidence.
        """

        if self.value is not None:
            return self.value
        if self.status != "asymmetric":
            return None
        components = (self.x.conservative_value, self.z.conservative_value)
        return min(components) if all(
            value is not None for value in components) else None


def _rows(value) -> tuple[tuple[int, ...], ...]:
    if value is None:
        return ()
    return tuple(tuple(int(item) for item in row) for row in value)


def _support_rows(value, *, basis: str,
                  what: str) -> tuple[tuple[int, ...], ...]:
    """Normalize CSS support declarations.

    Rows are ordinarily carrier-index tuples. The general ``PauliProduct``
    spelling is accepted when the product is pure ``X``- or pure ``Z``-type
    matching the declared basis; mixed products belong in ``stabilizers=`` /
    ``logicals=``.
    """
    from cudaq.logical.algebra.pauli import PauliProduct

    if value is None:
        return ()
    expected = basis.upper()
    rows = []
    for row in value:
        if isinstance(row, PauliProduct):
            if row.sign != 1:
                raise TypeError(
                    f"{what} rows are unsigned support declarations; declare "
                    "signed operators through stabilizers= or logicals=")
            support = []
            for factor in row.factors:
                if factor.pauli != expected:
                    raise TypeError(
                        f"{what} rows must be pure {expected}-type Pauli "
                        "products; declare mixed-type operators through "
                        "stabilizers= or logicals=")
                if not isinstance(factor.operand, int) or isinstance(
                        factor.operand, bool):
                    raise TypeError(
                        f"{what} rows must reference formal carrier indices")
                support.append(factor.operand)
            rows.append(tuple(sorted(support)))
            continue
        rows.append(tuple(int(item) for item in row))
    return tuple(rows)


def _validate_symplectic_closure(value, *, what: str) -> GF2Matrix:
    """Require a full-rank canonical symplectic GF(2) automorphism."""

    if not isinstance(value, GF2Matrix):
        raise TypeError(f"{what} must be a GF2Matrix")
    if value.nrows == 0 or value.nrows != value.ncols or value.ncols % 2:
        raise ValueError(
            f"{what} must be a nonempty even-dimensional square matrix")
    if value.rank != value.nrows:
        raise ValueError(f"{what} must be full rank")
    half = value.ncols // 2
    for left, left_row in enumerate(value.rows):
        for right, right_row in enumerate(value.rows):
            expected = ((left < half and right == left + half) or
                        (right < half and left == right + half))
            if bool(_symplectic_product(left_row, right_row, half)) != expected:
                raise ValueError(
                    f"{what} must preserve the canonical symplectic form")
    return value


def _in_span(row, basis) -> bool:
    basis = tuple(tuple(value for value in item) for item in basis)
    row = tuple(row)
    width = len(row)
    return GF2Matrix((*basis, row),
                     ncols=width).rank == GF2Matrix(basis, ncols=width).rank


def _independent_rows(rows,
                      *,
                      ncols: int,
                      _normalized: bool = False) -> tuple[tuple[int, ...], ...]:
    """Retain the first independent rows in declaration order."""

    pivots: dict[int, int] = {}
    result = []
    for raw in rows:
        row = raw if _normalized else _normalize_binary_values(
            raw, what="GF(2) row entries")
        if len(row) != ncols:
            raise ValueError(
                "GF(2) row width does not match the declared space")
        value = _row_bits(row)
        reduced = value
        while reduced:
            pivot = reduced.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = reduced
                result.append(row)
                break
            reduced ^= pivots[pivot]
    return tuple(result)


def _coordinates_in_basis_many(
        rows,
        basis,
        *,
        _normalized: bool = False) -> tuple[tuple[int, ...], ...]:
    """Express several rows in one independent basis with one elimination."""

    if _normalized:
        rows = tuple(rows)
        basis = tuple(basis)
    else:
        rows = tuple(
            _normalize_binary_values(row, what="GF(2) row entries")
            for row in rows)
        basis = tuple(
            _normalize_binary_values(item, what="GF(2) basis entries")
            for item in basis)
    if not basis:
        if any(any(row) for row in rows):
            raise ValueError("row is outside the empty GF(2) basis")
        return tuple(() for _ in rows)
    width = len(basis[0])
    if any(len(item) != width for item in basis):
        raise ValueError("basis rows have inconsistent widths")
    pivots: dict[int, tuple[int, int]] = {}
    for index, item in enumerate(basis):
        value = _row_bits(item)
        coefficients = 1 << index
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = (value, coefficients)
                break
            other, other_coefficients = pivots[pivot]
            value ^= other
            coefficients ^= other_coefficients
        if value == 0:
            raise ValueError("coordinate basis must be linearly independent")
    result = []
    for row in rows:
        if len(row) != width:
            raise ValueError("row width does not match its GF(2) basis")
        value = _row_bits(row)
        coefficients = 0
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                raise ValueError("row is outside the declared GF(2) basis")
            other, other_coefficients = pivots[pivot]
            value ^= other
            coefficients ^= other_coefficients
        result.append(
            tuple((coefficients >> index) & 1 for index in range(len(basis))))
    return tuple(result)


def _coordinates_in_basis(row, basis) -> tuple[int, ...]:
    return _coordinates_in_basis_many((row,), basis)[0]


def _independent_row_indices(rows) -> tuple[int, ...]:
    pivots: dict[int, int] = {}
    selected = []
    for index, row in enumerate(rows):
        value = _row_bits(row)
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = value
                selected.append(index)
                break
            value ^= pivots[pivot]
    return tuple(selected)


def _row_relations(rows,
                   *,
                   ncols: int,
                   _normalized: bool = False) -> tuple[tuple[int, ...], ...]:
    """Return an independent relation basis over an ordered row family."""

    rows = (tuple(rows) if _normalized else tuple(
        _normalize_binary_values(row, what="GF(2) relation entries")
        for row in rows))
    pivots: dict[int, tuple[int, int]] = {}
    relations = []
    for index, row in enumerate(rows):
        if len(row) != ncols:
            raise ValueError(
                "relation row width does not match the declared space")
        value = _row_bits(row)
        coefficients = 1 << index
        while value:
            pivot = value.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = (value, coefficients)
                break
            other, other_coefficients = pivots[pivot]
            value ^= other
            coefficients ^= other_coefficients
        if value == 0:
            relations.append(
                tuple((coefficients >> column) & 1
                      for column in range(len(rows))))
    return tuple(relations)


def _solve_gf2(equations, rhs, *, nvars: int) -> tuple[int, ...]:
    """Solve a consistent binary linear system with free variables set to zero."""

    equations = tuple(
        _normalize_binary_values(row, what="GF(2) equation entries")
        for row in equations)
    rhs = _normalize_binary_values(rhs, what="GF(2) right-hand side")
    if len(equations) != len(rhs):
        raise ValueError("GF(2) system row and right-hand-side counts differ")
    if any(len(row) != nvars for row in equations):
        raise ValueError("GF(2) system row width does not match nvars")
    rows = [
        _row_bits(row) | (value << nvars) for row, value in zip(equations, rhs)
    ]
    pivot_row = 0
    pivots = []
    for column in range(nvars):
        selected = next(
            (index for index in range(pivot_row, len(rows))
             if (rows[index] >> column) & 1),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        for index in range(len(rows)):
            if index != pivot_row and ((rows[index] >> column) & 1):
                rows[index] ^= rows[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    coefficient_mask = (1 << nvars) - 1
    if any(
        (row & coefficient_mask) == 0 and ((row >> nvars) & 1) for row in rows):
        raise ValueError("GF(2) system is inconsistent")
    solution = [0] * nvars
    for row, column in zip(rows, pivots):
        solution[column] = (row >> nvars) & 1
    return tuple(solution)


def _symplectic_product_bits(
    left: int,
    right: int,
    n: int,
    *,
    mask: int | None = None,
) -> int:
    """Return the symplectic product of two packed ``(X | Z)`` rows."""

    mask = (1 << n) - 1 if mask is None else mask
    left_x, left_z = left & mask, left >> n
    right_x, right_z = right & mask, right >> n
    return ((left_x & right_z).bit_count() ^ (left_z & right_x).bit_count()) & 1


def _symplectic_product(left, right, n: int) -> int:
    left = tuple(left)
    right = tuple(right)
    if len(left) != 2 * n or len(right) != 2 * n:
        raise ValueError("symplectic vectors must have width 2n")
    return _symplectic_product_bits(_row_bits(left), _row_bits(right), n)


def _symplectic_functional(row, n: int) -> tuple[int, ...]:
    row = tuple(row)
    if len(row) != 2 * n:
        raise ValueError("symplectic vector must have width 2n")
    return (*row[n:], *row[:n])


def _derive_anti_stabilizers(
    stabilizers,
    protected_pairs,
    *,
    n: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Symplectic Gram--Schmidt completion using integer bit operations.

    The retained stabilizer basis may change by invertible row operations, but
    its span is unchanged. This avoids solving one growing dense system per
    stabilizer and keeps large concatenated-code construction practical.
    """

    width = 2 * n
    mask = (1 << n) - 1

    def symplectic(left: int, right: int) -> int:
        left_x, left_z = left & mask, left >> n
        right_x, right_z = right & mask, right >> n
        return ((left_x & right_z).bit_count() ^
                (left_z & right_x).bit_count()) & 1

    remaining = [_row_bits(row) for row in stabilizers]
    pairs = [
        (_row_bits(left), _row_bits(right)) for left, right in protected_pairs
    ]
    anti = []
    for index in range(len(remaining)):
        stabilizer = remaining[index]
        x_mask = stabilizer & mask
        z_mask = stabilizer >> n
        if x_mask:
            qubit = (x_mask & -x_mask).bit_length() - 1
            partner = 1 << (n + qubit)
        elif z_mask:
            qubit = (z_mask & -z_mask).bit_length() - 1
            partner = 1 << qubit
        else:
            raise ValueError("cannot complete an identity stabilizer")
        for left, right in pairs:
            if symplectic(partner, right):
                partner ^= left
            if symplectic(partner, left):
                partner ^= right
        if symplectic(stabilizer, partner) != 1:
            raise ValueError("failed to derive an anti-stabilizer partner")
        for future in range(index + 1, len(remaining)):
            if symplectic(remaining[future], partner):
                remaining[future] ^= stabilizer
        pairs.append((stabilizer, partner))
        anti.append(partner)

    def bits(value: int) -> tuple[int, ...]:
        return tuple((value >> bit) & 1 for bit in range(width))

    return tuple(bits(value) for value in remaining), tuple(
        bits(value) for value in anti)


def _xor_rows(rows, coefficients, *, ncols: int) -> tuple[int, ...]:
    value = [0] * ncols
    for selected, row in zip(coefficients, rows):
        if selected:
            value = [left ^ right for left, right in zip(value, row)]
    return tuple(value)


def _support_bits(support, n: int) -> tuple[int, ...]:
    bits = [0] * n
    for index in support:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("Pauli support indices must be Python ints")
        if index < 0 or index >= n:
            raise ValueError(
                f"Pauli support index {index} is outside range({n})")
        if bits[index]:
            raise ValueError("Pauli support indices must be unique")
        bits[index] = 1
    return tuple(bits)


def _declared_pauli_row(value, n: int) -> tuple[int, ...]:
    """Normalize an explicit general-Pauli declaration to ``(x | z)`` bits."""

    from cudaq.logical.algebra.pauli import PauliProduct

    if isinstance(value, PauliProduct):
        if value.sign != 1:
            raise ValueError(
                "signed code generators are not representable in the current "
                "GF(2) code algebra; declare an unsigned generator or use a "
                "sign-preserving state/preparation contract")
        x_mask, z_mask = value.symplectic_for(range(n))
        return tuple((x_mask >> index) & 1 for index in range(n)) + tuple(
            (z_mask >> index) & 1 for index in range(n))
    row = tuple(value)
    if len(row) == 2 * n and all(bit in (0, 1) for bit in row):
        return tuple(int(bit) for bit in row)
    # Compatibility for the alpha ``gauges=((0, 1), ...)`` form.  A bare
    # support has no Pauli label, so it denotes a Z-type generator.  New
    # general subsystem-code definitions should use PauliProduct values.
    return (0,) * n + _support_bits(row, n)


def _code_symplectic_rows(code: "Code"):
    """Return the kept stabilizer basis and full gauge group."""

    stabilizers = code.stabilizer_basis.rows
    gauge_generators = (
        *code.gauge_x_basis.rows,
        *code.gauge_z_basis.rows,
        *(_declared_pauli_row(value, code.n) for value in code.gauges),
    )
    return stabilizers, (*stabilizers, *gauge_generators)


@dataclass(frozen=True, slots=True)
class _BoundaryMaps:
    """Private derived linear view of a code-profile boundary.

    Effective syndromes may be redundant. ``kept_from_effective`` is a left
    inverse on the valid syndrome subspace; ``effective_metachecks`` identifies
    the valid subspace. Logical coordinates are ordered X then Z.
    """

    effective_from_kept: GF2Matrix
    kept_from_effective: GF2Matrix
    physical_to_effective: GF2Matrix
    physical_to_logical: GF2Matrix
    kept_syndrome_to_physical: GF2Matrix
    logical_to_physical: GF2Matrix
    effective_metachecks: GF2Matrix

    def __post_init__(self) -> None:
        matrices = (
            self.effective_from_kept,
            self.kept_from_effective,
            self.physical_to_effective,
            self.physical_to_logical,
            self.kept_syndrome_to_physical,
            self.logical_to_physical,
            self.effective_metachecks,
        )
        if any(not isinstance(matrix, GF2Matrix) for matrix in matrices):
            raise TypeError("boundary-map fields must be GF2Matrix values")
        effective = self.effective_from_kept.nrows
        kept = self.effective_from_kept.ncols
        physical_width = self.physical_to_effective.ncols
        if self.kept_from_effective.nrows != kept or (
                self.kept_from_effective.ncols != effective):
            raise ValueError(
                "kept/effective boundary maps have incompatible shapes")
        if (self.kept_from_effective
                @ self.effective_from_kept).rows != GF2Matrix(
                    tuple(
                        tuple(int(row == column)
                              for column in range(kept))
                        for row in range(kept)),
                    ncols=kept,
                ).rows:
            raise ValueError(
                "kept_from_effective must left-invert effective_from_kept")
        if self.physical_to_effective.nrows != effective:
            raise ValueError(
                "physical_to_effective row count must equal effective checks")
        if self.kept_syndrome_to_physical.nrows != kept or (
                self.kept_syndrome_to_physical.ncols != physical_width):
            raise ValueError(
                "kept syndrome representatives have incompatible shape")
        if self.physical_to_logical.ncols != physical_width or (
                self.logical_to_physical.ncols != physical_width):
            raise ValueError(
                "logical boundary maps must share the physical width")
        if self.physical_to_logical.nrows != self.logical_to_physical.nrows:
            raise ValueError("logical boundary-map dimensions disagree")
        if self.effective_metachecks.ncols != effective:
            raise ValueError(
                "effective metachecks must index effective syndromes")

    def decode(
            self,
            pauli) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        width = self.physical_to_effective.ncols
        if width % 2:
            raise ValueError("physical symplectic width must be even")
        row = _declared_pauli_row(pauli, width // 2)
        column = GF2Matrix(tuple((bit,) for bit in row), ncols=1)
        effective = (self.physical_to_effective @ column).rows
        logical = (self.physical_to_logical @ column).rows
        values = tuple(item[0] for item in logical)
        half = len(values) // 2
        return (
            tuple(item[0] for item in effective),
            values[:half],
            values[half:],
        )

    def encode(
            self,
            effective_syndrome,
            logical_x=(),
            logical_z=(),
    ) -> tuple[int, ...]:
        effective = _normalize_binary_values(
            effective_syndrome, what="boundary-map effective syndrome")
        logical_x = _normalize_binary_values(
            logical_x, what="boundary-map logical-X coordinates")
        logical_z = _normalize_binary_values(
            logical_z, what="boundary-map logical-Z coordinates")
        if len(effective) != self.effective_from_kept.nrows:
            raise ValueError(
                "effective syndrome width does not match the profile")
        logical_width = self.logical_to_physical.nrows // 2
        if len(logical_x) != logical_width or len(logical_z) != logical_width:
            raise ValueError("logical coordinate width does not match the code")
        if self.effective_metachecks.nrows:
            column = GF2Matrix(tuple((bit,) for bit in effective), ncols=1)
            if any(item[0]
                   for item in (self.effective_metachecks @ column).rows):
                raise ValueError(
                    "effective syndrome violates the profile metachecks")
        effective_column = GF2Matrix(tuple((bit,) for bit in effective),
                                     ncols=1)
        kept = tuple(
            item[0]
            for item in (self.kept_from_effective @ effective_column).rows)
        physical_width = self.physical_to_effective.ncols
        syndrome_pauli = _xor_rows(
            self.kept_syndrome_to_physical.rows,
            kept,
            ncols=physical_width,
        )
        logical_pauli = _xor_rows(
            self.logical_to_physical.rows,
            (*logical_x, *logical_z),
            ncols=physical_width,
        )
        return tuple(
            left ^ right for left, right in zip(syndrome_pauli, logical_pauli))


def _derive_boundary_maps(
    code: "Code",
    effective_stabilizers: GF2Matrix,
    decomposition: GF2Matrix,
    effective_metachecks: GF2Matrix,
) -> _BoundaryMaps:
    kept = code.stabilizer_basis.nrows
    effective = effective_stabilizers.nrows
    if kept:
        units = tuple(
            tuple(int(row == column)
                  for column in range(kept))
            for row in range(kept))
        selected_indices = _independent_row_indices(decomposition.rows)
        if len(selected_indices) != kept:
            raise ValueError(
                "effective decomposition does not span kept syndromes")
        selected_rows = tuple(
            decomposition.rows[index] for index in selected_indices)
        selected_coordinates = _coordinates_in_basis_many(units,
                                                          selected_rows,
                                                          _normalized=True)
        selected_positions = {
            index: position for position, index in enumerate(selected_indices)
        }
        kept_from_effective_rows = tuple(
            tuple(coordinates[selected_positions[index]] if index in
                  selected_positions else 0
                  for index in range(effective))
            for coordinates in selected_coordinates)
        kept_from_effective = GF2Matrix._from_normalized_rows(
            kept_from_effective_rows,
            ncols=effective,
        )
    else:
        kept_from_effective = GF2Matrix((), ncols=effective)
    return _BoundaryMaps(
        effective_from_kept=decomposition,
        kept_from_effective=kept_from_effective,
        physical_to_effective=GF2Matrix._from_normalized_rows(
            tuple(
                _symplectic_functional(row, code.n)
                for row in effective_stabilizers.rows),
            ncols=2 * code.n,
        ),
        physical_to_logical=GF2Matrix._from_normalized_rows(
            tuple(
                _symplectic_functional(row, code.n) for row in (
                    *code.logical_z_basis.rows,
                    *code.logical_x_basis.rows,
                )),
            ncols=2 * code.n,
        ),
        kept_syndrome_to_physical=code.anti_stabilizers,
        logical_to_physical=GF2Matrix._from_normalized_rows(
            (*code.logical_x_basis.rows, *code.logical_z_basis.rows),
            ncols=2 * code.n,
        ),
        effective_metachecks=effective_metachecks,
    )


def _distance(value) -> Distance:
    """Normalize concise distance inputs without depending on Code values."""

    if isinstance(value, Distance):
        return value
    if value is None:
        return Distance.unknown("not established")
    return Distance.claim(value)
