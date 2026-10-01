# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Bivariate-bicycle code algebra and validated syndrome policy."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import sys

from ..errors import InvalidCodeAlgebra
from cudaq.logical.codes.structure import Block
from cudaq.logical.codes.definition import Code, CSSCode, _materialized_code_identity


@dataclass(frozen=True, slots=True)
class BinaryPolynomial:
    """A binary polynomial in x and y as a set of monomial exponents."""

    monomials: tuple[tuple[int, int], ...]
    _term_order: tuple[tuple[int, int], ...] = field(init=False,
                                                     repr=False,
                                                     compare=False,
                                                     hash=False)

    def __post_init__(self) -> None:
        try:
            raw_monomials = tuple(self.monomials)
        except TypeError as exc:
            raise TypeError(
                "BinaryPolynomial monomials must be exponent pairs") from exc
        authored = []
        for monomial in raw_monomials:
            if (not isinstance(monomial, (tuple, list)) or len(monomial) != 2):
                raise TypeError(
                    "BinaryPolynomial monomials must be exponent pairs")
            if any(type(exponent) is not int for exponent in monomial):
                raise TypeError(
                    "BinaryPolynomial exponents must be Python ints")
            authored.append(tuple(monomial))
        authored = tuple(authored)
        if not authored:
            raise ValueError("BinaryPolynomial needs at least one monomial")
        if any(i < 0 or j < 0 for i, j in authored):
            raise ValueError("BinaryPolynomial exponents must be nonnegative")
        seen = set()
        repeated = None
        for monomial in authored:
            if monomial in seen:
                repeated = monomial
                break
            seen.add(monomial)
        if repeated is not None:
            i, j = repeated
            raise InvalidCodeAlgebra(
                "binary-polynomial monomials must be authored once in reduced "
                f"GF(2) form; repeated exponent ({i}, {j}) would otherwise "
                "change the code while looking like cancellation")
        normalized = tuple(sorted(authored))
        object.__setattr__(self, "monomials", normalized)
        object.__setattr__(self, "_term_order", authored)

    @property
    def terms(self) -> tuple[tuple[int, int], ...]:
        """Distinct monomials in authored order.

        Polynomial equality and ``monomials`` remain algebraic/canonical.
        Code-specific constructions may use this order to label otherwise
        equivalent summands such as the BB paper's ``A1..A3`` maps.
        """

        return self._term_order

    def __str__(self) -> str:

        def term(i: int, j: int) -> str:
            parts = []
            if i:
                parts.append("x" if i == 1 else f"x**{i}")
            if j:
                parts.append("y" if j == 1 else f"y**{j}")
            return "*".join(parts) or "1"

        return " + ".join(term(i, j) for i, j in self.monomials)


def binary_polynomial(text: str) -> BinaryPolynomial:
    """Parse ``"1 + x + y**2"``-style binary-polynomial notation."""

    if not isinstance(text, str):
        raise TypeError("binary_polynomial expects a string")

    factor_pattern = re.compile(r"(?:1|[xy](?:(?:\^|\*\*)[0-9]+)?)")
    monomials = []
    for raw_term in text.split("+"):
        term = raw_term.strip()
        if not term:
            raise ValueError(f"empty term in binary polynomial {text!r}")
        i = j = 0
        position = 0
        while position < len(term):
            while position < len(term) and term[position].isspace():
                position += 1
            match = factor_pattern.match(term, position)
            if match is None:
                raise ValueError(
                    f"cannot parse binary-polynomial term {term!r}")
            factor = match.group(0)
            position = match.end()
            if factor != "1":
                name = factor[0]
                suffix = factor[1:]
                exponent = int(suffix[2:] if suffix.startswith("**") else
                               suffix[1:] if suffix.startswith("^") else "1")
                if name == "x":
                    i += exponent
                else:
                    j += exponent
            while position < len(term) and term[position].isspace():
                position += 1
            if position == len(term):
                break
            if term[position] != "*":
                raise ValueError(
                    f"cannot parse binary-polynomial term {term!r}")
            position += 1
            if position == len(term):
                raise ValueError(
                    f"cannot parse binary-polynomial term {term!r}")
        monomials.append((i, j))
    return BinaryPolynomial(tuple(monomials))


@dataclass(frozen=True, slots=True)
class CyclicProduct:
    """The abelian group Z_l x Z_m indexing bivariate-bicycle circulants."""

    l: int
    m: int

    def __post_init__(self) -> None:
        if any(not isinstance(value, int) or isinstance(value, bool)
               for value in (self.l, self.m)):
            raise TypeError("CyclicProduct dimensions must be Python ints")
        if min(self.l, self.m) < 1:
            raise ValueError("CyclicProduct dimensions must be positive")


@dataclass(frozen=True, slots=True)
class BBPermutationMap:
    """One typed BB monomial permutation using row-major group indices.

    Index ``i + l*j`` denotes group element ``(i, j)``. ``mapping[row]`` is
    the unique column selected by the monomial matrix at that row, while
    :meth:`transpose` applies the inverse map.
    """

    label: str
    group: CyclicProduct
    exponent: tuple[int, int]
    mapping: tuple[int, ...]
    _inverse: tuple[int, ...] = field(init=False,
                                      repr=False,
                                      compare=False,
                                      hash=False)

    def __post_init__(self) -> None:
        from ..errors import InvalidSyndromeSchedule

        if not isinstance(self.label, str) or not self.label:
            raise TypeError(
                "BB permutation-map label must be a nonempty string")
        if not isinstance(self.group, CyclicProduct):
            raise TypeError(
                "BB permutation-map group must be cudaq.logical.CyclicProduct")
        if (not isinstance(self.exponent, tuple) or len(self.exponent) != 2 or
                any(not isinstance(value, int) or isinstance(value, bool) or
                    value < 0 for value in self.exponent)):
            raise TypeError(
                "BB permutation-map exponent must be two nonnegative ints")
        mapping = tuple(self.mapping)
        width = self.group.l * self.group.m
        if len(mapping) != width:
            raise InvalidSyndromeSchedule(
                f"BB permutation map {self.label} has width {len(mapping)}; "
                f"expected group width {width}")
        if any(not isinstance(value, int) or isinstance(value, bool) or
               value < 0 or value >= width for value in mapping):
            raise InvalidSyndromeSchedule(
                f"BB permutation map {self.label} entries must lie in [0, {width})"
            )
        if len(set(mapping)) != width:
            raise InvalidSyndromeSchedule(
                f"BB permutation map {self.label} must be bijective")
        x_power, y_power = self.exponent
        expected = tuple((i + x_power) % self.group.l + self.group.l *
                         ((j + y_power) % self.group.m)
                         for j in range(self.group.m)
                         for i in range(self.group.l))
        if mapping != expected:
            raise InvalidSyndromeSchedule(
                f"BB permutation map {self.label} does not match monomial "
                f"exponent {self.exponent}")
        inverse = [0] * width
        for source, target in enumerate(mapping):
            inverse[target] = source
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "_inverse", tuple(inverse))

    @classmethod
    def from_monomial(
        cls,
        label: str,
        *,
        group: CyclicProduct,
        exponent: tuple[int, int],
    ) -> "BBPermutationMap":
        x_power, y_power = exponent
        mapping = tuple(
            (i + x_power) % group.l + group.l * ((j + y_power) % group.m)
            for j in range(group.m)
            for i in range(group.l))
        return cls(label, group, tuple(exponent), mapping)

    def __len__(self) -> int:
        return len(self.mapping)

    def __call__(self, index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("BB permutation-map index must be an int")
        if index < 0 or index >= len(self):
            raise IndexError(
                f"BB permutation-map index {index} exceeds width {len(self)}")
        return self.mapping[index]

    def transpose(self, index: int) -> int:
        """Apply the transposed permutation matrix (the inverse map)."""

        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("BB transpose-map index must be an int")
        if index < 0 or index >= len(self):
            raise IndexError(
                f"BB transpose-map index {index} exceeds width {len(self)}")
        return self._inverse[index]

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        """All ``(row, mapped_column)`` pairs in canonical row order."""

        return tuple(enumerate(self.mapping))

    @property
    def transpose_pairs(self) -> tuple[tuple[int, int], ...]:
        """All pairs for the transposed permutation matrix."""

        return tuple((row, self.transpose(row)) for row in range(len(self)))


@dataclass(frozen=True, slots=True)
class BBSyndromeMoment:
    """One authored moment of the BB Table-5 extraction cycle."""

    index: int
    x_cx: tuple[tuple[int, int], ...] = ()
    z_cx: tuple[tuple[int, int], ...] = ()
    initialize_x: bool = False
    measure_z: bool = False
    measure_x: bool = False
    initialize_z: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.index, int) or isinstance(self.index, bool) or
                self.index < 1):
            raise TypeError("BB syndrome moment index must be a positive int")
        for label, pairs in (("x_cx", self.x_cx), ("z_cx", self.z_cx)):
            normalized = tuple(tuple(pair) for pair in pairs)
            if any(
                    len(pair) != 2 or any(not isinstance(value, int) or
                                          isinstance(value, bool) or value < 0
                                          for value in pair)
                    for pair in normalized):
                raise TypeError(
                    f"BB syndrome moment {label} entries must be nonnegative "
                    "integer pairs")
            object.__setattr__(self, label, normalized)
        for label in (
                "initialize_x",
                "measure_z",
                "measure_x",
                "initialize_z",
        ):
            if not isinstance(getattr(self, label), bool):
                raise TypeError(f"BB syndrome moment {label} must be bool")


@dataclass(frozen=True, slots=True)
class BBSyndromeSchedule:
    """Validated eight-moment syndrome cycle derived from one BB code."""

    code: "BivariateBicycleCode"
    moments: tuple[BBSyndromeMoment, ...]

    def __post_init__(self) -> None:
        from ..errors import InvalidSyndromeSchedule

        if not isinstance(self.code, BivariateBicycleCode):
            raise TypeError(
                "BB syndrome schedule requires BivariateBicycleCode")
        moments = tuple(self.moments)
        if any(not isinstance(moment, BBSyndromeMoment) for moment in moments):
            raise InvalidSyndromeSchedule(
                "BB depth-8 schedule moments must be BBSyndromeMoment values")
        if tuple(moment.index for moment in moments) != tuple(range(1, 9)):
            raise InvalidSyndromeSchedule(
                "BB depth-8 schedule must contain moments 1 through 8 exactly once"
            )
        lifecycle = tuple((
            moment.initialize_x,
            moment.measure_z,
            moment.measure_x,
            moment.initialize_z,
        ) for moment in moments)
        expected_lifecycle = (
            (True, False, False, False),
            (False, False, False, False),
            (False, False, False, False),
            (False, False, False, False),
            (False, False, False, False),
            (False, False, False, False),
            (False, True, False, False),
            (False, False, True, True),
        )
        if lifecycle != expected_lifecycle:
            raise InvalidSyndromeSchedule(
                "BB depth-8 lifecycle must initialize X checks in moment 1, "
                "measure Z checks in moment 7, and measure X plus initialize "
                "Z checks in moment 8")
        for moment in moments:
            x_checks = [check for check, _ in moment.x_cx]
            z_checks = [check for _, check in moment.z_cx]
            x_data = [data for _, data in moment.x_cx]
            z_data = [data for data, _ in moment.z_cx]
            if len(set(x_checks)) != len(x_checks):
                raise InvalidSyndromeSchedule(
                    f"BB depth-8 moment {moment.index} reuses an X ancilla")
            if len(set(z_checks)) != len(z_checks):
                raise InvalidSyndromeSchedule(
                    f"BB depth-8 moment {moment.index} reuses a Z ancilla")
            if len(set((*x_data, *z_data))) != len(x_data) + len(z_data):
                raise InvalidSyndromeSchedule(
                    f"BB depth-8 moment {moment.index} reuses a data carrier")

        expected_x = sorted((check, data)
                            for check, support in enumerate(self.code.hx)
                            for data in support)
        expected_z = sorted((data, check)
                            for check, support in enumerate(self.code.hz)
                            for data in support)
        actual_x = sorted(pair for moment in moments for pair in moment.x_cx)
        actual_z = sorted(pair for moment in moments for pair in moment.z_cx)
        if actual_x != expected_x:
            raise InvalidSyndromeSchedule(
                "BB depth-8 X interactions must cover every Hx incidence "
                "exactly once")
        if actual_z != expected_z:
            raise InvalidSyndromeSchedule(
                "BB depth-8 Z interactions must cover every Hz incidence "
                "exactly once")

        expected_moments = _bb_table5_moments(self.code)
        if moments != expected_moments:
            raise InvalidSyndromeSchedule(
                "BB depth-8 interactions must match the ordered A1-A3/B1-B3 "
                "assignment in Table 5 exactly")
        object.__setattr__(self, "moments", moments)

    @property
    def interaction_depth(self) -> int:
        """Number of authored cycle moments, including lifecycle-only moment 8."""

        return len(self.moments)

    @property
    def cnot_depth(self) -> int:
        """Number of moments containing one or more CNOT interactions."""

        return sum(bool(moment.x_cx or moment.z_cx) for moment in self.moments)

    def _matches_code(self, code: "Code") -> bool:
        """Return whether ``code`` materializes to this schedule's BB code."""

        return (isinstance(code, BivariateBicycleCode) and
                _materialized_code_identity(
                    self.code) == _materialized_code_identity(code))


def _bb_table5_moments(
    code: "BivariateBicycleCode",) -> tuple[BBSyndromeMoment, ...]:
    from ..errors import InvalidSyndromeSchedule

    if len(code.a_maps) != 3 or len(code.b_maps) != 3:
        raise InvalidSyndromeSchedule(
            "BB depth-8 extraction requires exactly three distinct ordered "
            "monomials in each of A and B")
    for family, maps in (("A", code.a_maps), ("B", code.b_maps)):
        if len({mapping.mapping for mapping in maps}) != len(maps):
            raise InvalidSyndromeSchedule(
                f"BB depth-8 extraction requires three distinct modular "
                f"permutations in {family}")
    a1, a2, a3 = code.a_maps
    b1, b2, b3 = code.b_maps
    half = code.group.l * code.group.m
    checks = range(half)
    x_pairs = lambda mapping, offset=0: tuple(
        (check, offset + mapping(check)) for check in checks)
    z_pairs = lambda mapping, offset=0: tuple(
        (offset + mapping.transpose(check), check) for check in checks)
    return (
        BBSyndromeMoment(
            1,
            z_cx=z_pairs(a1, half),
            initialize_x=True,
        ),
        BBSyndromeMoment(
            2,
            x_cx=x_pairs(a2),
            z_cx=z_pairs(a3, half),
        ),
        BBSyndromeMoment(
            3,
            x_cx=x_pairs(b2, half),
            z_cx=z_pairs(b1),
        ),
        BBSyndromeMoment(
            4,
            x_cx=x_pairs(b1, half),
            z_cx=z_pairs(b2),
        ),
        BBSyndromeMoment(
            5,
            x_cx=x_pairs(b3, half),
            z_cx=z_pairs(b3),
        ),
        BBSyndromeMoment(
            6,
            x_cx=x_pairs(a1),
            z_cx=z_pairs(a2, half),
        ),
        BBSyndromeMoment(
            7,
            x_cx=x_pairs(a3),
            measure_z=True,
        ),
        BBSyndromeMoment(
            8,
            measure_x=True,
            initialize_z=True,
        ),
    )


class BivariateBicycleCode(CSSCode):
    __slots__ = ("group", "a", "b", "a_maps", "b_maps")

    @classmethod
    def from_polynomials(
        cls,
        *,
        group: CyclicProduct,
        a: BinaryPolynomial,
        b: BinaryPolynomial,
        d=None,
        name: str | None = None,
    ):
        """Construct the BB code ``Hx=[A|B], Hz=[B^T|A^T]`` over the group."""

        from .catalog import (
            _circulant_rows,
            _gf2_null_space,
            _quotient_basis,
            _symplectic_pair_css,
        )

        if not isinstance(group, CyclicProduct):
            raise TypeError("group= must be cudaq.logical.CyclicProduct")
        if not isinstance(a, BinaryPolynomial) or not isinstance(
                b, BinaryPolynomial):
            raise TypeError("a= and b= must be cudaq.logical.BinaryPolynomial")

        l, m = group.l, group.m
        half = l * m
        n = 2 * half
        a_rows = _circulant_rows(l, m, a.monomials)
        b_rows = _circulant_rows(l, m, b.monomials)
        transpose_a = [list(column) for column in zip(*a_rows)]
        transpose_b = [list(column) for column in zip(*b_rows)]
        hx_dense = [left + right for left, right in zip(a_rows, b_rows)]
        hz_dense = [
            left + right for left, right in zip(transpose_b, transpose_a)
        ]
        lz_dense = _quotient_basis(_gf2_null_space(hx_dense, n), hz_dense, n)
        lx_dense = _quotient_basis(_gf2_null_space(hz_dense, n), hx_dense, n)
        if len(lx_dense) != len(lz_dense):
            raise ValueError(
                "bivariate-bicycle construction produced asymmetric logicals")
        lx_dense, lz_dense = _symplectic_pair_css(lx_dense, lz_dense)
        support = lambda row: tuple(index for index, value in enumerate(row)
                                    if value)
        result = cls(
            name=name or f"bb_{l}x{m}",
            block=Block(data=n, sx=half, sz=half),
            d=d,
            hx=tuple(support(row) for row in hx_dense),
            hz=tuple(support(row) for row in hz_dense),
            lx=tuple(support(row) for row in lx_dense),
            lz=tuple(support(row) for row in lz_dense),
            metadata={
                "family": "bivariate_bicycle",
                "group": (l, m),
                "a": str(a),
                "b": str(b),
                "a_term_order": a.terms,
                "b_term_order": b.terms,
            },
        )
        object.__setattr__(result, "group", group)
        object.__setattr__(result, "a", a)
        object.__setattr__(result, "b", b)
        object.__setattr__(
            result, "a_maps",
            tuple(
                BBPermutationMap.from_monomial(
                    f"A{index}", group=group, exponent=term)
                for index, term in enumerate(a.terms, start=1)))
        object.__setattr__(
            result, "b_maps",
            tuple(
                BBPermutationMap.from_monomial(
                    f"B{index}", group=group, exponent=term)
                for index, term in enumerate(b.terms, start=1)))
        return result

    def depth8_syndrome_schedule(self) -> BBSyndromeSchedule:
        """Derive the published interleaved Table-5 syndrome cycle.

        The construction requires three ordered monomial maps in each of
        ``A`` and ``B``. The returned schedule contains eight authored moments:
        seven CNOT layers and the lifecycle-only eighth moment that measures
        ``q(X)`` and initializes ``q(Z)`` for the next cycle.
        """

        return BBSyndromeSchedule(self, _bb_table5_moments(self))


__all__ = [
    "BinaryPolynomial",
    "binary_polynomial",
    "CyclicProduct",
    "BBPermutationMap",
    "BBSyndromeMoment",
    "BBSyndromeSchedule",
    "BivariateBicycleCode",
]
