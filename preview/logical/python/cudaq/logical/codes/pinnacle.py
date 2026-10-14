# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Generalized-bicycle codes used by the Pinnacle architecture.

The five presets are Table I of arXiv:2602.11457v2.  Their distances are
paper claims (the family formula is explicitly conjectural), so the returned
``Code`` objects retain citation provenance instead of promoting those values
to exact distance evidence.  For lifts 31 and above, the logical bases are
selected from the four published seed orbits in Appendix A of
arXiv:2511.15989v1; the lift-15 instance predates that appendix and uses a
deterministic homology quotient basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType

from ..analysis.evidence import citation
from cudaq.logical.codes.structure import CSSBlock
from cudaq.logical.codes.distance import Distance
from cudaq.logical.codes.definition import CSSCode


@dataclass(frozen=True, slots=True)
class PinnacleGBInstance:
    """One published Pinnacle generalized-bicycle processing-block row."""

    name: str
    ell: int
    a: tuple[int, ...]
    b: tuple[int, ...]
    k: int
    distance: int
    code_block_qubits: int
    gadget_qubits: int
    bridge_qubits: int
    processing_block_qubits: int

    @property
    def n(self) -> int:
        return 2 * self.ell

    @property
    def logical_cycle_rounds(self) -> int:
        return self.distance + 2


PINNACLE_GB_INSTANCES = MappingProxyType({
    "gb30":
        PinnacleGBInstance("gb30", 15, (0, 6, 13), (0, 1, 4), 8, 4, 60, 13, 7,
                           140),
    "gb62":
        PinnacleGBInstance("gb62", 31, (0, 6, 15), (0, 5, 7), 10, 6, 124, 19,
                           11, 244),
    "gb126":
        PinnacleGBInstance("gb126", 63, (0, 4, 37), (0, 29, 49), 12, 10, 252,
                           31, 19, 452),
    "gb254":
        PinnacleGBInstance(
            "gb254",
            127,
            (0, 32, 100),
            (0, 28, 49),
            14,
            16,
            508,
            57,
            31,
            860,
        ),
    "gb510":
        PinnacleGBInstance(
            "gb510",
            255,
            (0, 39, 55),
            (0, 70, 127),
            16,
            24,
            1020,
            99,
            51,
            1620,
        ),
})

_BY_DISTANCE = {row.distance: row for row in PINNACLE_GB_INSTANCES.values()}

PUBLISHED_GB_SEEDS = MappingProxyType({
    31: {
        "x0": ((1, 6, 8, 10), (11, 26)),
        "z0": ((3, 12, 18, 19), (11, 18)),
        "x1": ((16, 23), (0, 15, 16, 22)),
        "z1": ((0, 16), (1, 3, 5, 10)),
    },
    63: {
        "x0": ((7, 12, 36, 41, 56), (1, 27, 31, 38, 61)),
        "z0": ((5, 15, 28, 35, 45, 61), (1, 11, 54, 57)),
        "x1": ((9, 19, 26, 29), (5, 15, 22, 38, 48, 55)),
        "z1": ((2, 25, 32, 36, 62), (7, 22, 27, 51, 56)),
    },
    127: {
        "x0": (
            (28, 47, 55, 75, 103, 114, 124),
            (4, 14, 15, 23, 50, 77, 83, 109, 123),
        ),
        "z0": (
            (1, 24, 33, 51, 60, 65, 107, 119, 124),
            (7, 8, 36, 85, 106, 114, 124),
        ),
        "x1": (
            (3, 31, 32, 42, 52, 60, 81),
            (6, 15, 38, 42, 47, 59, 101, 106, 115),
        ),
        "z1": (
            (0, 8, 9, 19, 27, 41, 67, 73, 100),
            (26, 36, 47, 75, 95, 103, 122),
        ),
    },
    255: {
        "x0": (
            (18, 31, 35, 36, 91, 126, 146, 163, 164, 180, 196, 216, 233, 253),
            (48, 52, 87, 101, 103, 106, 107, 125, 140, 156, 179, 211),
        ),
        "z0": (
            (38, 54, 57, 93, 112, 148, 164, 185, 197, 203, 213, 238, 240, 252),
            (18, 55, 59, 73, 129, 130, 142, 182, 187, 199, 244, 252),
        ),
        "x1": (
            (6, 27, 35, 80, 92, 97, 137, 149, 150, 206, 220, 224),
            (27, 39, 41, 66, 76, 82, 94, 115, 131, 167, 186, 222, 225, 241),
        ),
        "z1": (
            (10, 11, 14, 16, 30, 65, 69, 161, 193, 216, 232, 247),
            (26, 81, 82, 86, 99, 119, 139, 156, 176, 192, 208, 209, 226, 246),
        ),
    },
})


def _rref(rows: tuple[int, ...]) -> tuple[int, ...]:
    basis: dict[int, int] = {}
    for value in rows:
        row = value
        while row:
            pivot = row.bit_length() - 1
            if pivot not in basis:
                basis[pivot] = row
                break
            row ^= basis[pivot]
    for pivot in sorted(basis):
        for other in tuple(basis):
            if other != pivot and (basis[other] >> pivot) & 1:
                basis[other] ^= basis[pivot]
    return tuple(basis[pivot] for pivot in sorted(basis, reverse=True))


def _rank(rows: tuple[int, ...]) -> int:
    return len(_rref(rows))


def _in_span(row: int, rows: tuple[int, ...]) -> bool:
    return _rank(rows + (row,)) == _rank(rows)


def _nullspace(rows: tuple[int, ...], columns: int) -> tuple[int, ...]:
    reduced = _rref(rows)
    pivots = {row.bit_length() - 1: row for row in reduced}
    output = []
    for free in (column for column in range(columns) if column not in pivots):
        vector = 1 << free
        for pivot, row in pivots.items():
            if (row >> free) & 1:
                vector |= 1 << pivot
        output.append(vector)
    return tuple(output)


def _quotient_basis(centralizer: tuple[int, ...], stabilizers: tuple[int, ...],
                    count: int) -> tuple[int, ...]:
    span = list(_rref(stabilizers))
    output = []
    for row in centralizer:
        if not _in_span(row, tuple(span)):
            span.append(row)
            output.append(row)
            if len(output) == count:
                return tuple(output)
    raise ValueError(
        f"homology quotient supplied {len(output)}, expected {count}")


def _solve(rows: tuple[int, ...], rhs: tuple[int, ...], columns: int) -> int:
    equations = [row | (bit << columns) for row, bit in zip(rows, rhs)]
    pivot_row = 0
    pivots = []
    for column in range(columns):
        found = next(
            (index for index in range(pivot_row, len(equations))
             if (equations[index] >> column) & 1),
            None,
        )
        if found is None:
            continue
        equations[pivot_row], equations[found] = (
            equations[found],
            equations[pivot_row],
        )
        for index in range(len(equations)):
            if index != pivot_row and (equations[index] >> column) & 1:
                equations[index] ^= equations[pivot_row]
        pivots.append(column)
        pivot_row += 1
    mask = (1 << columns) - 1
    if any((equation & mask) == 0 and ((equation >> columns) & 1)
           for equation in equations):
        raise ValueError("logical pairing matrix is singular")
    solution = 0
    for equation, pivot in zip(equations[:pivot_row], pivots):
        if (equation >> columns) & 1:
            solution |= 1 << pivot
    return solution


def _canonicalize_lz(lx: tuple[int, ...], lz: tuple[int,
                                                    ...]) -> tuple[int, ...]:
    k = len(lx)
    pairing = tuple(
        sum(((x & z).bit_count() & 1) << column
            for column, z in enumerate(lz))
        for x in lx)
    output = []
    for logical in range(k):
        coefficients = _solve(
            pairing,
            tuple(int(index == logical) for index in range(k)),
            k,
        )
        row = 0
        for index, old in enumerate(lz):
            if (coefficients >> index) & 1:
                row ^= old
        output.append(row)
    return tuple(output)


def _gb_matrices(instance: PinnacleGBInstance):
    ell = instance.ell
    hx = tuple(
        tuple(sorted({(row + shift) % ell
                      for shift in instance.a})) +
        tuple(ell + qubit
              for qubit in sorted({(row + shift) % ell
                                   for shift in instance.b}))
        for row in range(ell))
    hz = tuple(
        tuple(sorted({(row - shift) % ell
                      for shift in instance.b})) +
        tuple(ell + qubit
              for qubit in sorted({(row - shift) % ell
                                   for shift in instance.a}))
        for row in range(ell))
    return hx, hz


def _pack(rows) -> tuple[int, ...]:
    return tuple(sum(1 << qubit for qubit in row) for row in rows)


def _unpack(row: int) -> tuple[int, ...]:
    return tuple(
        index for index in range(row.bit_length()) if (row >> index) & 1)


def _seed_row(ell: int, seed) -> int:
    left, right = seed
    return sum(1 << qubit for qubit in left) | sum(
        1 << (ell + qubit) for qubit in right)


def _orbit(row: int, ell: int) -> tuple[int, ...]:
    output = []
    for shift in range(ell):
        shifted = 0
        for qubit in range(2 * ell):
            if (row >> qubit) & 1:
                sector, index = divmod(qubit, ell)
                shifted |= 1 << (sector * ell + (index + shift) % ell)
        output.append(shifted)
    return tuple(output)


def _select_orbits(families, stabilizers, count_per_family):
    span = list(_rref(stabilizers))
    selected = []
    for family in families:
        before = len(selected)
        for row in family:
            if not _in_span(row, tuple(span)):
                span.append(row)
                selected.append(row)
                if len(selected) - before == count_per_family:
                    break
        if len(selected) - before != count_per_family:
            raise ValueError(
                "published seed orbit does not span its logical sector")
    return tuple(selected)


def pinnacle_gb_instance(value: str | int) -> PinnacleGBInstance:
    """Resolve a preset by canonical name or by its published distance."""

    if isinstance(value, str):
        try:
            return PINNACLE_GB_INSTANCES[value]
        except KeyError as exc:
            raise ValueError(f"unknown Pinnacle GB preset {value!r}; expected "
                             f"{tuple(PINNACLE_GB_INSTANCES)}") from exc
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            return _BY_DISTANCE[value]
        except KeyError as exc:
            raise ValueError(
                f"unknown Pinnacle GB distance {value!r}; expected "
                f"{tuple(sorted(_BY_DISTANCE))}") from exc
    raise TypeError("Pinnacle GB preset must be a name or distance int")


@lru_cache(maxsize=None)
def pinnacle_gb(value: str | int) -> CSSCode:
    """Return one native Mark III Pinnacle generalized-bicycle code."""

    instance = pinnacle_gb_instance(value)
    hx, hz = _gb_matrices(instance)
    hx_bits = _pack(hx)
    hz_bits = _pack(hz)
    actual_k = instance.n - _rank(hx_bits) - _rank(hz_bits)
    if actual_k != instance.k:
        raise ValueError(
            f"{instance.name} matrix rank gives k={actual_k}, expected {instance.k}"
        )

    seed_evidence = None
    if instance.ell == 15:
        lx_bits = _quotient_basis(_nullspace(hz_bits, instance.n), hx_bits,
                                  instance.k)
        lz_bits = _quotient_basis(_nullspace(hx_bits, instance.n), hz_bits,
                                  instance.k)
    else:
        seeds = PUBLISHED_GB_SEEDS[instance.ell]
        packed = {
            name: _seed_row(instance.ell, seed) for name, seed in seeds.items()
        }
        x_orbits = (_orbit(packed["x0"],
                           instance.ell), _orbit(packed["x1"], instance.ell))
        z_orbits = (_orbit(packed["z0"],
                           instance.ell), _orbit(packed["z1"], instance.ell))
        half = instance.k // 2
        lx_bits = _select_orbits(x_orbits, hx_bits, half)
        lz_bits = _select_orbits(z_orbits, hz_bits, half)
        seed_evidence = {
            "source":
                "arXiv:2511.15989v1 Appendix A",
            "x_sector_dimensions": (
                _rank(hx_bits + x_orbits[0]) - _rank(hx_bits),
                _rank(hx_bits + x_orbits[0] + x_orbits[1]) -
                _rank(hx_bits + x_orbits[0]),
            ),
        }

    lz_bits = _canonicalize_lz(lx_bits, lz_bits)
    metadata = {
        "family": "pinnacle_generalized_bicycle",
        "ell": instance.ell,
        "a": instance.a,
        "b": instance.b,
        "published_parameters": (instance.n, instance.k, instance.distance),
        "logical_cycle_rounds": instance.logical_cycle_rounds,
        "code_block_qubits": instance.code_block_qubits,
        "gadget_qubits": instance.gadget_qubits,
        "bridge_qubits": instance.bridge_qubits,
        "processing_block_qubits": instance.processing_block_qubits,
        "distance_status": "conjectured family value",
        "distance_source": "arXiv:2602.11457v2 Table I",
    }
    if seed_evidence is not None:
        metadata["seed_evidence"] = seed_evidence
    return CSSCode(
        name=f"pinnacle_{instance.name}",
        n=instance.n,
        k=instance.k,
        d=Distance.claimed(
            instance.distance,
            provenance=citation("arXiv:2602.11457v2 Table I"),
        ),
        block=CSSBlock(data=instance.n, sx=instance.ell, sz=instance.ell),
        hx=hx,
        hz=hz,
        lx=tuple(_unpack(row) for row in lx_bits),
        lz=tuple(_unpack(row) for row in lz_bits),
        metadata=metadata,
    )


__all__ = [
    "PINNACLE_GB_INSTANCES",
    "PUBLISHED_GB_SEEDS",
    "PinnacleGBInstance",
    "pinnacle_gb",
    "pinnacle_gb_instance",
]
