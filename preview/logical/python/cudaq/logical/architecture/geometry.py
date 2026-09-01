# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Geometry adapters for topological code families.

A geometry object carries everything a parametric code family derives from
its lattice — face supports tagged by geometric sublattice, the data
bipartition, logical chains, coordinates, and the provenance of its distance
argument — so each family states its construction once instead of restating
matrices per instance. The geometry is deliberately Pauli-free: it never
assigns an X/Z basis, so both a CSS surface code and a non-CSS (`ZXXZ`) surface
code can share one lattice and each own its stabilizers (see
:func:`cudaq.logical.codes.rotated_surface` and :func:`cudaq.logical.codes.zxxz_surface`).
"""

from __future__ import annotations

import itertools

from dataclasses import dataclass, field
from functools import lru_cache
from types import MappingProxyType
from typing import Any, Mapping

from ..analysis.evidence import Provenance


def _neighbors(point, offsets, data):
    return tuple((point[0] + dx, point[1] + dy)
                 for dx, dy in offsets
                 if (point[0] + dx, point[1] + dy) in data)


@dataclass(frozen=True, slots=True)
class SurfaceFace:
    """One rotated-surface plaquette as pure geometry.

    ``sublattice`` is 0 or 1 -- the two interleaved plaquette families of the
    lattice. It is a geometric label, *not* a Pauli type: a CSS code reads one
    family in X and the other in Z, a `ZXXZ` code reads every face as a mixed
    product. The geometry states neither.
    """

    support: tuple[int, ...]
    sublattice: int
    coordinate: tuple[int, int]


@dataclass(frozen=True, slots=True)
class SurfaceLattice:
    """Pauli-free rotated-surface lattice for one odd distance.

    Carries only geometry -- data coordinates, the plaquette ``faces`` tagged by
    geometric ``sublattice``, the two straight boundary ``logical_chains``, and
    the data-qubit ``bipartition`` (2-coloring). Codes assign their own
    operators on top: this is the shared tool for both the CSS surface code and
    the `ZXXZ` surface code, and it commits to no stabilizer basis.
    """

    distance: int
    n: int
    coordinates: tuple[tuple[int, int], ...]
    faces: tuple[SurfaceFace, ...]
    logical_chains: tuple[tuple[int, ...], tuple[int, ...]]
    bipartition: tuple[int, ...]
    distance_evidence: Provenance
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata",
                           MappingProxyType(dict(self.metadata or {})))

    @property
    def name(self) -> str:
        return f"rotated_surface_{self.distance}"

    def sublattice_faces(self, sublattice: int) -> tuple[tuple[int, ...], ...]:
        """Supports of the faces on one plaquette family, in geometric order."""
        return tuple(face.support
                     for face in self.faces
                     if face.sublattice == sublattice)


@lru_cache(maxsize=None)
def rotated_surface_lattice(distance: int) -> SurfaceLattice:
    """Derive the Pauli-free rotated-surface lattice for one odd ``distance``.

    The minimum-weight logical operators are the straight boundary chains, so
    the geometric distance argument is exact by construction.
    """
    if not isinstance(distance, int) or isinstance(distance, bool):
        raise TypeError("rotated-surface distance must be a Python int")
    if distance < 3 or distance % 2 == 0:
        raise ValueError("rotated-surface distance must be odd and at least 3")

    corners = ((1, 1), (-1, 1), (1, -1), (-1, -1))
    coordinates = tuple((2 * column + 1, 2 * row + 1)
                        for row in range(distance)
                        for column in range(distance))
    data = set(coordinates)
    data_index = {point: index for index, point in enumerate(coordinates)}

    # Two interleaved plaquette families (sublattice 1 first, then 0, each in
    # reading order) so a consumer recovers a stable per-family order by filter.
    faces: list[SurfaceFace] = []
    for wanted in (1, 0):
        family = []
        for y in range(0, 2 * distance + 1, 2):
            for x in range(0, 2 * distance + 1, 2):
                sublattice = (x // 2 + y // 2) % 2
                if sublattice != wanted:
                    continue
                support = _neighbors((x, y), corners, data)
                if len(support) < 2:
                    continue
                if sublattice == 1 and x in (0, 2 * distance):
                    continue
                if sublattice == 0 and y in (0, 2 * distance):
                    continue
                family.append((x, y, support))
        family.sort(key=lambda item: (item[1], item[0]))
        for x, y, support in family:
            faces.append(
                SurfaceFace(
                    support=tuple(sorted(data_index[item] for item in support)),
                    sublattice=wanted,
                    coordinate=(x, y),
                ))

    chain_vertical = tuple(
        data_index[item] for item in coordinates if item[0] == 1)
    chain_horizontal = tuple(
        data_index[item] for item in coordinates if item[1] == 1)
    bipartition = tuple(((index // distance) + (index % distance)) % 2
                        for index in range(distance * distance))
    return SurfaceLattice(
        distance=distance,
        n=distance * distance,
        coordinates=coordinates,
        faces=tuple(faces),
        logical_chains=(chain_vertical, chain_horizontal),
        bipartition=bipartition,
        distance_evidence=Provenance("geometry",
                                     f"rotated_surface(distance={distance})"),
        metadata={
            "family": "rotated_surface",
            "distance": distance
        },
    )


# Compatibility spellings keep the public geometry adapter intact while the
# returned value remains the new Pauli-free lattice. CSSCode.from_geometry owns
# the conventional sublattice-to-X/Z assignment.
SurfaceGeometry = SurfaceLattice


def rotated_surface(distance: int) -> SurfaceLattice:
    """Return the Pauli-free rotated-surface lattice (compatibility spelling)."""

    return rotated_surface_lattice(distance)


@dataclass(frozen=True, slots=True)
class HoneycombFloquetGeometry:
    """Derived honeycomb-torus data for the three-round Floquet schedule.

    Rows are GF(2) symplectic vectors of width ``2n`` (X part then Z part).
    ``gauges`` holds the two-body edge checks of each color in the stable
    per-phase record order ``g0..g{m-1}``; ``plaquettes`` the hexagon
    stabilizers; ``conserved_strips`` the two homologically non-trivial
    central operators; ``gauge_pairs`` a canonical hyperbolic basis of the
    full measured group whose span contains every edge check;
    ``temporal_recovery`` the within-period deterministic record closures
    over the flattened ``red | green | blue`` gauge-record order; and
    ``moving_frame`` the per-period byproduct-frame tracking data for the
    two dynamically protected logical representatives.
    """

    size: int
    n: int
    colors: tuple[str, ...]
    gauges: Mapping[str, tuple[tuple[int, ...], ...]]
    gauge_supports: Mapping[str, tuple[tuple[int, int], ...]]
    plaquettes: Mapping[str, tuple[tuple[int, ...], ...]]
    conserved_strips: tuple[tuple[int, ...], tuple[int, ...]]
    isg: Mapping[str, tuple[tuple[int, ...], ...]]
    gauge_pairs: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    temporal_recovery: tuple[tuple[int, ...], ...]
    temporal_recovery_labels: tuple[str, ...]
    moving_frame: Mapping[str, Mapping[str, Any]]
    period_closure: Mapping[str, tuple[str, ...]]
    distance_evidence: Provenance
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"honeycomb_floquet_{self.size}x{self.size}"


def _hf_sym(a: int, b: int, n: int) -> int:
    mask = (1 << n) - 1
    return (bin((a & mask) &
                (b >> n)).count("1") + bin((a >> n) &
                                           (b & mask)).count("1")) % 2


def _hf_reduce(rows):
    basis = []
    for row in rows:
        for existing in basis:
            row = min(row, row ^ existing)
        if row:
            basis.append(row)
            basis.sort(reverse=True)
    return basis


def _hf_in_span(row: int, basis) -> bool:
    for existing in basis:
        row = min(row, row ^ existing)
    return row == 0


def _hf_nullspace(equation_rows, ncols):
    pivot_columns = []
    reduced = []
    for row in equation_rows:
        current = row
        for column, existing in zip(pivot_columns, reduced):
            if (current >> column) & 1:
                current ^= existing
        if current:
            column = (current & -current).bit_length() - 1
            for index in range(len(reduced)):
                if (reduced[index] >> column) & 1:
                    reduced[index] ^= current
            reduced.append(current)
            pivot_columns.append(column)
    taken = set(pivot_columns)
    basis = []
    for free in range(ncols):
        if free in taken:
            continue
        vector = 1 << free
        for column, existing in zip(pivot_columns, reduced):
            if (existing >> free) & 1:
                vector |= 1 << column
        basis.append(vector)
    return basis


def _hf_commutant(rows, n):
    equations = []
    mask = (1 << n) - 1
    for generator in _hf_reduce(rows):
        gx, gz = generator & mask, generator >> n
        row = 0
        for bit in range(n):
            if (gz >> bit) & 1:
                row |= 1 << bit
            if (gx >> bit) & 1:
                row |= 1 << (n + bit)
        equations.append(row)
    return _hf_nullspace(equations, 2 * n)


def _hf_weight(row: int, n: int) -> int:
    mask = (1 << n) - 1
    return bin((row & mask) | (row >> n)).count("1")


def _hf_row_tuple(row: int, n: int) -> tuple[int, ...]:
    return tuple((row >> index) & 1 for index in range(2 * n))


def honeycomb_floquet(size: int) -> HoneycombFloquetGeometry:
    """Derive the honeycomb-torus Floquet-code data for one ``size``.

    Qubits sit on the ``2*size**2`` vertices of a honeycomb lattice wrapped
    on a torus; each edge carries a two-body check whose Pauli type follows
    its color (red = XX, green = YY, blue = ZZ) and each round measures one
    color family.  A proper three-coloring of hexagons — and therefore of
    edges — exists exactly when ``size`` is a multiple of three.

    Every structural claim (plaquette closure, center rank, instantaneous
    stabilizer ranks, canonical gauge pairing, span coverage, moving-frame
    transfer, period closure, and the weight-2 exhaustive logical search
    behind the distance evidence) is recomputed and checked here; the
    construction refuses to return unverified data.
    """

    if not isinstance(size, int) or isinstance(size, bool):
        raise TypeError("honeycomb size must be a Python int")
    if size < 3 or size % 3:
        raise ValueError(
            "the honeycomb torus is properly three-colorable only for "
            "size multiples of three (smallest honest instance: 3)")
    length = size
    n = 2 * length * length
    colors = ("red", "green", "blue")

    def vertex(i, j, sublattice):
        return 2 * ((j % length) * length + (i % length)) + sublattice

    def pauli_bits(site, color_index):
        x = 1 << site if color_index in (0, 1) else 0
        z = 1 << (n + site) if color_index in (1, 2) else 0
        return x | z

    # Edge combinatorics of the brickwork honeycomb: three edge families per
    # unit cell; hexagon (i, j) takes color (i - j) mod 3 and every edge takes
    # the color of the unique hexagon pair it does not bound.
    edges = {}
    for j in range(length):
        for i in range(length):
            edges[("e0", i, j)] = (vertex(i, j, 0), vertex(i, j,
                                                           1), (i - j) % 3)
            edges[("e1", i, j)] = (vertex(i + 1, j,
                                          0), vertex(i, j, 1), (i - j - 1) % 3)
            edges[("e2", i, j)] = (vertex(i, j + 1,
                                          0), vertex(i, j, 1), (i - j + 1) % 3)
    edge_keys = sorted(edges)
    edge_row = {
        key: pauli_bits(u, c) ^ pauli_bits(v, c)
        for key, (u, v, c) in edges.items()
    }
    by_color = {
        c: tuple(key for key in edge_keys if edges[key][2] == c)
        for c in range(3)
    }
    incident = {}
    for key, (u, v, c) in edges.items():
        incident.setdefault(u, []).append(c)
        incident.setdefault(v, []).append(c)
    if any(sorted(cs) != [0, 1, 2] for cs in incident.values()):
        raise AssertionError("vertex must touch one edge of each color")

    def hexagon(i, j):
        return (
            ("e1", i % length, j % length),
            ("e2", i % length, j % length),
            ("e0", i % length, (j + 1) % length),
            ("e1", i % length, (j + 1) % length),
            ("e2", (i + 1) % length, j % length),
            ("e0", (i + 1) % length, j % length),
        )

    hexes = {
        (i, j): (hexagon(i, j), (i - j) % 3) for j in range(length)
        for i in range(length)
    }
    plaq_row = {}
    for key, (boundary, c) in hexes.items():
        row = 0
        support = set()
        for edge in boundary:
            row ^= edge_row[edge]
            support.update(edges[edge][:2])
        pure = 0
        for site in support:
            pure ^= pauli_bits(site, c)
        if row != pure or len(support) != 6:
            raise AssertionError("plaquette is not the pure-color hexagon")
        if any(_hf_sym(row, e, n) for e in edge_row.values()):
            raise AssertionError("plaquette must be central in the gauge group")
        plaq_row[key] = row
    all_plaqs = [plaq_row[key] for key in sorted(plaq_row)]

    # Group structure: one global relation among the edge checks, hexagon
    # relations among the plaquettes, and a rank-(s^2/9*... ) center spanned
    # by the plaquettes plus two homologically non-trivial strips.
    edge_rows = [edge_row[key] for key in edge_keys]
    group_basis = _hf_reduce(edge_rows)
    plaq_basis = _hf_reduce(all_plaqs)
    gram = []
    for a in edge_rows:
        bits = 0
        for index, b in enumerate(edge_rows):
            if _hf_sym(a, b, n):
                bits |= 1 << index
        gram.append(bits)
    center = []
    for coefficients in _hf_nullspace(gram, len(edge_rows)):
        row = 0
        for index in range(len(edge_rows)):
            if (coefficients >> index) & 1:
                row ^= edge_rows[index]
        if row:
            center.append(row)
    center_basis = _hf_reduce(center)
    if len(center_basis) != len(plaq_basis) + 2:
        raise AssertionError("center must add exactly two strips to plaquettes")
    strips = []
    accumulated = list(plaq_basis)
    for row in center_basis:
        if not _hf_in_span(row, _hf_reduce(accumulated)):
            strips.append(row)
            accumulated.append(row)
    if len(strips) != 2:
        raise AssertionError("expected two conserved strip operators")
    # Choose minimum-weight coset representatives for readability.
    minimized = []
    for strip in strips:
        best = strip
        for count in range(1, len(plaq_basis) + 1):
            for combo in itertools.combinations(plaq_basis, count):
                candidate = strip
                for row in combo:
                    candidate ^= row
                if _hf_weight(candidate, n) < _hf_weight(
                        best, n) or (_hf_weight(candidate, n) == _hf_weight(
                            best, n) and candidate < best):
                    best = candidate
        minimized.append(best)
    strip_h, strip_v = sorted(minimized)

    # Canonical hyperbolic pairs spanning the measured group modulo center.
    work = list(group_basis)
    raw_pairs = []
    while True:
        found = None
        for a, b in itertools.combinations(work, 2):
            if _hf_sym(a, b, n):
                found = (a, b)
                break
        if found is None:
            break
        a, b = found
        raw_pairs.append((a, b))
        projected = []
        for row in work:
            if row is a or row is b:
                continue
            if _hf_sym(row, b, n):
                row ^= a
            if _hf_sym(row, a, n):
                row ^= b
            if row:
                projected.append(row)
        work = _hf_reduce(projected)
    if any(not _hf_in_span(row, _hf_reduce(center_basis)) for row in work):
        raise AssertionError("Gram-Schmidt residue must be the center")

    # Instantaneous stabilizer groups and the dynamically protected pairs.
    isg_rows = {}
    for c in range(3):
        rows = all_plaqs + [edge_row[key] for key in by_color[c]]
        isg_rows[c] = rows
        if len(_hf_reduce(rows)) != n - 2:
            raise AssertionError(
                "each round's ISG must leave two logical qubits")
    isg_red_basis = _hf_reduce(isg_rows[0])
    commutant = _hf_commutant(isg_rows[0], n)
    quotient = []
    accumulated = list(isg_red_basis)
    for row in commutant:
        if not _hf_in_span(row, _hf_reduce(accumulated)):
            quotient.append(row)
            accumulated.append(row)
    if len(quotient) != 4:
        raise AssertionError("ISG logical space must be four-dimensional")

    def complete_partner(anchor, exclusions):
        best = None
        for count in range(1, len(quotient) + 1):
            for combo in itertools.combinations(quotient, count):
                candidate = 0
                for row in combo:
                    candidate ^= row
                if _hf_sym(anchor, candidate, n) != 1:
                    continue
                if any(_hf_sym(other, candidate, n) for other in exclusions):
                    continue
                # Multiplying by ISG elements preserves every pairing, so a
                # reduced coset representative is equally valid; keep the
                # lighter spelling of the two.
                reduced = candidate
                for row in isg_red_basis:
                    reduced = min(reduced, reduced ^ row)
                candidate = min((candidate, reduced),
                                key=lambda r: _hf_weight(r, n))
                if best is None or _hf_weight(candidate, n) < _hf_weight(
                        best, n):
                    best = candidate
        return best

    sigma_h = complete_partner(strip_h, (strip_v,))
    sigma_v = complete_partner(strip_v, (strip_h, sigma_h))
    if sigma_h is None or sigma_v is None:
        raise AssertionError("moving-frame partners must exist")
    if (_hf_sym(strip_h, sigma_h, n) != 1 or
            _hf_sym(strip_v, sigma_v, n) != 1 or _hf_sym(strip_h, sigma_v, n) or
            _hf_sym(strip_v, sigma_h, n) or _hf_sym(sigma_h, sigma_v, n) or
            _hf_sym(strip_h, strip_v, n)):
        raise AssertionError("logical pairs must be canonically symplectic")

    # Correct the hyperbolic pairs so they commute with the moving-frame
    # partners: the correction only multiplies central strips in, so the
    # corrected pairs stay inside the measured group.
    pairs = []
    for a, b in raw_pairs:
        for anchor, partner in ((strip_h, sigma_h), (strip_v, sigma_v)):
            if _hf_sym(a, partner, n):
                a ^= anchor
            if _hf_sym(b, partner, n):
                b ^= anchor
        pairs.append((a, b))
    declared = list(plaq_basis) + [strip_h, strip_v, sigma_h, sigma_v]
    for a, b in pairs:
        declared.extend((a, b))
    declared_basis = _hf_reduce(declared)
    if any(not _hf_in_span(row, declared_basis) for row in edge_rows):
        raise AssertionError("every edge check must lie in the declared span")
    full_pairs = tuple(pairs) + ((strip_h, sigma_h), (strip_v, sigma_v))
    gauge_x_rows = [a for a, _ in full_pairs]
    gauge_z_rows = [b for _, b in full_pairs]
    for i, a in enumerate(gauge_x_rows):
        for j, b in enumerate(gauge_z_rows):
            if _hf_sym(a, b, n) != int(i == j):
                raise AssertionError("gauge pairs must be canonical")
    for family in (gauge_x_rows, gauge_z_rows):
        for a, b in itertools.combinations(family, 2):
            if _hf_sym(a, b, n):
                raise AssertionError("gauge families must commute internally")

    # Moving-frame transfer: multiply the tracked representative by recorded
    # link operators (this round's, and the previous round's whose outcomes
    # are still valid) until it commutes with the round being measured.
    def transfer(rep, target_color, previous_color):
        links = list(by_color[target_color]) + list(by_color[previous_color])
        generators = [edge_row[key] for key in links]
        targets = [edge_row[key] for key in by_color[target_color]]
        equations = []
        rhs = []
        for target in targets:
            row = 0
            for index, generator in enumerate(generators):
                if _hf_sym(generator, target, n):
                    row |= 1 << index
            equations.append(row)
            rhs.append(_hf_sym(rep, target, n))
        pivots, reduced, reduced_rhs = [], [], []
        for row, bit in zip(equations, rhs):
            current, value = row, bit
            for pivot, (existing,
                        existing_rhs) in zip(pivots, zip(reduced, reduced_rhs)):
                if (current >> pivot) & 1:
                    current ^= existing
                    value ^= existing_rhs
            if current:
                pivot = (current & -current).bit_length() - 1
                for index in range(len(reduced)):
                    if (reduced[index] >> pivot) & 1:
                        reduced[index] ^= current
                        reduced_rhs[index] ^= value
                reduced.append(current)
                reduced_rhs.append(value)
                pivots.append(pivot)
            elif value:
                raise AssertionError("moving-frame transfer must be solvable")
        solution = 0
        for pivot, value in zip(pivots, reduced_rhs):
            if value:
                solution |= 1 << pivot
        new_rep = rep
        chosen = []
        for index, key in enumerate(links):
            if (solution >> index) & 1:
                new_rep ^= edge_row[key]
                chosen.append(key)
        if any(_hf_sym(new_rep, target, n) for target in targets):
            raise AssertionError("transferred representative must commute")
        return new_rep, tuple(chosen)

    tracked = {"x1": strip_h, "z1": sigma_h, "x2": strip_v, "z2": sigma_v}
    frame = {}
    closing = {}
    for name, rep in tracked.items():
        current = rep
        updates = []
        # Close the complete red -> green -> blue -> red period. Every
        # selected link has a measurement record in that flattened period, so
        # this update list is a complete replayable transport proof.
        for target, previous in ((1, 0), (2, 1), (0, 2)):
            current, chosen = transfer(current, target, previous)
            updates.extend(chosen)
        closing[name] = current
        frame[name] = tuple(updates)
    if frame["x1"] or frame["x2"]:
        raise AssertionError("conserved strips must transfer without updates")

    # Period closure: express the already closed returning representative in
    # the tracked logical basis.
    closure = {}
    for name in tracked:
        final = closing[name]
        coordinates = tuple(label for label, (partner_sign) in (
            ("x1", _hf_sym(final, sigma_h, n)),
            ("z1", _hf_sym(final, strip_h, n)),
            ("x2", _hf_sym(final, sigma_v, n)),
            ("z2", _hf_sym(final, strip_v, n)),
        ) if partner_sign)
        residue = final
        for label, row in tracked.items():
            if label in coordinates:
                residue ^= row
        if not _hf_in_span(residue, isg_red_basis):
            raise AssertionError("period closure must return to the ISG coset")
        closure[name] = coordinates

    # Within-period deterministic closures over the flattened record order
    # red g0.. | green g0.. | blue g0..: hexagons whose two boundary colors
    # are both measured this period, plus the three full-color parities.
    per_phase = len(by_color[0])
    offsets = {c: index * per_phase for index, c in enumerate(range(3))}
    recovery_rows = []
    recovery_labels = []
    for key, (boundary, c) in sorted(hexes.items()):
        boundary_colors = {edges[edge][2] for edge in boundary}
        if c == 1:
            continue  # green hexagons close across the period boundary
        columns = [0] * (3 * per_phase)
        product = 0
        for edge in boundary:
            color = edges[edge][2]
            columns[offsets[color] + by_color[color].index(edge)] ^= 1
            product ^= edge_row[edge]
        if product != plaq_row[key]:
            raise AssertionError("recovery row must reconstruct its plaquette")
        recovery_rows.append(tuple(columns))
        recovery_labels.append(f"{colors[c]}_plaquette_{key[0]}_{key[1]}")
    for c in range(3):
        columns = [0] * (3 * per_phase)
        product = 0
        for index, key in enumerate(by_color[c]):
            columns[offsets[c] + index] = 1
            product ^= edge_row[key]
        if not _hf_in_span(product, _hf_reduce(plaq_basis)):
            raise AssertionError(
                "full-color parity must be a plaquette product")
        recovery_rows.append(tuple(columns))
        recovery_labels.append(f"{colors[c]}_color_parity")

    # Distance evidence: no weight-1 or weight-2 operator commutes with any
    # round's ISG while acting non-trivially on its logical quotient.
    for c in range(3):
        basis = _hf_reduce(isg_rows[c])
        comm_basis = _hf_reduce(_hf_commutant(isg_rows[c], n))
        for w in (1, 2):
            for sites in itertools.combinations(range(n), w):
                for paulis in itertools.product((1, 2, 3), repeat=w):
                    row = 0
                    for site, pauli in zip(sites, paulis):
                        if pauli & 1:
                            row |= 1 << site
                        if pauli & 2:
                            row |= 1 << (n + site)
                    if _hf_in_span(row,
                                   comm_basis) and not _hf_in_span(row, basis):
                        raise AssertionError(
                            "found a low-weight instantaneous logical")

    color_of = {0: "red", 1: "green", 2: "blue"}

    def record_names(update_keys):
        return tuple(
            (color_of[edges[key][2]], by_color[edges[key][2]].index(key))
            for key in update_keys)

    return HoneycombFloquetGeometry(
        size=length,
        n=n,
        colors=colors,
        gauges={
            color_of[c]:
                tuple(_hf_row_tuple(edge_row[key], n) for key in by_color[c])
            for c in range(3)
        },
        gauge_supports={
            color_of[c]:
                tuple((edges[key][0], edges[key][1]) for key in by_color[c])
            for c in range(3)
        },
        plaquettes={
            color_of[c]:
                tuple(
                    _hf_row_tuple(plaq_row[key], n)
                    for key in sorted(plaq_row)
                    if hexes[key][1] == c) for c in range(3)
        },
        conserved_strips=(
            _hf_row_tuple(strip_h, n),
            _hf_row_tuple(strip_v, n),
        ),
        isg={
            color_of[c]: tuple(_hf_row_tuple(row, n) for row in isg_rows[c])
            for c in range(3)
        },
        gauge_pairs=tuple(
            (_hf_row_tuple(a, n), _hf_row_tuple(b, n)) for a, b in full_pairs),
        temporal_recovery=tuple(recovery_rows),
        temporal_recovery_labels=tuple(recovery_labels),
        moving_frame={
            name: {
                "initial": _hf_row_tuple(tracked[name], n),
                "closing": _hf_row_tuple(closing[name], n),
                "updates": record_names(frame[name]),
            } for name in ("z1", "z2")
        },
        period_closure={name: closure[name] for name in sorted(closure)},
        distance_evidence=Provenance(
            "geometry",
            f"honeycomb_floquet(size={length}): exhaustive weight-2 search "
            "found no instantaneous-stabilizer logical below weight 3",
        ),
        metadata={
            "family": "honeycomb_floquet",
            "size": length,
            "static_subsystem_k": 0,
            "dynamic_logical_qubits": 2,
            "isg_rank": n - 2,
            "isg_min_logical_weight_exceeds": 2,
        },
    )


__all__ = [
    "SurfaceFace",
    "SurfaceGeometry",
    "SurfaceLattice",
    "rotated_surface",
    "rotated_surface_lattice",
    "HoneycombFloquetGeometry",
    "honeycomb_floquet",
]

from .._compat import preserve_legacy_module as _preserve_legacy_module

_preserve_legacy_module(globals(), "cudaq.logical.geometry")
