# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Reusable QLX QEC code definitions.

This module is deliberately an ordinary Python library.  Importing a code does
not register it in a process-global catalog; the selected definition becomes
part of a build only when a program, encoding, gadget, or protocol references
it.
"""

from __future__ import annotations

from typing import Iterable

from cudaq.logical.codes.bb import (
    BinaryPolynomial,
    BivariateBicycleCode,
    CyclicProduct,
)
from cudaq.logical.algebra.pauli import (
    X,
    Z,
)
from cudaq.logical.codes.structure import Block, CSSBlock
from cudaq.logical.codes.distance import Distance
from cudaq.logical.codes.definition import (
    CSSCode,
    StabilizerCode,
    SubsystemCode,
    code,
)
from .pinnacle import (
    PINNACLE_GB_INSTANCES,
    PUBLISHED_GB_SEEDS,
    PinnacleGBInstance,
    pinnacle_gb,
    pinnacle_gb_instance,
)


@code
class Repetition:
    """Three-carrier bit-flip repetition code embedded as a CSS code."""

    block = CSSBlock(data=3, sx=0, sz=2)
    d = Distance.asymmetric(x=3, z=1)
    hx = ()
    hz = ((0, 1), (1, 2))
    lx = ((0, 1, 2),)
    lz = ((0,),)


@code
class Steane:
    """The self-dual Steane ``[[7, 1, 3]]`` code."""

    block = CSSBlock(data=7, sx=3, sz=3)
    d = 3
    hx = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6))
    hz = hx
    lx = ((0, 1, 2, 3, 4, 5, 6),)
    lz = lx


@code
class BareQubit:
    """The trivial ``[[1, 1, 1]]`` code used at physical state boundaries.

    It has no checks or syndrome ancillas.  Treating a raw magic-state payload
    as this code keeps its one live qubit explicit without pretending it has
    already been encoded into the consumer's QEC code.
    """

    block = CSSBlock(data=1, sx=0, sz=0)
    d = 1
    hx = ()
    hz = ()
    lx = ((0,),)
    lz = ((0,),)


@code
class ReedMuller15:
    """The punctured quantum Reed--Muller ``[[15, 1, 3]]`` code.

    With this stabilizer/logical convention, physical ``T`` on every data
    carrier implements logical ``T†``. Consequently logical ``T`` is realized
    by carrierwise ``T†``. Keeping that direction explicit prevents examples
    and target lowerings from treating an arbitrary transversal phase as T.
    """

    block = CSSBlock(data=15, sx=4, sz=10)
    d = 3
    hx = (
        (0, 2, 4, 6, 8, 10, 12, 14),
        (1, 2, 5, 6, 9, 10, 13, 14),
        (3, 4, 5, 6, 11, 12, 13, 14),
        (7, 8, 9, 10, 11, 12, 13, 14),
    )
    hz = (
        (1, 2, 3, 4),
        (0, 2, 3, 5),
        (0, 1, 3, 6),
        (1, 2, 7, 8),
        (0, 2, 7, 9),
        (0, 1, 7, 10),
        (0, 1, 2, 3, 7, 11),
        (0, 3, 7, 12),
        (1, 3, 7, 13),
        (2, 3, 7, 14),
    )
    lx = (tuple(range(15)),)
    lz = ((0, 1, 2),)


@code
def rotated_surface(distance: int):
    """Return the rotated surface ``[[d**2, 1, d]]`` code.

    The CSS assignment over the shared lattice
    (:func:`cudaq.logical.architecture.geometry.
    rotated_surface_lattice`): one plaquette family is measured in X, the other
    in Z, and the two straight boundary chains are the X and Z logical
    representatives.
    """
    from ..architecture.geometry import rotated_surface_lattice

    lattice = rotated_surface_lattice(distance)
    hx = lattice.sublattice_faces(1)
    hz = lattice.sublattice_faces(0)
    lx = (lattice.logical_chains[0],)
    lz = (lattice.logical_chains[1],)
    return CSSCode(
        block=CSSBlock(data=lattice.n, sx=len(hx), sz=len(hz)),
        d=distance,
        hx=hx,
        hz=hz,
        lx=lx,
        lz=lz,
        metadata={
            "family": "rotated_surface",
            "distance": distance
        },
    )


@code
def zxxz_surface(distance: int):
    """Return the ZXXZ rotated surface ``[[d**2, 1, d]]`` code (non-CSS).

    Same lattice as :func:`rotated_surface`
    (:func:`cudaq.logical.architecture.geometry.
    rotated_surface_lattice`), but with a Hadamard applied to the checkerboard
    data sublattice named by the lattice's ``bipartition``.  Every stabilizer is
    therefore a *mixed* Pauli product: an X plaquette carries ``X`` on its
    off-checkerboard carriers and ``Z`` on its checkerboard carriers, and a Z
    plaquette the dual pattern -- so the algebra is stabilizer, not CSS.  The
    code is locally Clifford equivalent to the CSS rotated surface code and
    keeps its ``[[d**2, 1, d]]`` parameters. It is the XZZX-family construction
    of Bonilla Ataides et al. (arXiv:2009.07851), expressed in the repository's
    ZXXZ orientation convention.
    """
    from ..architecture.geometry import rotated_surface_lattice

    lattice = rotated_surface_lattice(distance)
    checker = frozenset(
        index for index, side in enumerate(lattice.bipartition) if side)

    def operator(support, base_is_x):
        product = None
        for carrier in support:
            is_x = base_is_x ^ (carrier in checker)
            factor = (X if is_x else Z)(carrier)
            product = factor if product is None else product @ factor
        return product

    stabilizers = tuple(
        operator(face.support, face.sublattice == 1) for face in lattice.faces)
    logicals = ((
        operator(lattice.logical_chains[0], base_is_x=True),
        operator(lattice.logical_chains[1], base_is_x=False),
    ),)
    # The ancillas live in the ``sx`` carrier partition. That name is pure
    # carrier layout, not a CSS-basis claim: each ancilla measures one mixed
    # ZXXZ stabilizer, not an X check. ``sx`` (rather than a bespoke name) is
    # required because Fabric gate/measure ops only address the native
    # partition enum ``{data, sx, sz, all}``.
    return StabilizerCode(
        block=Block(data=lattice.n, sx=len(stabilizers)),
        d=distance,
        stabilizers=stabilizers,
        logicals=logicals,
        metadata={
            "family": "zxxz_surface",
            "distance": distance
        },
    )


def _torus_index(linear_size: int, sublattice: int, i: int, j: int) -> int:
    offset = sublattice * linear_size * linear_size
    return offset + (i % linear_size) + linear_size * (j % linear_size)


@code
def toric(linear_size: int):
    """Return the square-lattice toric ``[[2L**2, 2, L]]`` code."""

    if not isinstance(linear_size, int) or isinstance(linear_size, bool):
        raise TypeError("toric linear size must be a Python int")
    if linear_size < 2:
        raise ValueError("toric linear size must be at least 2")
    index = lambda side, i, j: _torus_index(linear_size, side, i, j)
    hx = []
    hz = []
    for j in range(linear_size):
        for i in range(linear_size):
            hx.append((
                index(0, i, j),
                index(0, i, j + 1),
                index(1, i, j),
                index(1, i + 1, j),
            ))
            hz.append((
                index(0, i - 1, j),
                index(0, i, j),
                index(1, i, j - 1),
                index(1, i, j),
            ))
    lx = (
        tuple(index(0, i, 0) for i in range(linear_size)),
        tuple(index(1, 0, j) for j in range(linear_size)),
    )
    lz = (
        tuple(index(0, 0, j) for j in range(linear_size)),
        tuple(index(1, i, 0) for i in range(linear_size)),
    )
    return CSSCode(
        block=CSSBlock(
            data=2 * linear_size * linear_size,
            sx=linear_size * linear_size,
            sz=linear_size * linear_size,
        ),
        d=linear_size,
        hx=hx,
        hz=hz,
        lx=lx,
        lz=lz,
        metadata={
            "family": "toric",
            "linear_size": linear_size
        },
    )


def _gf2_rref(rows, ncols):
    matrix = [list(row) for row in rows]
    pivots = []
    row = 0
    for column in range(ncols):
        pivot = next(
            (candidate for candidate in range(row, len(matrix))
             if matrix[candidate][column]),
            None,
        )
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        for candidate in range(len(matrix)):
            if candidate != row and matrix[candidate][column]:
                matrix[candidate] = [
                    left ^ right
                    for left, right in zip(matrix[candidate], matrix[row])
                ]
        pivots.append(column)
        row += 1
        if row == len(matrix):
            break
    return matrix, pivots


def _gf2_rank(rows, ncols):
    return len(_gf2_rref(rows, ncols)[1])


def _gf2_null_space(rows, ncols):
    matrix, pivots = _gf2_rref(rows, ncols)
    free = [column for column in range(ncols) if column not in set(pivots)]
    basis = []
    for free_column in free:
        vector = [0] * ncols
        vector[free_column] = 1
        for row, pivot in enumerate(pivots):
            if matrix[row][free_column]:
                vector[pivot] = 1
        basis.append(vector)
    return basis


def _circulant_rows(l: int, m: int, monomials: Iterable[tuple[int, int]]):
    monomials = tuple(monomials)
    result = []
    for j in range(m):
        for i in range(l):
            row = [0] * (l * m)
            for x_power, y_power in monomials:
                column = (i + x_power) % l + l * ((j + y_power) % m)
                row[column] ^= 1
            result.append(row)
    return result


def _quotient_basis(kernel, stabilizers, ncols):

    def encode(row):
        value = 0
        for column in range(ncols):
            if row[column]:
                value |= 1 << column
        return value

    pivots = {}

    def add_to_span(row):
        value = encode(row)
        while value:
            pivot = value.bit_length() - 1
            existing = pivots.get(pivot)
            if existing is None:
                pivots[pivot] = value
                return True
            value ^= existing
        return False

    basis = []
    for row in stabilizers:
        add_to_span(row)
    for vector in kernel:
        if add_to_span(vector):
            basis.append(vector)
    return basis


def _symplectic_pair_css(lx_rows, lz_rows):
    """Canonically pair CSS logical representatives.

    Homology quotient bases span the right classes but are not paired;
    the code constructor requires ``<lx[i], lz[j]> = delta(i, j)``. Row
    additions stay within each side's coset space (kernel modulo checks),
    so the paired bases represent the same logical algebra.
    """
    lx = [list(row) for row in lx_rows]
    lz = [list(row) for row in lz_rows]
    if len(lx) != len(lz):
        raise ValueError("CSS logical bases must have equal rank")

    def odd_overlap(a, b):
        return sum(x & z for x, z in zip(a, b)) % 2

    for i in range(len(lx)):
        partner = next(
            (j for j in range(i, len(lz)) if odd_overlap(lx[i], lz[j])),
            None,
        )
        if partner is None:
            raise ValueError(
                "CSS logical bases are degenerate: no anticommuting partner")
        lz[i], lz[partner] = lz[partner], lz[i]
        for m in range(len(lx)):
            if m != i and odd_overlap(lx[m], lz[i]):
                lx[m] = [a ^ b for a, b in zip(lx[m], lx[i])]
        for m in range(len(lz)):
            if m != i and odd_overlap(lx[i], lz[m]):
                lz[m] = [a ^ b for a, b in zip(lz[m], lz[i])]
    return lx, lz


def bivariate_bicycle(
    l: int,
    m: int,
    a: Iterable[tuple[int, int]],
    b: Iterable[tuple[int, int]],
    *,
    d=None,
    name: str | None = None,
):
    """Compatibility constructor backed by the typed BB code artifact."""

    return BivariateBicycleCode.from_polynomials(
        group=CyclicProduct(l, m),
        a=BinaryPolynomial(tuple(a)),
        b=BinaryPolynomial(tuple(b)),
        d=d,
        name=name,
    )


BB3x3 = bivariate_bicycle(
    3,
    3,
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1)),
    d=3,
    name="BB3x3",
)

_tesseract_checks = (
    (0, 1, 2, 3, 4, 5, 6, 7),
    (0, 1, 2, 3, 8, 9, 10, 11),
    (0, 1, 4, 5, 8, 9, 12, 13),
    (0, 2, 4, 6, 8, 10, 12, 14),
    (8, 9, 10, 11, 12, 13, 14, 15),
)

Tesseract = SubsystemCode(
    name="Tesseract",
    block=CSSBlock(data=16, sx=5, sz=5),
    k=4,
    r=2,
    d=4,
    hx=_tesseract_checks,
    hz=_tesseract_checks,
    # Protected Path-4 order is (old L1, L0, L2, L5).  The remaining old
    # L3/L4 pair is explicit gauge workspace.  In this basis the paper's
    # column-swap automorphism preserves the gauge subsystem and implements
    # CNOT(path0 -> path1) * CNOT(path2 -> path3).
    lx=(
        (4, 5, 12, 13),
        (0, 2, 8, 10),
        (8, 9, 12, 13),
        (0, 2, 4, 6),
    ),
    lz=(
        (0, 2, 4, 6),
        (0, 1, 4, 5),
        (1, 3, 9, 11),
        (0, 1, 8, 9),
    ),
    gx=((8, 9, 10, 11), (0, 4, 8, 12)),
    gz=((1, 5, 9, 13), (0, 1, 2, 3)),
    metadata={
        "family": "tesseract",
        "distance_scope": "dressed",
        "logical_order": "path4:L1,L0,L2,L5;gauge:L3,L4",
    },
)


# Triangular 6.6.6 color codes. Qubits sit on the vertices of a trivalent,
# three-face-colorable lattice restricted to a triangular patch; every face
# carries one X and one Z stabilizer, so the codes are self-dual (hx == hz) and
# admit transversal Clifford gates. The distance-d code holds
# n = (3*d**2 + 1) / 4 data qubits. Color codes are the natural home for
# magic-state cultivation, and their non-graphlike weight-6 checks are exactly
# where a heralded-erasure advantage is most valuable: a located erasure is
# decoded against distance d-1 rather than the (d-1)//2 of an unlocated Pauli.
#
def _triangular_color_666_lattice(distance: int):
    """Construct the standard triangular honeycomb patch in row coordinates."""

    bound = 3 * (distance - 1) // 2

    # In this triangular coordinate system, (row, column) with
    # 0 <= column <= row <= bound is either a data vertex or a face center.
    # Face centers satisfy column mod 3 == 2 - row mod 3. Every other point is
    # a data vertex. This construction yields (3*d**2 + 1)/4 data vertices.
    def is_face(row: int, column: int) -> bool:
        return column % 3 == 2 - row % 3

    data_coordinates = tuple((row, column)
                             for row in range(bound + 1)
                             for column in range(row + 1)
                             if not is_face(row, column))
    positions = {
        coordinate: index for index, coordinate in enumerate(data_coordinates)
    }
    face_coordinates = tuple((row, column)
                             for row in range(bound + 1)
                             for column in range(row + 1)
                             if is_face(row, column))

    # A bulk hexagon touches these six coordinate offsets. At a triangular
    # boundary, the out-of-patch vertices are omitted, producing weight-four
    # boundary checks and weight-six bulk checks.
    checks = []
    for row, column in face_coordinates:
        neighbors = (
            (row - 1, column - 1),
            (row - 1, column),
            (row, column - 1),
            (row, column + 1),
            (row + 1, column),
            (row + 1, column + 1),
        )
        checks.append(
            tuple(positions[neighbor]
                  for neighbor in neighbors
                  if neighbor in positions))

    # One side of the triangle is a minimum logical string. It has exactly d
    # data vertices and works for both X and Z because the code is self-dual.
    boundary = tuple(positions[(row, 0)]
                     for row in range(bound + 1)
                     if (row, 0) in positions)
    return data_coordinates, face_coordinates, tuple(checks), boundary


@code
def triangular_color(distance: int):
    """Return the triangular 6.6.6 color code ``[[(3d**2+1)/4, 1, d]]``."""

    if not isinstance(distance, int) or isinstance(distance, bool):
        raise TypeError("triangular-color distance must be a Python int")
    if distance < 3 or distance % 2 == 0:
        raise ValueError(
            "triangular-color distance must be an odd Python int at least 3")
    data_coordinates, face_coordinates, checks, boundary = (
        _triangular_color_666_lattice(distance))
    n = len(data_coordinates)
    return CSSCode(
        name=f"triangular_color_{distance}",
        # Standard circuit extraction needs one ancilla for every X and Z
        # face.  Keeping these partitions explicit is what makes the public
        # code usable below the algebra-only/code-capacity level.
        block=CSSBlock(data=n, sx=len(checks), sz=len(checks)),
        n=n,
        k=1,
        d=distance,
        # Self-dual: the same faces carry the X and Z stabilizers.
        hx=checks,
        hz=checks,
        lx=(boundary,),
        lz=(boundary,),
        metadata={
            "family": "triangular_color",
            "lattice": "6.6.6",
            "distance": distance,
            "data_coordinates": data_coordinates,
            "face_coordinates": face_coordinates,
        },
    )


# Familiar family spellings without introducing a registry abstraction.
Surface = rotated_surface
Toric = toric
RM15 = ReedMuller15
TriangularColor = triangular_color

__all__ = [
    "BB3x3",
    "BareQubit",
    "Repetition",
    "ReedMuller15",
    "RM15",
    "Steane",
    "Surface",
    "Tesseract",
    "Toric",
    "TriangularColor",
    "bivariate_bicycle",
    "PINNACLE_GB_INSTANCES",
    "PUBLISHED_GB_SEEDS",
    "PinnacleGBInstance",
    "pinnacle_gb",
    "pinnacle_gb_instance",
    "rotated_surface",
    "toric",
    "triangular_color",
    "zxxz_surface",
]
