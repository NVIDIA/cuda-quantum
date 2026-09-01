# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Small catalog of concrete, generally reusable QEC codes."""

from .definition import CSSCode, code
from .distance import Distance
from .structure import CSSBlock


@code
class Repetition:
    block = CSSBlock(data=3, sx=0, sz=2)
    d = Distance.asymmetric(x=3, z=1)
    hx = ()
    hz = ((0, 1), (1, 2))
    lx = ((0, 1, 2),)
    lz = ((0,),)


@code
class Steane:
    block = CSSBlock(data=7, sx=3, sz=3)
    d = 3
    hx = ((0, 1, 2, 3), (0, 1, 4, 5), (0, 2, 4, 6))
    hz = hx
    lx = (tuple(range(7)),)
    lz = lx


@code
def rotated_surface(distance: int):
    """Return the rotated surface ``[[d**2, 1, d]]`` CSS code.

    Geometry derives the two plaquette families and logical boundary chains;
    this adapter owns only their X/Z assignment and encoded-block layout.
    """
    from ..architecture.geometry import rotated_surface_lattice

    lattice = rotated_surface_lattice(distance)
    hx = lattice.sublattice_faces(1)
    hz = lattice.sublattice_faces(0)
    return CSSCode(
        block=CSSBlock(data=lattice.n, sx=len(hx), sz=len(hz)),
        d=distance,
        hx=hx,
        hz=hz,
        lx=(lattice.logical_chains[0],),
        lz=(lattice.logical_chains[1],),
        metadata={
            "family": "rotated_surface",
            "distance": distance
        },
    )


@code
class BareQubit:
    block = CSSBlock(data=1, sx=0, sz=0)
    d = 1
    hx = ()
    hz = ()
    lx = ((0,),)
    lz = ((0,),)


@code
class ReedMuller15:
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


RM15 = ReedMuller15
Surface = rotated_surface

__all__ = [
    "BareQubit",
    "Repetition",
    "ReedMuller15",
    "RM15",
    "Steane",
    "Surface",
    "rotated_surface",
]
