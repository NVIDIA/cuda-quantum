# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Chain-complex cellulations for homological code construction.

A :class:`Cellulation` is a two-dimensional GF(2) chain complex — vertices,
edges, faces with boundary maps satisfying del1 . del2 = 0 — plus a chosen
homology basis. Qubits live on edges; faces give one check family and
vertex stars the other; the homology representatives become the logical
operators. Consume one with :meth:`cudaq.logical.CSSCode.from_chain_complex`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..analysis.evidence import Provenance


@dataclass(frozen=True, slots=True)
class Cellulation:
    """A 2D GF(2) chain complex with a chosen homology basis."""

    name: str
    num_vertices: int
    num_edges: int
    num_faces: int
    # Edge supports: one row per face (del2 columns) and per vertex star
    # (del1-transpose rows). Together they satisfy the CSS condition.
    face_boundaries: tuple[tuple[int, ...], ...]
    vertex_stars: tuple[tuple[int, ...], ...]
    homology_x: tuple[tuple[int, ...], ...]
    homology_z: tuple[tuple[int, ...], ...]
    distance_evidence: Provenance
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # del1 . del2 = 0 over GF(2): every face boundary meets every vertex
        # star in an even number of edges.
        for face in self.face_boundaries:
            face_set = set(face)
            for star in self.vertex_stars:
                if len(face_set.intersection(star)) % 2:
                    raise ValueError(
                        f"cellulation {self.name!r} violates del1.del2 = 0")


def square_torus(rows: int, columns: int) -> Cellulation:
    """Cellulate the square torus with ``rows x columns`` plaquettes.

    Edges carry qubits (two sublattices), faces give the X checks, vertex
    stars the Z checks, and the two nontrivial cycle pairs of the torus give
    the homology basis. The systole of this cellulation is
    ``min(rows, columns)``, which is the code distance.
    """
    if not isinstance(rows, int) or not isinstance(columns, int) or isinstance(
            rows, bool) or isinstance(columns, bool):
        raise TypeError("square-torus dimensions must be Python ints")
    if rows != columns:
        raise NotImplementedError(
            "this slice supports square tori (rows == columns)")
    if rows < 2:
        raise ValueError("square-torus size must be at least 2")
    size = rows
    from cudaq.logical.codes.catalog import _torus_index

    index = lambda side, i, j: _torus_index(size, side, i, j)
    faces = []
    stars = []
    for j in range(size):
        for i in range(size):
            faces.append((
                index(0, i, j),
                index(0, i, j + 1),
                index(1, i, j),
                index(1, i + 1, j),
            ))
            stars.append((
                index(0, i - 1, j),
                index(0, i, j),
                index(1, i, j - 1),
                index(1, i, j),
            ))
    homology_x = (
        tuple(index(0, i, 0) for i in range(size)),
        tuple(index(1, 0, j) for j in range(size)),
    )
    homology_z = (
        tuple(index(0, 0, j) for j in range(size)),
        tuple(index(1, i, 0) for i in range(size)),
    )
    return Cellulation(
        name=f"square_torus_{size}x{size}",
        num_vertices=size * size,
        num_edges=2 * size * size,
        num_faces=size * size,
        face_boundaries=tuple(faces),
        vertex_stars=tuple(stars),
        homology_x=homology_x,
        homology_z=homology_z,
        distance_evidence=Provenance("topology",
                                     f"square_torus systole = {size}"),
        metadata={
            "family": "toric",
            "linear_size": size
        },
    )


__all__ = ["Cellulation", "square_torus"]
