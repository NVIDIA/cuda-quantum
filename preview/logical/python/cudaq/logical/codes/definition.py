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

from .selection import (
    QECActionSelection,
    QECBlockBinding,
    QECBlockOwner,
    QECBlockRequest,
    QECSelectionWitness,
    _canonical_label_index,
    _deep_freeze,
    qec_block,
)
from .structure import Block, CSSBlock, CarrierRoleMap, PatchTransform
from .distance import (
    Distance,
    DistanceScope,
    Schedule,
    _coordinates_in_basis_many,
    _declared_pauli_row,
    _derive_anti_stabilizers,
    _independent_row_indices,
    _independent_rows,
    _in_span,
    _rows,
    _row_relations,
    _solve_gf2,
    _support_bits,
    _support_rows,
    _symplectic_functional,
    _symplectic_product,
    _symplectic_product_bits,
    _validate_symplectic_closure,
    _xor_rows,
)
from .profiles import (
    CodeProfile,
    GaugeMeasurementMap,
    MeasurementPhase,
    MetaChecks,
    RecordLogicalMap,
)
from .encodings import (
    Concatenated,
    Encoding,
    EncodingEpoch,
    EncodingEpochSchema,
    EncodingHierarchy,
    EncodingProjection,
    FixedPort,
    expose,
    fix,
    gauge,
)


def _distance(value) -> Distance:
    if isinstance(value, Distance):
        return value
    if value is None:
        return Distance.unknown("not established")
    return Distance.claim(value)


class Code(ImmutableValue):
    """General stabilizer/subsystem code algebra definition."""

    __slots__ = (
        "name",
        "n",
        "k",
        "r",
        "d",
        "block",
        "stabilizers",
        "gauges",
        "hx",
        "hz",
        "gx",
        "gz",
        "lx",
        "lz",
        "declared_stabilizers",
        "stabilizer_labels",
        "stabilizer_indices",
        "logical_pairs",
        "gauge_pairs",
        "stabilizer_basis",
        "logical_x_basis",
        "logical_z_basis",
        "gauge_x_basis",
        "gauge_z_basis",
        "anti_stabilizers",
        "encoding_clifford",
        "metadata",
        "default_profile",
        "default_encoding",
    )

    def __init__(
        self,
        *,
        name: str | None = None,
        n: int | None = None,
        k: int | None = None,
        r: int | None = None,
        d: Distance | int | None = None,
        distance: Distance | int | None = None,
        block: Block | None = None,
        stabilizers=(),
        stabilizer_labels=None,
        gauges=(),
        logicals=None,
        gauge_pairs=None,
        anti_stabilizers=(),
        hx=None,
        hz=None,
        gx=None,
        gz=None,
        lx=None,
        lz=None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name or "anonymous_code"
        self.block = block or Block(data=n or 0)
        inferred_n = self.block.partitions.get("data", self.block.size)
        self.n = inferred_n if n is None else n
        self.lx = _support_rows(lx, basis="x", what="lx")
        self.lz = _support_rows(lz, basis="z", what="lz")
        if logicals is not None and (self.lx or self.lz):
            raise TypeError("specify logicals= or lx=/lz=, not both")
        logicals = tuple(logicals or ())
        inferred_k = len(logicals) if logicals else max(len(self.lx),
                                                        len(self.lz), 0)
        self.k = inferred_k if k is None else k
        if gauge_pairs is not None and (gx or gz):
            raise TypeError("specify gauge_pairs= or gx=/gz=, not both")
        gauge_pairs = tuple(gauge_pairs or ())
        self.gx = _support_rows(gx, basis="x", what="gx")
        self.gz = _support_rows(gz, basis="z", what="gz")
        inferred_r = len(gauge_pairs) or max(len(self.gx), len(self.gz), 0)
        self.r = inferred_r if r is None else r
        if self.n < 1 or self.k < 0 or self.r < 0 or self.k + self.r > self.n:
            raise ValueError("Code requires n>=1, k>=0, r>=0, and k+r<=n")
        if d is not None and distance is not None:
            raise TypeError("specify d= or distance=, not both")
        self.d = _distance(d if d is not None else distance)
        if self.d.status == "circuit" or self.d.scope == "circuit":
            raise ValueError(
                "circuit distance belongs to a CodeProfile, not Code algebra")
        self.stabilizers = tuple(stabilizers)
        self.gauges = tuple(gauges)
        self.hx = _support_rows(hx, basis="x", what="hx")
        self.hz = _support_rows(hz, basis="z", what="hz")

        width = 2 * self.n
        zero = (0,) * self.n
        declared_stabilizers = (
            *(_support_bits(row, self.n) + zero for row in self.hx),
            *(zero + _support_bits(row, self.n) for row in self.hz),
            *(_declared_pauli_row(value, self.n) for value in self.stabilizers),
        )
        if any(not any(row) for row in declared_stabilizers):
            raise ValueError("stabilizer generators must be nonidentity")
        declared_stabilizer_bits = tuple(
            _row_bits(row) for row in declared_stabilizers)
        symplectic_mask = (1 << self.n) - 1
        for left in range(len(declared_stabilizers)):
            for right in range(left + 1, len(declared_stabilizers)):
                if _symplectic_product_bits(
                        declared_stabilizer_bits[left],
                        declared_stabilizer_bits[right],
                        self.n,
                        mask=symplectic_mask,
                ):
                    raise ValueError(
                        "stabilizer generators must mutually commute")
        self.declared_stabilizers = GF2Matrix._from_normalized_rows(
            declared_stabilizers, ncols=width)
        self.stabilizer_labels, self.stabilizer_indices = _canonical_label_index(
            range(len(declared_stabilizers))
            if stabilizer_labels is None else stabilizer_labels,
            len(declared_stabilizers),
            what="stabilizer_labels",
        )
        kept = _independent_rows(declared_stabilizers,
                                 ncols=width,
                                 _normalized=True)
        expected_stabilizers = self.n - self.k - self.r
        if len(kept) != expected_stabilizers:
            raise ValueError("stabilizer rank must equal n-k-r = "
                             f"{expected_stabilizers}, got {len(kept)}")
        self.stabilizer_basis = GF2Matrix._from_normalized_rows(kept,
                                                                ncols=width)

        if logicals:
            if len(logicals) != self.k or any(
                    len(pair) != 2 for pair in logicals):
                raise ValueError("logicals must contain exactly k Pauli pairs")
            logical_x = tuple(
                _declared_pauli_row(pair[0], self.n) for pair in logicals)
            logical_z = tuple(
                _declared_pauli_row(pair[1], self.n) for pair in logicals)
            # Preserve the CSS convenience representatives only when the
            # general symplectic declaration really is pure X / pure Z.
            if all(not any(row[self.n:]) for row in logical_x) and all(
                    not any(row[:self.n]) for row in logical_z):
                self.lx = tuple(
                    tuple(index
                          for index, bit in enumerate(row[:self.n])
                          if bit)
                    for row in logical_x)
                self.lz = tuple(
                    tuple(index
                          for index, bit in enumerate(row[self.n:])
                          if bit)
                    for row in logical_z)
            else:
                self.lx = self.lz = ()
        else:
            if len(self.lx) != self.k or len(self.lz) != self.k:
                raise ValueError(
                    "lx and lz must each contain exactly k logical operators")
            logical_x = tuple(
                _support_bits(row, self.n) + zero for row in self.lx)
            logical_z = tuple(
                zero + _support_bits(row, self.n) for row in self.lz)

        if gauge_pairs:
            if len(gauge_pairs) != self.r or any(
                    len(pair) != 2 for pair in gauge_pairs):
                raise ValueError(
                    "gauge_pairs must contain exactly r Pauli pairs")
            gauge_x = tuple(
                _declared_pauli_row(pair[0], self.n) for pair in gauge_pairs)
            gauge_z = tuple(
                _declared_pauli_row(pair[1], self.n) for pair in gauge_pairs)
            if all(not any(row[self.n:]) for row in gauge_x) and all(
                    not any(row[:self.n]) for row in gauge_z):
                self.gx = tuple(
                    tuple(index
                          for index, bit in enumerate(row[:self.n])
                          if bit)
                    for row in gauge_x)
                self.gz = tuple(
                    tuple(index
                          for index, bit in enumerate(row[self.n:])
                          if bit)
                    for row in gauge_z)
            else:
                self.gx = self.gz = ()
        else:
            if len(self.gx) != self.r or len(self.gz) != self.r:
                raise ValueError(
                    "gx and gz must each contain exactly r gauge operators")
            gauge_x = tuple(
                _support_bits(row, self.n) + zero for row in self.gx)
            gauge_z = tuple(
                zero + _support_bits(row, self.n) for row in self.gz)
        self.logical_pairs = tuple(zip(logical_x, logical_z))
        self.gauge_pairs = tuple(zip(gauge_x, gauge_z))
        for label, rows in (
            ("logical X", logical_x),
            ("logical Z", logical_z),
            ("gauge X", gauge_x),
            ("gauge Z", gauge_z),
        ):
            if any(not any(row) for row in rows):
                raise ValueError(f"{label} operators must be nonidentity")

        encoded_families = []

        def encode_family(rows):
            for family, encoded in encoded_families:
                if rows is family:
                    return encoded
            encoded = tuple(_row_bits(row) for row in rows)
            encoded_families.append((rows, encoded))
            return encoded

        def require_commutation(left_rows, right_rows, label, *, paired=False):
            left_encoded = encode_family(left_rows)
            right_encoded = encode_family(right_rows)
            for left_index, left in enumerate(left_encoded):
                for right_index, right in enumerate(right_encoded):
                    expected = int(paired and left_index == right_index)
                    actual = _symplectic_product_bits(
                        left,
                        right,
                        self.n,
                        mask=symplectic_mask,
                    )
                    if actual != expected:
                        relation = "canonical pairs" if paired else "commuting families"
                        raise ValueError(f"{label} must form {relation}")

        require_commutation(kept, kept, "stabilizers")
        require_commutation(kept, logical_x, "stabilizer/logical-X")
        require_commutation(kept, logical_z, "stabilizer/logical-Z")
        require_commutation(kept, gauge_x, "stabilizer/gauge-X")
        require_commutation(kept, gauge_z, "stabilizer/gauge-Z")
        require_commutation(logical_x, logical_x, "logical X")
        require_commutation(logical_z, logical_z, "logical Z")
        require_commutation(logical_x, logical_z, "logical X/Z", paired=True)
        require_commutation(gauge_x, gauge_x, "gauge X")
        require_commutation(gauge_z, gauge_z, "gauge Z")
        require_commutation(gauge_x, gauge_z, "gauge X/Z", paired=True)
        require_commutation(logical_x, gauge_x, "logical/gauge X")
        require_commutation(logical_x, gauge_z, "logical-X/gauge-Z")
        require_commutation(logical_z, gauge_x, "logical-Z/gauge-X")
        require_commutation(logical_z, gauge_z, "logical/gauge Z")

        self.logical_x_basis = GF2Matrix._from_normalized_rows(logical_x,
                                                               ncols=width)
        self.logical_z_basis = GF2Matrix._from_normalized_rows(logical_z,
                                                               ncols=width)
        self.gauge_x_basis = GF2Matrix._from_normalized_rows(gauge_x,
                                                             ncols=width)
        self.gauge_z_basis = GF2Matrix._from_normalized_rows(gauge_z,
                                                             ncols=width)

        gauge_group = (*kept, *gauge_x, *gauge_z)
        for value in self.gauges:
            row = _declared_pauli_row(value, self.n)
            if not any(row) or not _in_span(row, gauge_group):
                raise ValueError(
                    "supplementary gauge generators must be nonidentity members "
                    "of the declared gauge group")

        if anti_stabilizers:
            anti = tuple(
                _declared_pauli_row(value, self.n)
                for value in anti_stabilizers)
            if len(anti) != expected_stabilizers:
                raise ValueError(
                    "anti_stabilizers must contain one entry per kept stabilizer"
                )
        else:
            kept, anti = _derive_anti_stabilizers(
                kept,
                (
                    *zip(logical_x, logical_z),
                    *zip(gauge_x, gauge_z),
                ),
                n=self.n,
            )
            self.stabilizer_basis = GF2Matrix._from_normalized_rows(kept,
                                                                    ncols=width)
        require_commutation(kept,
                            anti,
                            "stabilizer/anti-stabilizer",
                            paired=True)
        require_commutation(anti, anti, "anti-stabilizers")
        require_commutation(anti, logical_x, "anti-stabilizer/logical-X")
        require_commutation(anti, logical_z, "anti-stabilizer/logical-Z")
        require_commutation(anti, gauge_x, "anti-stabilizer/gauge-X")
        require_commutation(anti, gauge_z, "anti-stabilizer/gauge-Z")
        self.anti_stabilizers = GF2Matrix._from_normalized_rows(anti,
                                                                ncols=width)

        encoding_rows = (
            *anti,
            *logical_x,
            *gauge_x,
            *kept,
            *logical_z,
            *gauge_z,
        )
        encoding_clifford = GF2Matrix._from_normalized_rows(encoding_rows,
                                                            ncols=width)
        if encoding_clifford.nrows != width or encoding_clifford.rank != width:
            raise ValueError(
                "code basis does not span the full symplectic space")
        self.encoding_clifford = encoding_clifford
        self.metadata = _deep_freeze(dict(metadata or {}), what="code metadata")
        self.default_profile = CodeProfile(self)
        self.default_encoding = Encoding(self, profile=self.default_profile)
        self._seal()

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)

    def encoding(self, **kwargs) -> Encoding:
        return Encoding(self, **kwargs)

    @property
    def num_x_checks(self) -> int:
        return len(self.hx)

    @property
    def num_z_checks(self) -> int:
        return len(self.hz)

    @property
    def gauge_generators(self) -> tuple:
        """Declared gauge representatives (X-type rows, then Z-type rows)."""
        if self.gauges:
            return tuple(self.gauges)
        return tuple(self.gx) + tuple(self.gz)

    def with_metachecks(self,
                        metachecks: MetaChecks,
                        *,
                        name: str | None = None) -> CodeProfile:
        return CodeProfile(
            self,
            name=name or f"{self.name}_metacheck_profile",
            metachecks=metachecks,
        )


class SubsystemCode(Code):
    __slots__ = ()


class StabilizerCode(SubsystemCode):

    __slots__ = ()

    def __init__(self, **kwargs) -> None:
        if kwargs.get("r", 0) != 0:
            raise ValueError(
                "StabilizerCode is the r=0 subsystem-code specialization")
        kwargs["r"] = 0
        super().__init__(**kwargs)


def _reorder_support(support, order):
    """Rank a check's data neighbors by ``order`` (bulk-degree checks only)."""
    if order is not None and len(support) == len(order):
        return tuple(support[position] for position in order)
    return tuple(support)


def _exact_edge_color(checks, order):
    """Optimally color a bipartite check/data incidence graph.

    Edges are inserted neighbor-rank by neighbor-rank. When an edge's
    endpoints have no common free color, swapping two colors along an
    alternating path frees one. Bipartite parity prevents that path from
    reaching the edge's data endpoint, so all incidences use exactly ``delta``
    colors. By the König line-coloring theorem, ``delta`` is minimum depth.
    """
    ordered_supports = tuple(
        _reorder_support(support, order) for support in checks)
    max_check_degree = max(map(len, ordered_supports), default=0)
    data_degrees = {}
    for support in ordered_supports:
        for data in support:
            data_degrees[data] = data_degrees.get(data, 0) + 1
    delta = max(max_check_degree, max(data_degrees.values(), default=0))
    if not delta:
        return ()

    incidences = []
    for rank in range(max_check_degree):
        for check, support in enumerate(ordered_supports):
            if rank < len(support):
                incidences.append((check, support[rank]))

    check_colors = [dict() for _ in ordered_supports]
    data_colors = {data: {} for data in data_degrees}
    edge_colors = []

    def assign(edge, color):
        check, data = incidences[edge]
        check_colors[check][color] = edge
        data_colors[data][color] = edge
        edge_colors[edge] = color

    for edge, (check, data) in enumerate(incidences):
        edge_colors.append(-1)
        common = next((color for color in range(delta)
                       if color not in check_colors[check] and
                       color not in data_colors[data]), None)
        if common is not None:
            assign(edge, common)
            continue

        check_free = next(
            color for color in range(delta) if color not in check_colors[check])
        data_free = next(
            color for color in range(delta) if color not in data_colors[data])
        path = []
        left_side = True
        vertex = check
        color = data_free
        while True:
            color_map = (check_colors[vertex]
                         if left_side else data_colors[vertex])
            path_edge = color_map.get(color)
            if path_edge is None:
                break
            path.append(path_edge)
            path_check, path_data = incidences[path_edge]
            left_side = not left_side
            vertex = path_data if not left_side else path_check
            color = check_free if color == data_free else data_free

        previous_colors = tuple(edge_colors[path_edge] for path_edge in path)
        for path_edge, previous in zip(path, previous_colors):
            path_check, path_data = incidences[path_edge]
            del check_colors[path_check][previous]
            del data_colors[path_data][previous]
        for path_edge, previous in zip(path, previous_colors):
            assign(path_edge,
                   check_free if previous == data_free else data_free)
        assign(edge, data_free)

    layers = [[] for _ in range(delta)]
    for incidence, color in zip(incidences, edge_colors):
        layers[color].append(incidence)
    return tuple(tuple(layer) for layer in layers)


class CSSCode(StabilizerCode):

    __slots__ = ()

    def colored_schedule(self, *, x_order=None, z_order=None):
        """A per-time-step layering of this code's syndrome-extraction CX gates.

        Returns plain data -- ``(x_layers, z_layers)``, where each basis's
        layers partition its ``(check, data)`` incidences into matchings (every
        incidence once, no ancilla or data qubit twice in a layer). The *order*
        of the layers is the schedule: the same stabilizers laid out two ways
        can have two circuit distances (a mid-round ancilla fault spreads to the
        data qubits it has yet to touch). ``x_order`` / ``z_order`` permute each
        bulk check's neighbor ranking before deterministic exact bipartite
        edge-coloring. Each basis uses the minimum possible number of layers:
        the maximum degree among its check and data vertices.

        Pass the result straight to
        ``cudaq.logical.extract_syndrome(..., cx_schedule=...)``; it is not a
        type, just
        the schedule input that lays the gadget's CXs into ``fabric.tick``
        moments.
        """
        return (_exact_edge_color(self.hx,
                                  x_order), _exact_edge_color(self.hz, z_order))

    @classmethod
    def from_geometry(cls, geometry, *, d=None, name: str | None = None):
        """Construct a CSS code from a derived lattice geometry.

        A Pauli-free
        :class:`cudaq.logical.architecture.geometry.SurfaceLattice` supplies
        two geometric
        face sublattices and two logical chains; this CSS adapter owns the
        conventional X/Z assignment. Legacy geometry objects with explicit
        ``hx/hz/lx/lz`` fields remain accepted. Distance evidence stays an
        explicit argument, never inferred by this adapter.
        """
        if hasattr(geometry, "sublattice_faces") and hasattr(
                geometry, "logical_chains"):
            hx = geometry.sublattice_faces(1)
            hz = geometry.sublattice_faces(0)
            lx = (tuple(geometry.logical_chains[0]),)
            lz = (tuple(geometry.logical_chains[1]),)
        else:
            hx, hz = geometry.hx, geometry.hz
            lx, lz = geometry.lx, geometry.lz
        block = Block(data=geometry.n, sx=len(hx), sz=len(hz))
        metadata = dict(getattr(geometry, "metadata", {}) or {})
        return cls(
            name=name or getattr(geometry, "name", None),
            block=block,
            d=d,
            hx=hx,
            hz=hz,
            lx=lx,
            lz=lz,
            metadata=metadata,
        )

    @classmethod
    def from_chain_complex(cls,
                           cellulation,
                           *,
                           d=None,
                           name: str | None = None):
        """Construct a homological CSS code from a 2D cellulation.

        Qubits live on edges, faces give the X checks, vertex stars the Z
        checks, and the cellulation's homology basis becomes the logical
        representatives (see :mod:`cudaq.logical.architecture.topology`).
        """
        n = cellulation.num_edges
        hx = cellulation.face_boundaries
        hz = cellulation.vertex_stars
        block = Block(data=n, sx=len(hx), sz=len(hz))
        metadata = dict(getattr(cellulation, "metadata", {}) or {})
        return cls(
            name=name or getattr(cellulation, "name", None),
            block=block,
            d=d,
            hx=hx,
            hz=hz,
            lx=cellulation.homology_x,
            lz=cellulation.homology_z,
            metadata=metadata,
        )


def _canonical_code_metadata_value(value, *, field: str):
    """Return one deterministic scalar accepted by ``fabric.code`` metadata.

    Code metadata is part of declaration equivalence and QEC-lowering manifest
    identity.  Falling back to ``str(object)`` would admit process-local object
    addresses into both MLIR and provenance hashes, so unsupported leaves fail
    closed here before either representation is created.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError(f"{field} must be finite canonical data")
        return value
    if isinstance(value, str):
        return value

    def tree(item):
        if item is None or isinstance(item, (str, bool, int)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise TypeError(f"{field} must be finite canonical data")
            return {"kind": "float", "value": repr(item)}
        if isinstance(item, Mapping):
            entries = []
            for key, child in item.items():
                if not isinstance(key, str) or not key:
                    raise TypeError(
                        f"{field} nested mapping keys must be nonempty strings")
                entries.append((key, tree(child)))
            return {"kind": "mapping", "entries": sorted(entries)}
        if isinstance(item, tuple):
            return {"kind": "tuple", "items": [tree(child) for child in item]}
        if isinstance(item, frozenset):
            items = [tree(child) for child in item]
            items.sort(key=lambda child: json.dumps(
                child,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ))
            return {"kind": "set", "items": items}
        raise TypeError(
            f"{field} must contain only canonical scalar/container data")

    return json.dumps(
        tree(value),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _materialized_code_metadata(code: "Code") -> Mapping[str, Any]:
    """Canonical metadata shared by code identity and MLIR materialization."""

    def evidence_text(value, *, field: str):
        if isinstance(value, str):
            return value
        from ..analysis.evidence import Provenance

        if isinstance(value, Provenance):
            return str(value)
        return _canonical_code_metadata_value(value, field=field)

    def distance_metadata(evidence, prefix="distance"):
        entries = {f"{prefix}_status": evidence.status}
        if evidence.reason:
            entries[f"{prefix}_reason"] = evidence.reason
        if evidence.method:
            entries[f"{prefix}_method"] = evidence.method
        if evidence.provenance is not None:
            entries[f"{prefix}_provenance"] = evidence_text(
                evidence.provenance,
                field=f"{prefix} provenance",
            )
        if evidence.certificate is not None:
            entries[f"{prefix}_certificate"] = evidence_text(
                evidence.certificate,
                field=f"{prefix} certificate",
            )
        if evidence.scope:
            entries[f"{prefix}_scope"] = evidence.scope
        if evidence.status == "asymmetric":
            entries.update(distance_metadata(evidence.x, f"{prefix}_x"))
            entries.update(distance_metadata(evidence.z, f"{prefix}_z"))
            if evidence.x.value is not None:
                entries[f"{prefix}_x_value"] = evidence.x.value
            if evidence.z.value is not None:
                entries[f"{prefix}_z_value"] = evidence.z.value
        return entries

    metadata = {}
    for key, value in code.metadata.items():
        if not isinstance(key, str) or not key:
            raise TypeError(
                "fabric.code metadata keys must be nonempty strings")
        metadata[key] = _canonical_code_metadata_value(
            value,
            field=f"fabric.code metadata {key!r}",
        )
    metadata["stabilizer_labels"] = _canonical_code_metadata_value(
        code.stabilizer_labels,
        field="fabric.code metadata 'stabilizer_labels'",
    )
    metadata.update(distance_metadata(code.d))
    return MappingProxyType(metadata)


def _materialized_code_identity(code: "Code") -> tuple[Any, ...]:
    """Mirror the label-independent payload of one ``fabric.code`` declaration."""

    def matrix_identity(matrix):
        return matrix.nrows, matrix.ncols, matrix.rows

    def metadata_identity(value):
        return type(value).__name__, value

    matrices = (
        code.stabilizer_basis,
        code.logical_x_basis,
        code.logical_z_basis,
        code.gauge_x_basis,
        code.gauge_z_basis,
        code.anti_stabilizers,
        code.encoding_clifford,
    )
    metadata = _materialized_code_metadata(code)
    return (
        tuple(sorted(code.block.partitions.items())),
        isinstance(code.block, CSSBlock),
        code.n,
        code.k,
        code.r,
        code.d.conservative_value or 0,
        code.hx,
        code.hz,
        code.gx,
        code.gz,
        code.lx,
        code.lz,
        tuple(str(value) for value in code.stabilizers),
        tuple(str(value) for value in code.gauges),
        tuple(matrix_identity(matrix) for matrix in matrices),
        tuple(
            sorted((key, metadata_identity(value))
                   for key, value in metadata.items())),
    )


_KEEP_CODE_FIELD = object()


def _derive_code_artifact(
    code: Code,
    *,
    name=_KEEP_CODE_FIELD,
    d=_KEEP_CODE_FIELD,
) -> Code:
    """Build a sealed sibling while preserving validated code algebra.

    This is the one controlled construction path for changes that do not alter
    the validated algebra.  Defaults must be rebuilt because both profiles and
    encodings retain the owning code by identity.
    """

    clone = object.__new__(type(code))
    object.__setattr__(clone, "_immutable_sealed", False)
    copied = set()
    for base in reversed(type(code).__mro__):
        slots = base.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in copied or slot in {
                    "__dict__",
                    "__weakref__",
                    "_immutable_sealed",
                    "name",
                    "d",
                    "default_profile",
                    "default_encoding",
            }:
                continue
            copied.add(slot)
            if hasattr(code, slot):
                object.__setattr__(clone, slot, getattr(code, slot))
    object.__setattr__(
        clone,
        "name",
        code.name if name is _KEEP_CODE_FIELD else str(name),
    )
    object.__setattr__(
        clone,
        "d",
        code.d if d is _KEEP_CODE_FIELD else d,
    )
    default_profile = CodeProfile(clone)
    object.__setattr__(clone, "default_profile", default_profile)
    object.__setattr__(
        clone,
        "default_encoding",
        Encoding(clone, profile=default_profile),
    )
    clone._seal()
    return clone


@dataclass(frozen=True, slots=True)
class _ParameterIdentity:
    """Type-disjoint cache identity for one public code-factory argument."""

    kind: type
    value: Any


def _parameter_identity(value):
    """Hashable specialization identity that preserves authored BB labels."""

    from cudaq.logical.codes.bb import BinaryPolynomial

    if isinstance(value, BinaryPolynomial):
        return _ParameterIdentity(
            type(value),
            (value.monomials, value.terms),
        )
    if isinstance(value, tuple):
        return _ParameterIdentity(
            tuple,
            tuple(_parameter_identity(item) for item in value),
        )
    if isinstance(value, Mapping):
        return _ParameterIdentity(
            type(value),
            frozenset((
                _parameter_identity(key),
                _parameter_identity(item),
            ) for key, item in value.items()),
        )
    if isinstance(value, list):
        return _ParameterIdentity(
            list,
            tuple(_parameter_identity(item) for item in value),
        )
    if isinstance(value, frozenset):
        return _ParameterIdentity(
            frozenset,
            frozenset(_parameter_identity(item) for item in value),
        )
    return _ParameterIdentity(type(value), value)


class ParameterizedCode:
    __slots__ = ("provider", "name", "signature", "_instances")

    def __init__(self,
                 provider: Callable[..., Code],
                 name: str | None = None) -> None:
        self.provider = provider
        self.name = name or provider.__name__
        self.signature = signature(provider)
        self._instances = {}

    def __getitem__(self, parameters):
        if not isinstance(parameters, tuple):
            parameters = (parameters,)
        return self(*parameters)

    def __call__(self, *args, **kwargs) -> Code:
        self.signature.bind(*args, **kwargs)
        key = (
            tuple(_parameter_identity(value) for value in args),
            tuple(
                sorted((name, _parameter_identity(value))
                       for name, value in kwargs.items())),
        )
        cached = self._instances.get(key)
        if cached is not None:
            return cached
        result = self.provider(*args, **kwargs)
        if not isinstance(result, Code):
            raise TypeError(
                "@cudaq.logical.code function must return a cudaq.logical.Code")
        if result.name == "anonymous_code":
            suffix = "_".join(str(value) for value in (*args, *kwargs.values()))
            result = _derive_code_artifact(
                result,
                name=f"{self.name}_{suffix}" if suffix else self.name,
            )
        self._instances[key] = result
        return result


def code(subject=None, *, name: str | None = None):

    def decorate(value):
        if isinstance(value, type):
            attrs = vars(value)
            return Code(
                name=name or value.__name__,
                n=attrs.get("n"),
                k=attrs.get("k"),
                r=attrs.get("r"),
                d=attrs.get("d", attrs.get("distance")),
                block=attrs.get("block"),
                stabilizers=attrs.get("stabilizers", ()),
                stabilizer_labels=attrs.get("stabilizer_labels"),
                gauges=attrs.get("gauges", ()),
                logicals=attrs.get("logicals"),
                gauge_pairs=attrs.get("gauge_pairs"),
                anti_stabilizers=attrs.get("anti_stabilizers", ()),
                hx=attrs.get("hx"),
                hz=attrs.get("hz"),
                gx=attrs.get("gx"),
                gz=attrs.get("gz"),
                lx=attrs.get("lx"),
                lz=attrs.get("lz"),
                metadata=attrs.get("metadata"),
            )
        if callable(value):
            return ParameterizedCode(value, name=name)
        raise TypeError(
            "@cudaq.logical.code decorates a class or code-constructor "
            "function")

    return decorate(subject) if subject is not None else decorate


_BB_COMPATIBILITY_EXPORTS = frozenset({
    "BinaryPolynomial",
    "binary_polynomial",
    "CyclicProduct",
    "BBPermutationMap",
    "BBSyndromeMoment",
    "BBSyndromeSchedule",
    "BivariateBicycleCode",
})

__all__ = [
    name for name, value in globals().items()
    if not name.startswith("_") and (isinstance(value, type) or callable(value))
] + sorted(_BB_COMPATIBILITY_EXPORTS)


def __getattr__(name: str):
    """Resolve historical BB-family imports from their typed owner."""

    if name not in _BB_COMPATIBILITY_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module("cudaq.logical.codes.bb")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
