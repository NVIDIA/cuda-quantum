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

from .structure import CSSBlock
from .distance import Distance
from .encodings import (
    Encoding,
    EncodingHierarchy,
    EncodingProjection,
    FixedPort,
    expose,
    gauge,
)
from .definition import CSSCode, Code, SubsystemCode


def _as_encoding(value) -> Encoding:
    if isinstance(value, Encoding):
        return value
    if isinstance(value, Code):
        return value.default_encoding
    raise TypeError("concatenation layers must be a Code or Encoding")


def _concatenate_encoding(
    *,
    outer,
    inner,
    carrier_map,
    unmapped,
    name,
) -> Encoding:
    """Derive the flat CSS algebra and retain a folded structural hierarchy."""
    outer_encoding = _as_encoding(outer)
    child_encoding = _as_encoding(inner)
    outer_code = outer_encoding.code
    child_code = child_encoding.code
    for layer, code_ in (("outer", outer_code), ("inner", child_code)):
        if len(code_.lx) != code_.k or len(code_.lz) != code_.k:
            raise ValueError(
                f"{layer} code must declare one lx/lz representative per logical"
            )
        if len(code_.gx) != code_.r or len(code_.gz) != code_.r:
            raise ValueError(
                f"{layer} subsystem code must declare one gx/gz representative "
                "per gauge qubit")

    if carrier_map is None:
        if child_code.k != 1:
            raise ValueError(
                "inner encodings with k>1 require carrier_map={outer: "
                "(child, logical_port)}")
        normalized_map = {
            outer_index: (outer_index, 0) for outer_index in range(outer_code.n)
        }
    else:
        normalized_map = {}
        for outer_index, target in carrier_map.items():
            if not isinstance(outer_index, int) or isinstance(
                    outer_index, bool):
                raise TypeError("outer carrier indices must be Python ints")
            try:
                child_index, port = target
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "carrier-map values must be (child_index, logical_port)"
                ) from exc
            if isinstance(port, str):
                try:
                    logical_index = child_encoding.logical_port_indices[port]
                except KeyError as exc:
                    raise ValueError(
                        f"unknown child logical port {port!r}") from exc
            else:
                logical_index = int(port)
            normalized_map[outer_index] = (int(child_index), logical_index)
    if set(normalized_map) != set(range(outer_code.n)):
        raise ValueError(
            "carrier_map must map every outer physical carrier exactly once")
    if any(child < 0 or port < 0 or port >= child_code.k
           for child, port in normalized_map.values()):
        raise ValueError(
            "carrier_map contains an invalid child or logical-port index")
    targets = tuple(normalized_map.values())
    if len(set(targets)) != len(targets):
        raise ValueError("carrier_map must be injective")
    child_count = max(child for child, _ in targets) + 1
    if set(child for child, _ in targets) != set(range(child_count)):
        raise ValueError(
            "carrier_map child indices must form a dense range from zero")

    all_ports = {(child, logical_index)
                 for child in range(child_count)
                 for logical_index in range(child_code.k)}
    unused = all_ports - set(targets)
    requested = dict(unmapped or {})
    normalized_dispositions = {}
    for key, disposition in requested.items():
        child, port = key
        if isinstance(port, str):
            port = child_encoding.logical_port_indices[port]
        normalized_dispositions[(int(child), int(port))] = disposition
    if set(normalized_dispositions) - unused:
        raise ValueError(
            "unmapped dispositions may name only unused child ports")
    exposed = tuple(
        sorted(port for port in unused
               if normalized_dispositions.get(port, expose) is expose))
    gauged = tuple(
        sorted(port for port in unused
               if normalized_dispositions.get(port, expose) is gauge))
    fixed = tuple(
        sorted(
            (child, port, disposition)
            for (child, port), disposition in normalized_dispositions.items()
            if (child, port) in unused and isinstance(disposition, FixedPort)))
    unsupported = [
        disposition for port, disposition in normalized_dispositions.items()
        if port in unused and disposition not in (
            expose, gauge) and not isinstance(disposition, FixedPort)
    ]
    if unsupported:
        raise TypeError(
            "unmapped-port dispositions must be cudaq.logical.expose, cudaq.logical.gauge, or "
            "cudaq.logical.fix(...)")

    def shift(child: int, support) -> tuple[int, ...]:
        return tuple(child * child_code.n + index for index in support)

    def lift_outer(row, basis_rows) -> tuple[int, ...]:
        # Substitution is multiplication of Pauli representatives, hence
        # addition over GF(2).  Packed high-rate children can place several
        # outer carriers in one block, so overlapping representative support
        # cancels instead of appearing twice in a support list.
        result = set()
        for outer_index in row:
            child, logical_index = normalized_map[outer_index]
            result.symmetric_difference_update(
                shift(child, basis_rows[logical_index]))
        return tuple(sorted(result))

    hx = [
        shift(child, row)
        for child in range(child_count)
        for row in child_code.hx
    ]
    hz = [
        shift(child, row)
        for child in range(child_count)
        for row in child_code.hz
    ]
    hx.extend(lift_outer(row, child_code.lx) for row in outer_code.hx)
    hz.extend(lift_outer(row, child_code.lz) for row in outer_code.hz)
    hx.extend(
        shift(child, child_code.lx[port])
        for child, port, disposition in fixed
        if disposition.basis == "x")
    hz.extend(
        shift(child, child_code.lz[port])
        for child, port, disposition in fixed
        if disposition.basis == "z")
    lx = [lift_outer(row, child_code.lx) for row in outer_code.lx]
    lz = [lift_outer(row, child_code.lz) for row in outer_code.lz]
    lx.extend(shift(child, child_code.lx[port]) for child, port in exposed)
    lz.extend(shift(child, child_code.lz[port]) for child, port in exposed)
    # Gauge structure composes exactly like the protected logical algebra.
    # Every child gauge pair remains a gauge pair in the leaf code, every
    # outer gauge pair is lifted through the selected child logical ports, and
    # an unused child logical pair becomes gauge only when the caller says so.
    # None of these rows may be folded into stabilizers: doing that would
    # silently fix a gauge and change the encoded subsystem.
    gx = [
        shift(child, row)
        for child in range(child_count)
        for row in child_code.gx
    ]
    gz = [
        shift(child, row)
        for child in range(child_count)
        for row in child_code.gz
    ]
    gx.extend(lift_outer(row, child_code.lx) for row in outer_code.gx)
    gz.extend(lift_outer(row, child_code.lz) for row in outer_code.gz)
    gx.extend(shift(child, child_code.lx[port]) for child, port in gauged)
    gz.extend(shift(child, child_code.lz[port]) for child, port in gauged)

    composite_r = child_count * child_code.r + outer_code.r + len(gauged)
    if len(gx) != composite_r or len(gz) != composite_r:
        raise ValueError(
            "concatenation produced an incomplete composite gauge basis")

    outer_layers = tuple(
        outer_encoding.metadata.get("concatenation_layers", (outer_code.name,)))
    inner_layers = tuple(
        child_encoding.metadata.get("concatenation_layers", (child_code.name,)))
    layers = (*outer_layers, *inner_layers)
    base_name = name or "_over_".join(layers)
    code_type = SubsystemCode if composite_r else CSSCode
    composite = code_type(
        name=f"{base_name}_code",
        n=child_count * child_code.n,
        k=outer_code.k + len(exposed),
        r=composite_r,
        d=Distance.unknown(
            "concatenated distance requires independent evidence; constituent "
            "distance claims are not multiplied automatically"),
        block=CSSBlock(
            data=child_count * child_code.n,
            sx=len(hx),
            sz=len(hz),
        ),
        hx=tuple(hx),
        hz=tuple(hz),
        gx=tuple(gx),
        gz=tuple(gz),
        lx=tuple(lx),
        lz=tuple(lz),
        metadata={
            "construction": "concatenation",
            "outer": outer_code.name,
            "inner": child_code.name,
            "outer_gauge_qubits": outer_code.r,
            "inner_gauge_qubits_per_child": child_code.r,
            "reclassified_gauge_ports": len(gauged),
            "fixed_logical_ports": len(fixed),
        },
    )
    flat = composite.default_encoding
    port_names = list(outer_encoding.logical_ports)
    for child, logical_index in exposed:
        port_names.append(
            f"child{child}_{child_encoding.logical_ports[logical_index]}")
    hierarchy = EncodingHierarchy(
        name=f"{base_name}_hierarchy",
        code=composite,
        outer=outer_encoding,
        child=child_encoding,
        multiplicity=child_count,
        carrier_map=tuple((outer_index, *normalized_map[outer_index])
                          for outer_index in range(outer_code.n)),
        exposed_ports=exposed,
        gauge_ports=gauged,
        flat_encoding=flat,
        depth=1 + (child_encoding.hierarchy.depth
                   if child_encoding.hierarchy is not None else 1),
        fixed_ports=fixed,
    )
    structural = Encoding(
        composite,
        name=base_name,
        logical_ports=tuple(port_names),
        hierarchy=hierarchy,
        metadata={
            "view": "structural",
            "concatenation_layers": layers
        },
    )
    projection = EncodingProjection(
        name=f"{base_name}_flat_projection",
        source=structural,
        destination=flat,
        carrier_map=tuple(range(composite.n)),
        logical_map=tuple(range(composite.k)),
    )
    # The projection is the second half of this private two-object
    # construction.  It is not a public mutation surface: the returned
    # Encoding remains sealed, while its derived structural view is completed
    # atomically here.
    object.__setattr__(structural, "flat_projection", projection)
    return structural
