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

from .selection import _canonical_label_index, _deep_freeze
from .structure import Block
from .profiles import CodeProfile, EncodingEpochSchema, EncodingEpoch


def _schema_from_dynamic_profile(profile: CodeProfile, *,
                                 name: str) -> EncodingEpochSchema:
    phases = profile.dynamic_phases
    if not phases:
        return EncodingEpochSchema.static(name=name)
    names = tuple(phase.input_epoch for phase in phases)
    if tuple(phase.name for phase in phases) != names:
        raise ValueError(
            "dynamic phase names must equal their input epochs so row and "
            "epoch identity cannot diverge")
    transitions = tuple(
        (phase.input_epoch, phase.output_epoch) for phase in phases)
    targets = tuple(target for _, target in transitions)
    if set(targets) != set(names):
        raise ValueError(
            "dynamic phase transitions must close over the declared epochs")
    logical_maps = {
        f"{source}->{target}": dict(phase.logical_map)
        for phase, (source, target) in zip(phases, transitions)
    }
    periodic = targets[-1] == names[0]
    return EncodingEpochSchema(
        name=name,
        phases=names,
        initial=names[0],
        transitions=transitions,
        logical_maps=logical_maps,
        is_periodic=periodic,
        closure=profile.period_closure,
    )


class Encoding(ImmutableValue):
    __slots__ = (
        "name",
        "code",
        "profile",
        "block",
        "logical_ports",
        "layout",
        "carrier_labels",
        "carrier_indices",
        "logical_port_indices",
        "hierarchy",
        "flat_projection",
        "metadata",
        "epoch_schema",
        "initial_epoch",
        "_epochs",
    )

    def __init__(
        self,
        code: "Code",
        *,
        profile: CodeProfile | None = None,
        name: str | None = None,
        block: str = "block0",
        logical_ports: Iterable[str] | Mapping[str, Any] | None = None,
        layout: Mapping[str, Any] | None = None,
        hierarchy: "EncodingHierarchy | None" = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.code = code
        if profile is not None and not isinstance(profile, CodeProfile):
            raise TypeError(
                "Encoding profile must be a cudaq.logical.CodeProfile")
        if profile is not None and profile.code is not code:
            raise ValueError("Encoding profile must belong to its code")
        self.profile = profile or code.default_profile
        self.name = name or f"{code.name}_default_encoding"
        self.block = block
        if logical_ports is None:
            port_names = tuple(f"q{i}" for i in range(code.k))
            port_indices = tuple(range(code.k))
        elif isinstance(logical_ports, Mapping):
            indexed_ports = tuple((str(port), int(index))
                                  for port, index in logical_ports.items())
            port_indices = tuple(index for _, index in indexed_ports)
            port_names = tuple(
                port
                for port, _ in sorted(indexed_ports, key=lambda item: item[1]))
        else:
            port_names = tuple(str(port) for port in logical_ports)
            port_indices = tuple(range(len(port_names)))
        if len(port_names) != code.k:
            raise ValueError(
                f"encoding for k={code.k} code requires {code.k} logical ports")
        if sorted(port_indices) != list(range(code.k)):
            raise ValueError(
                "logical port map must be a permutation of range(code.k)")
        if isinstance(logical_ports, Mapping):
            port_indices = tuple(range(len(port_names)))
        self.logical_ports = port_names
        self.logical_port_indices = MappingProxyType(
            dict(zip(port_names, port_indices)))
        layout_values = dict(layout or {})
        carrier_labels, carrier_indices = _canonical_label_index(
            layout_values.get("carrier_labels", range(code.n)),
            code.n,
            what="encoding carrier_labels",
        )
        if "carrier_labels" in layout_values:
            layout_values["carrier_labels"] = carrier_labels
        self.layout = _deep_freeze(layout_values, what="encoding layout")
        self.carrier_labels = carrier_labels
        self.carrier_indices = carrier_indices
        if hierarchy is not None and not isinstance(hierarchy,
                                                    EncodingHierarchy):
            raise TypeError(
                "Encoding hierarchy must be a cudaq.logical.EncodingHierarchy")
        if hierarchy is not None and hierarchy.code is not code:
            raise ValueError("Encoding hierarchy must belong to its code")
        self.hierarchy = hierarchy
        self.flat_projection = None
        self.metadata = _deep_freeze(metadata or {}, what="encoding metadata")
        self.epoch_schema = _schema_from_dynamic_profile(
            self.profile, name=f"{self.name}_epoch_schema")
        self.initial_epoch = EncodingEpoch(
            name=f"{self.name}_initial_epoch",
            encoding=self,
            schema=self.epoch_schema,
            phase=self.epoch_schema.initial,
        )
        self._epochs = {(self.initial_epoch.phase, 0): self.initial_epoch}
        self._seal()

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)

    @property
    def ports(self):
        """Typed protected-logical references exposed by this encoding."""

        from cudaq.logical.programs.binding import LogicalPorts

        return LogicalPorts(self)

    def flatten(self) -> "Encoding":
        """Return the flat view of the same derived code algebra."""
        if self.hierarchy is None:
            return self
        return self.hierarchy.flat_encoding

    def epoch(self,
              phase: str,
              *,
              index: int = 0,
              name: str | None = None) -> EncodingEpoch:
        """Return one stable phase instance for use in a patch type."""

        key = (str(phase), index)
        if key in self._epochs:
            return self._epochs[key]
        value = EncodingEpoch(
            name=name or f"{self.name}_{phase}_epoch{index}",
            encoding=self,
            schema=self.epoch_schema,
            phase=str(phase),
            index=index,
        )
        self._epochs[key] = value
        return value

    @classmethod
    def concatenate(
        cls,
        *,
        outer,
        inner,
        carrier_map: Mapping[int, tuple[int, int | str]] | None = None,
        unmapped: Mapping[tuple[int, int | str], Any] | None = None,
        name: str | None = None,
    ) -> "Encoding":
        from .composition import _concatenate_encoding

        return _concatenate_encoding(
            outer=outer,
            inner=inner,
            carrier_map=carrier_map,
            unmapped=unmapped,
            name=name,
        )


class Concatenated:
    """Concise canonical structural concatenation: ``Concatenated[O, I]``."""

    def __new__(cls, *args, **kwargs):
        raise TypeError("use cudaq.logical.Concatenated[Outer, Inner]")

    @classmethod
    def __class_getitem__(cls, layers) -> Encoding:
        from .composition import _as_encoding, _concatenate_encoding

        if not isinstance(layers, tuple):
            layers = (layers,)
        if len(layers) != 2:
            raise TypeError("cudaq.logical.Concatenated expects [Outer, Inner]")
        outer, inner = layers
        outer_code = _as_encoding(outer).code
        inner_code = _as_encoding(inner).code
        if inner_code.k < 1:
            raise ValueError(
                "cudaq.logical.Concatenated requires an inner code with a logical port"
            )
        carrier_map = {
            outer_index: (
                outer_index // inner_code.k,
                outer_index % inner_code.k,
            ) for outer_index in range(outer_code.n)
        }
        return _concatenate_encoding(
            outer=outer,
            inner=inner,
            carrier_map=carrier_map,
            unmapped=None,
            name=None,
        )


@dataclass(frozen=True, slots=True)
class EncodingHierarchy:
    name: str
    code: "Code"
    outer: "Encoding"
    child: Encoding
    multiplicity: int
    carrier_map: tuple[tuple[int, int, int], ...]
    exposed_ports: tuple[tuple[int, int], ...]
    gauge_ports: tuple[tuple[int, int], ...]
    flat_encoding: Encoding
    depth: int
    fixed_ports: tuple[tuple[int, int, "FixedPort"], ...] = ()

    def __post_init__(self) -> None:
        from .definition import Code

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("encoding hierarchy name must be nonempty")
        if not isinstance(self.code, Code):
            raise TypeError(
                "encoding hierarchy code must be a cudaq.logical.Code")
        if any(not isinstance(value, Encoding)
               for value in (self.outer, self.child, self.flat_encoding)):
            raise TypeError(
                "encoding hierarchy views must be cudaq.logical.Encoding values"
            )
        if self.flat_encoding.code is not self.code:
            raise ValueError(
                "encoding hierarchy flat view must belong to its code")
        if (isinstance(self.multiplicity, bool) or
                not isinstance(self.multiplicity, int) or
                self.multiplicity < 1):
            raise ValueError(
                "encoding hierarchy multiplicity must be a positive integer")
        if (isinstance(self.depth, bool) or not isinstance(self.depth, int) or
                self.depth < 1):
            raise ValueError(
                "encoding hierarchy depth must be a positive integer")

        def integer_rows(values, width, label):
            rows = []
            for raw in values:
                row = tuple(raw)
                if len(row) != width or any(
                        isinstance(item, bool) or not isinstance(item, int)
                        for item in row):
                    raise TypeError(
                        f"encoding hierarchy {label} entries must contain "
                        f"exactly {width} integers")
                rows.append(row)
            return tuple(rows)

        carrier_map = integer_rows(self.carrier_map, 3, "carrier_map")
        exposed_ports = integer_rows(self.exposed_ports, 2, "exposed_ports")
        gauge_ports = integer_rows(self.gauge_ports, 2, "gauge_ports")
        fixed_ports = []
        for raw in self.fixed_ports:
            row = tuple(raw)
            if (len(row) != 3 or isinstance(row[0], bool) or
                    not isinstance(row[0], int) or isinstance(row[1], bool) or
                    not isinstance(row[1], int) or
                    not isinstance(row[2], FixedPort)):
                raise TypeError(
                    "encoding hierarchy fixed_ports entries require child "
                    "index, logical index, and cudaq.logical.FixedPort evidence"
                )
            fixed_ports.append(row)
        fixed_ports = tuple(fixed_ports)

        if tuple(row[0] for row in carrier_map) != tuple(
                range(self.outer.code.n)):
            raise ValueError(
                "encoding hierarchy carrier_map must cover outer carriers "
                "once in canonical order")

        def validate_child_port(child, port, label):
            if not 0 <= child < self.multiplicity:
                raise ValueError(
                    f"encoding hierarchy {label} child index is out of range")
            if not 0 <= port < self.child.code.k:
                raise ValueError(
                    f"encoding hierarchy {label} logical index is out of range")

        for _, child, port in carrier_map:
            validate_child_port(child, port, "carrier_map")
        dispositions = (*exposed_ports, *gauge_ports,
                        *((child, port) for child, port, _ in fixed_ports))
        for child, port in dispositions:
            validate_child_port(child, port, "port disposition")
        if len(set(dispositions)) != len(dispositions):
            raise ValueError(
                "encoding hierarchy logical-port dispositions must be unique")

        object.__setattr__(self, "carrier_map", carrier_map)
        object.__setattr__(self, "exposed_ports", exposed_ports)
        object.__setattr__(self, "gauge_ports", gauge_ports)
        object.__setattr__(self, "fixed_ports", fixed_ports)

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)


@dataclass(frozen=True, slots=True)
class EncodingProjection:
    name: str
    source: Encoding
    destination: Encoding
    carrier_map: tuple[int, ...]
    logical_map: tuple[int, ...]
    evidence: str = "derived_concat_flattening"

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)


class _PortDisposition:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"qlx.{self.name}"


expose = _PortDisposition("expose")
gauge = _PortDisposition("gauge")


@dataclass(frozen=True, slots=True)
class FixedPort:
    """Evidence-bearing stabilizer constraint on one unused child logical."""

    basis: str
    eigenvalue: int
    evidence: str

    def __post_init__(self) -> None:
        basis = str(self.basis).lower()
        if basis not in {"x", "z"}:
            raise ValueError(
                "fixed concatenation ports currently require X or Z")
        if self.eigenvalue not in {-1, 1}:
            raise ValueError("fixed-port eigenvalue must be +1 or -1")
        if not isinstance(self.evidence, str) or not self.evidence:
            raise ValueError("fixed-port constraints require nonempty evidence")
        object.__setattr__(self, "basis", basis)


def fix(pauli, *, eigenvalue=1, evidence) -> FixedPort:
    """Fix one unused child logical Pauli instead of exposing or gauging it."""

    basis = getattr(pauli, "name", pauli)
    return FixedPort(str(basis), int(eigenvalue), evidence)
