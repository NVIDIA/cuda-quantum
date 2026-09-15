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


class Block:
    """Static named carrier-role partitions for each patch of a code."""

    __slots__ = ("partitions",)

    def __init__(self, **partitions: int) -> None:
        if not partitions:
            raise ValueError("Block requires at least one named partition")
        checked = {}
        for name, count in partitions.items():
            if not isinstance(count, int) or isinstance(count,
                                                        bool) or count < 0:
                raise TypeError(
                    f"partition {name!r} count must be a nonnegative int")
            checked[name] = count
        self.partitions = MappingProxyType(checked)

    def __getattr__(self, name: str) -> int:
        try:
            return self.partitions[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    @property
    def size(self) -> int:
        return sum(self.partitions.values())


class CSSBlock(Block):

    def __init__(self,
                 *,
                 data: int,
                 sx: int = 0,
                 sz: int = 0,
                 **extra: int) -> None:
        super().__init__(data=data, sx=sx, sz=sz, **extra)


@dataclass(frozen=True, slots=True)
class CarrierRoleMap:
    """Per-phase roles over one persistently owned carrier frame.

    Reservation ownership and encoded-support membership are intentionally
    independent: a measured/reset carrier may remain reserved while inactive
    and become active again in a later dynamic-code phase.
    """

    active: tuple[int, ...]
    measured: tuple[int, ...] = ()
    reset: tuple[int, ...] = ()
    scratch: tuple[int, ...] = ()
    dormant: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        names = ("active", "measured", "reset", "scratch", "dormant")
        groups = {}
        for name in names:
            values = tuple(getattr(self, name))
            if any(not isinstance(value, int) or isinstance(value, bool) or
                   value < 0 for value in values):
                raise TypeError(
                    f"carrier role {name} must contain nonnegative ints")
            if len(set(values)) != len(values):
                raise ValueError(
                    f"carrier role {name} contains duplicate indices")
            groups[name] = values
            object.__setattr__(self, name, values)
        assigned: dict[int, str] = {}
        for name, values in groups.items():
            for value in values:
                if value in assigned:
                    raise ValueError(
                        f"carrier {value} belongs to both {assigned[value]} and {name}"
                    )
                assigned[value] = name


@dataclass(frozen=True, slots=True)
class PatchTransform:
    """A linear QEC boundary change over an explicit local carrier frame.

    The transform is structural and evidentiary.  It does not stand in for the
    physical circuit: a gadget using it still contains the reset, gate, and
    measurement operations that realize the claimed code change.
    """

    source: "Encoding"
    destination: "Encoding"
    frame: Block
    source_support: tuple[int, ...]
    destination_support: tuple[int, ...]
    source_roles: CarrierRoleMap
    destination_roles: CarrierRoleMap
    logical_map: tuple[int, ...]
    evidence: str
    name: str | None = None

    def __post_init__(self) -> None:
        from .encodings import Encoding

        if not isinstance(self.source, Encoding) or not isinstance(
                self.destination, Encoding):
            raise TypeError(
                "PatchTransform source/destination must be Encoding values")
        if not isinstance(self.frame, Block):
            raise TypeError(
                "PatchTransform frame must be a cudaq.logical.Block")
        source_support = tuple(self.source_support)
        destination_support = tuple(self.destination_support)
        if len(source_support) != self.source.code.n:
            raise ValueError("source_support width must equal source code n")
        if len(destination_support) != self.destination.code.n:
            raise ValueError(
                "destination_support width must equal destination code n")
        for label, support in (
            ("source", source_support),
            ("destination", destination_support),
        ):
            if len(set(support)) != len(support):
                raise ValueError(
                    f"{label}_support contains duplicate frame indices")
            if any(not isinstance(value, int) or isinstance(value, bool) or
                   value < 0 or value >= self.frame.size for value in support):
                raise ValueError(
                    f"{label}_support must reference the local carrier frame")
        if not isinstance(self.source_roles, CarrierRoleMap) or not isinstance(
                self.destination_roles, CarrierRoleMap):
            raise TypeError(
                "PatchTransform roles must be CarrierRoleMap values")
        for roles in (self.source_roles, self.destination_roles):
            for group in (
                    roles.active,
                    roles.measured,
                    roles.reset,
                    roles.scratch,
                    roles.dormant,
            ):
                if any(value >= self.frame.size for value in group):
                    raise ValueError(
                        "carrier role index exceeds the local frame")
        if set(self.source_roles.active) != set(source_support):
            raise ValueError("source active role must equal source_support")
        if set(self.destination_roles.active) != set(destination_support):
            raise ValueError(
                "destination active role must equal destination_support")
        logical_map = tuple(self.logical_map)
        if len(logical_map) != self.source.code.k:
            raise ValueError("logical_map width must equal source code k")
        if any(not isinstance(value, int) or isinstance(value, bool) or
               value < 0 or value >= self.destination.code.k
               for value in logical_map):
            raise ValueError(
                "logical_map must reference destination logical ports")
        if len(set(logical_map)) != len(logical_map):
            raise ValueError("logical_map must be injective")
        if not isinstance(self.evidence, str) or not self.evidence:
            raise ValueError("PatchTransform requires nonempty evidence")
        object.__setattr__(self, "source_support", source_support)
        object.__setattr__(self, "destination_support", destination_support)
        object.__setattr__(self, "logical_map", logical_map)
        object.__setattr__(
            self,
            "name",
            self.name or f"{self.source.name}_to_{self.destination.name}",
        )

    @classmethod
    def infer(
        cls,
        source: "Encoding",
        destination: "Encoding",
        *,
        evidence: str = "derived_from_stable_carrier_labels",
    ) -> "PatchTransform":
        """Infer the common no-scratch frame from stable carrier labels."""

        source_labels = source.carrier_labels
        destination_labels = destination.carrier_labels
        try:
            labels = tuple(sorted(set((*source_labels, *destination_labels))))
        except TypeError as exc:
            raise TypeError(
                "source and destination carrier labels must be mutually sortable"
            ) from exc
        positions = {label: index for index, label in enumerate(labels)}
        source_support = tuple(positions[label] for label in source_labels)
        destination_support = tuple(
            positions[label] for label in destination_labels)
        frame = Block(data=len(labels))
        inactive_source = tuple(
            index for index in range(frame.size) if index not in source_support)
        inactive_destination = tuple(index for index in range(frame.size)
                                     if index not in destination_support)
        return cls(
            source=source,
            destination=destination,
            frame=frame,
            source_support=source_support,
            destination_support=destination_support,
            source_roles=CarrierRoleMap(active=source_support,
                                        dormant=inactive_source),
            destination_roles=CarrierRoleMap(active=destination_support,
                                             dormant=inactive_destination),
            logical_map=tuple(range(source.code.k)),
            evidence=evidence,
        )

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)
