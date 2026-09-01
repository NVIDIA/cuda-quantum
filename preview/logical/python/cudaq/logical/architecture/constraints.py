# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from ..architecture.logical import (
    CapabilityKey,
    LogicalValueRef,
    Space,
    SpaceSlot,
)


def _logical_value(value, *, what: str) -> LogicalValueRef:
    if not isinstance(value, LogicalValueRef):
        raise TypeError(f"{what} must be a logical value reference")
    if (type(value.path) is not tuple or len(value.path) != 1 or
            type(value.path[0]) is not int or value.path[0] < 0):
        raise ValueError(f"{what} must be one top-level logical value")
    return value


def _plural_tuple(value, *, what: str) -> tuple:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{what} must be a non-string iterable")
    try:
        return tuple(value)
    except TypeError as error:
        raise TypeError(f"{what} must be an iterable") from error


@dataclass(frozen=True, slots=True)
class AllowSpaces:
    spaces: tuple[Space, ...]

    def __post_init__(self) -> None:
        spaces = tuple(self.spaces)
        if any(not isinstance(space, Space) for space in spaces):
            raise TypeError("allowed placement spaces must be machine Spaces")
        object.__setattr__(self, "spaces", spaces)


@dataclass(frozen=True, slots=True)
class RequireCapability:
    capability: CapabilityKey

    def __post_init__(self) -> None:
        if not isinstance(self.capability, CapabilityKey):
            raise TypeError(
                "required placement capability must be a CapabilityKey")


@dataclass(frozen=True, slots=True)
class Colocate:
    values: tuple[LogicalValueRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "values",
            tuple(
                _logical_value(value, what="colocation value")
                for value in self.values),
        )


@dataclass(frozen=True, slots=True)
class LocalPlacement:
    value: LogicalValueRef
    space: Space
    slot: int | None = None
    witness: str | None = None

    def __post_init__(self) -> None:
        _logical_value(self.value, what="local placement value")
        if not isinstance(self.space, Space):
            raise TypeError("local placement space must be a machine Space")
        if self.slot is not None and (type(self.slot) is not int or
                                      self.slot < 0):
            raise TypeError("local placement slot must be a nonnegative int")
        if self.witness is not None and (type(self.witness) is not str or
                                         not self.witness):
            raise TypeError("local placement witness must be a nonempty string")


@dataclass(frozen=True, slots=True)
class Prefer:
    space: Space | None = None
    role: str | None = None
    for_: Any = None

    def __post_init__(self) -> None:
        if self.space is not None and not isinstance(self.space, Space):
            raise TypeError("preferred placement space must be a machine Space")
        if self.role is not None and (type(self.role) is not str or
                                      not self.role):
            raise TypeError(
                "preferred placement role must be a nonempty string")


@dataclass(frozen=True, slots=True)
class PlacementBinding:
    placement: str
    space: str
    slot: int
    source_allocation: int | None = None
    source_group: str | None = None
    source_path: tuple[int, ...] = ()
    binding_kind: str = "local"
    binding_data: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.binding_kind != "local":
            raise ValueError(
                "nonlocal placement bindings are not supported by CUDA-Q Logical"
            )


@dataclass(frozen=True, slots=True)
class PlacementWitness:
    machine: str
    input_p0: str
    bindings: tuple[PlacementBinding, ...]
    relaxed_preferences: tuple[str, ...] = ()
    objective: str = "first_fit"
    tie_break: str = "declaration_order"


def allow_spaces(*spaces: Space) -> AllowSpaces:
    return AllowSpaces(tuple(spaces))


def require_capability(key: CapabilityKey) -> RequireCapability:
    return RequireCapability(key)


def colocate(values) -> Colocate:
    if isinstance(values, LogicalValueRef):
        values = (values,)
    else:
        values = tuple(values)
    return Colocate(values)


def local(value, *, at, slot=None, witness=None) -> LocalPlacement:
    if isinstance(at, SpaceSlot):
        if slot is not None:
            raise TypeError("do not combine a SpaceSlot with slot=")
        at, slot = at.space, at.index
    if not isinstance(at, Space):
        raise TypeError(
            "cudaq.logical.local at= requires a machine Space or SpaceSlot")
    if slot is not None and (not isinstance(slot, int) or
                             isinstance(slot, bool) or slot < 0):
        raise TypeError("cudaq.logical.local slot= must be a nonnegative int")
    if witness is not None and (not isinstance(witness, str) or not witness):
        raise TypeError(
            "cudaq.logical.local witness= must be a nonempty string")
    return LocalPlacement(_one_value(value), at, slot, witness)


def prefer(*,
           space: Space | None = None,
           role: str | None = None,
           for_=None) -> Prefer:
    return Prefer(space=space, role=role, for_=for_)


def _one_value(value) -> LogicalValueRef:
    return _logical_value(value, what="placement descriptor")


class _Lifecycle:
    IDLE = "idle"
    ACTIVE = "active"


class _Metric:
    expected_spacetime_volume = "expected_spacetime_volume"
    first_fit = "first_fit"


lifecycle = _Lifecycle()
metric = _Metric()
