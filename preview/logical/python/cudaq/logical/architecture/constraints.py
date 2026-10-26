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

from cudaq.logical.architecture.logical import (
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
class DistributedPlacement:
    value: LogicalValueRef
    spaces: tuple[Space, ...]
    support_views: tuple[str, ...]
    ownership_witness: str
    link_obligations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _logical_value(self.value, what="distributed placement value")
        spaces = tuple(self.spaces)
        support_views = _plural_tuple(
            self.support_views,
            what="distributed support views",
        )
        obligations = _plural_tuple(
            self.link_obligations,
            what="distributed link obligations",
        )
        if not spaces or any(not isinstance(space, Space) for space in spaces):
            raise TypeError(
                "distributed placement requires one or more machine Spaces")
        if (len(spaces) != len(support_views) or any(
                type(view) is not str or not view for view in support_views)):
            raise ValueError(
                "distributed support views must align with its spaces")
        if (type(self.ownership_witness) is not str or
                not self.ownership_witness):
            raise TypeError(
                "distributed ownership witness must be a nonempty string")
        if any(type(item) is not str for item in obligations):
            raise TypeError("distributed link obligations must be strings")
        object.__setattr__(self, "spaces", spaces)
        object.__setattr__(self, "support_views", support_views)
        object.__setattr__(self, "link_obligations", obligations)


@dataclass(frozen=True, slots=True)
class TrajectoryPlacement:
    value: LogicalValueRef
    segments: tuple[Space, ...]
    transition_events: tuple[str, ...]
    continuity_witness: str

    def __post_init__(self) -> None:
        _logical_value(self.value, what="trajectory placement value")
        segments = tuple(self.segments)
        events = _plural_tuple(
            self.transition_events,
            what="trajectory transition events",
        )
        if not segments or any(
                not isinstance(space, Space) for space in segments):
            raise TypeError(
                "trajectory placement requires one or more machine Spaces")
        if (len(events) + 1 != len(segments) or
                any(type(event) is not str or not event for event in events)):
            raise ValueError(
                "trajectory placement requires one event between segments")
        if (type(self.continuity_witness) is not str or
                not self.continuity_witness):
            raise TypeError(
                "trajectory continuity witness must be a nonempty string")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(self, "transition_events", events)


@dataclass(frozen=True, slots=True)
class TopologicalPlacement:
    value: LogicalValueRef
    space: Space
    record: str
    frontier: tuple[str, ...]
    support_witness: str
    observable_witness: str

    def __post_init__(self) -> None:
        _logical_value(self.value, what="topological placement value")
        if not isinstance(self.space, Space):
            raise TypeError(
                "topological placement space must be a machine Space")
        strings = (
            self.record,
            self.support_witness,
            self.observable_witness,
        )
        if any(type(item) is not str or not item for item in strings):
            raise TypeError(
                "topological record and witnesses must be nonempty strings")
        frontier = _plural_tuple(
            self.frontier,
            what="topological placement frontier",
        )
        if not frontier or any(
                type(item) is not str or not item for item in frontier):
            raise ValueError(
                "topological placement frontier must be nonempty strings")
        object.__setattr__(self, "frontier", frontier)


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


def distributed(
        value,
        *,
        across,
        support_views,
        witness,
        link_obligations=(),
) -> DistributedPlacement:
    spaces = tuple(across)
    views = _plural_tuple(
        support_views,
        what="distributed support_views",
    )
    if not spaces or any(not isinstance(space, Space) for space in spaces):
        raise TypeError(
            "distributed across= requires one or more machine spaces")
    if len(spaces) != len(views) or any(not view for view in views):
        raise ValueError("distributed support_views must align with spaces")
    if not isinstance(witness, str) or not witness:
        raise TypeError("distributed witness must be a nonempty string")
    return DistributedPlacement(
        _one_value(value),
        spaces,
        views,
        witness,
        _plural_tuple(
            link_obligations,
            what="distributed link_obligations",
        ),
    )


def trajectory(value, *, through, transitions, witness) -> TrajectoryPlacement:
    segments = tuple(through)
    events = _plural_tuple(
        transitions,
        what="trajectory transitions",
    )
    if not segments or any(not isinstance(space, Space) for space in segments):
        raise TypeError(
            "trajectory through= requires one or more machine spaces")
    if len(events) + 1 != len(segments) or any(not event for event in events):
        raise ValueError("trajectory requires one transition between segments")
    if not isinstance(witness, str) or not witness:
        raise TypeError("trajectory witness must be a nonempty string")
    return TrajectoryPlacement(_one_value(value), segments, events, witness)


def topological_record(
    value,
    *,
    space,
    record,
    frontier,
    support_witness,
    observable_witness,
) -> TopologicalPlacement:
    if not isinstance(space, Space):
        raise TypeError("topological_record space= requires a machine Space")
    strings = (record, support_witness, observable_witness)
    if any(not isinstance(item, str) or not item for item in strings):
        raise TypeError(
            "topological record and witnesses must be nonempty strings")
    frontier = _plural_tuple(
        frontier,
        what="topological_record frontier",
    )
    if not frontier or any(not item for item in frontier):
        raise ValueError("topological_record frontier must be nonempty")
    return TopologicalPlacement(
        _one_value(value),
        space,
        record,
        frontier,
        support_witness,
        observable_witness,
    )


class _Lifecycle:
    IDLE = "idle"
    ACTIVE = "active"


class _Metric:
    expected_spacetime_volume = "expected_spacetime_volume"
    first_fit = "first_fit"


lifecycle = _Lifecycle()
metric = _Metric()
