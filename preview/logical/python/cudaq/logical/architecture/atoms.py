# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Neutral-atom conveniences over the generic QLX physical model.

Zones and shuttle routes are ordinary ``Topology`` values, pulses are ordinary
``PhysicalAction`` values, and the resulting program is still canonical
``phys`` IR.  This module adds useful vocabulary without a second physical IR.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from . import physical_actions, physical_instruments
from cudaq.logical.ops._impl import (
    apply,
    measure,
    move,
)
from cudaq.logical.architecture.physical_definition import (
    PhysicalMachine,
    ResourceClass,
    Topology,
)


class Role(str, Enum):
    STORAGE = "storage"
    ENTANGLING = "entangling"
    READOUT = "readout"
    LOADING = "loading"


def _role(value) -> str:
    if isinstance(value, Role):
        return value.value
    value = str(value)
    if value not in {item.value for item in Role}:
        raise ValueError(f"unknown neutral-atom zone role {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class Zone:
    name: str
    role: Role | str
    words: int
    sites_per_word: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("zone name must be nonempty")
        object.__setattr__(self, "role", Role(_role(self.role)))
        for field in ("words", "sites_per_word"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value,
                                                        bool) or value <= 0:
                raise TypeError(f"zone {field} must be a positive Python int")

    @property
    def capacity(self) -> int:
        return self.words * self.sites_per_word


class AtomLayout:
    __slots__ = ("name", "zones", "routes", "_by_role")

    def __init__(self, name: str, zones) -> None:
        self.name = str(name)
        self.zones = tuple(zones)
        if not self.zones or len({zone.name for zone in self.zones}) != len(
                self.zones):
            raise ValueError("atom layout requires uniquely named zones")
        self._by_role = {zone.role: zone for zone in self.zones}
        routes = {}
        for source in self.zones:
            for destination in self.zones:
                if source is destination:
                    continue
                name = f"{source.name}_to_{destination.name}"
                routes[name] = Topology(
                    "shuttle",
                    source=source.name,
                    destination=destination.name,
                )
        self.routes = routes

    def zone(self, role: Role | str) -> Zone:
        try:
            return self._by_role[Role(_role(role))]
        except KeyError as exc:
            raise KeyError(
                f"layout {self.name!r} has no {_role(role)!r} zone") from exc

    def route(self, source: Role | str, destination: Role | str) -> Topology:
        source_zone = self.zone(source)
        destination_zone = self.zone(destination)
        return self.routes[f"{source_zone.name}_to_{destination_zone.name}"]

    def topologies(self):
        zones = {
            zone.name:
                Topology(
                    "zone",
                    role=zone.role.value,
                    words=zone.words,
                    sites_per_word=zone.sites_per_word,
                    capacity=zone.capacity,
                ) for zone in self.zones
        }
        return {**zones, **self.routes}


def GeminiLogicalLayout() -> AtomLayout:
    return AtomLayout(
        "gemini_logical",
        (
            Zone("storage", Role.STORAGE, 2, 16),
            Zone("entangling", Role.ENTANGLING, 2, 16),
            Zone("readout", Role.READOUT, 2, 16),
        ),
    )


def GeminiFullLayout() -> AtomLayout:
    return AtomLayout(
        "gemini_full",
        (
            Zone("loading", Role.LOADING, 4, 32),
            Zone("storage", Role.STORAGE, 8, 32),
            Zone("entangling", Role.ENTANGLING, 4, 32),
            Zone("readout", Role.READOUT, 4, 32),
        ),
    )


def architecture(
        name: str,
        *,
        layout: AtomLayout,
        atom_count: int,
        capabilities=(),
) -> PhysicalMachine:
    """Build a generic physical architecture from a neutral-atom zone layout."""

    if not isinstance(layout, AtomLayout):
        raise TypeError("layout= must be an AtomLayout")
    atoms = ResourceClass(
        "atom",
        atom_count,
        native_actions=physical_actions.neutral_atom_set(),
        native_action_decompositions=(
            physical_actions.neutral_atom_decompositions()),
        native_instruments=(
            physical_instruments.MZ,
            physical_instruments.MX,
        ),
        capabilities=tuple(str(value) for value in capabilities),
    )
    return PhysicalMachine(
        name,
        resource_classes={"atoms": atoms},
        topologies=layout.topologies(),
        metadata={"layout": layout.name},
    )


class Pulse:
    HADAMARD = physical_actions.GLOBAL_H
    X = physical_actions.GLOBAL_X
    CZ_BLOCKADE = physical_actions.RYDBERG_CZ


def activate(values, *, via):
    return move(values, via=via, trajectory="activate")


def park(values, *, via):
    return move(values, via=via, trajectory="park")


def shift(values, *, via, by):
    if (not isinstance(by, (tuple, list)) or len(by) != 2 or
            any(not isinstance(value, int) or isinstance(value, bool)
                for value in by)):
        raise TypeError("neutral-atom shift by= requires two Python ints")
    return move(values, via=via, trajectory=f"shift:{by[0]},{by[1]}")


def fire(values, pulse):
    return apply(pulse, values)


def fire2(left, right, pulse=Pulse.CZ_BLOCKADE, *, pairs=None):
    return apply(pulse, left, right, pairs=pairs)


def read(values, *, basis="z", record="readout"):
    values = tuple(values) if isinstance(values, (tuple, list)) else (values,)
    return tuple(
        measure(value, basis=basis, record=f"{record}.{index}")
        for index, value in enumerate(values))


__all__ = [
    "Role",
    "Zone",
    "AtomLayout",
    "GeminiLogicalLayout",
    "GeminiFullLayout",
    "Pulse",
    "architecture",
    "activate",
    "park",
    "shift",
    "fire",
    "fire2",
    "read",
]
