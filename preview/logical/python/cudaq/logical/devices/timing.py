# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Physical timing values used by device operating points and schedules."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from cudaq.logical._core.immutable import freeze_mapping


def _nanoseconds(value, *, parameter: str) -> float:
    if isinstance(value, Duration):
        value = value.nanoseconds
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"parameter {parameter!r} must be a Duration or numeric "
            "nanosecond count")
    if not math.isfinite(value):
        raise ValueError(f"duration {parameter!r} must be finite")
    if value < 0:
        raise ValueError(f"duration {parameter!r} must be nonnegative")
    return float(value)


@dataclass(frozen=True, slots=True)
class Duration:
    """A physical duration in nanoseconds.

    Constructed by unit arithmetic, for example
    ``200 * cudaq.logical.devices.ns`` or
    ``1.5 * cudaq.logical.devices.us``.
    """

    nanoseconds: float

    def __post_init__(self):
        object.__setattr__(self, "nanoseconds",
                           _nanoseconds(self.nanoseconds, parameter="duration"))

    def __mul__(self, factor):
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            return NotImplemented
        return Duration(self.nanoseconds * factor)

    __rmul__ = __mul__

    def __add__(self, other):
        if not isinstance(other, Duration):
            return NotImplemented
        return Duration(self.nanoseconds + other.nanoseconds)

    def __str__(self) -> str:
        return f"{self.nanoseconds:g}ns"


ns = Duration(1)
us = Duration(1_000)
ms = Duration(1_000_000)


class TimingModel(Mapping):
    """Immutable physical durations with optional code-distance calibration.

    ``entries`` contains timings that do not depend on an encoded patch's code
    distance. ``by_code_distance`` groups the calibrated timings that do. The
    latter are serialized with canonical ``<name>_d<distance>_ns`` keys.
    """

    __slots__ = ("source", "_entries", "_by_code_distance")

    def __init__(
        self,
        entries: Mapping[str, Any] | None = None,
        *,
        by_code_distance: Mapping[int, Mapping[str, Any]] | None = None,
        source: str | None = None,
    ):
        if source is not None and (not isinstance(source, str) or not source):
            raise ValueError("TimingModel.source must be a nonempty string")
        if by_code_distance and source is None:
            raise ValueError(
                "code-distance timings require a calibration source")

        def validated_distance_timing(name: str, value: Any) -> float:
            if (not isinstance(name, str) or not name or
                    not name.endswith("_ns")):
                raise ValueError(
                    "code-distance timing names must be nonempty strings "
                    "ending in '_ns'")
            return _nanoseconds(value, parameter=name)

        flattened = dict(freeze_mapping(entries))
        for name, value in tuple(flattened.items()):
            if (isinstance(name, str) and re.fullmatch(r".+_d[0-9]+_ns", name)):
                raise ValueError(
                    "distance-qualified timing names are reserved for "
                    "by_code_distance")
            if isinstance(name, str) and name.endswith("_ns"):
                flattened[name] = _nanoseconds(value, parameter=name)
        qualified = {}
        for raw_distance, values in ({} if by_code_distance is None else
                                     by_code_distance).items():
            if (isinstance(raw_distance, bool) or
                    not isinstance(raw_distance, int) or raw_distance <= 0):
                raise TypeError(
                    "TimingModel code distances must be positive integers")
            distance = int(raw_distance)
            if not isinstance(values, Mapping) or not values:
                raise TypeError(
                    "each TimingModel code distance requires named timings")
            distance_entries = {}
            for name, value in values.items():
                numeric = validated_distance_timing(name, value)
                stem = name[:-3]
                canonical = f"{stem}_d{distance}_ns"
                if canonical in flattened:
                    raise ValueError(
                        f"duplicate canonical timing entry {canonical!r}")
                flattened[canonical] = numeric
                distance_entries[name] = numeric
            qualified[distance] = freeze_mapping(distance_entries)

        self.source = source
        self._entries = freeze_mapping(flattened)
        self._by_code_distance = MappingProxyType(
            dict(sorted(qualified.items())))

    @classmethod
    def from_calibration(cls, source: str) -> "TimingModel":
        """Reference a named calibration as the duration provider."""

        if not source or not isinstance(source, str):
            raise TypeError("TimingModel.from_calibration requires a name")
        return cls({}, source=source)

    @property
    def by_code_distance(self) -> Mapping[int, Mapping[str, float]]:
        """The inspectable, typed distance-qualified calibration table."""

        return self._by_code_distance

    def __getitem__(self, key):
        return self._entries[key]

    def __iter__(self):
        return iter(self._entries)

    def __len__(self):
        return len(self._entries)

    def __repr__(self):
        origin = f" source={self.source!r}" if self.source else ""
        return f"TimingModel({dict(self._entries)!r}{origin})"


__all__ = ["Duration", "TimingModel", "ms", "ns", "us"]
