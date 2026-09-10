# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tick intervals: the physical gate layers of a scheduled P3 build.

A *tick* is both the boundary the frontend authors (``cudaq.logical.tick``) and the time
interval it opens -- the set of physical gates that run together in one step. A
scheduled physical build already lays its events into non-overlapping time steps
(the typed :func:`cudaq.logical.schedule`, which orders events by their carrier-state SSA
edges and enforces carrier mutual exclusion), so the tick intervals are read
directly off that schedule: the entries sharing a start time are one tick, and
their carriers are the ones active in it.

This is the typed replacement for guessing moment membership in a backend
emitter: :func:`ticks` is what ``cudaq.logical.add_idle`` (idle carriers per tick) and
Stim TICK serialization both consume, derived from one source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schedule import PhysicalSchedule, schedule as _schedule


@dataclass(frozen=True, slots=True)
class Tick:
    """One physical gate layer: the carriers gated (and idle) in a time step."""

    index: int
    start_ns: float
    events: tuple[str, ...]
    active: tuple[str, ...]
    idle: tuple[str, ...]


def _is_carrier(resource: str) -> bool:
    # Synthetic scheduler resources (e.g. "control:tick21") carry a ":"; real
    # physical carriers are concrete resource-class/index binding keys.
    return ":" not in resource


def ticks(build) -> tuple[Tick, ...]:
    """The physical gate layers of a build, in time order.

    ``build`` is a physical :class:`~cudaq.logical.Build` (scheduled on demand) or an
    existing :class:`~cudaq.logical.PhysicalSchedule`. Each returned :class:`Tick` gives
    the events sharing that start time, the carriers they gate (``active``), and
    the in-play carriers that sit idle through the step (``idle`` = carriers whose
    lifetime spans the step but which are not gated in it).
    """
    physical_schedule = (build.canonical() if isinstance(
        build, PhysicalSchedule) else _schedule(build))
    # Real duration-bearing carrier events only. Zero-duration acquire/release,
    # fence, and state-carrying moment barriers constrain the schedule but are
    # not themselves physical gate layers.
    events = tuple((entry, tuple(r
                                 for r in entry.resources
                                 if _is_carrier(r)))
                   for entry in physical_schedule.entries
                   if entry.duration_ns > 0.0)
    events = tuple((entry, carriers) for entry, carriers in events if carriers)

    # Each carrier's lifetime [first start, last finish]; a carrier is "in play"
    # across that span and idle in any spanned step where it is not gated.
    first_seen: dict[str, float] = {}
    last_seen: dict[str, float] = {}
    for entry, carriers in events:
        for carrier in carriers:
            first_seen[carrier] = min(first_seen.get(carrier, entry.start_ns),
                                      entry.start_ns)
            last_seen[carrier] = max(last_seen.get(carrier, entry.finish_ns),
                                     entry.finish_ns)

    by_start: dict[float, list] = {}
    for entry, carriers in events:
        by_start.setdefault(entry.start_ns, []).append((entry, carriers))

    result = []
    for index, start in enumerate(sorted(by_start)):
        layer = by_start[start]
        active = tuple(sorted({c for _, carriers in layer for c in carriers}))
        active_set = set(active)
        idle = tuple(
            sorted(carrier for carrier in first_seen
                   if carrier not in active_set and
                   first_seen[carrier] <= start <= last_seen[carrier]))
        result.append(
            Tick(
                index=index,
                start_ns=float(start),
                events=tuple(entry.event_id for entry, _ in layer),
                active=active,
                idle=idle,
            ))
    return tuple(result)


__all__ = ["Tick", "ticks"]
