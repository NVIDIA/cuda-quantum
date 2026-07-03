# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scheduling passes for the pulse IR.

Implements ASAP, ALAP, and resource-constrained (RCP) scheduling.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    duration_of,
    line_id_of,
    tone_id_of,
    waveform_of,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass()
class ScheduledEvent:
    """A scheduled pulse event with absolute timing."""

    op_index: int
    kind: str
    start_vtu: float
    duration_vtu: float
    line_id: int | None = None
    tone_id: int | None = None
    waveform_id: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def end_vtu(self) -> float:
        return self.start_vtu + self.duration_vtu


@dataclass()
class ScheduleMetrics:
    """Summary statistics for a computed schedule."""

    total_length_vtu: float = 0.0
    total_length_ns: float = 0.0
    per_line_length_vtu: dict[int, float] = field(default_factory=dict)
    op_count: int = 0
    compile_time_ms: float = 0.0
    idle_total_vtu: float = 0.0
    idle_fraction: float = 0.0


@dataclass()
class MachineModel:
    """Hardware resource constraints for RCP scheduling."""

    max_concurrent_drives: int = 4
    max_concurrent_readouts: int = 2
    readout_latency_vtu: float = 0.0
    line_switch_penalty_vtu: float = 0.0
    qubit_connectivity: dict[int, list[int]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal: ASAP scheduling core
# ---------------------------------------------------------------------------


def _schedule_asap(program: Program) -> list[ScheduledEvent]:
    """Schedule all ops as soon as possible respecting line ordering.

    Loops: body ops are scheduled once per the ASAP walk; at END_FOR the
    per-line cost accumulated inside the body is multiplied by (count-1)
    and added to the line clocks, so total_length accounts for all iterations.
    """
    events: list[ScheduledEvent] = []
    line_clocks: dict[int, float] = {}
    global_clock: float = 0.0
    loop_stack: list[tuple[int, int, dict[int, float]]] = []

    for idx, op in enumerate(program.ops):
        lid = line_id_of(op)
        dur = duration_of(op)

        if op.kind == OpKind.SYNC:
            sync_lines = [
                v.vid
                for v in op.operands
                if v.vtype in (ValueType.DRIVE_LINE, ValueType.READOUT_LINE)
            ]
            if sync_lines:
                sync_time = max(line_clocks.get(l, 0.0) for l in sync_lines)
                for l in sync_lines:
                    line_clocks[l] = sync_time
            continue

        if op.kind == OpKind.FOR_LOOP:
            count = op.attrs.get("ub", op.attrs.get("count", 1))
            snap = dict(line_clocks)
            loop_stack.append((idx, count, snap))
            events.append(
                ScheduledEvent(
                    op_index=idx,
                    kind=op.kind,
                    start_vtu=global_clock,
                    duration_vtu=0.0,
                    line_id=lid,
                    attrs=dict(op.attrs),
                ))
            continue

        if op.kind == OpKind.END_FOR:
            if loop_stack:
                _, count, snap = loop_stack.pop()
                remaining = max(0, int(count) - 1)
                if remaining > 0:
                    for l in line_clocks:
                        body_cost = line_clocks[l] - snap.get(l, 0.0)
                        if body_cost > 0:
                            line_clocks[l] += body_cost * remaining
                    global_clock = max(
                        line_clocks.values()) if line_clocks else global_clock
            events.append(
                ScheduledEvent(
                    op_index=idx,
                    kind=op.kind,
                    start_vtu=global_clock,
                    duration_vtu=0.0,
                    line_id=lid,
                    attrs=dict(op.attrs),
                ))
            continue

        if op.kind in (OpKind.MAKE_WAVEFORM, OpKind.ALLOC_DRIVE,
                       OpKind.ALLOC_READOUT, OpKind.ALLOC_TONE):
            events.append(
                ScheduledEvent(
                    op_index=idx,
                    kind=op.kind,
                    start_vtu=global_clock,
                    duration_vtu=0.0,
                    line_id=lid,
                    tone_id=tone_id_of(op),
                    waveform_id=waveform_of(op),
                    attrs=dict(op.attrs),
                ))
            continue

        start = line_clocks.get(lid, 0.0) if lid is not None else global_clock
        events.append(
            ScheduledEvent(
                op_index=idx,
                kind=op.kind,
                start_vtu=start,
                duration_vtu=dur,
                line_id=lid,
                tone_id=tone_id_of(op),
                waveform_id=waveform_of(op),
                attrs=dict(op.attrs),
            ))

        if lid is not None:
            line_clocks[lid] = start + dur
            global_clock = max(global_clock, start + dur)
        elif dur > 0:
            global_clock += dur

    return events


# ---------------------------------------------------------------------------
# Internal: ASAP to ALAP conversion
# ---------------------------------------------------------------------------


def _to_alap(events: list[ScheduledEvent],
             total_length: float) -> list[ScheduledEvent]:
    """Shift events as late as possible within the total schedule length."""
    if not events:
        return events

    line_latest: dict[int | None, float] = {}
    alap_events: list[ScheduledEvent] = []

    for ev in reversed(events):
        if ev.duration_vtu == 0.0:
            alap_events.append(ev)
            continue

        lid = ev.line_id
        latest_end = line_latest.get(lid, total_length)
        new_start = latest_end - ev.duration_vtu

        alap_events.append(
            ScheduledEvent(
                op_index=ev.op_index,
                kind=ev.kind,
                start_vtu=max(new_start, 0.0),
                duration_vtu=ev.duration_vtu,
                line_id=ev.line_id,
                tone_id=ev.tone_id,
                waveform_id=ev.waveform_id,
                attrs=ev.attrs,
            ))
        line_latest[lid] = max(new_start, 0.0)

    alap_events.reverse()
    return alap_events


# ---------------------------------------------------------------------------
# Internal: RCP scheduling (resource-constrained)
# ---------------------------------------------------------------------------


def _rcp_repair_fixpoint(
    events: list[ScheduledEvent],
    machine: MachineModel,
    max_iterations: int = 100,
) -> list[ScheduledEvent]:
    """Iteratively repair resource violations until a fixpoint or limit."""
    import logging
    _log = logging.getLogger(__name__)
    for i in range(max_iterations):
        violations = _find_resource_violations(events, machine)
        if not violations:
            break
        events = _repair_violations(events, violations, machine)
    else:
        remaining = _find_resource_violations(events, machine)
        if remaining:
            _log.warning(
                "RCP scheduling did not converge after %d iterations; "
                "%d resource violations remain", max_iterations, len(remaining))
    return events


def _find_resource_violations(
    events: list[ScheduledEvent],
    machine: MachineModel,
) -> list[tuple[int, str]]:
    """Find events that violate resource constraints. Returns (event_idx, reason)."""
    violations: list[tuple[int, str]] = []

    time_slots: dict[float, list[int]] = {}
    for i, ev in enumerate(events):
        if ev.duration_vtu == 0.0:
            continue
        t = ev.start_vtu
        if t not in time_slots:
            time_slots[t] = []
        time_slots[t].append(i)

    for t, indices in time_slots.items():
        drives = [i for i in indices if events[i].kind == OpKind.DRIVE]
        readouts = [i for i in indices if events[i].kind == OpKind.READOUT]

        if len(drives) > machine.max_concurrent_drives:
            for extra in drives[machine.max_concurrent_drives:]:
                violations.append((extra, "drive_overflow"))

        if len(readouts) > machine.max_concurrent_readouts:
            for extra in readouts[machine.max_concurrent_readouts:]:
                violations.append((extra, "readout_overflow"))

    return violations


def _repair_violations(
    events: list[ScheduledEvent],
    violations: list[tuple[int, str]],
    machine: MachineModel,
) -> list[ScheduledEvent]:
    """Shift violating events later to fix resource conflicts."""
    result = list(events)
    for ev_idx, reason in violations:
        ev = result[ev_idx]
        penalty = machine.line_switch_penalty_vtu if machine.line_switch_penalty_vtu > 0 else ev.duration_vtu
        result[ev_idx] = ScheduledEvent(
            op_index=ev.op_index,
            kind=ev.kind,
            start_vtu=ev.start_vtu + penalty,
            duration_vtu=ev.duration_vtu,
            line_id=ev.line_id,
            tone_id=ev.tone_id,
            waveform_id=ev.waveform_id,
            attrs=ev.attrs,
        )
    return result


def _repair_ssa_readiness(events: list[ScheduledEvent],
                          program: Program) -> list[ScheduledEvent]:
    """Ensure all operands are ready before their consumers execute."""
    result_ready: dict[int, float] = {}

    for ev in events:
        op = program.ops[ev.op_index]
        for v in op.results:
            result_ready[v.vid] = ev.start_vtu + ev.duration_vtu

    adjusted = []
    for ev in events:
        op = program.ops[ev.op_index]
        min_start = ev.start_vtu
        for v in op.operands:
            if v.vid in result_ready:
                min_start = max(min_start, result_ready[v.vid])

        if min_start > ev.start_vtu:
            ev = ScheduledEvent(
                op_index=ev.op_index,
                kind=ev.kind,
                start_vtu=min_start,
                duration_vtu=ev.duration_vtu,
                line_id=ev.line_id,
                tone_id=ev.tone_id,
                waveform_id=ev.waveform_id,
                attrs=ev.attrs,
            )
        adjusted.append(ev)

    return adjusted


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------


def _compute_metrics(events: list[ScheduledEvent],
                     program: Program) -> ScheduleMetrics:
    """Compute schedule metrics from a list of events."""
    if not events:
        return ScheduleMetrics(op_count=program.op_count())

    total_length = max((ev.end_vtu for ev in events if ev.duration_vtu > 0),
                       default=0.0)
    per_line: dict[int, float] = {}
    active_per_line: dict[int, float] = {}

    for ev in events:
        if ev.line_id is not None and ev.duration_vtu > 0:
            per_line[ev.line_id] = max(per_line.get(ev.line_id, 0.0),
                                       ev.end_vtu)
            active_per_line[ev.line_id] = active_per_line.get(
                ev.line_id, 0.0) + ev.duration_vtu

    total_active = sum(active_per_line.values())
    total_possible = sum(per_line.values()) if per_line else 0.0
    idle_total = total_possible - total_active
    idle_fraction = idle_total / total_possible if total_possible > 0 else 0.0

    return ScheduleMetrics(
        total_length_vtu=total_length,
        total_length_ns=total_length * program.vtu_to_ns,
        per_line_length_vtu=per_line,
        op_count=program.op_count(),
        idle_total_vtu=idle_total,
        idle_fraction=idle_fraction,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def schedule_asap(
        program: Program) -> tuple[list[ScheduledEvent], ScheduleMetrics]:
    """Schedule the program using as-soon-as-possible policy."""
    t0 = time.perf_counter()
    events = _schedule_asap(program)
    events = _repair_ssa_readiness(events, program)
    metrics = _compute_metrics(events, program)
    metrics.compile_time_ms = (time.perf_counter() - t0) * 1000.0
    return events, metrics


def schedule_alap(
        program: Program) -> tuple[list[ScheduledEvent], ScheduleMetrics]:
    """Schedule the program using as-late-as-possible policy."""
    t0 = time.perf_counter()
    asap_events = _schedule_asap(program)
    asap_events = _repair_ssa_readiness(asap_events, program)
    total = max((ev.end_vtu for ev in asap_events if ev.duration_vtu > 0),
                default=0.0)
    events = _to_alap(asap_events, total)
    metrics = _compute_metrics(events, program)
    metrics.compile_time_ms = (time.perf_counter() - t0) * 1000.0
    return events, metrics


def schedule_rcp(
    program: Program,
    machine: MachineModel | None = None,
) -> tuple[list[ScheduledEvent], ScheduleMetrics]:
    """Schedule the program with resource-constrained placement."""
    if machine is None:
        machine = MachineModel()
    t0 = time.perf_counter()
    events = _schedule_asap(program)
    events = _repair_ssa_readiness(events, program)
    events = _rcp_repair_fixpoint(events, machine)
    events = _repair_ssa_readiness(events, program)
    metrics = _compute_metrics(events, program)
    metrics.compile_time_ms = (time.perf_counter() - t0) * 1000.0
    return events, metrics
