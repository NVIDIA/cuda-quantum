# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Dependency-correct scheduling passes for the lightweight pulse IR."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .ir_types import OpKind, Program, ValueType, duration_of, tone_id_of, waveform_of


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
    """Hardware resource constraints for resource-constrained scheduling."""

    max_concurrent_drives: int = 4
    max_concurrent_readouts: int = 2
    readout_latency_vtu: float = 0.0
    line_switch_penalty_vtu: float = 0.0
    qubit_connectivity: dict[int, list[int]] = field(default_factory=dict)


_TIMED_KINDS = frozenset({OpKind.DRIVE, OpKind.READOUT, OpKind.WAIT})
_LINE_TYPES = frozenset({ValueType.DRIVE_LINE, ValueType.READOUT_LINE})


def _validate_machine(machine: MachineModel) -> None:
    if machine.max_concurrent_drives <= 0:
        raise ValueError("max_concurrent_drives must be positive")
    if machine.max_concurrent_readouts <= 0:
        raise ValueError("max_concurrent_readouts must be positive")
    if machine.readout_latency_vtu < 0:
        raise ValueError("readout_latency_vtu cannot be negative")
    if machine.line_switch_penalty_vtu < 0:
        raise ValueError("line_switch_penalty_vtu cannot be negative")


def _line_roots(program: Program) -> dict[int, int]:
    """Map every line SSA value to its physical line allocation."""
    roots: dict[int, int] = {}
    for op in program.ops:
        line_operands = [
            value for value in op.operands if value.vtype in _LINE_TYPES
        ]
        line_results = [
            value for value in op.results if value.vtype in _LINE_TYPES
        ]
        if op.kind in (OpKind.ALLOC_DRIVE, OpKind.ALLOC_READOUT):
            for result in line_results:
                roots[result.vid] = result.vid
        elif op.kind == OpKind.SYNC:
            for operand, result in zip(line_operands, line_results):
                roots[result.vid] = roots.get(operand.vid, operand.vid)
        elif line_operands:
            root = roots.get(line_operands[0].vid, line_operands[0].vid)
            for result in line_results:
                roots[result.vid] = root
    return roots


def _event(program: Program, roots: dict[int, int], index: int, start: float,
           duration: float) -> ScheduledEvent:
    op = program.ops[index]
    line = next((value for value in op.operands if value.vtype in _LINE_TYPES),
                None)
    return ScheduledEvent(
        op_index=index,
        kind=op.kind,
        start_vtu=start,
        duration_vtu=duration,
        line_id=roots.get(line.vid, line.vid) if line is not None else None,
        tone_id=tone_id_of(op),
        waveform_id=waveform_of(op),
        attrs=dict(op.attrs),
    )


def _annotate(program: Program, events: list[ScheduledEvent]) -> None:
    for event in events:
        if event.kind in _TIMED_KINDS:
            op = program.ops[event.op_index]
            op.attrs["start_vtu"] = event.start_vtu
            op.attrs["duration_vtu"] = event.duration_vtu


def _schedule_forward(
        program: Program,
        machine: MachineModel | None = None) -> list[ScheduledEvent]:
    roots = _line_roots(program)
    ready: dict[int, float] = {}
    events: list[ScheduledEvent] = []
    drive_lanes = [
        0.0
    ] * machine.max_concurrent_drives if machine is not None else []
    readout_lanes = [
        0.0
    ] * machine.max_concurrent_readouts if machine is not None else []

    for index, op in enumerate(program.ops):
        operand_ready = max(
            (ready.get(value.vid, 0.0) for value in op.operands), default=0.0)
        duration = duration_of(op) if op.kind in _TIMED_KINDS else 0.0
        start = operand_ready
        result_ready = start + duration

        if op.kind == OpKind.SYNC:
            result_ready = operand_ready
        elif machine is not None and op.kind in (OpKind.DRIVE, OpKind.READOUT):
            lanes = drive_lanes if op.kind == OpKind.DRIVE else readout_lanes
            lane = min(range(len(lanes)), key=lanes.__getitem__)
            start = max(start, lanes[lane])
            result_ready = start + duration
            lanes[lane] = result_ready + machine.line_switch_penalty_vtu
            if op.kind == OpKind.READOUT:
                result_ready += machine.readout_latency_vtu

        events.append(_event(program, roots, index, start, duration))
        for result in op.results:
            ready[result.vid] = result_ready

    _annotate(program, events)
    return events


def _schedule_backward(program: Program,
                       forward: list[ScheduledEvent]) -> list[ScheduledEvent]:
    makespan = max((event.end_vtu for event in forward), default=0.0)
    roots = _line_roots(program)
    latest: dict[int, float] = {}
    reversed_events: list[ScheduledEvent] = []

    for index in range(len(program.ops) - 1, -1, -1):
        op = program.ops[index]
        duration = duration_of(op) if op.kind in _TIMED_KINDS else 0.0
        end = min((latest.get(value.vid, makespan) for value in op.results),
                  default=makespan)
        start = end - duration
        reversed_events.append(_event(program, roots, index, start, duration))
        for operand in op.operands:
            latest[operand.vid] = min(latest.get(operand.vid, makespan), start)

    events = list(reversed(reversed_events))
    _annotate(program, events)
    return events


def _compute_metrics(events: list[ScheduledEvent],
                     program: Program) -> ScheduleMetrics:
    total_length = max((event.end_vtu for event in events), default=0.0)
    per_line: dict[int, float] = {}
    active: dict[int, float] = {}
    first_start: dict[int, float] = {}
    for event in events:
        if event.line_id is None or event.duration_vtu <= 0:
            continue
        per_line[event.line_id] = max(per_line.get(event.line_id, 0.0),
                                      event.end_vtu)
        first_start[event.line_id] = min(
            first_start.get(event.line_id, event.start_vtu), event.start_vtu)
        active[event.line_id] = active.get(event.line_id,
                                           0.0) + event.duration_vtu
    available = sum(per_line[line] - first_start[line] for line in per_line)
    active_total = sum(active.values())
    idle_total = max(0.0, available - active_total)
    return ScheduleMetrics(
        total_length_vtu=total_length,
        total_length_ns=total_length * program.vtu_to_ns,
        per_line_length_vtu=per_line,
        op_count=program.op_count(),
        idle_total_vtu=idle_total,
        idle_fraction=idle_total / available if available else 0.0,
    )


def schedule_asap(
        program: Program) -> tuple[list[ScheduledEvent], ScheduleMetrics]:
    """Schedule operations at their earliest dependency-ready times."""
    started = time.perf_counter()
    events = _schedule_forward(program)
    metrics = _compute_metrics(events, program)
    metrics.compile_time_ms = (time.perf_counter() - started) * 1000.0
    return events, metrics


def schedule_alap(
        program: Program) -> tuple[list[ScheduledEvent], ScheduleMetrics]:
    """Schedule operations as late as dependencies allow at the ASAP makespan."""
    started = time.perf_counter()
    events = _schedule_backward(program, _schedule_forward(program))
    metrics = _compute_metrics(events, program)
    metrics.compile_time_ms = (time.perf_counter() - started) * 1000.0
    return events, metrics


def schedule_rcp(
    program: Program,
    machine: MachineModel | None = None,
) -> tuple[list[ScheduledEvent], ScheduleMetrics]:
    """List-schedule with interval-correct drive and readout resource limits."""
    machine = machine or MachineModel()
    _validate_machine(machine)
    started = time.perf_counter()
    events = _schedule_forward(program, machine)
    metrics = _compute_metrics(events, program)
    metrics.compile_time_ms = (time.perf_counter() - started) * 1000.0
    return events, metrics
