# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared IR types for the cudaq-pulse pass infrastructure.

Provides a lightweight dataclass-based IR: Value, Op, Program.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ValueType(enum.Enum):
    """Classification of SSA values flowing through the pulse IR."""

    DRIVE_LINE = "drive_line"
    READOUT_LINE = "readout_line"
    TONE = "tone"
    WAVEFORM = "waveform"
    IQ_DATA = "iq_data"
    MEASUREMENT = "measurement"
    QREF = "qref"


class OpKind:
    """Well-known operation kinds in the pulse IR."""

    ALLOC_DRIVE = "alloc_drive_line"
    ALLOC_READOUT = "alloc_readout_line"
    ALLOC_TONE = "alloc_tone"
    DRIVE = "drive"
    READOUT = "readout"
    SYNC = "sync"
    WAIT = "wait"
    SHIFT_PHASE = "shift_phase"
    SET_PHASE = "set_phase"
    MAKE_WAVEFORM = "make_waveform"
    FOR_LOOP = "for_loop"
    END_FOR = "end_for"

    # Operator dialect (pulse_to_operator lowering targets)
    QOP_SPIN = "qop.spin"
    QOP_CONST_SCALAR = "qop.const_scalar"
    QOP_MAKE_PRODUCT = "qop.make_product"
    QOP_MAKE_SUM = "qop.make_sum"
    QOP_CALLBACK_SCALAR = "qop.callback_scalar"
    QOP_LINDBLAD = "qop.lindblad"


@dataclass(frozen=True)
class Value:
    """An SSA value in the pulse IR."""

    vid: int
    vtype: ValueType
    name: str = ""

    def __repr__(self) -> str:
        tag = f":{self.name}" if self.name else ""
        return f"%{self.vid}{tag}:{self.vtype.value}"


@dataclass()
class Op:
    """A single operation in the pulse IR."""

    kind: str
    operands: tuple[Value, ...]
    results: tuple[Value, ...]
    attrs: dict[str, Any]

    def __repr__(self) -> str:
        res = ", ".join(repr(r) for r in self.results)
        ops = ", ".join(repr(o) for o in self.operands)
        return f"{res} = {self.kind}({ops})"


@dataclass()
class Program:
    """A complete pulse program — the unit of compilation."""

    name: str
    clock_ghz: float
    ops: list[Op]
    values: list[Value] = field(default_factory=list)
    qubit_freq_hz: dict[int, float] = field(default_factory=dict)

    @property
    def vtu_to_ns(self) -> float:
        """Virtual time unit to nanoseconds conversion."""
        if self.clock_ghz <= 0:
            raise ValueError(
                f"clock_ghz must be positive, got {self.clock_ghz}")
        return 1.0 / self.clock_ghz

    def op_count(self) -> int:
        return len(self.ops)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_next_vid: int = 0


def _reset_vid_counter(start: int = 0) -> None:
    global _next_vid
    _next_vid = start


def _mk(vtype: ValueType, name: str = "") -> Value:
    """Allocate a fresh Value with a unique vid."""
    global _next_vid
    v = Value(vid=_next_vid, vtype=vtype, name=name)
    _next_vid += 1
    return v


def duration_of(op: Op) -> float:
    """Extract duration from an Op's attrs; returns 0.0 if absent."""
    return float(op.attrs.get("duration_vtu", 0.0))


def line_id_of(op: Op) -> int | None:
    """Extract the line id from an Op's first operand if it is a line value."""
    if op.operands and op.operands[0].vtype in (
            ValueType.DRIVE_LINE,
            ValueType.READOUT_LINE,
    ):
        return op.operands[0].vid
    return None


def tone_id_of(op: Op) -> int | None:
    """Extract the tone id from an Op's operands."""
    for operand in op.operands:
        if operand.vtype == ValueType.TONE:
            return operand.vid
    return None


def is_linear_type(vtype: ValueType) -> bool:
    """Return True if this value type has linear (use-once) semantics."""
    return vtype in (ValueType.DRIVE_LINE, ValueType.READOUT_LINE,
                     ValueType.TONE)


_BARRIER_KINDS = frozenset({
    OpKind.FOR_LOOP,
    OpKind.END_FOR,
    OpKind.SYNC,
    "for_begin",
    "for_end",
})


def is_loop_or_barrier(op: Op) -> bool:
    """Return True for ops that act as scheduling/fusion barriers."""
    return op.kind in _BARRIER_KINDS


def waveform_of(op: Op) -> int | None:
    """Extract waveform vid from an Op's operands."""
    for operand in op.operands:
        if operand.vtype == ValueType.WAVEFORM:
            return operand.vid
    return None


def collect_values(program: Program) -> dict[int, Value]:
    """Build vid -> Value map for all values in a program."""
    table: dict[int, Value] = {}
    for v in program.values:
        table[v.vid] = v
    for op in program.ops:
        for v in op.results:
            table[v.vid] = v
        for v in op.operands:
            table[v.vid] = v
    return table


def clone_program(program: Program) -> Program:
    """Deep-copy a program."""
    return Program(
        name=program.name,
        clock_ghz=program.clock_ghz,
        ops=[
            Op(o.kind, o.operands, o.results, dict(o.attrs))
            for o in program.ops
        ],
        values=list(program.values),
        qubit_freq_hz=dict(program.qubit_freq_hz),
    )
