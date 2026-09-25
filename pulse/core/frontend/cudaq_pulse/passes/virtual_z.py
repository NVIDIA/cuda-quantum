# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Virtual-Z gate elimination pass.

Folds shift_phase/set_phase ops into downstream drive ops by adjusting the
drive's persistent frame phase. Tracks phase through SSA tone lineage so that
shift_phase(tone_%2) -> tone_%3 followed by drive(..., tone_%3) correctly
absorbs the accumulated phase.
"""

from __future__ import annotations

import math
from .ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    clone_program,
)


def _normalize_phase(phase: float) -> float:
    """Normalize phase to [0, 2*pi)."""
    return phase % (2.0 * math.pi)


def _tone_lineage(op: Op) -> tuple[int | None, int | None]:
    """Return (input_tone_vid, output_tone_vid) for an op."""
    in_tid = None
    out_tid = None
    for v in op.operands:
        if v.vtype == ValueType.TONE:
            in_tid = v.vid
            break
    for v in op.results:
        if v.vtype == ValueType.TONE:
            out_tid = v.vid
            break
    return in_tid, out_tid


def run_virtual_z(program: Program) -> Program:
    """Fold shift_phase/set_phase ops into subsequent drive ops.

    Rules:
      - Two consecutive shift_phase on the same tone merge into one.
      - set_phase followed by shift_phase -> single set_phase.
      - Accumulated phase is applied to every downstream drive in that frame.
      - Phase state tracks through SSA tone lineage (shift_phase produces a new
        tone VID, and the accumulated phase transfers to that new VID).
    """
    result = clone_program(program)

    # Phase state keyed by tone VID: (mode, accumulated_phase)
    tone_phase: dict[int, tuple[str, float]] = {}
    tone_alias: dict[int, int] = {}

    new_ops: list[Op] = []

    def resolved_tone(tone_vid: int) -> int:
        seen: set[int] = set()
        while tone_vid in tone_alias and tone_vid not in seen:
            seen.add(tone_vid)
            replacement = tone_alias[tone_vid]
            if replacement == tone_vid:
                break
            tone_vid = replacement
        return tone_vid

    for op in result.ops:
        in_tid, out_tid = _tone_lineage(op)

        if op.kind == OpKind.SHIFT_PHASE and in_tid is not None:
            delta = float(
                op.attrs.get(
                    "delta_rad",
                    op.attrs.get("phase", op.attrs.get("phase_rad", 0.0))))
            current = tone_phase.pop(in_tid, None)

            if current is None:
                new_phase = ("shift", delta)
            elif current[0] == "shift":
                new_phase = ("shift", current[1] + delta)
            else:
                new_phase = ("set", current[1] + delta)

            target_tid = out_tid if out_tid is not None else in_tid
            tone_phase[target_tid] = new_phase
            tone_alias[target_tid] = resolved_tone(in_tid)
            continue

        if op.kind == OpKind.SET_PHASE and in_tid is not None:
            phase_val = float(
                op.attrs.get("phase_rad", op.attrs.get("phase", 0.0)))
            tone_phase.pop(in_tid, None)
            target_tid = out_tid if out_tid is not None else in_tid
            tone_phase[target_tid] = ("set", phase_val)
            tone_alias[target_tid] = resolved_tone(in_tid)
            continue

        operands = list(op.operands)
        if in_tid is not None:
            resolved_vid = resolved_tone(in_tid)
            if resolved_vid != in_tid:
                resolved_value = _find_tone_value(result.ops, resolved_vid)
                if resolved_value is None:
                    raise ValueError(
                        f"cannot resolve tone %{resolved_vid} after virtual-Z")
                operands = [
                    resolved_value if value.vid == in_tid else value
                    for value in operands
                ]

        new_attrs = dict(op.attrs)
        if op.kind in (OpKind.DRIVE, OpKind.READOUT) and in_tid is not None:
            phase_info = tone_phase.pop(in_tid, None)
            if phase_info is not None:
                mode, accumulated = phase_info
                existing_phase = float(new_attrs.get("frame_phase_offset", 0.0))

                if mode == "shift":
                    new_attrs["frame_phase_offset"] = _normalize_phase(
                        existing_phase + accumulated)
                elif mode == "set":
                    new_attrs["frame_phase_offset"] = _normalize_phase(
                        accumulated)

                new_attrs["virtual_z_applied"] = True

                if out_tid is not None:
                    tone_phase[out_tid] = phase_info

        elif in_tid is not None and out_tid is not None:
            phase_info = tone_phase.pop(in_tid, None)
            if phase_info is not None:
                tone_phase[out_tid] = phase_info

        if out_tid is not None:
            # Kept operations define a new, valid SSA tone. Future aliases
            # should stop here rather than bypassing the operation.
            tone_alias[out_tid] = out_tid

        new_ops.append(Op(op.kind, tuple(operands), op.results, new_attrs))

    # A phase operation whose tone never reaches a drive/readout is dead. No
    # residual operation is needed because pulse kernels do not return tones.
    result.ops = new_ops
    return result


def _find_tone_value(ops: list[Op], tid: int) -> Value | None:
    """Find the Value instance for a given tone vid."""
    for op in ops:
        for v in op.results:
            if v.vid == tid and v.vtype == ValueType.TONE:
                return v
        for v in op.operands:
            if v.vid == tid and v.vtype == ValueType.TONE:
                return v
    return None
