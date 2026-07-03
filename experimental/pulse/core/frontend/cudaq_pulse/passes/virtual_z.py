# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Virtual-Z gate elimination pass.

Folds shift_phase/set_phase ops into subsequent drive ops by adjusting the
drive's waveform phase. Tracks phase through SSA tone lineage so that
shift_phase(tone_%2) -> tone_%3 followed by drive(..., tone_%3) correctly
absorbs the accumulated phase.
"""

from __future__ import annotations

import math
from typing import Any

from .ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    clone_program,
    is_loop_or_barrier,
    tone_id_of,
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
      - Accumulated phase is applied to the next drive op's waveform phase attr.
      - Phase state tracks through SSA tone lineage (shift_phase produces a new
        tone VID, and the accumulated phase transfers to that new VID).
    """
    result = clone_program(program)

    # Phase state keyed by tone VID: (mode, accumulated_phase)
    tone_phase: dict[int, tuple[str, float]] = {}

    new_ops: list[Op] = []
    skip_indices: set[int] = set()

    for idx, op in enumerate(result.ops):
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
            skip_indices.add(idx)
            continue

        if op.kind == OpKind.SET_PHASE and in_tid is not None:
            phase_val = float(
                op.attrs.get("phase_rad", op.attrs.get("phase", 0.0)))
            tone_phase.pop(in_tid, None)
            target_tid = out_tid if out_tid is not None else in_tid
            tone_phase[target_tid] = ("set", phase_val)
            skip_indices.add(idx)
            continue

        if op.kind == OpKind.DRIVE and in_tid is not None:
            phase_info = tone_phase.pop(in_tid, None)
            if phase_info is not None:
                mode, accumulated = phase_info
                new_attrs = dict(op.attrs)
                existing_phase = float(new_attrs.get("phase", 0.0))

                if mode == "shift":
                    new_attrs["phase"] = _normalize_phase(existing_phase +
                                                          accumulated)
                elif mode == "set":
                    new_attrs["phase"] = _normalize_phase(accumulated)

                new_attrs["virtual_z_applied"] = True
                new_ops.append(
                    Op(
                        kind=op.kind,
                        operands=op.operands,
                        results=op.results,
                        attrs=new_attrs,
                    ))

                if out_tid is not None and out_tid != in_tid:
                    pass
                continue

        if is_loop_or_barrier(op):
            tone_phase.clear()

        new_ops.append(op)

    # Emit residual phase ops that couldn't be folded
    residual_ops: list[Op] = []
    for tid, (mode, phase) in tone_phase.items():
        tone_value = _find_tone_value(result.ops, tid)
        if tone_value is None:
            continue

        if mode == "set":
            residual_ops.append(
                Op(
                    kind=OpKind.SET_PHASE,
                    operands=(tone_value,),
                    results=(),
                    attrs={"phase_rad": _normalize_phase(phase)},
                ))
        elif mode == "shift" and abs(phase) > 1e-12:
            residual_ops.append(
                Op(
                    kind=OpKind.SHIFT_PHASE,
                    operands=(tone_value,),
                    results=(),
                    attrs={"delta_rad": _normalize_phase(phase)},
                ))

    # Rebuild: non-skipped original ops replaced by new_ops, then residuals
    final_ops: list[Op] = []
    new_iter = iter(new_ops)
    for idx in range(len(result.ops)):
        if idx in skip_indices:
            continue
        final_ops.append(next(new_iter))

    final_ops.extend(residual_ops)
    result.ops = final_ops
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
