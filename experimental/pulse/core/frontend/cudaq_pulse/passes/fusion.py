# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pulse fusion pass.

Merges adjacent same-line constant-amplitude (square) pulses into a single
longer pulse when constraints are satisfied.
"""

from __future__ import annotations

from .ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    clone_program,
    duration_of,
    is_loop_or_barrier,
    line_id_of,
    tone_id_of,
)


def _is_square_pulse(op: Op) -> bool:
    """Check if an op represents a constant-amplitude (square) drive."""
    if op.kind != OpKind.DRIVE:
        return False
    wf_type = op.attrs.get("waveform_type", "")
    return wf_type in ("square", "constant", "const", "")


def _can_fuse(a: Op, b: Op) -> bool:
    """Check if two drive ops can be fused.

    Conditions:
      - Both are square pulses
      - Same line
      - Same tone
      - Same amplitude
      - No phase difference (or both have identical phase)
    """
    if not (_is_square_pulse(a) and _is_square_pulse(b)):
        return False

    if line_id_of(a) != line_id_of(b):
        return False

    if tone_id_of(a) != tone_id_of(b):
        return False

    amp_a = a.attrs.get("amplitude", 1.0)
    amp_b = b.attrs.get("amplitude", 1.0)
    if abs(float(amp_a) - float(amp_b)) > 1e-12:
        return False

    phase_a = float(a.attrs.get("phase", 0.0))
    phase_b = float(b.attrs.get("phase", 0.0))
    if abs(phase_a - phase_b) > 1e-12:
        return False

    return True


def _fuse_ops(a: Op, b: Op) -> Op:
    """Create a fused op from two adjacent compatible drive ops."""
    dur_a = duration_of(a)
    dur_b = duration_of(b)
    merged_attrs = dict(a.attrs)
    merged_attrs["duration_vtu"] = dur_a + dur_b
    merged_attrs["fused"] = True
    merged_attrs["fused_count"] = a.attrs.get("fused_count", 1) + 1

    return Op(
        kind=OpKind.DRIVE,
        operands=a.operands,
        results=a.results,
        attrs=merged_attrs,
    )


def run_fusion(program: Program) -> Program:
    """Merge adjacent same-line constant-amplitude pulses into single longer pulses.

    Only merges when:
      - Same line
      - Same tone
      - Same amplitude
      - No intervening ops on the line between the two drives
    """
    result = clone_program(program)
    ops = result.ops

    if len(ops) < 2:
        return result

    changed = True
    while changed:
        changed = False
        new_ops: list[Op] = []
        last_drive_per_line: dict[int, int] = {}
        intervened: set[int] = set()

        i = 0
        while i < len(ops):
            op = ops[i]
            lid = line_id_of(op)

            if is_loop_or_barrier(op):
                intervened.update(last_drive_per_line.keys())
            elif op.kind == OpKind.DRIVE and _is_square_pulse(op):
                lid_key = lid if lid is not None else -1
                if lid_key in last_drive_per_line and lid_key not in intervened:
                    prev_idx = last_drive_per_line[lid_key]
                    prev_op = new_ops[prev_idx]
                    if _can_fuse(prev_op, op):
                        new_ops[prev_idx] = _fuse_ops(prev_op, op)
                        changed = True
                        i += 1
                        continue

                last_drive_per_line[lid_key] = len(new_ops)
                intervened.discard(lid_key)
            elif lid is not None and op.kind not in (OpKind.MAKE_WAVEFORM,
                                                     OpKind.ALLOC_TONE):
                intervened.add(lid)

            new_ops.append(op)
            i += 1

        ops = new_ops

    result.ops = ops
    return result
