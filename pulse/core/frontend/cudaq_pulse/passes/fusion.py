# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
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
)


def _is_square_pulse(op: Op) -> bool:
    """Check if an op represents a constant-amplitude (square) drive."""
    if op.kind != OpKind.DRIVE:
        return False
    wf_type = op.attrs.get("waveform_type", "")
    return wf_type in ("square", "constant", "const")


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

    a_line_results = [
        value for value in a.results if value.vtype in (
            ValueType.DRIVE_LINE,
            ValueType.READOUT_LINE,
        )
    ]
    b_line_operands = [
        value for value in b.operands if value.vtype in (
            ValueType.DRIVE_LINE,
            ValueType.READOUT_LINE,
        )
    ]
    if not a_line_results or not b_line_operands or a_line_results[
            0].vid != b_line_operands[0].vid:
        return False

    a_tone_results = [
        value for value in a.results if value.vtype == ValueType.TONE
    ]
    b_tone_operands = [
        value for value in b.operands if value.vtype == ValueType.TONE
    ]
    if not a_tone_results or not b_tone_operands or a_tone_results[
            0].vid != b_tone_operands[0].vid:
        return False

    def complex_amplitude(value) -> complex:
        if isinstance(value, (list, tuple)):
            real = float(value[0]) if value else 0.0
            imaginary = float(value[1]) if len(value) > 1 else 0.0
            return complex(real, imaginary)
        return complex(value)

    amp_a = complex_amplitude(a.attrs.get("amplitude", 1.0))
    amp_b = complex_amplitude(b.attrs.get("amplitude", 1.0))
    if abs(amp_a - amp_b) > 1e-12:
        return False

    for key in ("phase", "phase_offset", "frame_phase_offset"):
        phase_a = float(a.attrs.get(key, 0.0))
        phase_b = float(b.attrs.get(key, 0.0))
        if abs(phase_a - phase_b) > 1e-12:
            return False

    return True


def _fuse_ops(a: Op, b: Op, waveform: Value) -> Op:
    """Create a fused op from two adjacent compatible drive ops."""
    dur_a = duration_of(a)
    dur_b = duration_of(b)
    merged_attrs = dict(a.attrs)
    merged_attrs["duration_vtu"] = dur_a + dur_b
    merged_attrs["fused"] = True
    merged_attrs["fused_count"] = a.attrs.get("fused_count", 1) + 1

    return Op(
        kind=OpKind.DRIVE,
        operands=(a.operands[0], waveform, a.operands[2]),
        results=b.results,
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
    next_vid = 1 + max(
        (value.vid
         for op in result.ops
         for value in (*op.operands, *op.results)),
        default=-1,
    )
    new_ops: list[Op] = []
    drive_by_output_line: dict[int, int] = {}

    for op in result.ops:
        input_lines = [
            value for value in op.operands if value.vtype in (
                ValueType.DRIVE_LINE,
                ValueType.READOUT_LINE,
            )
        ]
        previous_index = drive_by_output_line.get(
            input_lines[0].vid) if input_lines else None

        if (op.kind == OpKind.DRIVE and previous_index is not None and
                _can_fuse(new_ops[previous_index], op)):
            previous = new_ops[previous_index]
            duration = duration_of(previous) + duration_of(op)
            waveform = Value(next_vid, ValueType.WAVEFORM,
                             f"fused_square_{next_vid}")
            next_vid += 1
            waveform_op = Op(
                OpKind.MAKE_WAVEFORM,
                (),
                (waveform,),
                {
                    "waveform_type": "square",
                    "duration_vtu": duration,
                    "amplitude": previous.attrs.get("amplitude", 0.0),
                },
            )
            for line_vid, index in list(drive_by_output_line.items()):
                if index >= previous_index:
                    drive_by_output_line[line_vid] = index + 1
            new_ops.insert(previous_index, waveform_op)
            previous_index += 1
            fused = _fuse_ops(previous, op, waveform)
            new_ops[previous_index] = fused
            for value in fused.results:
                if value.vtype in (ValueType.DRIVE_LINE,
                                   ValueType.READOUT_LINE):
                    drive_by_output_line[value.vid] = previous_index
            continue

        new_index = len(new_ops)
        new_ops.append(op)
        if op.kind == OpKind.DRIVE:
            for value in op.results:
                if value.vtype in (ValueType.DRIVE_LINE,
                                   ValueType.READOUT_LINE):
                    drive_by_output_line[value.vid] = new_index

    result.ops = new_ops
    return result
