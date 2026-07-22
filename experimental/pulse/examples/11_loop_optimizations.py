# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loop optimization passes: LICM and loop strength reduction.

Shows how loop-invariant code motion hoists waveform construction out of
for-loops, and how strength reduction annotates linear phase progressions
for more efficient downstream lowering.
"""

import math

from cudaq_pulse.passes.ir_types import (
    Program,
    Value,
    ValueType,
    Op,
    OpKind,
    _mk,
    _reset_vid_counter,
)
from cudaq_pulse.passes.loop_passes import run_licm, run_loop_strength_reduction


def build_loop_program() -> Program:
    """A for-loop that constructs the same waveform every iteration.

    LICM should hoist the waveform construction before the loop.
    Strength reduction should annotate the constant shift_phase.
    """
    _reset_vid_counter()

    drive_line_0 = _mk(ValueType.DRIVE_LINE, "d0")
    tone_0 = _mk(ValueType.TONE, "t0")

    ops = [
        Op(OpKind.ALLOC_DRIVE, (), (drive_line_0, tone_0), {
            "qubit": 0,
            "freq_hz": 5.0e9
        }),
        Op(OpKind.FOR_LOOP, (), (), {
            "lb": 0,
            "ub": 10,
            "step": 1,
            "var": "i",
            "count": 10
        }),
    ]

    prev_drive, prev_tone = drive_line_0, tone_0
    for _ in range(1):
        waveform = _mk(ValueType.WAVEFORM, "wf_loop")
        drive_out = _mk(ValueType.DRIVE_LINE)
        tone_out = _mk(ValueType.TONE)

        ops.append(
            Op(
                OpKind.MAKE_WAVEFORM, (), (waveform,), {
                    "waveform_type": "gaussian",
                    "duration_vtu": 40,
                    "amplitude": 0.3,
                    "sigma": 10.0
                }))
        ops.append(
            Op(OpKind.DRIVE, (prev_drive, waveform, prev_tone),
               (drive_out, tone_out), {
                   "duration_vtu": 40,
                   "amplitude": 0.3
               }))
        ops.append(
            Op(OpKind.SHIFT_PHASE, (tone_out,), (),
               {"phase_rad": math.pi / 10}))

        prev_drive, prev_tone = drive_out, tone_out

    ops.append(Op(OpKind.END_FOR, (), (), {}))

    return Program(
        name="loop_opt_demo",
        clock_ghz=1.0,
        ops=ops,
        values=[drive_line_0, tone_0],
        qubit_freq_hz={0: 5.0e9},
    )


def find_op_index(ops: list[Op], kind: str) -> int:
    for i, op in enumerate(ops):
        if op.kind == kind:
            return i
    return -1


def main():
    prog = build_loop_program()

    print("=== Original program ===")
    for i, op in enumerate(prog.ops):
        print(f"  [{i:2d}] {op.kind}" +
              (f"  attrs={op.attrs}" if op.attrs else ""))

    # LICM: hoist waveform construction
    hoisted = run_licm(prog)
    print(f"\n=== After LICM ({len(hoisted.ops)} ops) ===")
    for_idx = find_op_index(hoisted.ops, OpKind.FOR_LOOP)
    wf_idx = find_op_index(hoisted.ops, OpKind.MAKE_WAVEFORM)
    for i, op in enumerate(hoisted.ops):
        marker = ""
        if op.kind == OpKind.MAKE_WAVEFORM and i < for_idx:
            marker = "  <-- HOISTED before loop"
        print(f"  [{i:2d}] {op.kind}{marker}")

    if wf_idx < for_idx:
        print(
            "\n  Waveform construction successfully hoisted above the for-loop."
        )
    else:
        print(
            "\n  (No hoisting occurred — waveform may depend on loop variables.)"
        )

    # Strength reduction: annotate constant shift_phase
    reduced = run_loop_strength_reduction(hoisted)
    print(f"\n=== After strength reduction ({len(reduced.ops)} ops) ===")
    for i, op in enumerate(reduced.ops):
        if op.attrs.get("strength_reduced"):
            print(f"  [{i:2d}] {op.kind}  ** strength-reduced: "
                  f"delta={op.attrs['increment_delta']:.4f}, "
                  f"loop_count={op.attrs.get('loop_count', '?')}")
        else:
            print(f"  [{i:2d}] {op.kind}")


if __name__ == "__main__":
    main()
