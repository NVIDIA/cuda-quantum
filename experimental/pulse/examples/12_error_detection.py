# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verification error detection: intentionally bad programs.

Demonstrates every category of error the pulse-verify pass can catch:
  - Linearity violations (double consumption, unconsumed values)
  - Backward time travel
  - Drive exclusivity overlap
  - Cross-resonance miscalibration heuristic
"""

from cudaq_pulse.passes.ir_types import (
    Program,
    Value,
    ValueType,
    Op,
    OpKind,
    _mk,
    _reset_vid_counter,
)
from cudaq_pulse.passes.verify import (
    verify,
    check_linearity,
    check_monotone_time,
    check_drive_exclusivity,
    check_cr_miscalibration,
    LinearityViolation,
    BackwardTimeTravelError,
    UnintentionalOverlapError,
    CrossResonanceMiscalibrationError,
)

# ── 1. Linearity violation: double consumption ──────────────────────────────


def linearity_double_consume() -> Program:
    """Use the same linear drive_line value in two drive ops."""
    _reset_vid_counter()

    drive_line_0 = _mk(ValueType.DRIVE_LINE, "d0")
    tone_0 = _mk(ValueType.TONE, "t0")
    waveform = _mk(ValueType.WAVEFORM, "wf")
    drive_out_1 = _mk(ValueType.DRIVE_LINE)
    tone_out_1 = _mk(ValueType.TONE)
    drive_out_2 = _mk(ValueType.DRIVE_LINE)
    tone_out_2 = _mk(ValueType.TONE)

    ops = [
        Op(OpKind.ALLOC_DRIVE, (), (drive_line_0, tone_0), {"qubit": 0}),
        Op(OpKind.MAKE_WAVEFORM, (), (waveform,), {"duration_vtu": 40}),
        Op(OpKind.DRIVE, (drive_line_0, waveform, tone_0),
           (drive_out_1, tone_out_1), {"duration_vtu": 40}),
        Op(OpKind.DRIVE, (drive_line_0, waveform, tone_0),
           (drive_out_2, tone_out_2), {"duration_vtu": 40}),
    ]

    return Program(name="bad_linearity",
                   clock_ghz=1.0,
                   ops=ops,
                   values=[drive_line_0, tone_0, waveform])


# ── 2. Backward time travel ─────────────────────────────────────────────────


def backward_time() -> Program:
    """An op references a start time earlier than its predecessor's end."""
    _reset_vid_counter()

    drive_line_0 = _mk(ValueType.DRIVE_LINE, "d0")
    tone_0 = _mk(ValueType.TONE, "t0")
    waveform = _mk(ValueType.WAVEFORM, "wf")
    drive_line_1 = _mk(ValueType.DRIVE_LINE)
    tone_1 = _mk(ValueType.TONE)
    drive_line_2 = _mk(ValueType.DRIVE_LINE)
    tone_2 = _mk(ValueType.TONE)

    ops = [
        Op(OpKind.ALLOC_DRIVE, (), (drive_line_0, tone_0), {"qubit": 0}),
        Op(OpKind.MAKE_WAVEFORM, (), (waveform,), {"duration_vtu": 100}),
        Op(OpKind.DRIVE, (drive_line_0, waveform, tone_0),
           (drive_line_1, tone_1), {
               "duration_vtu": 100,
               "start_vtu": 0
           }),
        Op(OpKind.DRIVE, (drive_line_1, waveform, tone_1),
           (drive_line_2, tone_2), {
               "duration_vtu": 40,
               "start_vtu": 50
           }),
    ]

    return Program(name="bad_time",
                   clock_ghz=1.0,
                   ops=ops,
                   values=[drive_line_0, tone_0, waveform])


# ── 3. Drive exclusivity overlap ────────────────────────────────────────────


def overlapping_drives() -> Program:
    """Two drive ops on the same line with overlapping time intervals."""
    _reset_vid_counter()

    drive_line_0 = _mk(ValueType.DRIVE_LINE, "d0")
    tone_0 = _mk(ValueType.TONE, "t0")
    waveform = _mk(ValueType.WAVEFORM, "wf")
    drive_line_1 = _mk(ValueType.DRIVE_LINE)
    tone_1 = _mk(ValueType.TONE)
    drive_line_2 = _mk(ValueType.DRIVE_LINE)
    tone_2 = _mk(ValueType.TONE)

    ops = [
        Op(OpKind.ALLOC_DRIVE, (), (drive_line_0, tone_0), {"qubit": 0}),
        Op(OpKind.MAKE_WAVEFORM, (), (waveform,), {"duration_vtu": 100}),
        Op(OpKind.DRIVE, (drive_line_0, waveform, tone_0),
           (drive_line_1, tone_1), {
               "duration_vtu": 100,
               "start_vtu": 0
           }),
        Op(OpKind.DRIVE, (drive_line_0, waveform, tone_0),
           (drive_line_2, tone_2), {
               "duration_vtu": 100,
               "start_vtu": 50
           }),
    ]

    return Program(name="bad_overlap",
                   clock_ghz=1.0,
                   ops=ops,
                   values=[drive_line_0, tone_0, waveform])


# ── 4. Cross-resonance miscalibration ───────────────────────────────────────


def cr_miscalibration() -> Program:
    """A CR drive tagged for qubit 1 but using a tone labeled for qubit 2."""
    _reset_vid_counter()

    drive_line_0 = _mk(ValueType.DRIVE_LINE, "d0")
    tone_0 = _mk(ValueType.TONE, "t0")
    waveform = _mk(ValueType.WAVEFORM, "wf")
    drive_line_1 = _mk(ValueType.DRIVE_LINE)
    tone_1 = _mk(ValueType.TONE)

    ops = [
        Op(OpKind.ALLOC_DRIVE, (), (drive_line_0, tone_0), {"qubit": 0}),
        Op(OpKind.MAKE_WAVEFORM, (), (waveform,), {"duration_vtu": 200}),
        Op(OpKind.DRIVE, (drive_line_0, waveform, tone_0),
           (drive_line_1, tone_1), {
               "duration_vtu": 200,
               "cr_target": 1,
               "tone_tag": "q2_tone"
           }),
    ]

    return Program(name="bad_cr",
                   clock_ghz=1.0,
                   ops=ops,
                   values=[drive_line_0, tone_0, waveform])


# ── 5. Clean program (control) ──────────────────────────────────────────────


def clean_program() -> Program:
    """A well-formed program that should pass all checks."""
    _reset_vid_counter()

    drive_line_0 = _mk(ValueType.DRIVE_LINE, "d0")
    tone_0 = _mk(ValueType.TONE, "t0")
    waveform = _mk(ValueType.WAVEFORM, "wf")
    drive_line_1 = _mk(ValueType.DRIVE_LINE)
    tone_1 = _mk(ValueType.TONE)

    ops = [
        Op(OpKind.ALLOC_DRIVE, (), (drive_line_0, tone_0), {"qubit": 0}),
        Op(OpKind.MAKE_WAVEFORM, (), (waveform,), {"duration_vtu": 40}),
        Op(OpKind.DRIVE, (drive_line_0, waveform, tone_0),
           (drive_line_1, tone_1), {"duration_vtu": 40}),
    ]

    return Program(name="clean",
                   clock_ghz=1.0,
                   ops=ops,
                   values=[drive_line_0, tone_0, waveform])


def main():
    test_cases = [
        ("Linearity violation (double consume)", linearity_double_consume,
         LinearityViolation),
        ("Backward time travel", backward_time, BackwardTimeTravelError),
        ("Overlapping drives", overlapping_drives, UnintentionalOverlapError),
        ("CR miscalibration", cr_miscalibration,
         CrossResonanceMiscalibrationError),
        ("Clean program (should pass)", clean_program, None),
    ]

    for label, builder, expected_type in test_cases:
        prog = builder()
        issues = verify(prog)
        status = "PASS" if not issues else f"CAUGHT {len(issues)} issue(s)"
        print(f"\n{'='*60}")
        print(f"  {label}: {status}")
        print(f"{'='*60}")

        for issue in issues:
            matched = expected_type and isinstance(issue, expected_type)
            tag = "EXPECTED" if matched else "OTHER"
            print(f"  [{tag}] {issue}")

        if not issues:
            if expected_type is None:
                print("  All checks passed (as expected).")
            else:
                print(
                    f"  WARNING: Expected {expected_type.__name__} but no issues found."
                )


if __name__ == "__main__":
    main()
