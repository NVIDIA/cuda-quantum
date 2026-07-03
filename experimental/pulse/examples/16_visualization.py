# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pulse schedule visualization with matplotlib.

Shows how to use the viz module to plot per-line Gantt charts of
scheduled pulse programs — useful for debugging timing, identifying
idle gaps, and verifying sync alignment.

Requires matplotlib: pip install matplotlib
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
from cudaq_pulse.passes.scheduling import schedule_asap, schedule_alap, ScheduledEvent


def build_vis_program() -> Program:
    """A 3-qubit program with varied timing for interesting visualization."""
    _reset_vid_counter()

    lines, tones = [], []
    for i in range(3):
        lines.append(_mk(ValueType.DRIVE_LINE, f"d{i}"))
        tones.append(_mk(ValueType.TONE, f"t{i}"))

    waveform_short = _mk(ValueType.WAVEFORM, "wf_short")
    waveform_long = _mk(ValueType.WAVEFORM, "wf_long")
    waveform_readout = _mk(ValueType.WAVEFORM, "wf_ro")

    readout_line = _mk(ValueType.READOUT_LINE, "r0")
    readout_tone = _mk(ValueType.TONE, "rt0")

    ops = [
        Op(OpKind.ALLOC_DRIVE, (), (lines[0], tones[0]), {
            "qubit": 0,
            "freq_hz": 5.0e9
        }),
        Op(OpKind.ALLOC_DRIVE, (), (lines[1], tones[1]), {
            "qubit": 1,
            "freq_hz": 5.1e9
        }),
        Op(OpKind.ALLOC_DRIVE, (), (lines[2], tones[2]), {
            "qubit": 2,
            "freq_hz": 5.2e9
        }),
        Op(OpKind.ALLOC_READOUT, (), (readout_line, readout_tone),
           {"qubit": 0}),
        Op(OpKind.MAKE_WAVEFORM, (), (waveform_short,), {
            "waveform_type": "drag",
            "duration_vtu": 40
        }),
        Op(OpKind.MAKE_WAVEFORM, (), (waveform_long,), {
            "waveform_type": "gaussian_square",
            "duration_vtu": 300
        }),
        Op(OpKind.MAKE_WAVEFORM, (), (waveform_readout,), {
            "waveform_type": "square",
            "duration_vtu": 1000
        }),
    ]

    out = [[], [], []]
    for i in range(3):
        drive_out = _mk(ValueType.DRIVE_LINE)
        tone_out = _mk(ValueType.TONE)
        ops.append(
            Op(OpKind.DRIVE, (lines[i], waveform_short, tones[i]),
               (drive_out, tone_out), {
                   "duration_vtu": 40,
                   "amplitude": 0.25
               }))
        out[i] = [drive_out, tone_out]

    ops.append(Op(OpKind.SYNC, (out[0][0], out[1][0], out[2][0]), (), {}))

    drive_out_0_cr = _mk(ValueType.DRIVE_LINE)
    tone_out_0_cr = _mk(ValueType.TONE)
    ops.append(
        Op(OpKind.DRIVE, (out[0][0], waveform_long, out[1][1]),
           (drive_out_0_cr, tone_out_0_cr), {
               "duration_vtu": 300,
               "amplitude": 0.05
           }))

    drive_wait_2 = _mk(ValueType.DRIVE_LINE)
    ops.append(
        Op(OpKind.WAIT, (out[2][0],), (drive_wait_2,), {"duration_vtu": 200}))

    drive_out_1_2 = _mk(ValueType.DRIVE_LINE)
    tone_out_1_2 = _mk(ValueType.TONE)
    ops.append(
        Op(OpKind.DRIVE, (out[1][0], waveform_short, out[1][1]),
           (drive_out_1_2, tone_out_1_2), {
               "duration_vtu": 40,
               "amplitude": 0.25
           }))

    ops.append(
        Op(OpKind.SYNC, (drive_out_0_cr, drive_out_1_2, drive_wait_2), (), {}))

    readout_out = _mk(ValueType.READOUT_LINE)
    readout_tone_out = _mk(ValueType.TONE)
    ops.append(
        Op(OpKind.READOUT, (readout_line, waveform_readout, readout_tone),
           (readout_out, readout_tone_out), {
               "duration_vtu": 1000,
               "amplitude": 0.05
           }))

    return Program(
        name="vis_demo",
        clock_ghz=1.0,
        ops=ops,
        values=lines + tones + [
            waveform_short, waveform_long, waveform_readout, readout_line,
            readout_tone
        ],
        qubit_freq_hz={
            0: 5.0e9,
            1: 5.1e9,
            2: 5.2e9
        },
    )


def print_ascii_timeline(events: list[ScheduledEvent], width: int = 70) -> None:
    """Render a simple ASCII timeline for terminals without matplotlib."""
    active = [
        ev for ev in events if ev.duration_vtu > 0 and ev.line_id is not None
    ]
    if not active:
        print("  (no timed events)")
        return

    max_t = max(ev.end_vtu for ev in active)
    line_ids = sorted({ev.line_id for ev in active})

    for lid in line_ids:
        line_evs = [ev for ev in active if ev.line_id == lid]
        row = [" "] * width
        for ev in line_evs:
            start = int(ev.start_vtu / max_t * (width - 1))
            end = int(ev.end_vtu / max_t * (width - 1))
            char = "D" if ev.kind == "drive" else (
                "R" if ev.kind == "readout" else ".")
            for c in range(start, min(end + 1, width)):
                row[c] = char
        print(f"  line {lid}: |{''.join(row)}|")

    print(f"  {'':>8} 0{' ' * (width - 8)}{max_t:.0f} VTU")


def main():
    prog = build_vis_program()
    print(f"Program: {prog.name}, {len(prog.ops)} ops, 3 qubits + readout\n")

    # ASAP schedule
    events_asap, metrics_asap = schedule_asap(prog)
    print(f"=== ASAP Schedule ({metrics_asap.total_length_vtu:.0f} VTU) ===")
    print_ascii_timeline(events_asap)

    # ALAP schedule
    events_alap, metrics_alap = schedule_alap(prog)
    print(f"\n=== ALAP Schedule ({metrics_alap.total_length_vtu:.0f} VTU) ===")
    print_ascii_timeline(events_alap)

    # Try matplotlib plot
    try:
        from cudaq_pulse.viz.timeline import plot_schedule, save_schedule
        fig = plot_schedule(events_asap,
                            program=prog,
                            title="ASAP Schedule (3 qubits)")
        print("\n  matplotlib figure created successfully.")
        print("  Call save_schedule(events, 'output.png') to save to file.")
    except ImportError:
        print(
            "\n  matplotlib not installed. Install with: pip install matplotlib"
        )


if __name__ == "__main__":
    main()
