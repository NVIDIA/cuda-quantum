# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Side-by-side comparison of ASAP, ALAP, and RCP scheduling policies.

Builds a realistic 4-qubit program with varying op durations and resource
contention, then schedules it under all three policies and prints the
resulting timelines and metrics.
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
from cudaq_pulse.passes.scheduling import (
    schedule_asap,
    schedule_alap,
    schedule_rcp,
    MachineModel,
    ScheduledEvent,
)


def build_four_qubit_program() -> Program:
    """A 4-qubit program with mixed durations and syncs."""
    _reset_vid_counter()

    lines, tones, waveforms = [], [], []
    for i in range(4):
        lines.append(_mk(ValueType.DRIVE_LINE, f"d{i}"))
        tones.append(_mk(ValueType.TONE, f"t{i}"))

    for i in range(4):
        waveforms.append(_mk(ValueType.WAVEFORM, f"wf{i}"))

    ops = []
    for i in range(4):
        ops.append(
            Op(OpKind.ALLOC_DRIVE, (), (lines[i], tones[i]), {
                "qubit": i,
                "freq_hz": (5.0 + 0.1 * i) * 1e9
            }))

    for i in range(4):
        ops.append(
            Op(
                OpKind.MAKE_WAVEFORM, (), (waveforms[i],), {
                    "waveform_type": "gaussian",
                    "duration_vtu": 40 + 20 * i,
                    "amplitude": 0.25
                }))

    out_lines = []
    out_tones = []
    for i in range(4):
        drive_out = _mk(ValueType.DRIVE_LINE, f"d{i}'")
        tone_out = _mk(ValueType.TONE, f"t{i}'")
        ops.append(
            Op(OpKind.DRIVE, (lines[i], waveforms[i], tones[i]),
               (drive_out, tone_out), {
                   "duration_vtu": 40 + 20 * i,
                   "amplitude": 0.25
               }))
        out_lines.append(drive_out)
        out_tones.append(tone_out)

    ops.append(Op(OpKind.SYNC, tuple(out_lines), (), {}))

    out_lines2 = []
    for i in range(4):
        drive_out_2 = _mk(ValueType.DRIVE_LINE, f"d{i}''")
        tone_out_2 = _mk(ValueType.TONE, f"t{i}''")
        ops.append(
            Op(OpKind.DRIVE, (out_lines[i], waveforms[i], out_tones[i]),
               (drive_out_2, tone_out_2), {
                   "duration_vtu": 40 + 20 * i,
                   "amplitude": 0.25
               }))
        out_lines2.append(drive_out_2)

    return Program(
        name="four_qubit",
        clock_ghz=1.0,
        ops=ops,
        values=lines + tones + waveforms,
        qubit_freq_hz={i: (5.0 + 0.1 * i) * 1e9 for i in range(4)},
    )


def print_timeline(events: list[ScheduledEvent], label: str) -> None:
    """Pretty-print a timeline of scheduled events."""
    print(f"\n  --- {label} ---")
    active = [ev for ev in events if ev.duration_vtu > 0]
    for ev in sorted(active, key=lambda e: (e.start_vtu, e.line_id or 0)):
        lid = f"line {ev.line_id}" if ev.line_id is not None else "global"
        print(
            f"    [{ev.start_vtu:7.1f} - {ev.end_vtu:7.1f}] {lid:>8}  {ev.kind}"
        )


def main():
    prog = build_four_qubit_program()
    print(f"Program: {prog.name}, {len(prog.ops)} ops, 4 qubits")

    # ASAP
    events_asap, metrics_asap = schedule_asap(prog)
    print_timeline(events_asap, "ASAP")
    print(f"\n  Total: {metrics_asap.total_length_vtu:.0f} VTU, "
          f"idle: {metrics_asap.idle_fraction:.1%}, "
          f"compile: {metrics_asap.compile_time_ms:.3f} ms")

    # ALAP
    events_alap, metrics_alap = schedule_alap(prog)
    print_timeline(events_alap, "ALAP")
    print(f"\n  Total: {metrics_alap.total_length_vtu:.0f} VTU, "
          f"idle: {metrics_alap.idle_fraction:.1%}, "
          f"compile: {metrics_alap.compile_time_ms:.3f} ms")

    # RCP with hardware constraints
    machine = MachineModel(
        max_concurrent_drives=2,
        max_concurrent_readouts=1,
        line_switch_penalty_vtu=5.0,
    )
    events_rcp, metrics_rcp = schedule_rcp(prog, machine)
    print_timeline(
        events_rcp,
        f"RCP (max {machine.max_concurrent_drives} concurrent drives)")
    print(f"\n  Total: {metrics_rcp.total_length_vtu:.0f} VTU, "
          f"idle: {metrics_rcp.idle_fraction:.1%}, "
          f"compile: {metrics_rcp.compile_time_ms:.3f} ms")

    # Summary table
    print("\n  ┌──────────┬───────────┬──────────┬────────────┐")
    print("  │ Policy   │ Total VTU │ Idle %   │ Compile ms │")
    print("  ├──────────┼───────────┼──────────┼────────────┤")
    for name, metrics in [("ASAP", metrics_asap), ("ALAP", metrics_alap),
                          ("RCP", metrics_rcp)]:
        print(
            f"  │ {name:<8} │ {metrics.total_length_vtu:>9.0f} │ {metrics.idle_fraction:>7.1%} │ {metrics.compile_time_ms:>10.3f} │"
        )
    print("  └──────────┴───────────┴──────────┴────────────┘")


if __name__ == "__main__":
    main()
