#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QEC scaling benchmark: compile time vs distance for surface-code-like programs.

Generates synthetic surface-code syndrome extraction cycles at various
code distances and measures verify + schedule time to confirm near-linear
scaling in qubit/op count.
"""

from __future__ import annotations

import time

from cudaq_pulse.passes.ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    _mk,
    _reset_vid_counter,
)
from cudaq_pulse.passes.verify import verify
from cudaq_pulse.passes.scheduling import schedule_alap


def build_surface_code_cycle(d: int) -> Program:
    """Build a d-distance surface-code-like syndrome extraction program.

    Layout:
      d^2 data qubits + (d^2 - 1) ancilla qubits = 2d^2 - 1 total
      Each ancilla interacts with up to 4 data neighbors via echoed CR.
    """
    _reset_vid_counter(0)
    n_data = d * d
    n_ancilla = d * d - 1
    n_total = n_data + n_ancilla
    clock_ghz = 2.0

    lines: list[Value] = []
    tones: list[Value] = []
    ops: list[Op] = []
    freq_hz: dict[int, float] = {}

    for qi in range(n_total):
        dl = _mk(ValueType.DRIVE_LINE, f"d{qi}")
        t = _mk(ValueType.TONE, f"t{qi}")
        lines.append(dl)
        tones.append(t)
        fhz = 5.0e9 + 0.05e9 * (qi % 8)
        freq_hz[qi] = fhz
        ops.append(
            Op(OpKind.ALLOC_DRIVE, (), (dl, t), {
                "qubit": qi,
                "frequency_hz": fhz
            }))

    def _cr_interaction(ctrl: int, tgt: int):
        """Emit an echoed CR between ctrl and tgt (4 ops)."""
        wf_cr = _mk(ValueType.WAVEFORM, "cr")
        ops.append(
            Op(OpKind.MAKE_WAVEFORM, (), (wf_cr,), {
                "waveform_type": "gaussian",
                "duration_vtu": 98,
                "amplitude": 0.32
            }))
        new_lc = _mk(ValueType.DRIVE_LINE)
        new_tc = _mk(ValueType.TONE)
        ops.append(
            Op(OpKind.DRIVE, (lines[ctrl], wf_cr, tones[ctrl]),
               (new_lc, new_tc), {"duration_vtu": 98}))
        lines[ctrl] = new_lc
        tones[ctrl] = new_tc

        wf_x = _mk(ValueType.WAVEFORM, "x_echo")
        ops.append(
            Op(OpKind.MAKE_WAVEFORM, (), (wf_x,), {
                "waveform_type": "drag",
                "duration_vtu": 20,
                "amplitude": 0.44
            }))
        new_lt = _mk(ValueType.DRIVE_LINE)
        new_tt = _mk(ValueType.TONE)
        ops.append(
            Op(OpKind.DRIVE, (lines[tgt], wf_x, tones[tgt]), (new_lt, new_tt),
               {"duration_vtu": 20}))
        lines[tgt] = new_lt
        tones[tgt] = new_tt

    def _hadamard(qi: int):
        """Phase-SX-Phase on qubit qi (3 ops)."""
        new_t1 = _mk(ValueType.TONE)
        ops.append(
            Op(OpKind.SHIFT_PHASE, (tones[qi],), (new_t1,),
               {"delta_rad": 1.5708}))
        tones[qi] = new_t1

        wf_sx = _mk(ValueType.WAVEFORM, "sx")
        ops.append(
            Op(OpKind.MAKE_WAVEFORM, (), (wf_sx,), {
                "waveform_type": "drag",
                "duration_vtu": 20,
                "amplitude": 0.44
            }))
        new_l = _mk(ValueType.DRIVE_LINE)
        new_t2 = _mk(ValueType.TONE)
        ops.append(
            Op(OpKind.DRIVE, (lines[qi], wf_sx, tones[qi]), (new_l, new_t2),
               {"duration_vtu": 20}))
        lines[qi] = new_l
        tones[qi] = new_t2

        new_t3 = _mk(ValueType.TONE)
        ops.append(
            Op(OpKind.SHIFT_PHASE, (tones[qi],), (new_t3,),
               {"delta_rad": 1.5708}))
        tones[qi] = new_t3

    # Syndrome cycle: each ancilla does H, then CR with up to 4 data neighbors, then H.
    for anc_idx in range(n_ancilla):
        anc = n_data + anc_idx
        row, col = anc_idx // (d - 1), anc_idx % (d - 1)
        is_z = (row + col) % 2 == 0

        if is_z:
            _hadamard(anc)

        neighbors = []
        r, c = row, col
        for dr, dc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < d and 0 <= nc < d:
                neighbors.append(nr * d + nc)

        for data_q in neighbors:
            _cr_interaction(data_q, anc)

        if is_z:
            _hadamard(anc)

    all_values = []
    for op in ops:
        all_values.extend(op.results)

    return Program(
        name=f"surface_d{d}",
        clock_ghz=clock_ghz,
        ops=ops,
        values=all_values,
        qubit_freq_hz=freq_hz,
    )


def main():
    distances = [3, 5, 7, 9, 11, 15, 21]
    print(f"  {'d':>3s}  {'Qubits':>7s}  {'Ops':>7s}  {'Build (ms)':>10s}  "
          f"{'Verify (ms)':>11s}  {'Sched (ms)':>10s}  {'Total (ms)':>10s}")
    print(
        f"  {'---':>3s}  {'-------':>7s}  {'-------':>7s}  {'----------':>10s}  "
        f"{'-----------':>11s}  {'----------':>10s}  {'----------':>10s}")

    rows = []
    for d in distances:
        t0 = time.monotonic()
        prog = build_surface_code_cycle(d)
        t_build = time.monotonic() - t0

        t0 = time.monotonic()
        issues = verify(prog)
        t_verify = time.monotonic() - t0

        t0 = time.monotonic()
        events, metrics = schedule_alap(prog)
        t_sched = time.monotonic() - t0

        total = t_build + t_verify + t_sched
        n_qubits = 2 * d * d - 1
        n_ops = len(prog.ops)
        rows.append((d, n_qubits, n_ops, t_build, t_verify, t_sched, total))
        print(
            f"  {d:3d}  {n_qubits:7d}  {n_ops:7d}  {t_build*1000:10.2f}  "
            f"{t_verify*1000:11.2f}  {t_sched*1000:10.2f}  {total*1000:10.2f}")

    # Print scaling ratios relative to d=3 baseline
    base_d, base_q, base_ops, _, _, _, base_t = rows[0]
    print(
        f"\n  Scaling relative to d={base_d} ({base_q} qubits, {base_ops} ops, {base_t*1000:.2f} ms):"
    )
    print(
        f"  {'d':>3s}  {'Qubits':>7s}  {'Q ratio':>8s}  {'Ops':>7s}  {'Op ratio':>9s}  {'Time ratio':>10s}"
    )
    print(
        f"  {'---':>3s}  {'-------':>7s}  {'--------':>8s}  {'-------':>7s}  {'---------':>9s}  {'----------':>10s}"
    )
    for d, nq, nops, _, _, _, t in rows:
        qr = nq / base_q
        opr = nops / base_ops
        tr = t / base_t
        print(
            f"  {d:3d}  {nq:7d}  {qr:8.1f}x  {nops:7d}  {opr:9.1f}x  {tr:10.1f}x"
        )


if __name__ == "__main__":
    main()
