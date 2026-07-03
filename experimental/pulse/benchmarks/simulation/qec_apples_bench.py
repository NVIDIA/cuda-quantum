#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Apples-to-apples QEC benchmark: cudaq-pulse vs pulse_ref.

Runs identical workloads through matched pipelines:
  - Both: build program + ALAP schedule (the only pass pulse_ref runs)
  - cudaq-pulse additionally: MLIR text emission (the overhead we care about)

Focuses on large QEC workloads where overhead amortizes:
  - Surface code d=3,5,7,9,11,15,21
  - qLDPC bivariate bicycle [[72,12,6]], [[144,12,12]], [[288,12,18]] (Tour de Gross)

Usage:
  PYTHONPATH=python python benchmarks/qec_apples_bench.py \
      --pulse-ref-dir /path/to/pulse_mlir_qce26/benchmarks
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------------------
# cudaq-pulse imports
# ---------------------------------------------------------------------------
from cudaq_pulse.passes.ir_types import (
    Op,
    OpKind,
    Program,
    Value,
    ValueType,
    _mk,
    _reset_vid_counter,
)
from cudaq_pulse.passes.scheduling import schedule_alap
from cudaq_pulse.passes.to_pulse_mlir import program_to_pulse_mlir

_HAS_NATIVE = False
_HAS_PACKED = False
try:
    from cudaq_pulse.native_emit import emit_pulse_module
    _HAS_NATIVE = True
except ImportError:
    pass
try:
    from cudaq_pulse.packed_emit import pack_program, emit_pulse_module_packed
    _HAS_PACKED = True
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


# ---------------------------------------------------------------------------
# cudaq-pulse surface-code builder (matched to pulse_ref's layout)
# ---------------------------------------------------------------------------


def _echo_cr(ops, lines, tones, ctrl, tgt, sx_wf, cr_pos_wf, cr_neg_wf, x_wf):
    """Echo-CR CNOT: 5 drives, matching pulse_ref's _echo_cr_pair."""
    # SX on target
    new_lt = _mk(ValueType.DRIVE_LINE)
    new_tt = _mk(ValueType.TONE)
    ops.append(
        Op(OpKind.DRIVE, (lines[tgt], sx_wf, tones[tgt]), (new_lt, new_tt),
           {"duration_vtu": 40}))
    lines[tgt] = new_lt
    tones[tgt] = new_tt

    # CR+ on ctrl using target tone
    new_lc = _mk(ValueType.DRIVE_LINE)
    new_tc = _mk(ValueType.TONE)
    ops.append(
        Op(OpKind.DRIVE, (lines[ctrl], cr_pos_wf, tones[tgt]), (new_lc, new_tc),
           {"duration_vtu": 200}))
    lines[ctrl] = new_lc
    tones[tgt] = new_tc

    # X on ctrl using ctrl tone
    new_lc2 = _mk(ValueType.DRIVE_LINE)
    new_tc2 = _mk(ValueType.TONE)
    ops.append(
        Op(OpKind.DRIVE, (lines[ctrl], x_wf, tones[ctrl]), (new_lc2, new_tc2),
           {"duration_vtu": 40}))
    lines[ctrl] = new_lc2
    tones[ctrl] = new_tc2

    # CR- on ctrl using target tone
    new_lc3 = _mk(ValueType.DRIVE_LINE)
    new_tc3 = _mk(ValueType.TONE)
    ops.append(
        Op(OpKind.DRIVE, (lines[ctrl], cr_neg_wf, tones[tgt]),
           (new_lc3, new_tc3), {"duration_vtu": 200}))
    lines[ctrl] = new_lc3
    tones[tgt] = new_tc3

    # SX on target
    new_lt2 = _mk(ValueType.DRIVE_LINE)
    new_tt2 = _mk(ValueType.TONE)
    ops.append(
        Op(OpKind.DRIVE, (lines[tgt], sx_wf, tones[tgt]), (new_lt2, new_tt2),
           {"duration_vtu": 40}))
    lines[tgt] = new_lt2
    tones[tgt] = new_tt2


def build_surface_code(d: int, cycles: int = 1) -> Program:
    """Build surface-code syndrome extraction matching pulse_ref layout."""
    _reset_vid_counter(0)
    n_data = d * d
    n_anc = d * d - 1
    n_total = n_data + n_anc

    lines: list[Value] = []
    tones: list[Value] = []
    ops: list[Op] = []
    freq_hz: dict[int, float] = {}

    for qi in range(n_total):
        dl = _mk(ValueType.DRIVE_LINE, f"d{qi}")
        t = _mk(ValueType.TONE, f"t{qi}")
        lines.append(dl)
        tones.append(t)
        fhz = 5.0e9 + 1e5 * qi
        freq_hz[qi] = fhz
        ops.append(
            Op(OpKind.ALLOC_DRIVE, (), (dl, t), {
                "qubit": qi,
                "frequency_hz": fhz
            }))

    ro_lines: list[Value] = []
    ro_tones: list[Value] = []
    for anc_idx in range(n_anc):
        qi = n_data + anc_idx
        rl = _mk(ValueType.READOUT_LINE, f"r{qi}")
        rt = _mk(ValueType.TONE, f"rt{qi}")
        ro_lines.append(rl)
        ro_tones.append(rt)
        ops.append(
            Op(OpKind.ALLOC_READOUT, (), (rl, rt), {
                "qubit": qi,
                "frequency_hz": 6.5e9 + 1e5 * qi
            }))

    # Shared waveforms
    sx_wf = _mk(ValueType.WAVEFORM, "sx")
    ops.append(
        Op(
            OpKind.MAKE_WAVEFORM, (), (sx_wf,), {
                "waveform_type": "drag",
                "duration_vtu": 40,
                "amplitude": 0.025,
                "sigma": 10.0,
                "beta": 0.5
            }))
    cr_pos_wf = _mk(ValueType.WAVEFORM, "cr_pos")
    ops.append(
        Op(
            OpKind.MAKE_WAVEFORM, (), (cr_pos_wf,), {
                "waveform_type": "gaussian",
                "duration_vtu": 200,
                "amplitude": 0.10,
                "sigma": 50.0
            }))
    cr_neg_wf = _mk(ValueType.WAVEFORM, "cr_neg")
    ops.append(
        Op(
            OpKind.MAKE_WAVEFORM, (), (cr_neg_wf,), {
                "waveform_type": "gaussian",
                "duration_vtu": 200,
                "amplitude": -0.10,
                "sigma": 50.0
            }))
    x_wf = _mk(ValueType.WAVEFORM, "x")
    ops.append(
        Op(OpKind.MAKE_WAVEFORM, (), (x_wf,), {
            "waveform_type": "square",
            "duration_vtu": 40,
            "amplitude": 0.047
        }))
    ro_wf = _mk(ValueType.WAVEFORM, "ro")
    ops.append(
        Op(OpKind.MAKE_WAVEFORM, (), (ro_wf,), {
            "waveform_type": "square",
            "duration_vtu": 500,
            "amplitude": 0.05
        }))
    sx_prep_wf = _mk(ValueType.WAVEFORM, "sx_prep")
    ops.append(
        Op(
            OpKind.MAKE_WAVEFORM, (), (sx_prep_wf,), {
                "waveform_type": "drag",
                "duration_vtu": 40,
                "amplitude": 0.25,
                "sigma": 10.0,
                "beta": 0.5
            }))

    # Ancilla positions (matching pulse_ref)
    anc_positions: list[tuple[int, int]] = []
    for r in range(d - 1):
        for c in range(d - 1):
            anc_positions.append((r, c))
    for i in range(d - 1):
        anc_positions.append((-1, i))
        if len(anc_positions) >= n_anc:
            break
    while len(anc_positions) < n_anc:
        anc_positions.append((d - 1, len(anc_positions) - 2 * (d - 1)))

    for cycle in range(cycles):
        for idx, (ar, ac) in enumerate(anc_positions[:n_anc]):
            q_anc = n_data + idx
            # Prep
            new_l = _mk(ValueType.DRIVE_LINE)
            new_t = _mk(ValueType.TONE)
            ops.append(
                Op(OpKind.DRIVE, (lines[q_anc], sx_prep_wf, tones[q_anc]),
                   (new_l, new_t), {"duration_vtu": 40}))
            lines[q_anc] = new_l
            tones[q_anc] = new_t

            # 4 CNOTs
            neighbors = []
            for dr in (0, 1):
                for dc in (0, 1):
                    r, c = ar + dr, ac + dc
                    if 0 <= r < d and 0 <= c < d:
                        neighbors.append(r * d + c)

            for data_q in neighbors:
                s_anc = _mk(ValueType.DRIVE_LINE)
                s_data = _mk(ValueType.DRIVE_LINE)
                ops.append(
                    Op(OpKind.SYNC, (lines[q_anc], lines[data_q]),
                       (s_anc, s_data), {}))
                lines[q_anc] = s_anc
                lines[data_q] = s_data
                _echo_cr(ops, lines, tones, q_anc, data_q, sx_wf, cr_pos_wf,
                         cr_neg_wf, x_wf)

            # Close prep
            new_l2 = _mk(ValueType.DRIVE_LINE)
            new_t2 = _mk(ValueType.TONE)
            ops.append(
                Op(OpKind.DRIVE, (lines[q_anc], sx_prep_wf, tones[q_anc]),
                   (new_l2, new_t2), {"duration_vtu": 40}))
            lines[q_anc] = new_l2
            tones[q_anc] = new_t2

            # Sync + readout
            s_anc2 = _mk(ValueType.DRIVE_LINE)
            s_ro = _mk(ValueType.READOUT_LINE)
            ops.append(
                Op(OpKind.SYNC, (lines[q_anc], ro_lines[idx]), (s_anc2, s_ro),
                   {}))
            lines[q_anc] = s_anc2
            ro_lines[idx] = s_ro
            new_ro = _mk(ValueType.READOUT_LINE)
            new_rot = _mk(ValueType.TONE)
            meas = _mk(ValueType.MEASUREMENT)
            ops.append(
                Op(OpKind.READOUT, (ro_lines[idx], ro_wf, ro_tones[idx]),
                   (new_ro, new_rot, meas), {
                       "duration_vtu": 500,
                       "mode": "iq"
                   }))
            ro_lines[idx] = new_ro
            ro_tones[idx] = new_rot

    all_values = []
    for op in ops:
        all_values.extend(op.results)

    return Program(
        name=f"surface_d{d}_c{cycles}",
        clock_ghz=2.0,
        ops=ops,
        values=all_values,
        qubit_freq_hz=freq_hz,
    )


def build_qldpc_bb(l: int, m: int, cycles: int = 1) -> Program:
    """Build bivariate bicycle qLDPC syndrome extraction matching pulse_ref."""
    _reset_vid_counter(0)
    n_physical = 2 * l * m
    n_x_checks = l * m
    n_z_checks = l * m
    total_q = n_physical + n_x_checks + n_z_checks

    lines: list[Value] = []
    tones: list[Value] = []
    ops: list[Op] = []
    freq_hz: dict[int, float] = {}

    for qi in range(total_q):
        dl = _mk(ValueType.DRIVE_LINE, f"d{qi}")
        t = _mk(ValueType.TONE, f"t{qi}")
        lines.append(dl)
        tones.append(t)
        fhz = 5.0e9 + 1e5 * qi
        freq_hz[qi] = fhz
        ops.append(
            Op(OpKind.ALLOC_DRIVE, (), (dl, t), {
                "qubit": qi,
                "frequency_hz": fhz
            }))

    ro_lines: list[Value] = []
    ro_tones: list[Value] = []
    for anc_idx in range(n_x_checks + n_z_checks):
        qi = n_physical + anc_idx
        rl = _mk(ValueType.READOUT_LINE, f"r{qi}")
        rt = _mk(ValueType.TONE, f"rt{qi}")
        ro_lines.append(rl)
        ro_tones.append(rt)
        ops.append(
            Op(OpKind.ALLOC_READOUT, (), (rl, rt), {
                "qubit": qi,
                "frequency_hz": 6.5e9 + 1e5 * qi
            }))

    # Shared waveforms
    sx_prep_wf = _mk(ValueType.WAVEFORM, "sx_prep")
    ops.append(
        Op(
            OpKind.MAKE_WAVEFORM, (), (sx_prep_wf,), {
                "waveform_type": "drag",
                "duration_vtu": 40,
                "amplitude": 0.25,
                "sigma": 10.0,
                "beta": 0.5
            }))
    sx_wf = _mk(ValueType.WAVEFORM, "sx")
    ops.append(
        Op(
            OpKind.MAKE_WAVEFORM, (), (sx_wf,), {
                "waveform_type": "drag",
                "duration_vtu": 40,
                "amplitude": 0.025,
                "sigma": 10.0,
                "beta": 0.5
            }))
    cr_pos_wf = _mk(ValueType.WAVEFORM, "cr_pos")
    ops.append(
        Op(
            OpKind.MAKE_WAVEFORM, (), (cr_pos_wf,), {
                "waveform_type": "gaussian",
                "duration_vtu": 200,
                "amplitude": 0.10,
                "sigma": 50.0
            }))
    cr_neg_wf = _mk(ValueType.WAVEFORM, "cr_neg")
    ops.append(
        Op(
            OpKind.MAKE_WAVEFORM, (), (cr_neg_wf,), {
                "waveform_type": "gaussian",
                "duration_vtu": 200,
                "amplitude": -0.10,
                "sigma": 50.0
            }))
    x_wf = _mk(ValueType.WAVEFORM, "x")
    ops.append(
        Op(OpKind.MAKE_WAVEFORM, (), (x_wf,), {
            "waveform_type": "square",
            "duration_vtu": 40,
            "amplitude": 0.047
        }))
    ro_wf = _mk(ValueType.WAVEFORM, "ro")
    ops.append(
        Op(OpKind.MAKE_WAVEFORM, (), (ro_wf,), {
            "waveform_type": "square",
            "duration_vtu": 500,
            "amplitude": 0.05
        }))

    def x_check_neighbours(check_idx):
        cr, cc = divmod(check_idx, m)
        shifts = [(0, 0), (0, 1), (3, 0), (0, 2), (1, 0), (2, 2)]
        return [((cr + dr) % l) * m + ((cc + dc) % m) for (dr, dc) in shifts]

    def z_check_neighbours(check_idx):
        cr, cc = divmod(check_idx, m)
        shifts = [(0, 0), (1, 0), (2, 1), (0, 1), (1, 2), (2, 2)]
        return [((cr + dr) % l) * m + ((cc + dc) % m) + l * m if
                ((cr + dr) % l) * m + ((cc + dc) % m) + l * m < n_physical else
                ((cr + dr) % l) * m + ((cc + dc) % m) for (dr, dc) in shifts]

    for cycle in range(cycles):
        for xi in range(n_x_checks):
            q_anc = n_physical + xi
            new_l = _mk(ValueType.DRIVE_LINE)
            new_t = _mk(ValueType.TONE)
            ops.append(
                Op(OpKind.DRIVE, (lines[q_anc], sx_prep_wf, tones[q_anc]),
                   (new_l, new_t), {"duration_vtu": 40}))
            lines[q_anc] = new_l
            tones[q_anc] = new_t

            for data_q in x_check_neighbours(xi):
                s_anc = _mk(ValueType.DRIVE_LINE)
                s_data = _mk(ValueType.DRIVE_LINE)
                ops.append(
                    Op(OpKind.SYNC, (lines[q_anc], lines[data_q]),
                       (s_anc, s_data), {}))
                lines[q_anc] = s_anc
                lines[data_q] = s_data
                _echo_cr(ops, lines, tones, q_anc, data_q, sx_wf, cr_pos_wf,
                         cr_neg_wf, x_wf)

            new_l2 = _mk(ValueType.DRIVE_LINE)
            new_t2 = _mk(ValueType.TONE)
            ops.append(
                Op(OpKind.DRIVE, (lines[q_anc], sx_prep_wf, tones[q_anc]),
                   (new_l2, new_t2), {"duration_vtu": 40}))
            lines[q_anc] = new_l2
            tones[q_anc] = new_t2

            s_anc2 = _mk(ValueType.DRIVE_LINE)
            s_ro = _mk(ValueType.READOUT_LINE)
            ops.append(
                Op(OpKind.SYNC, (lines[q_anc], ro_lines[xi]), (s_anc2, s_ro),
                   {}))
            lines[q_anc] = s_anc2
            ro_lines[xi] = s_ro
            new_ro = _mk(ValueType.READOUT_LINE)
            new_rot = _mk(ValueType.TONE)
            meas = _mk(ValueType.MEASUREMENT)
            ops.append(
                Op(OpKind.READOUT, (ro_lines[xi], ro_wf, ro_tones[xi]),
                   (new_ro, new_rot, meas), {
                       "duration_vtu": 500,
                       "mode": "iq"
                   }))
            ro_lines[xi] = new_ro
            ro_tones[xi] = new_rot

        for zi in range(n_z_checks):
            q_anc = n_physical + n_x_checks + zi
            for data_q in z_check_neighbours(zi):
                s_anc = _mk(ValueType.DRIVE_LINE)
                s_data = _mk(ValueType.DRIVE_LINE)
                ops.append(
                    Op(OpKind.SYNC, (lines[q_anc], lines[data_q]),
                       (s_anc, s_data), {}))
                lines[q_anc] = s_anc
                lines[data_q] = s_data
                _echo_cr(ops, lines, tones, data_q, q_anc, sx_wf, cr_pos_wf,
                         cr_neg_wf, x_wf)

            ro_idx = n_x_checks + zi
            s_anc2 = _mk(ValueType.DRIVE_LINE)
            s_ro = _mk(ValueType.READOUT_LINE)
            ops.append(
                Op(OpKind.SYNC, (lines[q_anc], ro_lines[ro_idx]),
                   (s_anc2, s_ro), {}))
            lines[q_anc] = s_anc2
            ro_lines[ro_idx] = s_ro
            new_ro = _mk(ValueType.READOUT_LINE)
            new_rot = _mk(ValueType.TONE)
            meas = _mk(ValueType.MEASUREMENT)
            ops.append(
                Op(OpKind.READOUT, (ro_lines[ro_idx], ro_wf, ro_tones[ro_idx]),
                   (new_ro, new_rot, meas), {
                       "duration_vtu": 500,
                       "mode": "iq"
                   }))
            ro_lines[ro_idx] = new_ro
            ro_tones[ro_idx] = new_rot

    all_values = []
    for op in ops:
        all_values.extend(op.results)

    return Program(
        name=f"bb_{l}x{m}_c{cycles}",
        clock_ghz=2.0,
        ops=ops,
        values=all_values,
        qubit_freq_hz=freq_hz,
    )


# ---------------------------------------------------------------------------
# cudaq-pulse pipeline: build + schedule (+ optional MLIR emit)
# ---------------------------------------------------------------------------


@dataclass
class CPResult:
    name: str
    qubits: int
    ops: int
    build_ms: float
    sched_ms: float
    mlir_ms: float
    total_no_mlir_ms: float
    total_with_mlir_ms: float
    mlir_lines: int
    mlir_chars: int
    native_mlir_ms: float = 0.0
    total_with_native_ms: float = 0.0
    packed_mlir_ms: float = 0.0
    total_with_packed_ms: float = 0.0


def bench_cudaq_pulse_surface(d: int, cycles: int = 1) -> CPResult:
    t0 = time.monotonic()
    prog = build_surface_code(d, cycles)
    build_ms = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    schedule_alap(prog)
    sched_ms = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    mlir_text = program_to_pulse_mlir(prog)
    mlir_ms = (time.monotonic() - t0) * 1000

    native_mlir_ms = 0.0
    if _HAS_NATIVE:
        t0 = time.monotonic()
        mod = emit_pulse_module(prog)
        _ = mod.print()
        native_mlir_ms = (time.monotonic() - t0) * 1000

    packed_mlir_ms = 0.0
    if _HAS_PACKED:
        t0 = time.monotonic()
        mod = emit_pulse_module_packed(prog)
        _ = mod.print()
        packed_mlir_ms = (time.monotonic() - t0) * 1000

    nq = 2 * d * d - 1
    return CPResult(
        name=f"surface_d{d}",
        qubits=nq,
        ops=len(prog.ops),
        build_ms=build_ms,
        sched_ms=sched_ms,
        mlir_ms=mlir_ms,
        total_no_mlir_ms=build_ms + sched_ms,
        total_with_mlir_ms=build_ms + sched_ms + mlir_ms,
        mlir_lines=mlir_text.count("\n"),
        mlir_chars=len(mlir_text),
        native_mlir_ms=native_mlir_ms,
        total_with_native_ms=build_ms + sched_ms + native_mlir_ms,
        packed_mlir_ms=packed_mlir_ms,
        total_with_packed_ms=build_ms + sched_ms + packed_mlir_ms,
    )


def bench_cudaq_pulse_qldpc(l: int, m: int, cycles: int = 1) -> CPResult:
    t0 = time.monotonic()
    prog = build_qldpc_bb(l, m, cycles)
    build_ms = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    schedule_alap(prog)
    sched_ms = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    mlir_text = program_to_pulse_mlir(prog)
    mlir_ms = (time.monotonic() - t0) * 1000

    native_mlir_ms = 0.0
    if _HAS_NATIVE:
        t0 = time.monotonic()
        mod = emit_pulse_module(prog)
        _ = mod.print()
        native_mlir_ms = (time.monotonic() - t0) * 1000

    packed_mlir_ms = 0.0
    if _HAS_PACKED:
        t0 = time.monotonic()
        mod = emit_pulse_module_packed(prog)
        _ = mod.print()
        packed_mlir_ms = (time.monotonic() - t0) * 1000

    nq = 2 * l * m
    return CPResult(
        name=f"bb_{l}x{m}",
        qubits=nq,
        ops=len(prog.ops),
        build_ms=build_ms,
        sched_ms=sched_ms,
        mlir_ms=mlir_ms,
        total_no_mlir_ms=build_ms + sched_ms,
        total_with_mlir_ms=build_ms + sched_ms + mlir_ms,
        mlir_lines=mlir_text.count("\n"),
        mlir_chars=len(mlir_text),
        native_mlir_ms=native_mlir_ms,
        total_with_native_ms=build_ms + sched_ms + native_mlir_ms,
        packed_mlir_ms=packed_mlir_ms,
        total_with_packed_ms=build_ms + sched_ms + packed_mlir_ms,
    )


# ---------------------------------------------------------------------------
# pulse_ref pipeline: build + schedule
# ---------------------------------------------------------------------------


@dataclass
class PRResult:
    name: str
    qubits: int
    ops: int
    build_ms: float
    sched_ms: float
    total_ms: float


def bench_pulse_ref(ref_dir: str) -> tuple[list[PRResult], list[PRResult]]:
    bench_dir = Path(ref_dir)
    if not bench_dir.is_dir():
        return [], []

    sys.path.insert(0, str(bench_dir))
    try:
        from workloads import surface_code, qldpc_bicycle
        from pulse_ref.sched import schedule as pr_schedule
    except (ImportError, TypeError) as e:
        print(f"  [note] pulse_ref import failed: {e}")
        return [], []

    surface_results = []
    for d in [3, 5, 7, 9, 11, 15, 21]:
        nq = surface_code.qubit_count(d)
        t0 = time.monotonic()
        p = surface_code.build(d=d, cycles=1)
        build_ms = (time.monotonic() - t0) * 1000
        t0 = time.monotonic()
        _, metrics = pr_schedule(p, policy="alap")
        sched_ms = (time.monotonic() - t0) * 1000
        surface_results.append(
            PRResult(
                name=f"surface_d{d}",
                qubits=nq,
                ops=len(p.ops),
                build_ms=build_ms,
                sched_ms=sched_ms,
                total_ms=build_ms + sched_ms,
            ))

    qldpc_results = []
    for inst_name, (l, m, k, d_log) in qldpc_bicycle.INSTANCES.items():
        nq = qldpc_bicycle.physical_qubit_count(inst_name)
        t0 = time.monotonic()
        p = qldpc_bicycle.build(inst_name, cycles=1)
        build_ms = (time.monotonic() - t0) * 1000
        t0 = time.monotonic()
        _, metrics = pr_schedule(p, policy="alap")
        sched_ms = (time.monotonic() - t0) * 1000
        qldpc_results.append(
            PRResult(
                name=f"bb_{inst_name}",
                qubits=nq,
                ops=len(p.ops),
                build_ms=build_ms,
                sched_ms=sched_ms,
                total_ms=build_ms + sched_ms,
            ))

    return surface_results, qldpc_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Apples-to-apples QEC benchmark")
    parser.add_argument("--pulse-ref-dir", default=None)
    args = parser.parse_args()

    ref_dir = args.pulse_ref_dir
    if ref_dir is None:
        for candidate in [
                "../pulse_ref_benchmarks",
                os.path.expanduser("~/Downloads/pulse_mlir_qce26/benchmarks"),
        ]:
            if Path(candidate).is_dir():
                ref_dir = candidate
                break

    W = 110
    native_tag = " (native C++ builder available)" if _HAS_NATIVE else " (Python text only)"
    print("=" * W)
    print("  QEC Apples-to-Apples Benchmark")
    print("  Pipeline: build program + ALAP schedule (matched passes)")
    print(
        f"  MLIR emission paths: Python text{' + C++ native' if _HAS_NATIVE else ''}"
    )
    print("=" * W)

    def _print_surface_row(r: CPResult, label: str | None = None):
        lbl = label or r.name
        native_col = f"  {r.native_mlir_ms:>8.1f}" if _HAS_NATIVE else ""
        packed_col = f"  {r.packed_mlir_ms:>8.1f}  {r.total_with_packed_ms:>8.1f}" if _HAS_PACKED else ""
        print(f"  {lbl:<16s}  {r.qubits:>7d}  {r.ops:>8d}  "
              f"{r.build_ms:>8.1f}  {r.sched_ms:>8.1f}  {r.mlir_ms:>8.1f}"
              f"{native_col}{packed_col}  {r.total_with_mlir_ms:>8.1f}")

    def _print_header():
        native_hdr = f"  {'DictML':>8s}" if _HAS_NATIVE else ""
        native_unit = f"  {'(ms)':>8s}" if _HAS_NATIVE else ""
        packed_hdr = f"  {'Packed':>8s}  {'Tot+Pk':>8s}" if _HAS_PACKED else ""
        packed_unit = f"  {'(ms)':>8s}  {'(ms)':>8s}" if _HAS_PACKED else ""
        print(f"  {'Name':<16s}  {'Qubits':>7s}  {'Ops':>8s}  "
              f"{'Build':>8s}  {'Sched':>8s}  {'PyMLIR':>8s}"
              f"{native_hdr}{packed_hdr}  {'Tot+Py':>8s}")
        print(f"  {'':16s}  {'':>7s}  {'':>8s}  "
              f"{'(ms)':>8s}  {'(ms)':>8s}  {'(ms)':>8s}"
              f"{native_unit}{packed_unit}  {'(ms)':>8s}")
        print("  " + "-" * (W - 2))

    # ── Surface code ──────────────────────────────────────────────────
    print("\n--- Surface Code (1 cycle) ---")
    _print_header()

    cp_surface: list[CPResult] = []
    for d in [3, 5, 7, 9, 11, 15, 21]:
        r = bench_cudaq_pulse_surface(d)
        cp_surface.append(r)
        _print_surface_row(r)

    # ── qLDPC ─────────────────────────────────────────────────────────
    print(f"\n--- qLDPC Bivariate Bicycle (1 cycle) ---")
    _print_header()

    cp_qldpc: list[CPResult] = []
    for (l, m, label) in [(6, 6, "72_12_6"), (12, 6, "144_12_12"),
                          (12, 12, "288_12_18")]:
        r = bench_cudaq_pulse_qldpc(l, m)
        cp_qldpc.append(r)
        _print_surface_row(r, label)

    # ── pulse_ref comparison ──────────────────────────────────────────
    pr_surface, pr_qldpc = [], []
    if ref_dir:
        pr_surface, pr_qldpc = bench_pulse_ref(ref_dir)

    def _print_comparison(cp_list, pr_list, labels=None):
        native_hdr = f"  {'cp+Nat':>8s}  {'Ratio+N':>8s}" if _HAS_NATIVE else ""
        print(f"  {'Name':<16s}  {'Ops':>8s}  "
              f"{'cp (ms)':>8s}  {'ref (ms)':>8s}  {'Ratio':>8s}  "
              f"{'cp+PyML':>8s}  {'Rat+Py':>8s}"
              f"{native_hdr}")
        print("  " + "-" * (W - 2))

        for i, (cp_r, pr_r) in enumerate(zip(cp_list, pr_list)):
            lbl = labels[i] if labels else cp_r.name
            ratio = cp_r.total_no_mlir_ms / pr_r.total_ms if pr_r.total_ms > 0 else 0
            ratio_py = cp_r.total_with_mlir_ms / pr_r.total_ms if pr_r.total_ms > 0 else 0
            native_col = ""
            if _HAS_NATIVE:
                ratio_nat = cp_r.total_with_native_ms / pr_r.total_ms if pr_r.total_ms > 0 else 0
                native_col = f"  {cp_r.total_with_native_ms:>8.1f}  {ratio_nat:>7.2f}x"
            print(
                f"  {lbl:<16s}  {cp_r.ops:>8d}  "
                f"{cp_r.total_no_mlir_ms:>8.1f}  {pr_r.total_ms:>8.1f}  {ratio:>7.2f}x  "
                f"{cp_r.total_with_mlir_ms:>8.1f}  {ratio_py:>7.2f}x"
                f"{native_col}")

    if pr_surface:
        print("\n" + "=" * W)
        print("  COMPARISON: cudaq-pulse vs pulse_ref (build + ALAP schedule)")
        print("=" * W)

        print("\n  Surface Code:")
        _print_comparison(cp_surface, pr_surface)

    if pr_qldpc:
        print(f"\n  qLDPC Bivariate Bicycle:")
        _print_comparison(cp_qldpc, pr_qldpc,
                          ["72_12_6", "144_12_12", "288_12_18"])

    # ── Scaling analysis ──────────────────────────────────────────────
    native_thru_hdr = f"  {'Dict ops/ms':>14s}" if _HAS_NATIVE else ""
    packed_thru_hdr = f"  {'Packed ops/ms':>14s}" if _HAS_PACKED else ""
    print("\n" + "-" * W)
    print("  SCALING ANALYSIS (ops/ms throughput)")
    print("-" * W)
    print(f"  {'Name':<16s}  {'Ops':>8s}  "
          f"{'cp ops/ms':>10s}  {'ref ops/ms':>10s}  "
          f"{'cp+PyML ops/ms':>14s}{native_thru_hdr}{packed_thru_hdr}")
    print("  " + "-" * (W - 2))

    all_cp = cp_surface + cp_qldpc
    all_pr = (pr_surface + pr_qldpc) if pr_surface else [None] * len(all_cp)
    all_labels = [r.name for r in cp_surface
                 ] + ["72_12_6", "144_12_12", "288_12_18"]

    for cp_r, pr_r, lbl in zip(all_cp, all_pr, all_labels):
        cp_thru = cp_r.ops / cp_r.total_no_mlir_ms if cp_r.total_no_mlir_ms > 0 else 0
        cp_pyml_thru = cp_r.ops / cp_r.total_with_mlir_ms if cp_r.total_with_mlir_ms > 0 else 0
        pr_thru = pr_r.ops / pr_r.total_ms if pr_r and pr_r.total_ms > 0 else 0
        pr_str = f"{pr_thru:>10.0f}" if pr_r else f"{'N/A':>10s}"
        native_col = ""
        if _HAS_NATIVE:
            nat_thru = cp_r.ops / cp_r.total_with_native_ms if cp_r.total_with_native_ms > 0 else 0
            native_col = f"  {nat_thru:>14.0f}"
        packed_col = ""
        if _HAS_PACKED:
            pk_thru = cp_r.ops / cp_r.total_with_packed_ms if cp_r.total_with_packed_ms > 0 else 0
            packed_col = f"  {pk_thru:>14.0f}"
        print(f"  {lbl:<16s}  {cp_r.ops:>8d}  "
              f"{cp_thru:>10.0f}  {pr_str}  "
              f"{cp_pyml_thru:>14.0f}{native_col}{packed_col}")

    print("\n" + "=" * W)
    print("  Done.")
    print("=" * W)


if __name__ == "__main__":
    main()
