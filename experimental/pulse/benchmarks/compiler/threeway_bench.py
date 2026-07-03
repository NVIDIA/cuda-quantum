#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three-way performance benchmark: cudaq-pulse (MLIR path) vs cudaq-pulse
(pure Python) vs pulse_ref.

Measures wall-clock time for each stage:
  1. Bytecode capture (kernel -> IR)
  2. IR lowering   (to_program)
  3. Python passes  (verify + canonicalize + virtual-z + fusion + LICM)
  4. Scheduling     (ALAP)
  5. MLIR emission  (program_to_pulse_mlir)  -- cudaq-pulse only
  6. End-to-end     (sum of all stages)

Usage:
  PYTHONPATH=python python benchmarks/threeway_bench.py \
      [--pulse-ref-dir /path/to/pulse_mlir_qce26/benchmarks] \
      [--iterations 100]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# cudaq-pulse imports
# ---------------------------------------------------------------------------
import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes import (
    verify,
    run_canonicalize,
    run_virtual_z,
    run_fusion,
    run_licm,
    schedule_alap,
)
from cudaq_pulse.passes.to_pulse_mlir import program_to_pulse_mlir

# ---------------------------------------------------------------------------
# Canonical workloads (same kernels used by compile_time.py)
# ---------------------------------------------------------------------------


@pulse.kernel
def bell(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    shift_phase(t0, math.pi / 2)
    sx = drag(40, 0.25, 10.0, 0.5)
    drive(d0, sx, t0)
    shift_phase(t0, math.pi / 2)
    sync(d0, d1)
    drive(d1, sx, t1)
    cr = gaussian(200, 0.10, 50.0)
    drive(d0, cr, t1)
    x = square(40, 0.047 + 0j)
    drive(d0, x, t0)
    cr_neg = gaussian(200, -0.10, 50.0)
    drive(d0, cr_neg, t1)
    drive(d1, sx, t1)


@pulse.kernel
def cnot_cr(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    sync(d0, d1)
    sx = drag(40, 0.025, 10.0, 0.5)
    cr = gaussian(200, 0.10, 50.0)
    cr_neg = gaussian(200, -0.10, 50.0)
    x = square(40, 0.047 + 0j)
    drive(d1, sx, t1)
    drive(d0, cr, t1)
    drive(d0, x, t0)
    drive(d0, cr_neg, t1)
    drive(d1, sx, t1)


@pulse.kernel
def qaoa4(q0, q1, q2, q3):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)
    lines = [d0, d1, d2, d3]
    tones = [t0, t1, t2, t3]
    sync(lines[0], lines[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[1])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    shift_phase(tones[1], 0.6)
    sync(lines[1], lines[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    drive(lines[1], gaussian(200, 0.10, 50.0), tones[2])
    drive(lines[1], square(40, 0.047 + 0j), tones[1])
    drive(lines[1], gaussian(200, -0.10, 50.0), tones[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    shift_phase(tones[2], 0.6)
    sync(lines[2], lines[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    drive(lines[2], gaussian(200, 0.10, 50.0), tones[3])
    drive(lines[2], square(40, 0.047 + 0j), tones[2])
    drive(lines[2], gaussian(200, -0.10, 50.0), tones[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    shift_phase(tones[3], 0.6)
    sync(lines[3], lines[0])
    drive(lines[0], drag(40, 0.025, 10.0, 0.5), tones[0])
    drive(lines[3], gaussian(200, 0.10, 50.0), tones[0])
    drive(lines[3], square(40, 0.047 + 0j), tones[3])
    drive(lines[3], gaussian(200, -0.10, 50.0), tones[0])
    drive(lines[0], drag(40, 0.025, 10.0, 0.5), tones[0])
    shift_phase(tones[0], 0.6)


@pulse.kernel
def syndrome(q0, q1, q2, q3, q4):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)
    d4, t4 = get_drive_line(q4)
    lines = [d0, d1, d2, d3, d4]
    tones = [t0, t1, t2, t3, t4]
    sx = drag(40, 0.25, 10.0, 0.5)
    shift_phase(tones[0], math.pi / 2)
    drive(lines[0], sx, tones[0])
    shift_phase(tones[0], math.pi / 2)
    for data in range(1, 5):
        sync(lines[0], lines[data])
        drive(lines[data], drag(40, 0.025, 10.0, 0.5), tones[data])
        drive(lines[0], gaussian(200, 0.10, 50.0), tones[data])
        drive(lines[0], square(40, 0.047 + 0j), tones[0])
        drive(lines[0], gaussian(200, -0.10, 50.0), tones[data])
        drive(lines[data], drag(40, 0.025, 10.0, 0.5), tones[data])


@pulse.kernel
def dd_cpmg8(q0):
    d0, t0 = get_drive_line(q0)
    x = square(40, 0.047 + 0j)
    wait(d0, 100)
    for _ in range(8):
        drive(d0, x, t0)
        wait(d0, 200)


@pulse.kernel
def vqe_hea(q0, q1, q2, q3):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)
    lines = [d0, d1, d2, d3]
    tones = [t0, t1, t2, t3]
    sx = drag(40, 0.25, 10.0, 0.5)
    thetas = [0.3, 0.5, 0.7, 0.9]
    for q in range(4):
        shift_phase(tones[q], -math.pi / 2)
        drive(lines[q], sx, tones[q])
        shift_phase(tones[q], thetas[q])
        drive(lines[q], sx, tones[q])
        shift_phase(tones[q], math.pi / 2)
    sync(lines[0], lines[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[1])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    sync(lines[1], lines[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    drive(lines[1], gaussian(200, 0.10, 50.0), tones[2])
    drive(lines[1], square(40, 0.047 + 0j), tones[1])
    drive(lines[1], gaussian(200, -0.10, 50.0), tones[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    sync(lines[2], lines[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    drive(lines[2], gaussian(200, 0.10, 50.0), tones[3])
    drive(lines[2], square(40, 0.047 + 0j), tones[2])
    drive(lines[2], gaussian(200, -0.10, 50.0), tones[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])


WORKLOADS = [
    ("bell", bell, 2, {
        0: 5.0e9,
        1: 5.1e9
    }),
    ("cnot_cr", cnot_cr, 2, {
        0: 5.0e9,
        1: 5.1e9
    }),
    ("qaoa4", qaoa4, 4, {
        i: 5.0e9 + 0.1e9 * i for i in range(4)
    }),
    ("syndrome", syndrome, 5, {
        i: 5.0e9 + 0.1e9 * i for i in range(5)
    }),
    ("dd_cpmg8", dd_cpmg8, 1, {
        0: 5.0e9
    }),
    ("vqe_hea", vqe_hea, 4, {
        i: 5.0e9 + 0.1e9 * i for i in range(4)
    }),
]

# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _bench_us(fn: Callable, *args, repeats: int = 50) -> float:
    """Return median wall time in microseconds."""
    times = []
    for _ in range(3):
        fn(*args)
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn(*args)
        t1 = time.perf_counter_ns()
        times.append((t1 - t0) / 1000.0)
    return _median(times)


# ---------------------------------------------------------------------------
# cudaq-pulse pipeline (with MLIR emission)
# ---------------------------------------------------------------------------


@dataclass
class CudaqPulseTimings:
    name: str
    n_qubits: int
    op_count: int
    capture_us: float
    lower_us: float
    passes_us: float
    schedule_us: float
    mlir_emit_us: float
    total_us: float
    mlir_lines: int
    mlir_chars: int


def bench_cudaq_pulse(name: str, kern, n_qubits: int, freq_dict: dict,
                      iterations: int) -> CudaqPulseTimings:
    qubits = [pulse.qudit_ref() for _ in range(n_qubits)]

    def do_capture():
        return kern(*qubits)

    def do_lower(ir):
        return to_program(ir, clock_ghz=2.0, qubit_freq_hz=freq_dict)

    def do_passes(prog):
        p = run_canonicalize(prog)
        p = run_virtual_z(p)
        p = run_fusion(p)
        p = run_licm(p)
        return p

    def do_schedule(prog):
        schedule_alap(prog)

    def do_mlir(prog):
        return program_to_pulse_mlir(prog)

    ir = do_capture()
    prog = do_lower(ir)
    verify(prog)
    prog = do_passes(prog)
    do_schedule(prog)
    mlir_text = do_mlir(prog)

    capture_us = _bench_us(do_capture, repeats=iterations)
    ir_for_lower = do_capture()
    lower_us = _bench_us(do_lower, ir_for_lower, repeats=iterations)
    prog_for_passes = do_lower(do_capture())
    passes_us = _bench_us(do_passes, prog_for_passes, repeats=iterations)
    opt_prog = do_passes(do_lower(do_capture()))
    schedule_us = _bench_us(do_schedule, opt_prog, repeats=iterations)
    sched_prog = do_passes(do_lower(do_capture()))
    do_schedule(sched_prog)
    mlir_us = _bench_us(do_mlir, sched_prog, repeats=iterations)

    total_us = capture_us + lower_us + passes_us + schedule_us + mlir_us

    return CudaqPulseTimings(
        name=name,
        n_qubits=n_qubits,
        op_count=prog.op_count(),
        capture_us=capture_us,
        lower_us=lower_us,
        passes_us=passes_us,
        schedule_us=schedule_us,
        mlir_emit_us=mlir_us,
        total_us=total_us,
        mlir_lines=mlir_text.count("\n"),
        mlir_chars=len(mlir_text),
    )


# ---------------------------------------------------------------------------
# cudaq-pulse pipeline (pure Python only, no MLIR)
# ---------------------------------------------------------------------------


@dataclass
class PurePythonTimings:
    name: str
    n_qubits: int
    op_count: int
    capture_us: float
    lower_us: float
    passes_us: float
    schedule_us: float
    total_us: float


def bench_pure_python(name: str, kern, n_qubits: int, freq_dict: dict,
                      iterations: int) -> PurePythonTimings:
    qubits = [pulse.qudit_ref() for _ in range(n_qubits)]

    def do_e2e():
        ir = kern(*qubits)
        prog = to_program(ir, clock_ghz=2.0, qubit_freq_hz=freq_dict)
        verify(prog)
        prog = run_canonicalize(prog)
        prog = run_virtual_z(prog)
        prog = run_fusion(prog)
        prog = run_licm(prog)
        schedule_alap(prog)
        return prog

    # Warmup
    for _ in range(3):
        do_e2e()

    # Fine-grained timing
    def do_capture():
        return kern(*qubits)

    def do_lower(ir):
        return to_program(ir, clock_ghz=2.0, qubit_freq_hz=freq_dict)

    def do_passes(prog):
        p = run_canonicalize(prog)
        p = run_virtual_z(p)
        p = run_fusion(p)
        p = run_licm(p)
        return p

    def do_schedule(prog):
        schedule_alap(prog)

    capture_us = _bench_us(do_capture, repeats=iterations)
    ir_for_lower = do_capture()
    lower_us = _bench_us(do_lower, ir_for_lower, repeats=iterations)
    prog_for_passes = do_lower(do_capture())
    passes_us = _bench_us(do_passes, prog_for_passes, repeats=iterations)
    opt_prog = do_passes(do_lower(do_capture()))
    schedule_us = _bench_us(do_schedule, opt_prog, repeats=iterations)

    total_us = capture_us + lower_us + passes_us + schedule_us

    return PurePythonTimings(
        name=name,
        n_qubits=n_qubits,
        op_count=opt_prog.op_count(),
        capture_us=capture_us,
        lower_us=lower_us,
        passes_us=passes_us,
        schedule_us=schedule_us,
        total_us=total_us,
    )


# ---------------------------------------------------------------------------
# pulse_ref benchmark
# ---------------------------------------------------------------------------


@dataclass
class PulseRefTimings:
    name: str
    n_qubits: int
    op_count: int
    build_us: float
    schedule_us: float
    total_us: float
    schedule_length_ns: float


def bench_pulse_ref(ref_dir: str,
                    iterations: int) -> dict[str, PulseRefTimings]:
    bench_dir = Path(ref_dir)
    if not bench_dir.is_dir():
        return {}

    sys.path.insert(0, str(bench_dir))
    try:
        from workloads import ALL_WORKLOADS as PR_WL
        from pulse_ref.sched import schedule as pr_schedule
    except (ImportError, TypeError) as e:
        print(f"  [note] pulse_ref import failed: {e}")
        return {}

    results = {}
    name_map = {
        "bell": "bell",
        "cnot_cr": "cnot_cr",
        "qaoa4": "qaoa4",
        "syndrome": "syndrome",
        "dd_cpmg8": "dd_cpmg8",
        "vqe_hea": "vqe_hea",
    }

    for our_name, ref_key in name_map.items():
        if ref_key not in PR_WL:
            continue
        mod = PR_WL[ref_key]

        # Warmup
        for _ in range(3):
            p = mod.build()
            pr_schedule(p, policy="alap")

        def do_build():
            return mod.build()

        def do_sched(p):
            return pr_schedule(p, policy="alap")

        build_us = _bench_us(do_build, repeats=iterations)
        p = mod.build()
        schedule_us = _bench_us(do_sched, p, repeats=iterations)

        _, metrics = pr_schedule(mod.build(), policy="alap")

        results[our_name] = PulseRefTimings(
            name=our_name,
            n_qubits=mod.NUM_QUBITS,
            op_count=metrics.op_count,
            build_us=build_us,
            schedule_us=schedule_us,
            total_us=build_us + schedule_us,
            schedule_length_ns=metrics.total_length_ns,
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Three-way benchmark: cudaq-pulse (MLIR) vs "
        "cudaq-pulse (Python) vs pulse_ref")
    parser.add_argument("--pulse-ref-dir",
                        default=None,
                        help="Path to pulse_mlir_qce26/benchmarks")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    ref_dir = args.pulse_ref_dir
    if ref_dir is None:
        for candidate in [
                "../pulse_ref_benchmarks",
                "../../pulse_ref_benchmarks",
                os.path.expanduser("~/Downloads/pulse_mlir_qce26/benchmarks"),
        ]:
            if Path(candidate).is_dir():
                ref_dir = candidate
                break

    iters = args.iterations
    W = 72

    print("=" * W)
    print("  cudaq-pulse Three-Way Performance Benchmark")
    print(f"  {iters} iterations per measurement (median)")
    print("=" * W)

    # ── cudaq-pulse with MLIR ─────────────────────────────────────────
    print("\n[1/3] cudaq-pulse (bytecode -> Python passes -> MLIR emission)")
    print("-" * W)
    hdr = (f"  {'Workload':<14s}  {'Capture':>8s}  {'Lower':>8s}  "
           f"{'Passes':>8s}  {'Sched':>8s}  {'MLIR':>8s}  {'Total':>10s}")
    print(hdr)
    print(f"  {'':14s}  {'(us)':>8s}  {'(us)':>8s}  {'(us)':>8s}  "
          f"{'(us)':>8s}  {'(us)':>8s}  {'(us)':>10s}")
    print("  " + "-" * (W - 2))

    mlir_results: list[CudaqPulseTimings] = []
    for name, kern, nq, freqs in WORKLOADS:
        r = bench_cudaq_pulse(name, kern, nq, freqs, iters)
        mlir_results.append(r)
        print(f"  {r.name:<14s}  {r.capture_us:>8.1f}  {r.lower_us:>8.1f}  "
              f"{r.passes_us:>8.1f}  {r.schedule_us:>8.1f}  "
              f"{r.mlir_emit_us:>8.1f}  {r.total_us:>10.1f}")

    # ── cudaq-pulse pure Python ───────────────────────────────────────
    print(f"\n[2/3] cudaq-pulse (bytecode -> Python passes only, no MLIR)")
    print("-" * W)
    hdr2 = (f"  {'Workload':<14s}  {'Capture':>8s}  {'Lower':>8s}  "
            f"{'Passes':>8s}  {'Sched':>8s}  {'Total':>10s}")
    print(hdr2)
    print(f"  {'':14s}  {'(us)':>8s}  {'(us)':>8s}  {'(us)':>8s}  "
          f"{'(us)':>8s}  {'(us)':>10s}")
    print("  " + "-" * (W - 2))

    py_results: list[PurePythonTimings] = []
    for name, kern, nq, freqs in WORKLOADS:
        r = bench_pure_python(name, kern, nq, freqs, iters)
        py_results.append(r)
        print(f"  {r.name:<14s}  {r.capture_us:>8.1f}  {r.lower_us:>8.1f}  "
              f"{r.passes_us:>8.1f}  {r.schedule_us:>8.1f}  "
              f"{r.total_us:>10.1f}")

    # ── cudaq-pulse compile() API (single-call, uses native if available) ──
    @dataclass
    class CompileTimings:
        name: str
        total_us: float
        capture_us: float
        lower_us: float
        passes_us: float
        schedule_us: float
        mlir_emit_us: float

    print(
        f"\n[3/4] cudaq-pulse compile() API (single call, native if available)")
    print("-" * W)
    hdr_c = (f"  {'Workload':<14s}  {'Capture':>8s}  {'Lower':>8s}  "
             f"{'Passes':>8s}  {'Sched':>8s}  {'MLIR':>8s}  {'Total':>10s}")
    print(hdr_c)
    print(f"  {'':14s}  {'(us)':>8s}  {'(us)':>8s}  {'(us)':>8s}  "
          f"{'(us)':>8s}  {'(us)':>8s}  {'(us)':>10s}")
    print("  " + "-" * (W - 2))

    compile_results: list[CompileTimings] = []
    for name, kern, nq, freqs in WORKLOADS:
        qubits = [pulse.qudit_ref() for _ in range(nq)]
        for _ in range(3):
            pulse.compile(kern,
                          qubits,
                          clock_ghz=2.0,
                          qubit_freq_hz=freqs,
                          schedule="alap")
        times = []
        for _ in range(iters):
            t0 = time.perf_counter_ns()
            ck = pulse.compile(kern,
                               qubits,
                               clock_ghz=2.0,
                               qubit_freq_hz=freqs,
                               schedule="alap")
            times.append((time.perf_counter_ns() - t0) / 1000.0)
        total_us = _median(times)
        m = ck.metrics
        cr = CompileTimings(
            name=name,
            total_us=total_us,
            capture_us=m.capture_ms * 1000,
            lower_us=m.lower_ms * 1000,
            passes_us=m.passes_ms * 1000,
            schedule_us=m.schedule_ms * 1000,
            mlir_emit_us=m.mlir_emit_ms * 1000,
        )
        compile_results.append(cr)
        print(f"  {cr.name:<14s}  {cr.capture_us:>8.1f}  {cr.lower_us:>8.1f}  "
              f"{cr.passes_us:>8.1f}  {cr.schedule_us:>8.1f}  "
              f"{cr.mlir_emit_us:>8.1f}  {cr.total_us:>10.1f}")

    # ── pulse_ref ─────────────────────────────────────────────────────
    ref_results: dict[str, PulseRefTimings] = {}
    if ref_dir:
        print(f"\n[4/4] pulse_ref (reference Python implementation)")
        print("-" * W)
        ref_results = bench_pulse_ref(ref_dir, iters)
        if ref_results:
            hdr3 = (f"  {'Workload':<14s}  {'Build':>8s}  {'Sched':>8s}  "
                    f"{'Total':>10s}  {'Ops':>6s}")
            print(hdr3)
            print(f"  {'':14s}  {'(us)':>8s}  {'(us)':>8s}  "
                  f"{'(us)':>10s}  {'':>6s}")
            print("  " + "-" * (W - 2))
            for name, _, _, _ in WORKLOADS:
                if name in ref_results:
                    rr = ref_results[name]
                    print(f"  {rr.name:<14s}  {rr.build_us:>8.1f}  "
                          f"{rr.schedule_us:>8.1f}  {rr.total_us:>10.1f}  "
                          f"{rr.op_count:>6d}")
        else:
            print("  pulse_ref not available (import failed or dir missing).")
    else:
        print(f"\n[4/4] pulse_ref: SKIPPED (--pulse-ref-dir not provided)")

    # ── Summary comparison table ──────────────────────────────────────
    print("\n" + "=" * W)
    print("  COMPARISON SUMMARY  (all times in microseconds)")
    print("=" * W)
    hdr_cmp = (f"  {'Workload':<14s}  {'Python':>10s}  {'+ PyMLIR':>10s}  "
               f"{'compile()':>10s}  {'pulse_ref':>10s}  "
               f"{'comp/Ref':>8s}")
    print(hdr_cmp)
    print("  " + "-" * (W - 2))

    for i, (name, _, _, _) in enumerate(WORKLOADS):
        py_us = py_results[i].total_us
        mlir_us = mlir_results[i].total_us
        comp_us = compile_results[i].total_us
        ref_us = ref_results[name].total_us if name in ref_results else None

        comp_ref = comp_us / ref_us if ref_us and ref_us > 0 else None

        ref_str = f"{ref_us:>10.1f}" if ref_us is not None else f"{'N/A':>10s}"
        comp_ratio = f"{comp_ref:>7.2f}x" if comp_ref else f"{'N/A':>8s}"

        print(f"  {name:<14s}  {py_us:>10.1f}  {mlir_us:>10.1f}  "
              f"{comp_us:>10.1f}  {ref_str}  {comp_ratio}")

    # ── Overhead analysis ─────────────────────────────────────────────
    print("\n" + "-" * W)
    print("  MLIR EMISSION OVERHEAD")
    print("-" * W)
    for i, r in enumerate(mlir_results):
        pct = (r.mlir_emit_us / r.total_us * 100) if r.total_us > 0 else 0
        print(f"  {r.name:<14s}  MLIR emit: {r.mlir_emit_us:>8.1f} us  "
              f"({pct:>5.1f}% of total)  "
              f"  [{r.mlir_lines} lines, {r.mlir_chars} chars]")

    # ── Stage breakdown ───────────────────────────────────────────────
    print("\n" + "-" * W)
    print("  STAGE BREAKDOWN (% of total, cudaq-pulse + MLIR path)")
    print("-" * W)
    for r in mlir_results:
        tot = r.total_us if r.total_us > 0 else 1
        print(f"  {r.name:<14s}  "
              f"cap={r.capture_us / tot * 100:>4.1f}%  "
              f"low={r.lower_us / tot * 100:>4.1f}%  "
              f"pass={r.passes_us / tot * 100:>4.1f}%  "
              f"sched={r.schedule_us / tot * 100:>4.1f}%  "
              f"mlir={r.mlir_emit_us / tot * 100:>4.1f}%")

    print("\n" + "=" * W)
    print("  Done.")
    print("=" * W)


if __name__ == "__main__":
    main()
