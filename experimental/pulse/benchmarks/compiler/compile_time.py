# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile-time benchmark: cudaq-pulse on canonical workloads.

Usage:
    PYTHONPATH=python python benchmarks/compile_time.py [--gate FACTOR] [--iterations N]

Compiles each of the 6 canonical workloads, verifies, and schedules.
If any workload's compile time regresses beyond the gate factor, exit 1.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import cudaq_pulse as pulse

from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.verify import verify
from cudaq_pulse.passes.scheduling import schedule_alap


@dataclass
class BenchResult:
    name: str
    ms: float
    op_count: int


# ── Workload kernels ─────────────────────────────────────────────────────────


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

    # Edge (0, 1)
    sync(lines[0], lines[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[1])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    shift_phase(tones[1], 0.6)

    # Edge (1, 2)
    sync(lines[1], lines[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    drive(lines[1], gaussian(200, 0.10, 50.0), tones[2])
    drive(lines[1], square(40, 0.047 + 0j), tones[1])
    drive(lines[1], gaussian(200, -0.10, 50.0), tones[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    shift_phase(tones[2], 0.6)

    # Edge (2, 3)
    sync(lines[2], lines[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    drive(lines[2], gaussian(200, 0.10, 50.0), tones[3])
    drive(lines[2], square(40, 0.047 + 0j), tones[2])
    drive(lines[2], gaussian(200, -0.10, 50.0), tones[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    shift_phase(tones[3], 0.6)

    # Edge (3, 0)
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

    # data = 1
    sync(lines[0], lines[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[1])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])

    # data = 2
    sync(lines[0], lines[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[2])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])

    # data = 3
    sync(lines[0], lines[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[3])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])

    # data = 4
    sync(lines[0], lines[4])
    drive(lines[4], drag(40, 0.025, 10.0, 0.5), tones[4])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[4])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[4])
    drive(lines[4], drag(40, 0.025, 10.0, 0.5), tones[4])


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

    # Edge (0, 1)
    sync(lines[0], lines[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])
    drive(lines[0], gaussian(200, 0.10, 50.0), tones[1])
    drive(lines[0], square(40, 0.047 + 0j), tones[0])
    drive(lines[0], gaussian(200, -0.10, 50.0), tones[1])
    drive(lines[1], drag(40, 0.025, 10.0, 0.5), tones[1])

    # Edge (1, 2)
    sync(lines[1], lines[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])
    drive(lines[1], gaussian(200, 0.10, 50.0), tones[2])
    drive(lines[1], square(40, 0.047 + 0j), tones[1])
    drive(lines[1], gaussian(200, -0.10, 50.0), tones[2])
    drive(lines[2], drag(40, 0.025, 10.0, 0.5), tones[2])

    # Edge (2, 3)
    sync(lines[2], lines[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])
    drive(lines[2], gaussian(200, 0.10, 50.0), tones[3])
    drive(lines[2], square(40, 0.047 + 0j), tones[2])
    drive(lines[2], gaussian(200, -0.10, 50.0), tones[3])
    drive(lines[3], drag(40, 0.025, 10.0, 0.5), tones[3])


# ── Workload registry: (name, kernel_fn, n_qubits, freq_dict) ───────────────

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


def _bench(kernel_fn, n_qubits: int, freq_dict: dict,
           iterations: int) -> tuple[float, int]:
    total = 0.0
    op_count = 0
    for _ in range(iterations):
        t0 = time.monotonic()
        qubits = [pulse.qudit_ref() for _ in range(n_qubits)]
        ir = kernel_fn(*qubits)
        program = to_program(ir, clock_ghz=2.0, qubit_freq_hz=freq_dict)
        verify(program)
        schedule_alap(program)
        total += time.monotonic() - t0
        op_count = program.op_count()
    return (total / iterations) * 1000.0, op_count


def main():
    parser = argparse.ArgumentParser(description="Compile-time benchmark")
    parser.add_argument("--gate",
                        type=float,
                        default=1.05,
                        help="Regression gate factor (default 1.05 = 5%%)")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    print(f"Running {args.iterations} iterations per workload...\n")
    print(f"  {'Workload':20s}  {'Time (ms)':>10s}  {'Ops':>6s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*6}")

    results: list[BenchResult] = []
    for name, kernel_fn, n_qubits, freq_dict in WORKLOADS:
        ms, ops = _bench(kernel_fn, n_qubits, freq_dict, args.iterations)
        results.append(BenchResult(name=name, ms=ms, op_count=ops))
        print(f"  {name:20s}  {ms:10.3f}  {ops:6d}")

    print(f"\n--- Summary ---")
    for r in results:
        print(f"  {r.name}: {r.ms:.3f} ms ({r.op_count} ops)")

    print(f"\nAll {len(results)} workloads compiled successfully.")


if __name__ == "__main__":
    main()
