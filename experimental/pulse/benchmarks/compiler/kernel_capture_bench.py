#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Microbenchmark: bytecode kernel capture latency.

Measures the time to compile a pulse kernel into IR using the bytecode
backend across the 6 canonical workloads from the paper benchmarks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cudaq_pulse as pulse

# ── Canonical workloads ──────────────────────────────────────────────


@pulse.kernel
def wl_bell(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    wf_pi2 = gaussian(40, 0.3, 10.0)
    wf_cr = square(246, 0.04)
    drive(d0, wf_pi2, t0)
    drive(d1, wf_cr, t1)
    drive(d0, wf_pi2, t0)


@pulse.kernel
def wl_cnot_cr(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    wf_pi2 = drag(40, 0.435, 5.0, 0.75)
    wf_cr = square(246, 0.04)
    drive(d0, wf_pi2, t0)
    drive(d1, wf_cr, t1)
    drive(d0, wf_pi2, t0)


@pulse.kernel
def wl_qaoa4(q0, q1, q2, q3):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)
    wf_pi2 = gaussian(40, 0.3, 10.0)
    wf_zz = square(100, 0.1)
    drive(d0, wf_pi2, t0)
    drive(d1, wf_pi2, t1)
    drive(d2, wf_pi2, t2)
    drive(d3, wf_pi2, t3)
    drive(d0, wf_zz, t0)
    drive(d1, wf_zz, t1)
    drive(d2, wf_zz, t2)
    drive(d3, wf_zz, t3)
    shift_phase(t0, 0.5)
    shift_phase(t1, 0.5)
    shift_phase(t2, 0.5)
    shift_phase(t3, 0.5)
    drive(d0, wf_pi2, t0)
    drive(d1, wf_pi2, t1)
    drive(d2, wf_pi2, t2)
    drive(d3, wf_pi2, t3)


@pulse.kernel
def wl_syndrome(q0, q1, q2, q3, q4):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)
    d4, t4 = get_drive_line(q4)
    wf_pi2 = gaussian(40, 0.3, 10.0)
    wf_cz = square(98, 0.5)
    drive(d4, wf_pi2, t4)
    drive(d0, wf_cz, t0)
    drive(d4, wf_pi2, t4)
    drive(d1, wf_cz, t1)
    drive(d4, wf_pi2, t4)
    drive(d2, wf_cz, t2)
    drive(d4, wf_pi2, t4)
    drive(d3, wf_cz, t3)
    r4, tr4 = get_readout_line(q4)
    wf_ro = square(600, 0.1)
    readout(r4, wf_ro, tr4)


@pulse.kernel
def wl_dd_cpmg8(q0):
    d0, t0 = get_drive_line(q0)
    wf_pi = gaussian(40, 0.6, 10.0)
    wf_pi2 = gaussian(40, 0.3, 10.0)
    drive(d0, wf_pi2, t0)
    for i in range(8):
        wait(d0, 100)
        drive(d0, wf_pi, t0)
    wait(d0, 100)
    drive(d0, wf_pi2, t0)


@pulse.kernel
def wl_vqe_hea(q0, q1, q2, q3):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    d2, t2 = get_drive_line(q2)
    d3, t3 = get_drive_line(q3)
    wf_rx = gaussian(40, 0.3, 10.0)
    wf_cr = square(246, 0.04)
    for i in range(3):
        drive(d0, wf_rx, t0)
        drive(d1, wf_rx, t1)
        drive(d2, wf_rx, t2)
        drive(d3, wf_rx, t3)
        shift_phase(t0, 0.5)
        shift_phase(t1, 0.5)
        shift_phase(t2, 0.5)
        shift_phase(t3, 0.5)
        drive(d0, wf_cr, t0)
        drive(d1, wf_cr, t1)
        drive(d2, wf_cr, t2)


WORKLOADS = [
    ("bell", wl_bell, 2),
    ("cnot_cr", wl_cnot_cr, 2),
    ("qaoa4", wl_qaoa4, 4),
    ("syndrome", wl_syndrome, 5),
    ("dd_cpmg8", wl_dd_cpmg8, 1),
    ("vqe_hea", wl_vqe_hea, 4),
]

# ── Benchmark runner ─────────────────────────────────────────────────


@dataclass
class BenchResult:
    name: str
    bc_us: float
    ops: int


def bench_one(name: str,
              fn,
              n_qubits: int,
              iterations: int = 200) -> BenchResult:
    qubits = [pulse.qudit_ref() for _ in range(n_qubits)]

    for _ in range(5):
        fn(*qubits)
    t0 = time.perf_counter_ns()
    for _ in range(iterations):
        ir = fn(*qubits)
    bc_ns = (time.perf_counter_ns() - t0) / iterations

    return BenchResult(name=name, bc_us=bc_ns / 1000, ops=len(ir.ops))


def main():
    print(
        f"Running bytecode kernel capture benchmarks (200 iterations each)...\n"
    )
    print(f"  {'Workload':<20s}  {'Latency (us)':>12s}  {'Ops':>5s}")
    print(f"  {'-'*20}  {'-'*12}  {'-'*5}")

    results = []
    for name, fn, nq in WORKLOADS:
        r = bench_one(name, fn, nq)
        results.append(r)
        print(f"  {r.name:<20s}  {r.bc_us:>12.1f}  {r.ops:>5d}")

    print()
    bc_total = sum(r.bc_us for r in results)
    print(f"  Total: {bc_total:.1f} us")


if __name__ == "__main__":
    main()
