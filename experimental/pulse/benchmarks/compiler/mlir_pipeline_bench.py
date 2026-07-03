#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmarks for the cudaq-pulse MLIR pipeline.

Measures wall-clock time for each stage of the compilation pipeline:
  1. Python passes (verify + canonicalize + vz + fusion + schedule)
  2. program_to_pulse_mlir() emission
  3. Full pipeline timing (if GPU available)

Usage:
  python benchmarks/mlir_pipeline_bench.py
"""

import math
import time
from typing import Any, Callable, Tuple

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
from cudaq_pulse.passes.ir_types import Program


def _time_fn(fn: Callable, *args: Any, repeats: int = 10) -> float:
    """Time a function, return median wall time in ms."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


# --- 1Q Rabi kernel ---
@pulse.kernel
def rabi_1q(q0):
    d0, t0 = get_drive_line(q0)
    wf = gaussian(100, 0.1, 25.0)
    drive(d0, wf, t0)


# --- 2Q CR-CNOT kernel ---
@pulse.kernel
def cr_cnot_2q(q0, q1):
    d0, t0 = get_drive_line(q0)
    d1, t1 = get_drive_line(q1)
    x90 = gaussian(40, 0.25, 10.0)
    drive(d0, x90, t0)
    sync(d0, d1)
    cr = square(160, 0.05)
    drive(d0, cr, t0)
    sync(d0, d1)
    drive(d1, x90, t1)


# --- Echo DD kernel ---
@pulse.kernel
def echo_dd(q0):
    d0, t0 = get_drive_line(q0)
    for i in range(10):
        wf = gaussian(40, 0.3, 10.0)
        drive(d0, wf, t0)
        wait(d0, 100)
        shift_phase(t0, math.pi)
        wf_neg = gaussian(40, 0.3, 10.0)
        drive(d0, wf_neg, t0)
        wait(d0, 100)


def _build_program(kern, *refs, n_qubits=1):
    ir = kern(*refs)
    freq_map = {i: 5.0e9 + i * 0.1e9 for i in range(n_qubits)}
    return to_program(ir, clock_ghz=2.0, qubit_freq_hz=freq_map)


def _run_passes(prog: Program) -> Program:
    prog = run_canonicalize(prog)
    prog = run_virtual_z(prog)
    prog = run_fusion(prog)
    prog = run_licm(prog)
    schedule_alap(prog)
    return prog


def bench_python_passes(name: str, prog: Program) -> None:
    ms = _time_fn(_run_passes, prog)
    print(f"  Python passes ({name}): {ms:.2f} ms")


def bench_mlir_emission(name: str, prog: Program) -> None:
    optimized = _run_passes(prog)
    ms = _time_fn(program_to_pulse_mlir, optimized)
    print(f"  MLIR emission ({name}): {ms:.2f} ms")


def bench_mlir_text_size(name: str, prog: Program) -> None:
    optimized = _run_passes(prog)
    mlir = program_to_pulse_mlir(optimized)
    lines = mlir.count("\n")
    chars = len(mlir)
    print(f"  MLIR size ({name}): {lines} lines, {chars} chars")


def main():
    print("=" * 60)
    print("cudaq-pulse MLIR Pipeline Benchmarks")
    print("=" * 60)

    benchmarks = [
        ("1Q Rabi", _build_program(rabi_1q, pulse.qudit_ref(), n_qubits=1)),
        ("2Q CR-CNOT",
         _build_program(cr_cnot_2q,
                        pulse.qudit_ref(),
                        pulse.qudit_ref(),
                        n_qubits=2)),
        ("Echo DD (10x)", _build_program(echo_dd, pulse.qudit_ref(),
                                         n_qubits=1)),
    ]

    print("\n--- Python Pass Timings ---")
    for name, prog in benchmarks:
        bench_python_passes(name, prog)

    print("\n--- MLIR Emission Timings ---")
    for name, prog in benchmarks:
        bench_mlir_emission(name, prog)

    print("\n--- MLIR Text Size ---")
    for name, prog in benchmarks:
        bench_mlir_text_size(name, prog)

    print("\n--- Program Statistics ---")
    for name, prog in benchmarks:
        print(f"  {name}: {prog.op_count()} ops, "
              f"{len(prog.qubit_freq_hz)} qubits")

    print("\n" + "=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
