# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-comparison: cudaq-pulse vs pulse_ref compile times.

Usage:
    PYTHONPATH=python python benchmarks/compare_vs_pulse_ref.py \
        [--pulse-ref-dir /path/to/pulse_mlir_qce26/benchmarks] \
        [--iterations N]

Runs both cudaq-pulse and pulse_ref (if available) on the 6 canonical
workloads and prints a side-by-side comparison table.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cudaq_pulse as pulse
from cudaq_pulse.lower import _to_program as to_program
from cudaq_pulse.passes.verify import verify
from cudaq_pulse.passes.scheduling import schedule_alap

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent / "compiler"))
from compile_time import WORKLOADS


@dataclass
class CompareResult:
    name: str
    cudaq_pulse_ms: float
    pulse_ref_ms: float | None
    ratio: float | None


def _bench_cudaq_pulse(kernel_fn, n_qubits: int, freq_dict: dict,
                       iterations: int) -> float:
    total = 0.0
    for _ in range(iterations):
        t0 = time.monotonic()
        qubits = [pulse.qudit_ref() for _ in range(n_qubits)]
        ir = kernel_fn(*qubits)
        program = to_program(ir, clock_ghz=2.0, qubit_freq_hz=freq_dict)
        verify(program)
        schedule_alap(program)
        total += time.monotonic() - t0
    return (total / iterations) * 1000.0


def _try_bench_pulse_ref(pulse_ref_dir: str, workload_names: list[str],
                         iterations: int) -> dict[str, float] | None:
    """Attempt to import and benchmark pulse_ref. Returns None if unavailable."""
    bench_dir = Path(pulse_ref_dir)
    if not bench_dir.is_dir():
        return None

    sys.path.insert(0, str(bench_dir))
    try:
        from pulse_ref.sched import schedule as pr_schedule
        from workloads import ALL_WORKLOADS as PR_WORKLOADS
    except (ImportError, TypeError) as e:
        print(f"  [note] pulse_ref import failed: {e}")
        return None

    results = {}
    for key, mod in PR_WORKLOADS.items():
        if key not in workload_names:
            continue
        total = 0.0
        for _ in range(iterations):
            t0 = time.monotonic()
            p = mod.build()
            pr_schedule(p, policy="alap")
            total += time.monotonic() - t0
        results[key] = (total / iterations) * 1000.0

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare cudaq-pulse vs pulse_ref")
    parser.add_argument(
        "--pulse-ref-dir",
        default="/Users/anthonys/Downloads/pulse_mlir_qce26/benchmarks",
        help="Path to pulse_mlir_qce26/benchmarks directory")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    iterations = args.iterations
    print(f"Benchmarking with {iterations} iterations per workload...\n")

    workload_names = [name for name, _, _, _ in WORKLOADS]

    # cudaq-pulse
    cp_results = {}
    for name, kernel_fn, n_qubits, freq_dict in WORKLOADS:
        cp_results[name] = _bench_cudaq_pulse(kernel_fn, n_qubits, freq_dict,
                                              iterations)

    # pulse_ref
    pr_results = _try_bench_pulse_ref(args.pulse_ref_dir, workload_names,
                                      iterations)

    # Comparison table
    print(
        f"  {'Workload':20s}  {'cudaq-pulse':>12s}  {'pulse_ref':>12s}  {'Ratio':>8s}"
    )
    print(f"  {'-'*20}  {'-'*12}  {'-'*12}  {'-'*8}")

    comparisons = []
    for name in workload_names:
        cp_ms = cp_results[name]
        pr_ms = pr_results.get(name) if pr_results else None
        ratio = cp_ms / pr_ms if pr_ms else None
        comparisons.append(
            CompareResult(name=name,
                          cudaq_pulse_ms=cp_ms,
                          pulse_ref_ms=pr_ms,
                          ratio=ratio))
        pr_str = f"{pr_ms:.3f} ms" if pr_ms is not None else "N/A"
        ratio_str = f"{ratio:.2f}x" if ratio is not None else "N/A"
        print(f"  {name:20s}  {cp_ms:9.3f} ms  {pr_str:>12s}  {ratio_str:>8s}")

    print()
    if pr_results is None:
        print(
            "  pulse_ref not found. To enable comparison, pass --pulse-ref-dir."
        )
    else:
        ratios = [c.ratio for c in comparisons if c.ratio is not None]
        avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
        print(f"  Average ratio (cudaq-pulse / pulse_ref): {avg_ratio:.2f}x")
        within_gate = all(
            c.ratio is not None and c.ratio <= 1.05 for c in comparisons)
        print(f"  Within 5% gate: {'YES' if within_gate else 'NO'}")


if __name__ == "__main__":
    main()
