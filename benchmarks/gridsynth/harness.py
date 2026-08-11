# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Benchmark runner for GridSynth tuning.

Runs a (implementation x angle x epsilon x options) grid and writes one tidy
JSON record per case. Every result is verified against the requested
tolerance before it counts as a success -- a fast wrong answer is a failure,
not a win.

Each case runs in its own process so a single pathological case can be capped
and killed rather than hanging the sweep. At the current defaults one call at
eps=1e-50 takes ~25 s and eps=1e-60 does not finish in minutes, so this cap is
load-bearing, not defensive. Timing is measured inside the child around the
synthesis call alone, so interpreter and import cost stay out of the number.

Usage:
    python3 -m benchmarks.gridsynth.harness --out results.json
    python3 -m benchmarks.gridsynth.harness --split tuning --sweep-timeouts
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import corpus

# Sentinel for a case that exceeded its wall-clock cap. Distinct from a
# synthesis failure: the algorithm did not decline, we stopped it.
OUTCOME_TIMEOUT = "harness_timeout"
OUTCOME_SUCCESS = "success"
OUTCOME_ERROR = "error"
OUTCOME_TOLERANCE_VIOLATED = "tolerance_violated"


@dataclass
class Case:
    """One unit of benchmark work."""

    implementation: str
    angle: str
    family: str
    split: str
    theta: str
    epsilon: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class Result:
    """Outcome of one case. Serialized verbatim to JSON."""

    implementation: str
    angle: str
    family: str
    split: str
    theta: str
    epsilon: str
    options: dict[str, Any]
    outcome: str
    seconds: float | None = None
    t_count: int | None = None
    gate_count: int | None = None
    achieved_error: float | None = None
    detail: str | None = None


# --------------------------------------------------------------------------- #
# Implementations
# --------------------------------------------------------------------------- #


def _run_cudaq(theta: str, epsilon: str, options: dict[str, Any]) -> dict:
    """Synthesize with the in-tree CUDA-Q gridsynth.

    Runs in the child process. Returns the raw measurement; verification
    against the tolerance happens in the caller so it is applied identically
    to every implementation.
    """
    from cudaq import synth

    kwargs = {}
    for key in ("diophantine_timeout_ms", "factoring_timeout_ms"):
        if key in options:
            kwargs[key] = options[key]

    start = time.perf_counter()
    sequence = synth.gridsynth(theta, epsilon, **kwargs)
    seconds = time.perf_counter() - start

    gates = str(sequence)
    return {
        "seconds": seconds,
        "gates": gates,
        "t_count": gates.count("T"),
        "gate_count": len(sequence),
        # Measured with CUDA-Q's exact operator-norm oracle. It reconstructs
        # the unitary from the gate string, so it validates any
        # implementation's output, not just our own.
        "achieved_error": synth.rz_error(theta, gates),
    }


IMPLEMENTATIONS = {
    "cudaq": _run_cudaq,
}

# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #


def _child(queue: mp.Queue, implementation: str, theta: str, epsilon: str,
           options: dict[str, Any]) -> None:
    """Process entry point: run one case, put the raw measurement on `queue`."""
    try:
        queue.put(("ok", IMPLEMENTATIONS[implementation](theta, epsilon,
                                                         options)))
    except Exception as exc:  # noqa: BLE001 - reported, not handled
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


def run_case(case: Case, timeout_s: float) -> Result:
    """Run one case in a subprocess, capped at `timeout_s`."""
    base = {
        "implementation": case.implementation,
        "angle": case.angle,
        "family": case.family,
        "split": case.split,
        "theta": case.theta,
        "epsilon": case.epsilon,
        "options": case.options,
    }

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_child,
                       args=(queue, case.implementation, case.theta,
                             case.epsilon, case.options))
    proc.start()
    proc.join(timeout_s)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        return Result(**base,
                      outcome=OUTCOME_TIMEOUT,
                      detail=f"exceeded {timeout_s}s wall-clock cap")

    if queue.empty():
        return Result(**base,
                      outcome=OUTCOME_ERROR,
                      detail=f"child exited with code {proc.exitcode}")

    status, payload = queue.get()
    if status == "error":
        return Result(**base, outcome=OUTCOME_ERROR, detail=payload)

    # Correctness gate, applied uniformly: the requested tolerance is a
    # contract, so exceeding it is a failed case regardless of how fast it was.
    tolerance = float(case.epsilon)
    achieved = payload["achieved_error"]
    outcome = (OUTCOME_SUCCESS
               if achieved <= tolerance else OUTCOME_TOLERANCE_VIOLATED)

    return Result(**base,
                  outcome=outcome,
                  seconds=payload["seconds"],
                  t_count=payload["t_count"],
                  gate_count=payload["gate_count"],
                  achieved_error=achieved)


# --------------------------------------------------------------------------- #
# Grid construction
# --------------------------------------------------------------------------- #

# Per-candidate budget settings swept in the tuning phase. The current
# shipped default is 200/50.
TIMEOUT_SWEEP = [{
    "diophantine_timeout_ms": d,
    "factoring_timeout_ms": f
} for d, f in ((5, 2), (20, 5), (50, 10), (200, 50), (1000, 200))]


def build_cases(implementations: list[str], split: str | None,
                epsilons: list[str], option_sets: list[dict]) -> list[Case]:
    """Expand the corpus into the full case grid."""
    cases = []
    for angle in corpus.angles():
        if split and angle.split != split:
            continue
        for epsilon in epsilons:
            for implementation in implementations:
                for options in option_sets:
                    cases.append(
                        Case(implementation=implementation,
                             angle=angle.name,
                             family=angle.family,
                             split=angle.split,
                             theta=angle.theta,
                             epsilon=epsilon,
                             options=dict(options)))
    return cases


# --------------------------------------------------------------------------- #
# Host metadata
# --------------------------------------------------------------------------- #


def host_metadata() -> dict[str, Any]:
    """Provenance recorded alongside results.

    Absolute timings are only interpretable against the machine that produced
    them, so a results file without this is not evidence.
    """

    def _cmd(args: list[str]) -> str | None:
        try:
            return subprocess.run(
                args, capture_output=True, text=True,
                timeout=10).stdout.strip() or None
        except Exception:  # noqa: BLE001 - metadata is best-effort
            return None

    cpu = None
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass

    return {
        "cpu": cpu,
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_cmd(["git", "status", "--porcelain"])),
        "corpus_seed": corpus.CORPUS_SEED,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default="gridsynth-results.json",
                        help="Output JSON path.")
    parser.add_argument("--split",
                        choices=("tuning", "heldout"),
                        default=None,
                        help="Restrict to one corpus split (default: both).")
    parser.add_argument("--implementations",
                        default="cudaq",
                        help="Comma-separated implementation names.")
    parser.add_argument("--timeout",
                        type=float,
                        default=60.0,
                        help="Per-case wall-clock cap in seconds.")
    parser.add_argument("--deep",
                        action="store_true",
                        help="Include the deep-epsilon tolerances. Slow: "
                        "budget a large --timeout.")
    parser.add_argument("--sweep-timeouts",
                        action="store_true",
                        help="Sweep the per-candidate budget settings "
                        "instead of using the shipped defaults.")
    parser.add_argument("--limit",
                        type=int,
                        default=None,
                        help="Run only the first N cases (smoke testing).")
    args = parser.parse_args(argv)

    epsilons = list(corpus.EPSILONS)
    if args.deep:
        epsilons += corpus.DEEP_EPSILONS

    option_sets = TIMEOUT_SWEEP if args.sweep_timeouts else [{}]
    implementations = args.implementations.split(",")

    unknown = set(implementations) - set(IMPLEMENTATIONS)
    if unknown:
        parser.error(f"unknown implementation(s): {sorted(unknown)}; "
                     f"available: {sorted(IMPLEMENTATIONS)}")

    cases = build_cases(implementations, args.split, epsilons, option_sets)
    if args.limit:
        cases = cases[:args.limit]

    print(corpus.summary())
    print(f"{len(cases)} cases, {args.timeout}s cap each", flush=True)

    results = []
    for index, case in enumerate(cases, 1):
        result = run_case(case, args.timeout)
        results.append(result)
        elapsed = "     -" if result.seconds is None else f"{result.seconds * 1000:6.1f}ms"
        print(
            f"[{index:5d}/{len(cases)}] {case.angle:<28s} {case.epsilon:<6s} "
            f"{elapsed}  T={result.t_count}  {result.outcome}",
            flush=True)

    payload = {
        "metadata": host_metadata(),
        "results": [asdict(r) for r in results],
    }
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)

    counts: dict[str, int] = {}
    for result in results:
        counts[result.outcome] = counts.get(result.outcome, 0) + 1
    print(f"\nwrote {args.out}: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # Any non-success is worth a non-zero exit so CI notices.
    return 0 if set(counts) <= {OUTCOME_SUCCESS} else 1


if __name__ == "__main__":
    sys.exit(main())
