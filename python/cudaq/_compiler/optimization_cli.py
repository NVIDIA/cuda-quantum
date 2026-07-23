# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Command-line interface for the Optimization Validation Core.

Agents and CI invoke this from a standalone CUDA-Q checkout.
(Taken this example from Thomas's plan)

    `python3 -m cudaq._compiler.optimization_cli \\`
      `--input cudaq/test/Transforms/commutation_cancellation.qke \\`
      `--prepare 'builtin.module(func.func(memtoreg))' \\`
      `--candidate 'builtin.module(func.func(quake-commutation-cancellation))' \\`
      `--oracle strict-unitary \\`
      `--metric operation-count:nonincreasing \\`
      `--metric two-qubit-count:nonincreasing \\`
      `--fixed-point-runs 1 \\`
      `--preset quick \\`
      `--seed 184467 \\`
      `--result /tmp/commutation-cancellation/result.json \\`
      `--artifacts /tmp/commutation-cancellation`

The JSON result is always emitted (to ``--result`` if given, else `stdout`). 
The shell exit status is a category 
    0 success
    1 invariant failure 
    2 unsupported domain
    3 invalid request
    4 infrastructure failure
The JSON ``status`` field carries the same category so callers never have to parse human prose.
"""

import argparse
import json
import sys
from pathlib import Path

from .optimization_validation import (
    PREDICATES,
    MetricSpec,
    OracleSpec,
    PipelineTarget,
    ValidationRequest,
    ValidationStatus,
    capabilities,
    result_to_dict,
    validate,
)

# Shell exit status per outcome category.
EXIT_STATUS = {
    ValidationStatus.PASSED: 0,
    ValidationStatus.INVARIANT_FAILURE: 1,
    ValidationStatus.UNSUPPORTED_DOMAIN: 2,
    ValidationStatus.INVALID_REQUEST: 3,
    ValidationStatus.INFRASTRUCTURE_FAILURE: 4,
}


def _parse_metric(raw: str) -> MetricSpec:
    """Parse a ``name[:predicate]`` metric flag. Predicate defaults to
    `nonincreasing`.

    Metric names may themselves contain a colon (``gate:rz``), so the predicate
    is only split off when the final ``:``-separated token is a known predicate.
    """
    head, sep, tail = raw.rpartition(":")
    if sep and tail in PREDICATES:
        return MetricSpec(name=head, predicate=tail)
    return MetricSpec(name=raw, predicate="nonincreasing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m cudaq._compiler.optimization_cli",
        description="Validate a candidate CUDA-Q pass/pipeline against a "
        "baseline over one or more Quake inputs.")
    parser.add_argument("--input",
                        action="append",
                        default=[],
                        metavar="FILE",
                        help="Quake (.qke) input file; repeatable.")
    parser.add_argument("--prepare",
                        default="",
                        help="Pipeline applied before the candidate.")
    parser.add_argument("--candidate",
                        default="",
                        help="Candidate pipeline under test.")
    parser.add_argument("--observe",
                        default="",
                        help="Pipeline applied identically to baseline and "
                        "candidate outputs before comparison.")
    parser.add_argument("--oracle",
                        default="strict-unitary",
                        help="Equivalence oracle: strict-unitary or "
                        "up-to-global-phase.")
    parser.add_argument("--metric",
                        action="append",
                        default=[],
                        metavar="NAME[:PREDICATE]",
                        help="Declared metric and predicate; repeatable.")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--fixed-point-runs", type=int, default=1)
    parser.add_argument("--exact-qubit-bound", type=int, default=14)
    parser.add_argument("--kernel-name",
                        default=None,
                        help="Kernel symbol to compare when a module has more "
                        "than one.")
    parser.add_argument("--preset", default="quick")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--result",
                        default=None,
                        metavar="FILE",
                        help="Write the JSON result here (default: stdout).")
    parser.add_argument("--artifacts",
                        default=None,
                        metavar="DIR",
                        help="Directory for failure artifacts.")
    parser.add_argument("--capabilities",
                        action="store_true",
                        help="Print machine-readable capabilities and exit.")
    return parser


def _emit(payload: dict, result_path) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if result_path:
        path = Path(result_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    else:
        print(text)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.capabilities:
        import dataclasses
        _emit(dataclasses.asdict(capabilities()), args.result)
        return EXIT_STATUS[ValidationStatus.PASSED]

    if args.artifacts:
        Path(args.artifacts).mkdir(parents=True, exist_ok=True)

    target = PipelineTarget(prepare=args.prepare, observe=args.observe)
    request = ValidationRequest(
        inputs=tuple(Path(p) for p in args.input),
        pipeline=target.with_pipeline(args.candidate),
        oracle=OracleSpec(kind=args.oracle, rtol=args.rtol, atol=args.atol),
        metrics=tuple(_parse_metric(m) for m in args.metric),
        seed=args.seed,
        fixed_point_runs=args.fixed_point_runs,
        exact_qubit_bound=args.exact_qubit_bound,
        kernel_name=args.kernel_name,
        preset=args.preset,
    )

    result = validate(request)
    _emit(result_to_dict(result), args.result)
    return EXIT_STATUS.get(result.status,
                           EXIT_STATUS[ValidationStatus.INFRASTRUCTURE_FAILURE])


if __name__ == "__main__":
    sys.exit(main())
