# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Thin command-line wrapper over the Optimization Validation Core API.

The public API (:func:`cudaq._compiler.validate` / :func:`~cudaq._compiler.capabilities`)
is the product. Every flag maps 1:1 onto a :class:`ValidationRequest` field.
The CLI holds no validation logic of its own.

    `python3 -m cudaq._compiler \\`
      `--input my_kernels.py \\`
      `--prepare 'builtin.module(func.func(cc-loop-unroll,canonicalize,memtoreg))' \\`
      `--candidate 'builtin.module(func.func(quake-commutation-cancellation))' \\`
      `--oracle strict-unitary \\`
      `--metric operation-count:nonincreasing \\`
      `--metric two-qubit-count:nonincreasing \\`
      `--fixed-point-runs 1 \\`
      `--result /tmp/commutation-cancellation/result.json`

An ``--input`` may be either a Quake ``.qke`` file or a Python file of
``@cudaq.kernel`` definitions. A ``.py`` input is lowered through the CUDA-Q
frontend. Every kernel in the file becomes one validation input, or ``FILE.py::name``
selects a single kernel by its Python function name. Kernels that use a Python
``for`` loop lower to ``cc.loop`` and need ``cc-loop-unroll`` in ``--prepare`` to
enter the bounded-unitary domain.

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
import contextlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from .optimization_validation import (
    PREDICATES,
    MetricSpec,
    OracleSpec,
    PipelineTarget,
    ValidationRequest,
    ValidationResult,
    ValidationStatus,
    capabilities,
    result_to_dict,
    validate,
)

# Monotonic counter so each imported user kernel file gets a unique module name.
_kernel_import_counter = 0


class InputError(Exception):
    """A malformed ``--input`` (bad kernel file, unknown kernel name, ...)."""


def _is_kernel_spec(spec: str) -> bool:
    """True if ``--input`` names a Python kernel file rather than a ``.qke``.

    A ``FILE.py`` path or an explicit ``FILE::name`` selector is a kernel file.
    Anything else is treated as raw Quake IR to parse directly.
    """
    file = spec.split("::", 1)[0]
    return file.endswith(".py") or "::" in spec


def _import_kernel_file(path: str):
    """Import a user Python file so its ``@cudaq.kernel`` definitions exist.

    Registered under a unique module name so repeated ``--input`` files with the
    same stem do not collide, and so :mod:`inspect` can recover kernel source.
    """
    global _kernel_import_counter
    resolved = Path(path)
    if not resolved.is_file():
        raise InputError(f"kernel file not found: {path}")
    _kernel_import_counter += 1
    modname = f"_cudaq_user_kernels_{_kernel_import_counter}_{resolved.stem}"
    spec = importlib.util.spec_from_file_location(modname, str(resolved))
    module = importlib.util.module_from_spec(spec)
    sys.modules[modname] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise InputError(f"failed to import kernel file {path}: {exc}")
    return module


def _lower_kernel_spec(spec: str) -> list:
    """Lower a ``FILE.py[::name]`` input to ``[(kernel_name, quake_text), ...]``.

    Every ``@cudaq.kernel`` in the file is lowered through the frontend, or a
    single one when ``::name`` selects it by Python function name.
    """
    from cudaq.kernel.kernel_decorator import PyKernelDecorator

    file, _, name = spec.partition("::")
    module = _import_kernel_file(file)
    found = [(v.name, v)
             for v in vars(module).values()
             if isinstance(v, PyKernelDecorator)]
    if name:
        found = [(n, k) for n, k in found if n == name]
        if not found:
            raise InputError(f"kernel '{name}' not found in {file}")
    if not found:
        raise InputError(f"no @cudaq.kernel definitions found in {file}")
    return [(n, str(k)) for n, k in found]


def _resolve_inputs(specs: list, stack: contextlib.ExitStack) -> list:
    """Resolve ``--input`` specs to Quake file paths.

    Raw ``.qke`` paths pass through unchanged. Python kernel files are lowered
    and each kernel is materialized as ``<name>.qke`` in a temporary directory
    so the label carries the kernel name.
    """
    paths = []
    for spec in specs:
        if not _is_kernel_spec(spec):
            paths.append(Path(spec))
            continue
        tmpdir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        for name, text in _lower_kernel_spec(spec):
            path = tmpdir / f"{name}.qke"
            path.write_text(text)
            paths.append(path)
    return paths


# Exit status per outcome category.
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
        prog="python3 -m cudaq._compiler",
        description="Validate a candidate CUDA-Q pass/pipeline against a "
        "baseline over one or more Quake inputs.")
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="FILE",
        help="Quake (.qke) file, or a Python @cudaq.kernel file "
        "(FILE.py, or FILE.py::name for one kernel); repeatable.")
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
                        help="Equivalence oracle: strict-unitary, "
                        "up-to-global-phase, or clifford-tableau (Clifford "
                        "circuits only, but no qubit bound).")
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
    parser.add_argument("--result",
                        default=None,
                        metavar="FILE",
                        help="Write the JSON result here (default: stdout).")
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

    with contextlib.ExitStack() as stack:
        try:
            inputs = _resolve_inputs(args.input, stack)
        except InputError as exc:
            result = ValidationResult(
                status=ValidationStatus.INVALID_REQUEST,
                cases=(),
                aggregate_metrics={},
                messages=(str(exc),),
            )
            _emit(result_to_dict(result), args.result)
            return EXIT_STATUS[ValidationStatus.INVALID_REQUEST]

        target = PipelineTarget(prepare=args.prepare, observe=args.observe)
        request = ValidationRequest(
            inputs=tuple(inputs),
            pipeline=target.with_pipeline(args.candidate),
            oracle=OracleSpec(kind=args.oracle, rtol=args.rtol, atol=args.atol),
            metrics=tuple(_parse_metric(m) for m in args.metric),
            fixed_point_runs=args.fixed_point_runs,
            exact_qubit_bound=args.exact_qubit_bound,
            kernel_name=args.kernel_name,
        )

        result = validate(request)
        _emit(result_to_dict(result), args.result)
        return EXIT_STATUS.get(
            result.status, EXIT_STATUS[ValidationStatus.INFRASTRUCTURE_FAILURE])


if __name__ == "__main__":
    sys.exit(main())
