# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Request/result contracts and the ``validate()`` entry point for the
Optimization Validation Core."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from cudaq.mlir.ir import Context, Module
from cudaq.mlir.passmanager import PassManager
from cudaq.mlir._mlir_libs._quakeDialects import (cudaq_runtime,
                                                  register_all_dialects, quake,
                                                  cc)

# Schema versions are part of the machine-readable contract.
RESULT_SCHEMA_VERSION = 1
CAPABILITY_SCHEMA_VERSION = 1

ASSURANCE_TIER_EXACT = "exact"

DEFAULT_EXACT_QUBIT_BOUND = 14


class ValidationStatus:
    """Outcome categories, ordered from least to most severe by SEVERITY."""

    PASSED = "passed"
    INVARIANT_FAILURE = "invariant-failure"
    UNSUPPORTED_DOMAIN = "unsupported-domain"
    INVALID_REQUEST = "invalid-request"
    INFRASTRUCTURE_FAILURE = "infrastructure-failure"


# Higher number == more severe. The aggregate status is the most severe case.
_SEVERITY = {
    ValidationStatus.PASSED: 0,
    ValidationStatus.INVARIANT_FAILURE: 1,
    ValidationStatus.UNSUPPORTED_DOMAIN: 2,
    ValidationStatus.INVALID_REQUEST: 3,
    ValidationStatus.INFRASTRUCTURE_FAILURE: 4,
}

_ORACLE_KINDS = ("strict-unitary", "up-to-global-phase")
PREDICATES = ("nonincreasing", "decreasing", "unchanged", "any")
_PRESETS = ("smoke", "quick", "ci", "full", "single-reproducer")


# Request contracts
@dataclass(frozen=True)
class PipelineSpec:
    """The three pass pipelines applied around the candidate.

    Each is a full MLIR pipeline string, e.g.
    ``builtin.module(func.func(memtoreg))``. An empty string is a no-op stage.
    """

    prepare: str = ""
    candidate: str = ""
    observe: str = ""


@dataclass(frozen=True)
class OracleSpec:
    """Which exact oracle decides semantic equivalence, and its tolerances."""

    kind: str = "strict-unitary"
    rtol: float = 1e-5
    atol: float = 1e-8


@dataclass(frozen=True)
class MetricSpec:
    """A declared metric and the predicate its delta must satisfy.

    ``name`` is one of ``operation-count``, ``two-qubit-count``,
    ``multi-qubit-count``, ``depth``, ``t-count``, or ``gate:<name>``.
    ``predicate`` is one of ``nonincreasing``, ``decreasing``, ``unchanged``,
    ``any``.
    """

    name: str
    predicate: str = "nonincreasing"


@dataclass(frozen=True)
class ValidationRequest:
    inputs: tuple[Path, ...]
    pipeline: PipelineSpec
    oracle: OracleSpec = field(default_factory=OracleSpec)
    metrics: tuple[MetricSpec, ...] = ()
    seed: int = 0
    fixed_point_runs: int = 1
    exact_qubit_bound: int = DEFAULT_EXACT_QUBIT_BOUND
    kernel_name: Optional[str] = None
    preset: str = "quick"


# Result contracts
@dataclass(frozen=True)
class MetricDelta:
    name: str
    predicate: str
    baseline: int
    candidate: int
    delta: int
    satisfied: bool


@dataclass(frozen=True)
class CaseResult:
    input: str
    status: str
    assurance_tier: str
    strict_equal: bool
    equal_up_to_global_phase: bool
    phase: float
    phase_is_zero: bool
    deterministic: bool
    fixed_point: bool
    metrics: tuple[MetricDelta, ...]
    messages: tuple[str, ...]


@dataclass(frozen=True)
class ValidationResult:
    status: str
    cases: tuple[CaseResult, ...]
    aggregate_metrics: Mapping[str, MetricDelta]
    seed: int
    preset: str
    messages: tuple[str, ...] = ()
    result_schema_version: int = RESULT_SCHEMA_VERSION


@dataclass(frozen=True)
class ValidationCapabilities:
    oracles: tuple[str, ...]
    metrics: tuple[str, ...]
    predicates: tuple[str, ...]
    presets: tuple[str, ...]
    assurance_tiers: tuple[str, ...]
    result_schema_version: int
    capability_schema_version: int


class InvalidRequest(Exception):
    """Raised when a request is malformed. Maps to INVALID_REQUEST."""


# MLIR plumbing
def _make_context() -> Context:
    """Create an MLIR context with the CUDA-Q dialects and passes registered.

    Mirrors ``cudaq.kernel.utils.getMLIRContext`` but stays self-contained so
    this internal package does not pull in the kernel-building machinery.
    """
    ctx = Context()
    register_all_dialects(ctx)
    quake.register_dialect(context=ctx)
    cc.register_dialect(context=ctx)
    cudaq_runtime.registerLLVMDialectTranslation(ctx)
    return ctx


def _run_pipeline(pipeline: str, module: Module, ctx: Context) -> Module:
    """Run ``pipeline`` on ``module`` in place and return it. No-op if empty."""
    if not pipeline.strip():
        return module
    pm = PassManager.parse(pipeline, context=ctx)
    cudaq_runtime.runPassManager(pm, module)
    return module


def _clone(module: Module) -> Module:
    return cudaq_runtime.cloneModule(module)


# Metrics
def _metric_value(counts: dict, name: str) -> int:
    if name == "operation-count":
        return counts["gate_count"]
    if name == "two-qubit-count":
        return counts["two_qubit_count"]
    if name == "multi-qubit-count":
        return counts["multi_qubit_count"]
    if name == "depth":
        return counts["depth"]
    if name == "t-count":
        return counts["per_gate"].get("t", 0)
    if name.startswith("gate:"):
        return counts["per_gate"].get(name[len("gate:"):], 0)
    raise InvalidRequest(f"unknown metric '{name}'")


def _predicate_ok(predicate: str, baseline: int, candidate: int) -> bool:
    if predicate == "nonincreasing":
        return candidate <= baseline
    if predicate == "decreasing":
        return candidate < baseline
    if predicate == "unchanged":
        return candidate == baseline
    if predicate == "any":
        return True
    raise InvalidRequest(f"unknown predicate '{predicate}'")


# Request validation
def _validate_request(request: ValidationRequest, ctx: Context) -> None:
    if not request.inputs:
        raise InvalidRequest("no inputs provided")
    for path in request.inputs:
        if not Path(path).is_file():
            raise InvalidRequest(f"input not found: {path}")
    if request.oracle.kind not in _ORACLE_KINDS:
        raise InvalidRequest(f"unknown oracle '{request.oracle.kind}'")
    if request.preset not in _PRESETS:
        raise InvalidRequest(f"unknown preset '{request.preset}'")
    if request.fixed_point_runs < 0:
        raise InvalidRequest("fixed_point_runs must be non-negative")
    for metric in request.metrics:
        if metric.predicate not in PREDICATES:
            raise InvalidRequest(
                f"unknown predicate '{metric.predicate}' for '{metric.name}'")
    for stage in (request.pipeline.prepare, request.pipeline.candidate,
                  request.pipeline.observe):
        if stage.strip():
            try:
                PassManager.parse(stage, context=ctx)
            except Exception as exc:
                raise InvalidRequest(f"invalid pipeline '{stage}': {exc}")


# Per-input evaluation
def _equivalent(oracle_kind: str, comparison: dict) -> bool:
    if oracle_kind == "strict-unitary":
        return bool(comparison["strict_equal"])
    return bool(comparison["equal_up_to_global_phase"])


def _evaluate_input(path: Path, request: ValidationRequest,
                    ctx: Context) -> CaseResult:
    messages: list[str] = []
    pipeline = request.pipeline
    oracle = request.oracle

    module = Module.parse(Path(path).read_text(), ctx)

    prepared = _run_pipeline(pipeline.prepare, module, ctx)

    candidate_raw = _run_pipeline(pipeline.candidate, _clone(prepared), ctx)

    baseline_obs = _run_pipeline(pipeline.observe, _clone(prepared), ctx)
    candidate_obs = _run_pipeline(pipeline.observe, _clone(candidate_raw), ctx)

    base_pf = cudaq_runtime.preflight_bounded_unitary(baseline_obs,
                                                      request.exact_qubit_bound)
    cand_pf = cudaq_runtime.preflight_bounded_unitary(candidate_obs,
                                                      request.exact_qubit_bound)
    if not base_pf["supported"] or not cand_pf["supported"]:
        for side, pf in (("baseline", base_pf), ("candidate", cand_pf)):
            for rej in pf["rejections"]:
                messages.append(f"{side} unsupported: {rej['kind']} "
                                f"in {rej['kernel']} ({rej['detail']})")
        return CaseResult(
            input=str(path),
            status=ValidationStatus.UNSUPPORTED_DOMAIN,
            assurance_tier=ASSURANCE_TIER_EXACT,
            strict_equal=False,
            equal_up_to_global_phase=False,
            phase=0.0,
            phase_is_zero=False,
            deterministic=False,
            fixed_point=False,
            metrics=(),
            messages=tuple(messages),
        )

    comparison = cudaq_runtime.compare_unitaries(baseline_obs, candidate_obs,
                                                 request.kernel_name,
                                                 oracle.rtol, oracle.atol)
    status = ValidationStatus.PASSED
    if not comparison["computed"]:
        status = ValidationStatus.INVARIANT_FAILURE
        messages.append(f"comparison failed: {comparison['error']}")
    elif not _equivalent(oracle.kind, comparison):
        status = ValidationStatus.INVARIANT_FAILURE
        messages.append(f"not equivalent under oracle '{oracle.kind}'")

    base_counts = cudaq_runtime.count_resources_checkpoint(baseline_obs)
    cand_counts = cudaq_runtime.count_resources_checkpoint(candidate_obs)
    metrics: list[MetricDelta] = []
    if base_counts["computed"] and cand_counts["computed"]:
        for spec in request.metrics:
            base_val = _metric_value(base_counts, spec.name)
            cand_val = _metric_value(cand_counts, spec.name)
            ok = _predicate_ok(spec.predicate, base_val, cand_val)
            metrics.append(
                MetricDelta(name=spec.name,
                            predicate=spec.predicate,
                            baseline=base_val,
                            candidate=cand_val,
                            delta=cand_val - base_val,
                            satisfied=ok))
            if not ok:
                status = _worst(status, ValidationStatus.INVARIANT_FAILURE)
                messages.append(f"metric '{spec.name}' violates "
                                f"'{spec.predicate}': {base_val} -> {cand_val}")
    elif request.metrics:
        status = _worst(status, ValidationStatus.INFRASTRUCTURE_FAILURE)
        messages.append("resource counting failed; metrics unavailable")

    rerun = _run_pipeline(pipeline.candidate, _clone(prepared), ctx)
    deterministic = str(rerun) == str(candidate_raw)
    if not deterministic:
        status = _worst(status, ValidationStatus.INVARIANT_FAILURE)
        messages.append("candidate output is not deterministic")

    fixed_point = True
    reference = candidate_raw
    for _ in range(request.fixed_point_runs):
        again = _run_pipeline(pipeline.candidate, _clone(reference), ctx)
        if str(again) != str(reference):
            fixed_point = False
            break
        reference = again
    if not fixed_point:
        status = _worst(status, ValidationStatus.INVARIANT_FAILURE)
        messages.append("candidate is not at a fixed point")

    return CaseResult(
        input=str(path),
        status=status,
        assurance_tier=ASSURANCE_TIER_EXACT,
        strict_equal=bool(comparison.get("strict_equal", False)),
        equal_up_to_global_phase=bool(
            comparison.get("equal_up_to_global_phase", False)),
        phase=float(comparison.get("phase", 0.0)),
        phase_is_zero=bool(comparison.get("phase_is_zero", False)),
        deterministic=deterministic,
        fixed_point=fixed_point,
        metrics=tuple(metrics),
        messages=tuple(messages),
    )


def _worst(a: str, b: str) -> str:
    return a if _SEVERITY[a] >= _SEVERITY[b] else b


# Public API
def validate(request: ValidationRequest) -> ValidationResult:
    """Validate a candidate pipeline against a baseline over the request inputs.

    Never raises for a validation failure. Failures are reported through the
    returned :class:`ValidationResult`. A malformed request yields a result with
    status ``invalid-request``.
    """
    ctx = _make_context()
    try:
        _validate_request(request, ctx)
    except InvalidRequest as exc:
        return ValidationResult(
            status=ValidationStatus.INVALID_REQUEST,
            cases=(),
            aggregate_metrics={},
            seed=request.seed,
            preset=request.preset,
            messages=(str(exc),),
        )

    cases: list[CaseResult] = []
    for path in request.inputs:
        try:
            cases.append(_evaluate_input(path, request, ctx))
        except InvalidRequest:
            raise
        except Exception as exc:
            cases.append(
                CaseResult(
                    input=str(path),
                    status=ValidationStatus.INFRASTRUCTURE_FAILURE,
                    assurance_tier=ASSURANCE_TIER_EXACT,
                    strict_equal=False,
                    equal_up_to_global_phase=False,
                    phase=0.0,
                    phase_is_zero=False,
                    deterministic=False,
                    fixed_point=False,
                    metrics=(),
                    messages=(f"infrastructure error: {exc}",),
                ))

    status = ValidationStatus.PASSED
    for case in cases:
        status = _worst(status, case.status)

    return ValidationResult(
        status=status,
        cases=tuple(cases),
        aggregate_metrics=_aggregate_metrics(cases),
        seed=request.seed,
        preset=request.preset,
    )


def _aggregate_metrics(cases) -> dict:
    """Worst-case (least favorable) delta per metric across all cases."""
    aggregate: dict[str, MetricDelta] = {}
    for case in cases:
        for metric in case.metrics:
            current = aggregate.get(metric.name)
            if current is None or metric.delta > current.delta:
                aggregate[metric.name] = metric
    return aggregate


def capabilities() -> ValidationCapabilities:
    """Return the machine-readable capabilities of this validator.

    This is the authoritative source for which oracles, metrics, and presets
    have executable support.
    """
    return ValidationCapabilities(
        oracles=_ORACLE_KINDS,
        metrics=("operation-count", "two-qubit-count", "multi-qubit-count",
                 "depth", "t-count", "gate:<name>"),
        predicates=PREDICATES,
        presets=_PRESETS,
        assurance_tiers=(ASSURANCE_TIER_EXACT,),
        result_schema_version=RESULT_SCHEMA_VERSION,
        capability_schema_version=CAPABILITY_SCHEMA_VERSION,
    )


def result_to_dict(result: ValidationResult) -> dict:
    """Convert a result to a JSON-serializable dict."""
    return dataclasses.asdict(result)
