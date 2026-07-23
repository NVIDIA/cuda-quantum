# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

import abc
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional, Union

from cudaq.mlir.ir import Context, Module
from cudaq.mlir.passmanager import PassManager
from cudaq.mlir._mlir_libs._quakeDialects import (cudaq_runtime,
                                                  register_all_dialects, quake,
                                                  cc)

# Schema versions are part of the contract.
# Each case now carries a structured `invariants` list
# (equivalence / determinism / fixed-point).
RESULT_SCHEMA_VERSION = 3
# Capabilities now advertise the `invariants` names.
CAPABILITY_SCHEMA_VERSION = 4

# Assurance tiers, from strongest guarantee to weakest. The tier records what
# kind of equivalence evidence an oracle produces, independent of the circuit it
# ran on. Names are deliberately concrete about the backing method, so a new
# strategy gets a new named tier rather than overloading an abstract one.
#
#   exact-unitary       Full-operator equivalence built directly from the IR
#                       (dense unitary, up to global phase). Basis- and
#                       input-independent: checks the whole operator, not one
#                       input state. Safest, but bounded by the 2^n dense-matrix
#                       cost (see DEFAULT_EXACT_QUBIT_BOUND).
#   exact-clifford-sim  Exact equivalence that scales past the dense bound by
#                       exploiting circuit structure, specifically Clifford
#                       circuits via a tableau/`stim` simulator. Same strength as
#                       exact-unitary on its (narrower) domain, but usable at many
#                       more qubits -- a scaling axis, not a generality superset.
#   exact-density-sim   Density-matrix equivalence for small circuits. Handles
#                       measurement/reset/noise that the pure-unitary oracles
#                       reject, at the same ~small-qubit cost ceiling.
#   advisory            Sampled or expectation-value evidence (e.g. a statevector
#                       compared on a fixed input state). Weaker: it can witness a
#                       difference but cannot certify equivalence, so per the
#                       oracle hardening rules it may only turn passed -> failed,
#                       never failed -> passed.
ASSURANCE_TIER_EXACT_UNITARY = "exact-unitary"
ASSURANCE_TIER_EXACT_CLIFFORD_SIM = "exact-clifford-sim"
ASSURANCE_TIER_EXACT_DENSITY_SIM = "exact-density-sim"
ASSURANCE_TIER_ADVISORY = "advisory"

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

# The first-class boolean invariants checked on every in-domain case, reported as
# structured `InvariantResult`s. These are the semantic guarantees the validator
# enforces beyond the declared resource metrics.
INVARIANT_EQUIVALENCE = "equivalence"
INVARIANT_DETERMINISM = "determinism"
INVARIANT_FIXED_POINT = "fixed-point"
INVARIANT_KINDS = (INVARIANT_EQUIVALENCE, INVARIANT_DETERMINISM,
                   INVARIANT_FIXED_POINT)


@dataclass(frozen=True)
class OracleDescriptor:
    """One equivalence oracle in the `roadmap`, executable or deferred.

    This is the authoritative, machine-readable description of how an oracle
    decides equivalence and how far it scales, so a caller (or the agent-facing
    skill) never has to infer an oracle's strength or domain from prose.
    """

    kind: str
    tier: str
    # "supported" if executable in this build, else "deferred".
    status: str
    # The simulation/analysis method backing the decision.
    method: str
    # domain + scaling note.
    note: str


# The oracle `roadmap`.
ORACLE_ROADMAP = (
    OracleDescriptor(
        kind="strict-unitary",
        tier=ASSURANCE_TIER_EXACT_UNITARY,
        status="supported",
        method="dense-unitary-from-ir",
        note="Element-wise unitary equality. Bounded by the dense 2^n cost.",
    ),
    OracleDescriptor(
        kind="up-to-global-phase",
        tier=ASSURANCE_TIER_EXACT_UNITARY,
        status="supported",
        method="dense-unitary-from-ir",
        note="Unitary equality after dividing out a global phase. Bounded by "
        "the dense 2^n cost.",
    ),
    OracleDescriptor(
        kind="clifford-tableau",
        tier=ASSURANCE_TIER_EXACT_CLIFFORD_SIM,
        status="deferred",
        method="tableau-stim",
        note="Exact equivalence for Clifford-only circuits, scalable well past "
        "the dense bound. Requires a Clifford-domain preflight.",
    ),
    OracleDescriptor(
        kind="density-matrix",
        tier=ASSURANCE_TIER_EXACT_DENSITY_SIM,
        status="deferred",
        method="density-matrix-sim",
        note="Small-circuit equivalence that also covers measurement/reset/"
        "noise. Same small-qubit ceiling as the dense oracle.",
    ),
    OracleDescriptor(
        kind="statevector-expectation",
        tier=ASSURANCE_TIER_ADVISORY,
        status="deferred",
        method="statevector-sim",
        note="Expectation-value evidence on fixed input states.",
    ),
)


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
    """Declarative selection of a built-in oracle, and its tolerances.

    This is the config-level way to ask for the built-in
    :class:`DenseUnitaryOracle`. It is bound to an :class:`Oracle` instance by
    the runner. To supply your own oracle, pass an :class:`Oracle` instance as
    the request's ``oracle`` instead.
    """

    kind: str = "strict-unitary"
    rtol: float = 1e-5
    atol: float = 1e-8


@dataclass(frozen=True)
class OracleDecision:
    """One oracle's verdict on the equivalence invariant for a single case.

    An oracle answers exactly one question 
        are these two observed modules semantically equivalent?
    and reports the assurance tier of the evidence it produced. 
        ``supported`` is the oracle's own domain preflight
        ``computed`` is whether the comparison actually ran
        ``equivalent`` is the verdict. 
    The unitary-specific evidence fields are populated by dense-unitary oracles and
    left at their defaults by others.
    """

    supported: bool
    computed: bool
    equivalent: bool
    tier: str
    detail: str
    rejections: tuple[dict, ...] = ()
    strict_equal: bool = False
    equal_up_to_global_phase: bool = False
    phase: float = 0.0
    phase_is_zero: bool = False


class Oracle(abc.ABC):
    """The equivalence-oracle extension point.

    An oracle owns exactly one invariant semantic equivalence between the observed 
    baseline and candidate modules together with its own domain preflight and the
    assurance tier of the evidence it produces. Resource metrics, determinism, and 
    fixed-point are runner-owned invariants and are deliberately NOT an oracle's concern.

    Users implement this for the fast optimization loop (a user's own fast test
    oracle is the common case). The built-in :class:`DenseUnitaryOracle` backs
    the trusted validation gate. Subclasses set :attr:`kind` and :attr:`tier` and
    implement :meth:`decide`.
    """

    kind: str = ""
    tier: str = ""

    @abc.abstractmethod
    def decide(self, baseline: Module, candidate: Module,
               kernel_name: Optional[str]) -> OracleDecision:
        """Decide equivalence of the two observed modules.

        Must not raise for an ordinary negative verdict or an unsupported domain;
        report those through the returned :class:`OracleDecision`.
        """


class DenseUnitaryOracle(Oracle):
    """Built-in exact-unitary oracle (the V1 default).

    Builds the dense operator directly from the IR (no simulator, no target)
    and compares operators either strictly or up to a global phase. Basis and
    input-independent, bounded by the dense 2^n cost. Wraps the reused
    ``preflight_bounded_unitary`` and ``compare_unitaries`` bindings.
    """

    tier = ASSURANCE_TIER_EXACT_UNITARY

    def __init__(self,
                 kind: str = "strict-unitary",
                 rtol: float = 1e-5,
                 atol: float = 1e-8,
                 qubit_bound: int = DEFAULT_EXACT_QUBIT_BOUND):
        if kind not in _ORACLE_KINDS:
            raise InvalidRequest(f"unknown oracle '{kind}'")
        self.kind = kind
        self.rtol = rtol
        self.atol = atol
        self.qubit_bound = qubit_bound

    def decide(self, baseline: Module, candidate: Module,
               kernel_name: Optional[str]) -> OracleDecision:
        base_pf = cudaq_runtime.preflight_bounded_unitary(
            baseline, self.qubit_bound)
        cand_pf = cudaq_runtime.preflight_bounded_unitary(
            candidate, self.qubit_bound)
        if not base_pf["supported"] or not cand_pf["supported"]:
            rejections = tuple({
                **rej, "side": side
            }
                               for side, pf in (("baseline", base_pf),
                                                ("candidate", cand_pf))
                               for rej in pf["rejections"])
            return OracleDecision(supported=False,
                                  computed=False,
                                  equivalent=False,
                                  tier=self.tier,
                                  detail="unsupported domain",
                                  rejections=rejections)

        comparison = cudaq_runtime.compare_unitaries(baseline, candidate,
                                                     kernel_name, self.rtol,
                                                     self.atol)
        computed = bool(comparison["computed"])
        equivalent = computed and _equivalent(self.kind, comparison)
        if not computed:
            detail = f"comparison failed: {comparison['error']}"
        elif not equivalent:
            detail = f"not equivalent under oracle '{self.kind}'"
        else:
            detail = f"equivalent under oracle '{self.kind}'"
        return OracleDecision(
            supported=True,
            computed=computed,
            equivalent=equivalent,
            tier=self.tier,
            detail=detail,
            strict_equal=bool(comparison.get("strict_equal", False)),
            equal_up_to_global_phase=bool(
                comparison.get("equal_up_to_global_phase", False)),
            phase=float(comparison.get("phase", 0.0)),
            phase_is_zero=bool(comparison.get("phase_is_zero", False)))


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
    oracle: Union[OracleSpec, Oracle] = field(default_factory=OracleSpec)
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
class InvariantResult:
    """A named boolean invariant checked on the candidate, with its outcome.

    ``name`` is one of :data:`INVARIANT_KINDS`. ``detail`` carries context (the
    deciding oracle, the fixed-point bound, or a failure reason).
    """

    name: str
    satisfied: bool
    detail: str = ""


@dataclass(frozen=True)
class CaseResult:
    input: str
    status: str
    assurance_tier: str
    strict_equal: bool
    equal_up_to_global_phase: bool
    phase: float
    phase_is_zero: bool
    # Semantic invariants (equivalence, determinism, fixed-point).
    invariants: tuple[InvariantResult, ...]
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
    # Oracle kinds executable in this build.
    oracles: tuple[str, ...]
    metrics: tuple[str, ...]
    predicates: tuple[str, ...]
    presets: tuple[str, ...]
    # Invariants checked on every in-domain case.
    invariants: tuple[str, ...]
    # Assurance tiers this `validator` can accept at.
    assurance_tiers: tuple[str, ...]
    # Full oracle `roadmap`, tier and method for each.
    oracle_roadmap: tuple[OracleDescriptor, ...]
    result_schema_version: int
    capability_schema_version: int


class InvalidRequest(Exception):
    """Raised when a request is malformed. Maps to INVALID_REQUEST."""


class _StageFailure(Exception):
    """Internal: a pipeline stage (or the input) failed to parse, run, or verify.

    Carries the outcome ``status`` to attribute to the case and a message. Caught
    inside :func:`_evaluate_input` and turned into a failed case. It never escapes.
    """

    def __init__(self, status: str, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


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


def _module_is_valid(module: Module) -> bool:
    """True iff ``module`` passes MLIR verification.

    ``operation.verify()`` raises ``MLIRError`` on failure rather than returning
    ``False``. It normalize both to a bool so callers can branch on it.
    """
    try:
        return bool(module.operation.verify())
    except Exception:
        return False


def _run_stage(pipeline: str, module: Module, ctx: Context, *, stage: str,
               failure_status: str) -> Module:
    """Run one stage and verify its output IR, failing closed on either error.

    Implements the ``verify`` gate of the per-input flow at a stage boundary:
    running the pipeline and confirming the result is structurally valid IR. A
    run error or an invalid result raises :class:`_StageFailure` tagged with
    ``failure_status`` so the case is attributed to the right category (a bad
    candidate is an invariant failure; a bad prepare/observe is infrastructure).
    Covers no-op (empty) stages too, which never reach the pass manager's own
    verifier.
    """
    try:
        out = _run_pipeline(pipeline, module, ctx)
    except Exception as exc:
        raise _StageFailure(failure_status, f"{stage} stage failed: {exc}")
    if not _module_is_valid(out):
        raise _StageFailure(failure_status,
                            f"{stage} stage produced invalid IR")
    return out


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
    if isinstance(request.oracle, OracleSpec):
        if request.oracle.kind not in _ORACLE_KINDS:
            raise InvalidRequest(f"unknown oracle '{request.oracle.kind}'")
    elif not isinstance(request.oracle, Oracle):
        raise InvalidRequest(
            "oracle must be an OracleSpec or an Oracle instance")
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


def _failed_case(path: Path,
                 status: str,
                 messages,
                 tier: str = ASSURANCE_TIER_EXACT_UNITARY) -> CaseResult:
    """A case that never reached comparison: no invariants were established."""
    return CaseResult(
        input=str(path),
        status=status,
        assurance_tier=tier,
        strict_equal=False,
        equal_up_to_global_phase=False,
        phase=0.0,
        phase_is_zero=False,
        invariants=(),
        metrics=(),
        messages=tuple(messages),
    )


def _parse_input(path: Path, ctx: Context) -> Module:
    """Parse and verify an input module. A malformed input is a bad request.

    ``Module.parse`` runs the verifier, so this is the ``verify`` gate for the
    input: invalid IR fails here rather than being mistaken for an internal
    infrastructure error later.
    """
    try:
        module = Module.parse(Path(path).read_text(), ctx)
    except Exception as exc:
        raise _StageFailure(ValidationStatus.INVALID_REQUEST,
                            f"input IR failed to parse or verify: {exc}")
    if not _module_is_valid(module):
        raise _StageFailure(ValidationStatus.INVALID_REQUEST,
                            "input IR failed verification")
    return module


def _resolve_oracle(oracle: Union[OracleSpec, Oracle],
                    qubit_bound: int) -> Oracle:
    """Resolve the request's oracle field to an executable :class:`Oracle`.

    An :class:`Oracle` instance is used as-is (user-supplied, the common case); a
    declarative :class:`OracleSpec` is bound to the built-in
    :class:`DenseUnitaryOracle` using the request's qubit bound.
    """
    if isinstance(oracle, Oracle):
        return oracle
    return DenseUnitaryOracle(kind=oracle.kind,
                              rtol=oracle.rtol,
                              atol=oracle.atol,
                              qubit_bound=qubit_bound)


def _evaluate_input(path: Path, request: ValidationRequest,
                    ctx: Context) -> CaseResult:
    messages: list[str] = []
    pipeline = request.pipeline
    oracle = _resolve_oracle(request.oracle, request.exact_qubit_bound)

    # Verify -> prepare -> candidate -> observe, verifying IR at each boundary. A
    # bad candidate fails closed as an invariant failure; a bad prepare/observe
    # is infrastructure.
    try:
        module = _parse_input(path, ctx)
        prepared = _run_stage(
            pipeline.prepare,
            module,
            ctx,
            stage="prepare",
            failure_status=ValidationStatus.INFRASTRUCTURE_FAILURE)
        candidate_raw = _run_stage(
            pipeline.candidate,
            _clone(prepared),
            ctx,
            stage="candidate",
            failure_status=ValidationStatus.INVARIANT_FAILURE)
        baseline_obs = _run_stage(
            pipeline.observe,
            _clone(prepared),
            ctx,
            stage="observe (baseline)",
            failure_status=ValidationStatus.INFRASTRUCTURE_FAILURE)
        candidate_obs = _run_stage(
            pipeline.observe,
            _clone(candidate_raw),
            ctx,
            stage="observe (candidate)",
            failure_status=ValidationStatus.INFRASTRUCTURE_FAILURE)
    except _StageFailure as failure:
        return _failed_case(path, failure.status, [failure.message])

    # The oracle owns the equivalence invariant and its own domain preflight.
    decision = oracle.decide(baseline_obs, candidate_obs, request.kernel_name)
    if not decision.supported:
        for rej in decision.rejections:
            messages.append(f"{rej['side']} unsupported: {rej['kind']} "
                            f"in {rej['kernel']} ({rej['detail']})")
        return _failed_case(path,
                            ValidationStatus.UNSUPPORTED_DOMAIN,
                            messages,
                            tier=decision.tier)

    status = ValidationStatus.PASSED
    equivalent = decision.equivalent
    equivalence_detail = decision.detail
    if not decision.computed or not equivalent:
        status = ValidationStatus.INVARIANT_FAILURE
        messages.append(decision.detail)

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

    invariants = (
        InvariantResult(name=INVARIANT_EQUIVALENCE,
                        satisfied=equivalent,
                        detail=equivalence_detail),
        InvariantResult(name=INVARIANT_DETERMINISM, satisfied=deterministic),
        InvariantResult(name=INVARIANT_FIXED_POINT,
                        satisfied=fixed_point,
                        detail=f"{request.fixed_point_runs} run(s)"),
    )

    return CaseResult(
        input=str(path),
        status=status,
        assurance_tier=decision.tier,
        strict_equal=decision.strict_equal,
        equal_up_to_global_phase=decision.equal_up_to_global_phase,
        phase=decision.phase,
        phase_is_zero=decision.phase_is_zero,
        invariants=invariants,
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
                _failed_case(path, ValidationStatus.INFRASTRUCTURE_FAILURE,
                             [f"infrastructure error: {exc}"]))

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
    """Return the machine-readable capabilities of this `validator`.

    This is the authoritative source for which oracles, metrics, and presets
    have executable support.
    """
    return ValidationCapabilities(
        oracles=_ORACLE_KINDS,
        metrics=("operation-count", "two-qubit-count", "multi-qubit-count",
                 "depth", "t-count", "gate:<name>"),
        predicates=PREDICATES,
        presets=_PRESETS,
        invariants=INVARIANT_KINDS,
        assurance_tiers=(ASSURANCE_TIER_EXACT_UNITARY,),
        oracle_roadmap=ORACLE_ROADMAP,
        result_schema_version=RESULT_SCHEMA_VERSION,
        capability_schema_version=CAPABILITY_SCHEMA_VERSION,
    )


def result_to_dict(result: ValidationResult) -> dict:
    """Convert a result to a JSON-serializable dict."""
    return dataclasses.asdict(result)
