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

# The assurance tier records what kind of equivalence evidence an oracle
# produces, independent of the circuit it ran on. Two tiers are executable:
#
#   exact-unitary       Full-operator equivalence built directly from the IR
#                       (dense unitary, up to global phase). Basis- and
#                       input-independent: checks the whole operator, not one
#                       input state. Safest, but bounded by the 2^n dense-matrix
#                       cost (see DEFAULT_EXACT_QUBIT_BOUND).
#
#   `exact-clifford-sim`  Full-operator equivalence via stabilizer tableaux. Also
#                       exact and input-independent, and it has no qubit bound
#                       (the tableau is O(n^2)), but it can only represent
#                       Clifford circuits: a single T or arbitrary-angle
#                       rotation puts a kernel out of domain. Reach is bought by
#                       restricting the circuit class, so read a verdict from
#                       this tier against that class.
ASSURANCE_TIER_EXACT_UNITARY = "exact-unitary"
ASSURANCE_TIER_EXACT_CLIFFORD_SIM = "exact-clifford-sim"

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

# The kinds `DenseUnitaryOracle` answers to: how strictly it compares the two
# dense operators.
_DENSE_ORACLE_KINDS = ("strict-unitary", "up-to-global-phase")
CLIFFORD_TABLEAU_ORACLE_KIND = "clifford-tableau"
# Every built-in kind an `OracleSpec` may declare.
_ORACLE_KINDS = _DENSE_ORACLE_KINDS + (CLIFFORD_TABLEAU_ORACLE_KIND,)
PREDICATES = ("nonincreasing", "decreasing", "unchanged", "any")

# The first-class boolean `invariants` checked on every in-domain case, reported as
# structured `InvariantResult`s. These are the semantic guarantees the `validator`
# enforces beyond the declared resource metrics.
INVARIANT_EQUIVALENCE = "equivalence"
INVARIANT_DETERMINISM = "determinism"
INVARIANT_FIXED_POINT = "fixed-point"
INVARIANT_KINDS = (INVARIANT_EQUIVALENCE, INVARIANT_DETERMINISM,
                   INVARIANT_FIXED_POINT)

# What an equivalence verdict is a statement about.
#
#   `exact`           Neither kernel used `ancillas`. The verdict covers the
#                     whole operator.
#
#   `clean-ancilla`   A kernel introduced ancilla qubits (allocations marked
#                     `quake.ancilla`). They were checked to be returned to the
#                     basis state they came in as and projected out, so the
#                     verdict is that the two kernels agree on the system qubits
#                     when the `ancillas` start in |0>. It is not a claim about
#                     `ancillas` that arrive in any other state.
#
#   `borrowed-ancilla` The wider kernel was shown to be the narrower one
#                     `tensored` with the identity, so it never touches the
#                     extra qubits.
#                     Stronger than `clean-ancilla`: it holds whatever state the
#                     `ancillas` arrive in.
GUARANTEE_EXACT = "exact"
GUARANTEE_CLEAN_ANCILLA = "clean-ancilla"
GUARANTEE_BORROWED_ANCILLA = "borrowed-ancilla"
GUARANTEE_KINDS = (GUARANTEE_EXACT, GUARANTEE_CLEAN_ANCILLA,
                   GUARANTEE_BORROWED_ANCILLA)


@dataclass(frozen=True)
class OracleDescriptor:
    """One executable equivalence oracle.

    The authoritative, machine-readable description of how an oracle decides
    equivalence and how far it scales, so a caller (or the agent-facing skill)
    never has to infer an oracle's strength or domain from prose.
    """

    kind: str
    tier: str
    # The simulation/analysis method backing the decision.
    method: str
    # domain + scaling note.
    note: str


# The equivalence oracles executable in this build.
ORACLE_ROADMAP = (
    OracleDescriptor(
        kind="strict-unitary",
        tier=ASSURANCE_TIER_EXACT_UNITARY,
        method="dense-unitary-from-ir",
        note="Element-wise unitary equality. Bounded by the dense 2^n cost.",
    ),
    OracleDescriptor(
        kind="up-to-global-phase",
        tier=ASSURANCE_TIER_EXACT_UNITARY,
        method="dense-unitary-from-ir",
        note="Unitary equality after dividing out a global phase. Bounded by "
        "the dense 2^n cost.",
    ),
    OracleDescriptor(
        kind=CLIFFORD_TABLEAU_ORACLE_KIND,
        tier=ASSURANCE_TIER_EXACT_CLIFFORD_SIM,
        method="stabilizer-tableau-from-ir",
        note="Stabilizer-tableau equality, up to a global phase. No qubit "
        "bound (O(n^2) tableau), but Clifford circuits only.",
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
class PipelineTarget:
    """The fixed baseline compilation target a candidate is validated against."""

    prepare: str = ""
    observe: str = ""

    def with_pipeline(self, candidate: str) -> PipelineSpec:
        """Derive the concrete runner pipeline by substituting ``candidate``.

        Returns the :class:`PipelineSpec` the runner consumes: this target's
        fixed ``prepare``/``observe`` stages with ``candidate`` injected between
        them. The baseline is this target with no candidate substitution.
        """
        return PipelineSpec(prepare=self.prepare,
                            candidate=candidate,
                            observe=self.observe)


@dataclass(frozen=True)
class OracleSpec:
    """Declarative selection of a built-in oracle, and its tolerances.

    This is the config-level way to ask for a built-in oracle:
    ``strict-unitary``/``up-to-global-phase`` bind to
    :class:`DenseUnitaryOracle`, ``clifford-tableau`` to
    :class:`CliffordTableauOracle` (which ignores the tolerances - it is exact).
    The runner binds it to an :class:`Oracle` instance. To supply your own
    oracle, pass an :class:`Oracle` instance as the request's ``oracle``
    instead.
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
        ``supported`` is the oracle's own domain `preflight`
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
    guarantee: str = GUARANTEE_EXACT


class Oracle(abc.ABC):
    """The equivalence-oracle extension point.

    An oracle owns exactly one invariant semantic equivalence between the observed
    baseline and candidate modules together with its own domain `preflight` and the
    assurance tier of the evidence it produces. Resource metrics, determinism, and
    fixed-point are runner-owned `invariants` and are deliberately NOT an oracle's concern.

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
        if kind not in _DENSE_ORACLE_KINDS:
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
        elif comparison.get("ancilla_not_restored", False):
            # A determinate negative verdict, not a failure to compare. The
            # candidate's `ancillas` come back entangled with the system qubits,
            # so it does not implement the baseline operator on them.
            detail = (f"not equivalent under oracle '{self.kind}': "
                      "ancilla-not-restored")
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
            phase_is_zero=bool(comparison.get("phase_is_zero", False)),
            guarantee=str(comparison.get("guarantee", GUARANTEE_EXACT)))


class CliffordTableauOracle(Oracle):
    """Built-in exact-Clifford oracle: equivalence past the dense qubit bound.

    Compiles both modules to stabilizer tableaux directly from the IR (no
    simulator, no target) and compares them. Like the dense oracle this is
    exact, basis- and input-independent, and up to a global phase, but the
    tableau is O(n^2) rather than 2^n, so there is deliberately no qubit bound:
    Clifford kernels of hundreds or thousands of qubits are certifiable.

    The price is domain. A kernel is in domain only if every operation is
    Clifford (see ``preflight_clifford``); a single T, an arbitrary-angle
    rotation, a measurement/reset, or a Toffoli-class control puts it out. Out
    of domain is reported as an unsupported domain, never as a negative verdict.
    Wraps the ``preflight_clifford`` and ``compare_tableaux`` bindings.

    Kernels of different widths are compared by padding the narrower tableau
    with identity, so a candidate that took on `ancillas` certifies exactly when
    it leaves them untouched. That earns ``borrowed-ancilla``, which is a
    stronger claim than the dense oracle's ``clean-ancilla``: the dense
    oracle projects onto `ancillas` starting in |0>, whereas an untouched qubit
    is untouched whatever state it arrives in.
    """

    kind = CLIFFORD_TABLEAU_ORACLE_KIND
    tier = ASSURANCE_TIER_EXACT_CLIFFORD_SIM

    def decide(self, baseline: Module, candidate: Module,
               kernel_name: Optional[str]) -> OracleDecision:
        base_pf = cudaq_runtime.preflight_clifford(baseline)
        cand_pf = cudaq_runtime.preflight_clifford(candidate)
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

        comparison = cudaq_runtime.compare_tableaux(baseline, candidate,
                                                    kernel_name)
        computed = bool(comparison["computed"])
        equivalent = computed and bool(comparison["equivalent"])
        if not computed:
            detail = f"comparison failed: {comparison['error']}"
        elif not equivalent:
            detail = f"not equivalent under oracle '{self.kind}'"
        else:
            detail = f"equivalent under oracle '{self.kind}'"
        # The unitary-specific evidence fields stay at their defaults: a tableau
        # carries no global phase, so there is no phase to report and no
        # strict-vs-up-to-phase distinction to draw.
        return OracleDecision(supported=True,
                              computed=computed,
                              equivalent=equivalent,
                              tier=self.tier,
                              detail=detail,
                              guarantee=str(
                                  comparison.get("guarantee", GUARANTEE_EXACT)))


@dataclass(frozen=True)
class MetricSpec:
    """A declared metric and the predicate its delta must satisfy.

    ``name`` is one of ``operation-count``, ``single-qubit-count``,
    ``two-qubit-count``, ``multi-qubit-count``, ``three-plus-qubit-count``,
    ``qubit-count``, ``depth``, ``t-count``, ``rotation-count``, or
    ``gate:<name>``. ``predicate`` is one of ``nonincreasing``, ``decreasing``,
    ``unchanged``, ``any``.

    ``gating`` decides whether a violated predicate fails the case. A gating
    metric (the default) is a hard correctness gate. Violating its predicate is
    an ``invariant-failure``. An informational metric (``gating=False``) has
    its predicate outcome computed and reported, but never changes the case
    status. This is the metric/objective boundary. The core reports the metric
    tuple and each predicate outcome. The caller's (`untrusted`) objective owns the
    weighted scalar reward. The core never reduces metrics to a single reward,
    because that weighted reduction is exactly what an optimizer could game.
    """

    name: str
    predicate: str = "nonincreasing"
    gating: bool = True


@dataclass(frozen=True)
class ValidationRequest:
    inputs: tuple[Path, ...]
    pipeline: PipelineSpec
    oracle: Union[OracleSpec, Oracle] = field(default_factory=OracleSpec)
    metrics: tuple[MetricSpec, ...] = ()
    fixed_point_runs: int = 1
    exact_qubit_bound: int = DEFAULT_EXACT_QUBIT_BOUND
    kernel_name: Optional[str] = None


# Result contracts
@dataclass(frozen=True)
class MetricDelta:
    name: str
    predicate: str
    baseline: int
    candidate: int
    delta: int
    satisfied: bool
    # Whether a violated predicate gated (failed) the case. Informational
    # metrics (gating=False) report `satisfied` but never change case status.
    gating: bool = True


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
    # What the verdict is a statement about; one of GUARANTEE_KINDS.
    guarantee: str
    # Semantic `invariants` (equivalence, determinism, fixed-point).
    invariants: tuple[InvariantResult, ...]
    metrics: tuple[MetricDelta, ...]
    messages: tuple[str, ...]


@dataclass(frozen=True)
class ValidationResult:
    status: str
    cases: tuple[CaseResult, ...]
    aggregate_metrics: Mapping[str, MetricDelta]
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ValidationCapabilities:
    # Oracle kinds executable in this build.
    oracles: tuple[str, ...]
    metrics: tuple[str, ...]
    predicates: tuple[str, ...]
    # Invariants checked on every in-domain case.
    invariants: tuple[str, ...]
    # Assurance tiers this `validator` can accept at.
    assurance_tiers: tuple[str, ...]
    # The executable oracles, with tier and method for each.
    oracle_roadmap: tuple[OracleDescriptor, ...]


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

# Gates the resource counter names for a parameterized rotation.
_ROTATION_GATES = frozenset(("rx", "ry", "rz", "r1", "u3", "u2", "crx", "cry",
                             "crz", "cr1", "rxx", "ryy", "rzz", "phased_rx"))

# The metric names with executable support.
_METRIC_NAMES = ("operation-count", "single-qubit-count", "two-qubit-count",
                 "multi-qubit-count", "three-plus-qubit-count", "qubit-count",
                 "depth", "t-count", "rotation-count")
_GATE_METRIC_PREFIX = "gate:"


def _is_known_metric(name: str) -> bool:
    return name in _METRIC_NAMES or (name.startswith(_GATE_METRIC_PREFIX) and
                                     len(name) > len(_GATE_METRIC_PREFIX))


def _rotation_count(counts: dict) -> int:
    return sum(
        n for gate, n in counts["per_gate"].items() if gate in _ROTATION_GATES)


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
    if name == "qubit-count":
        return counts["num_qubits"]
    if name == "single-qubit-count":
        return counts["gate_count"] - counts["multi_qubit_count"]
    if name == "three-plus-qubit-count":
        return counts["multi_qubit_count"] - counts["two_qubit_count"]
    if name == "rotation-count":
        return _rotation_count(counts)
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


def _compute_metrics(baseline_obs: Module, candidate_obs: Module,
                     metric_specs) -> tuple[list, str, list]:
    """Count declared metrics on two observed modules and check predicates.

    Returns ``(metrics, status_delta, messages)``. ``status_delta`` is the worst
    status this contributes (a violated predicate is an invariant failure, failed
    counting is infrastructure). Counting runs only when metrics are requested,
    since the resource counter mutates the modules it is handed.
    """
    metrics: list[MetricDelta] = []
    messages: list[str] = []
    status = ValidationStatus.PASSED
    if not metric_specs:
        return metrics, status, messages
    base_counts = cudaq_runtime.count_resources_checkpoint(baseline_obs)
    cand_counts = cudaq_runtime.count_resources_checkpoint(candidate_obs)
    if base_counts["computed"] and cand_counts["computed"]:
        for spec in metric_specs:
            base_val = _metric_value(base_counts, spec.name)
            cand_val = _metric_value(cand_counts, spec.name)
            ok = _predicate_ok(spec.predicate, base_val, cand_val)
            metrics.append(
                MetricDelta(name=spec.name,
                            predicate=spec.predicate,
                            baseline=base_val,
                            candidate=cand_val,
                            delta=cand_val - base_val,
                            satisfied=ok,
                            gating=spec.gating))
            # Only a gating metric can fail the case. An informational metric
            # reports its predicate outcome but never escalates status. The
            # weighted reward is the caller's objective, not the core's.
            if not ok and spec.gating:
                status = _worst(status, ValidationStatus.INVARIANT_FAILURE)
                messages.append(f"metric '{spec.name}' violates "
                                f"'{spec.predicate}': {base_val} -> {cand_val}")
    else:
        status = _worst(status, ValidationStatus.INFRASTRUCTURE_FAILURE)
        messages.append("resource counting failed; metrics unavailable")
    return metrics, status, messages


def _format_rejections(decision: OracleDecision) -> list:
    """An oracle's domain rejections."""
    return [
        f"{rej['side']} unsupported: {rej['kind']} "
        f"in {rej['kernel']} ({rej['detail']})" for rej in decision.rejections
    ]


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
    if request.fixed_point_runs < 0:
        raise InvalidRequest("fixed_point_runs must be non-negative")
    for metric in request.metrics:
        if not _is_known_metric(metric.name):
            raise InvalidRequest(f"unknown metric '{metric.name}'")
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
    """No `invariants` were established."""
    return CaseResult(
        input=str(path),
        status=status,
        assurance_tier=tier,
        strict_equal=False,
        equal_up_to_global_phase=False,
        phase=0.0,
        phase_is_zero=False,
        guarantee=GUARANTEE_EXACT,
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


def _coerce_module(obj, ctx: Context) -> Module:
    """Coerce a caller-supplied artifact to a verified :class:`Module`.

    Accepts an already-parsed ``Module``, a `filesystem` ``Path``, or
    a string of IR text. Malformed IR is attributed to ``invalid-request``.
    """
    if isinstance(obj, Module):
        if not _module_is_valid(obj):
            raise _StageFailure(ValidationStatus.INVALID_REQUEST,
                                "provided module failed verification")
        return obj
    if isinstance(obj, Path):
        return _parse_input(obj, ctx)
    if isinstance(obj, str):
        try:
            module = Module.parse(obj, ctx)
        except Exception as exc:
            raise _StageFailure(
                ValidationStatus.INVALID_REQUEST,
                f"artifact IR failed to parse or verify: {exc}")
        if not _module_is_valid(module):
            raise _StageFailure(ValidationStatus.INVALID_REQUEST,
                                "artifact IR failed verification")
        return module
    raise _StageFailure(ValidationStatus.INVALID_REQUEST,
                        f"unsupported artifact type: {type(obj).__name__}")


def _resolve_oracle(oracle: Union[OracleSpec, Oracle],
                    qubit_bound: int) -> Oracle:
    """Resolve the request's oracle field to an executable :class:`Oracle`.

    An :class:`Oracle` instance is used as-is (user-supplied, the common case). A
    declarative :class:`OracleSpec` is bound to the built-in oracle that answers
    to its ``kind``: :class:`CliffordTableauOracle` for ``clifford-tableau``,
    otherwise :class:`DenseUnitaryOracle` using the request's qubit bound (the
    tableau oracle has no qubit bound, and no tolerances - it is exact).
    """
    if isinstance(oracle, Oracle):
        return oracle
    if oracle.kind == CLIFFORD_TABLEAU_ORACLE_KIND:
        return CliffordTableauOracle()
    return DenseUnitaryOracle(kind=oracle.kind,
                              rtol=oracle.rtol,
                              atol=oracle.atol,
                              qubit_bound=qubit_bound)


def _evaluate_observed(baseline_obs: Module, candidate_obs: Module,
                       oracle: Oracle, metric_specs, kernel_name: Optional[str],
                       *, input_label: str) -> CaseResult:
    """Trusted core over two observed modules: oracle equivalence + metrics.

    Runs no passes, so it cannot be crashed by a candidate pipeline. Reports the
    equivalence invariant and the declared metrics only. Determinism and
    fixed-point are pipeline `invariants` and are added by :func:`_evaluate_input`.
    An unsupported domain returns a failed case with no established `invariants`
    (empty ``invariants``), which is how the pipeline path detects it.
    """
    # The oracle owns the equivalence invariant and its own domain `preflight`.
    decision = oracle.decide(baseline_obs, candidate_obs, kernel_name)
    if not decision.supported:
        return _failed_case(input_label,
                            ValidationStatus.UNSUPPORTED_DOMAIN,
                            _format_rejections(decision),
                            tier=decision.tier)

    messages: list[str] = []
    equivalent = decision.equivalent
    status = ValidationStatus.PASSED
    if not decision.computed or not equivalent:
        status = ValidationStatus.INVARIANT_FAILURE
        messages.append(decision.detail)

    metrics, metric_status, metric_messages = _compute_metrics(
        baseline_obs, candidate_obs, metric_specs)
    status = _worst(status, metric_status)
    messages.extend(metric_messages)

    invariants = (InvariantResult(name=INVARIANT_EQUIVALENCE,
                                  satisfied=equivalent,
                                  detail=decision.detail),)

    return CaseResult(
        input=str(input_label),
        status=status,
        assurance_tier=decision.tier,
        strict_equal=decision.strict_equal,
        equal_up_to_global_phase=decision.equal_up_to_global_phase,
        phase=decision.phase,
        phase_is_zero=decision.phase_is_zero,
        guarantee=decision.guarantee,
        invariants=invariants,
        metrics=tuple(metrics),
        messages=tuple(messages),
    )


def _evaluate_input(path: Path, request: ValidationRequest,
                    ctx: Context) -> CaseResult:
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

    # Equivalence + metrics on the observed modules (shared with the artifact-in
    # path). An unsupported domain establishes no `invariants`, so return as-is.
    core = _evaluate_observed(baseline_obs,
                              candidate_obs,
                              oracle,
                              request.metrics,
                              request.kernel_name,
                              input_label=str(path))
    if not core.invariants:
        return core

    # Pipeline-only `invariants`: determinism and fixed-point require re-running
    # the candidate pipeline, so they belong to this path, not the trusted core.
    status = core.status
    messages = list(core.messages)

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

    invariants = core.invariants + (
        InvariantResult(name=INVARIANT_DETERMINISM, satisfied=deterministic),
        InvariantResult(name=INVARIANT_FIXED_POINT,
                        satisfied=fixed_point,
                        detail=f"{request.fixed_point_runs} run(s)"),
    )

    return dataclasses.replace(core,
                               status=status,
                               invariants=invariants,
                               messages=tuple(messages))


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
    )


def validate_artifacts(pairs,
                       *,
                       oracle: Optional[Union[OracleSpec, Oracle]] = None,
                       metrics: tuple = (),
                       exact_qubit_bound: int = DEFAULT_EXACT_QUBIT_BOUND,
                       kernel_name: Optional[str] = None) -> ValidationResult:
    """Validate already-compiled ``(baseline, candidate)`` Quake artifacts.

    The trusted, crash-isolated primitive: it runs no passes. The caller is
    responsible for compiling the (`untrusted`) candidate out-of-process (e.g. in
    a `subprocess` it owns) and hands the two observed modules here. This entry
    point only `preflights`, compares under the oracle, and counts metrics. Because
    no candidate pipeline runs in-process, a crashing candidate compile cannot
    take down the `validator`.

    ``pairs`` is an iterable of ``(baseline, candidate)``. Each side may be a
    parsed ``Module``, a `filesystem` ``Path``, or a string of IR text. Determinism
    and fixed-point are pipeline `invariants` and are therefore not reported here.
    """
    ctx = _make_context()
    if oracle is None:
        oracle = DenseUnitaryOracle(qubit_bound=exact_qubit_bound)
    try:
        resolved = _resolve_oracle(oracle, exact_qubit_bound)
    except InvalidRequest as exc:
        return ValidationResult(
            status=ValidationStatus.INVALID_REQUEST,
            cases=(),
            aggregate_metrics={},
            messages=(str(exc),),
        )

    cases: list[CaseResult] = []
    for index, pair in enumerate(pairs):
        label = f"artifact[{index}]"
        try:
            baseline, candidate = pair
        except (TypeError, ValueError):
            cases.append(
                _failed_case(label, ValidationStatus.INVALID_REQUEST,
                             ["expected a (baseline, candidate) pair"]))
            continue
        try:
            baseline_obs = _coerce_module(baseline, ctx)
            candidate_obs = _coerce_module(candidate, ctx)
        except _StageFailure as failure:
            cases.append(_failed_case(label, failure.status, [failure.message]))
            continue
        try:
            cases.append(
                _evaluate_observed(baseline_obs,
                                   candidate_obs,
                                   resolved,
                                   metrics,
                                   kernel_name,
                                   input_label=label))
        except Exception as exc:
            cases.append(
                _failed_case(label, ValidationStatus.INFRASTRUCTURE_FAILURE,
                             [f"infrastructure error: {exc}"]))

    status = ValidationStatus.PASSED
    for case in cases:
        status = _worst(status, case.status)

    return ValidationResult(
        status=status,
        cases=tuple(cases),
        aggregate_metrics=_aggregate_metrics(cases),
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

    This is the authoritative source for which oracles and metrics have
    executable support.
    """
    return ValidationCapabilities(
        oracles=_ORACLE_KINDS,
        metrics=_METRIC_NAMES + (f"{_GATE_METRIC_PREFIX}<name>",),
        predicates=PREDICATES,
        invariants=INVARIANT_KINDS,
        assurance_tiers=(ASSURANCE_TIER_EXACT_UNITARY,
                         ASSURANCE_TIER_EXACT_CLIFFORD_SIM),
        oracle_roadmap=ORACLE_ROADMAP,
    )


def result_to_dict(result: ValidationResult) -> dict:
    """Convert a result to a JSON-serializable dict."""
    return dataclasses.asdict(result)
