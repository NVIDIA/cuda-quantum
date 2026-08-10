# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Unit tests for ``cudaq._compiler.optimization_validation.validate`` and the
capabilities/oracle-tier contract.

Run with::

    PYTHONPATH=build/python python3 -m pytest -v \\
        python/tests/compiler/optimization_validation/
"""

import dataclasses
import json
from pathlib import Path

import pytest

from cudaq._compiler import optimization_corpus as corpus
from cudaq._compiler.optimization_validation import (
    ASSURANCE_TIER_EXACT_CLIFFORD_SIM,
    ASSURANCE_TIER_EXACT_UNITARY,
    CLIFFORD_TABLEAU_ORACLE_KIND,
    DEFAULT_EXACT_QUBIT_BOUND,
    GUARANTEE_BORROWED_ANCILLA,
    GUARANTEE_CLEAN_ANCILLA,
    GUARANTEE_EXACT,
    INVARIANT_KINDS,
    ORACLE_ROADMAP,
    CliffordTableauOracle,
    DenseUnitaryOracle,
    InvalidRequest,
    MetricSpec,
    Oracle,
    OracleDecision,
    OracleSpec,
    PipelineSpec,
    PipelineTarget,
    ValidationRequest,
    ValidationStatus,
    _make_context,
    _resolve_oracle,
    _run_pipeline,
    capabilities,
    result_to_dict,
    validate,
    validate_artifacts,
)
from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.mlir.ir import Module

_INPUTS = Path(__file__).parent / "Inputs"

# Reused pipelines: memtoreg normalizes to wire form, phase-folding is a real,
# semantics-preserving candidate to exercise the happy path against.
_PREPARE = "builtin.module(func.func(memtoreg))"
_PHASE_FOLDING = "builtin.module(func.func(phase-folding))"
# A candidate that genuinely shrinks the seeded corpus kernel (it cancels the
# adjacent involution pair and merges the adjacent rotations, 10 gates -> 7).
# Tests about metric deltas need a pass that actually moves the count.
_QUAKE_SIMPLIFY = "builtin.module(func.func(quake-simplify))"


def _write(tmp_path, name, text) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def _good_input(tmp_path, seed=184467) -> Path:
    """A seeded, straight-line bounded-unitary kernel from the corpus."""
    return _write(tmp_path, f"generated_{seed}.qke",
                  corpus.generate_module_text(seed, num_qubits=2, length=6))


def _request(inputs, **kwargs) -> ValidationRequest:
    defaults = dict(
        pipeline=PipelineSpec(prepare=_PREPARE, candidate=_PHASE_FOLDING),
        oracle=OracleSpec(kind="up-to-global-phase"),
        metrics=(MetricSpec("operation-count", "nonincreasing"),),
        fixed_point_runs=1,
    )
    defaults.update(kwargs)
    return ValidationRequest(inputs=tuple(inputs), **defaults)


# Happy path
def test_semantics_preserving_candidate_passes(tmp_path):
    result = validate(_request([_good_input(tmp_path)]))
    assert result.status == ValidationStatus.PASSED
    assert len(result.cases) == 1
    case = result.cases[0]
    assert case.status == ValidationStatus.PASSED
    assert case.assurance_tier == ASSURANCE_TIER_EXACT_UNITARY
    assert case.equal_up_to_global_phase
    # Invariants are present and satisfied on the happy path.
    by_name = {inv.name: inv for inv in case.invariants}
    assert set(by_name) == set(INVARIANT_KINDS)
    assert all(inv.satisfied for inv in case.invariants)
    assert case.metrics
    assert all(m.satisfied for m in case.metrics)


def test_strict_oracle_is_recorded_on_the_case(tmp_path):
    result = validate(_request([_good_input(tmp_path)]))
    case = result.cases[0]
    assert isinstance(case.strict_equal, bool)
    assert isinstance(case.phase, float)
    assert isinstance(case.phase_is_zero, bool)


def test_canonical_corpus_input_validates(tmp_path):
    text = corpus.canonical_module_text("bell_pair")
    path = _write(tmp_path, "bell_pair.qke", text)
    result = validate(_request([path]))
    assert result.status == ValidationStatus.PASSED
    assert result.cases[0].assurance_tier == ASSURANCE_TIER_EXACT_UNITARY


# Target-shaped input: a baseline PipelineTarget + a candidate substitution.
def test_target_with_pipeline_injects_the_candidate_substitution():
    target = PipelineTarget(prepare=_PREPARE, observe="")
    spec = target.with_pipeline(_PHASE_FOLDING)
    assert spec == PipelineSpec(prepare=_PREPARE,
                                candidate=_PHASE_FOLDING,
                                observe="")


def test_target_derived_pipeline_matches_raw_pipeline_spec(tmp_path):
    inputs = [_good_input(tmp_path)]
    target = PipelineTarget(prepare=_PREPARE)
    via_target = validate(
        _request(inputs, pipeline=target.with_pipeline(_PHASE_FOLDING)))
    via_spec = validate(
        _request(inputs,
                 pipeline=PipelineSpec(prepare=_PREPARE,
                                       candidate=_PHASE_FOLDING)))
    assert via_target.status == via_spec.status == ValidationStatus.PASSED
    assert result_to_dict(via_target)["cases"] == result_to_dict(
        via_spec)["cases"]


# Oracle extension point: a user-supplied Oracle plugs in via the same contract
class _StubOracle(Oracle):
    """A minimal user-supplied oracle: returns a fixed equivalence verdict."""

    tier = ASSURANCE_TIER_EXACT_UNITARY

    def __init__(self, kind, equivalent):
        self.kind = kind
        self._equivalent = equivalent
        self.called = False

    def decide(self, baseline, candidate, kernel_name):
        self.called = True
        detail = "stub: equivalent" if self._equivalent else "stub: not equivalent"
        return OracleDecision(supported=True,
                              computed=True,
                              equivalent=self._equivalent,
                              tier=self.tier,
                              detail=detail)


def test_user_supplied_oracle_is_used(tmp_path):
    oracle = _StubOracle("always-equivalent", equivalent=True)
    result = validate(
        _request([_good_input(tmp_path)], oracle=oracle, metrics=()))
    assert oracle.called
    assert result.status == ValidationStatus.PASSED
    eq = {inv.name: inv for inv in result.cases[0].invariants}["equivalence"]
    assert eq.satisfied and eq.detail == "stub: equivalent"


def test_user_oracle_negative_verdict_is_invariant_failure(tmp_path):
    oracle = _StubOracle("never-equivalent", equivalent=False)
    result = validate(
        _request([_good_input(tmp_path)], oracle=oracle, metrics=()))
    assert result.status == ValidationStatus.INVARIANT_FAILURE


# Built-in Clifford-tableau oracle: exact, no qubit bound, Clifford-only domain
_CLIFFORD_KERNEL = """
func.func @kernel() {
  %q = quake.alloca !quake.veq<2>
  %q0 = quake.extract_ref %q[0] : (!quake.veq<2>) -> !quake.ref
  %q1 = quake.extract_ref %q[1] : (!quake.veq<2>) -> !quake.ref
  quake.h %q0 : (!quake.ref) -> ()
  quake.x [%q0] %q1 : (!quake.ref, !quake.ref) -> ()
  cc.return
}
"""

# The same kernel with one extra X on q1, so it is genuinely inequivalent.
_CLIFFORD_KERNEL_PERTURBED = _CLIFFORD_KERNEL.replace(
    "  cc.return", "  quake.x %q1 : (!quake.ref) -> ()\n  cc.return")


def _clifford_request(inputs, **kwargs):
    return _request(inputs,
                    oracle=OracleSpec(kind=CLIFFORD_TABLEAU_ORACLE_KIND),
                    **kwargs)


def test_clifford_tableau_oracle_certifies_a_clifford_kernel(tmp_path):
    path = _write(tmp_path, "clifford_mix.qke",
                  corpus.canonical_module_text("clifford_mix"))
    result = validate(_clifford_request([path], metrics=()))
    assert result.status == ValidationStatus.PASSED
    case = result.cases[0]
    assert case.assurance_tier == ASSURANCE_TIER_EXACT_CLIFFORD_SIM
    eq = {inv.name: inv for inv in case.invariants}["equivalence"]
    assert eq.satisfied
    # A tableau carries no global phase, so the dense-only evidence fields stay
    # at their defaults rather than claiming an equality they never checked.
    assert not case.strict_equal
    assert not case.equal_up_to_global_phase
    assert case.phase == 0.0


def test_clifford_tableau_oracle_rejects_a_non_clifford_kernel(tmp_path):
    path = _write(tmp_path, "t_ladder.qke",
                  corpus.canonical_module_text("t_ladder"))
    result = validate(_clifford_request([path], metrics=()))
    assert result.status == ValidationStatus.UNSUPPORTED_DOMAIN
    case = result.cases[0]
    assert case.assurance_tier == ASSURANCE_TIER_EXACT_CLIFFORD_SIM
    assert any("non-clifford-gate" in msg for msg in case.messages)
    # Fail-closed: an out-of-domain kernel establishes no invariants.
    assert not case.invariants


def test_clifford_tableau_oracle_detects_inequivalence():
    baseline, _ctx = _observed(_CLIFFORD_KERNEL)
    candidate, _ctx2 = _observed(_CLIFFORD_KERNEL_PERTURBED)
    result = validate_artifacts(
        [(str(baseline), str(candidate))],
        oracle=OracleSpec(kind=CLIFFORD_TABLEAU_ORACLE_KIND))
    assert result.status == ValidationStatus.INVARIANT_FAILURE
    case = result.cases[0]
    assert case.assurance_tier == ASSURANCE_TIER_EXACT_CLIFFORD_SIM
    assert not case.invariants[0].satisfied


def test_clifford_spec_resolves_to_the_tableau_oracle():
    resolved = _resolve_oracle(OracleSpec(kind=CLIFFORD_TABLEAU_ORACLE_KIND),
                               qubit_bound=14)
    assert isinstance(resolved, CliffordTableauOracle)
    assert resolved.kind == CLIFFORD_TABLEAU_ORACLE_KIND
    assert resolved.tier == ASSURANCE_TIER_EXACT_CLIFFORD_SIM
    # The dense kinds still resolve to the dense oracle.
    assert isinstance(
        _resolve_oracle(OracleSpec(kind="strict-unitary"), qubit_bound=14),
        DenseUnitaryOracle)


def test_dense_oracle_does_not_answer_to_the_clifford_kind():
    with pytest.raises(InvalidRequest):
        DenseUnitaryOracle(kind=CLIFFORD_TABLEAU_ORACLE_KIND)


# Why the tableau oracle exists: the same input, past the dense qubit bound.
_BEYOND_DENSE_BOUND_QUBITS = 20


def _ghz_input(tmp_path, num_qubits=_BEYOND_DENSE_BOUND_QUBITS, **kwargs):
    return _write(tmp_path, f"ghz_{num_qubits}.qke",
                  corpus.clifford_ghz_module_text(num_qubits, **kwargs))


def test_dense_oracle_rejects_a_kernel_past_its_qubit_bound(tmp_path):
    assert _BEYOND_DENSE_BOUND_QUBITS > DEFAULT_EXACT_QUBIT_BOUND
    result = validate(
        _request([_ghz_input(tmp_path)],
                 oracle=OracleSpec(kind="up-to-global-phase"),
                 metrics=()))
    assert result.status == ValidationStatus.UNSUPPORTED_DOMAIN
    assert any("too-many-qubits" in msg for msg in result.cases[0].messages)


def test_clifford_tableau_oracle_certifies_past_the_dense_bound(tmp_path):
    """The same 20-qubit kernel the dense oracle just refused, certified."""
    result = validate(_clifford_request([_ghz_input(tmp_path)], metrics=()))
    assert result.status == ValidationStatus.PASSED
    case = result.cases[0]
    assert case.assurance_tier == ASSURANCE_TIER_EXACT_CLIFFORD_SIM
    assert {inv.name: inv for inv in case.invariants}["equivalence"].satisfied


def test_clifford_tableau_oracle_is_not_vacuous_past_the_dense_bound(tmp_path):
    """Reach is worthless if it accepts everything: drop one CX and it fails."""
    baseline, _ctx = _observed(
        corpus.clifford_ghz_module_text(_BEYOND_DENSE_BOUND_QUBITS))
    candidate, _ctx2 = _observed(
        corpus.clifford_ghz_module_text(
            _BEYOND_DENSE_BOUND_QUBITS,
            chain_length=_BEYOND_DENSE_BOUND_QUBITS - 2))
    result = validate_artifacts(
        [(str(baseline), str(candidate))],
        oracle=OracleSpec(kind=CLIFFORD_TABLEAU_ORACLE_KIND))
    assert result.status == ValidationStatus.INVARIANT_FAILURE
    assert not result.cases[0].invariants[0].satisfied


# Artifact-in: validate already-compiled modules with no pass execution
def _observed(text) -> tuple:
    ctx = _make_context()
    module = Module.parse(text, ctx)
    _run_pipeline("builtin.module(func.func(memtoreg))", module, ctx)
    return module, ctx


def test_validate_artifacts_equivalent_pair_passes():
    observed, _ctx = _observed(
        corpus.generate_module_text(184467, num_qubits=2, length=6))
    text = str(observed)
    result = validate_artifacts([(text, text)],
                                oracle=OracleSpec(kind="up-to-global-phase"),
                                metrics=(MetricSpec("operation-count",
                                                    "nonincreasing"),))
    assert result.status == ValidationStatus.PASSED
    case = result.cases[0]
    assert [inv.name for inv in case.invariants] == ["equivalence"]
    assert case.invariants[0].satisfied
    assert case.assurance_tier == ASSURANCE_TIER_EXACT_UNITARY
    assert case.metrics and case.metrics[0].satisfied


def test_validate_artifacts_accepts_module_objects():
    observed, _ctx = _observed(
        corpus.generate_module_text(184467, num_qubits=2, length=6))
    result = validate_artifacts([(observed, observed)])
    assert result.status == ValidationStatus.PASSED
    assert len(result.cases[0].invariants) == 1


def test_validate_artifacts_reports_unsupported_domain():
    observed, _ctx = _observed((_INPUTS / "measurement.qke").read_text())
    text = str(observed)
    result = validate_artifacts([(text, text)])
    assert result.status == ValidationStatus.UNSUPPORTED_DOMAIN
    case = result.cases[0]
    assert any("measurement" in msg for msg in case.messages)
    assert not case.invariants


def test_validate_artifacts_bad_input_is_invalid_request():
    result = validate_artifacts([("not valid IR (((", "also bad")])
    assert result.status == ValidationStatus.INVALID_REQUEST


# Fail-closed on out-of-domain inputs
@pytest.mark.parametrize("fixture,reason", [
    ("measurement.qke", "measurement"),
    ("reset.qke", "reset"),
    ("dynamic_control_flow.qke", "dynamic-control-flow"),
])
def test_out_of_domain_input_fails_closed(fixture, reason):
    result = validate(_request([_INPUTS / fixture]))
    assert result.status == ValidationStatus.UNSUPPORTED_DOMAIN
    case = result.cases[0]
    assert case.status == ValidationStatus.UNSUPPORTED_DOMAIN
    assert any(reason in msg for msg in case.messages)
    # Fail-closed: never reports an equivalence it could not establish.
    assert not case.strict_equal
    assert not case.equal_up_to_global_phase


# Pipeline isolation: the baseline must never see the candidate pipeline
def test_baseline_is_isolated_from_candidate(tmp_path):
    good = _good_input(tmp_path)
    result = validate(
        _request([good],
                 pipeline=PipelineSpec(
                     prepare="builtin.module(func.func(memtoreg))",
                     candidate=_QUAKE_SIMPLIFY),
                 metrics=(MetricSpec("operation-count", "nonincreasing"),)))
    case = result.cases[0]
    assert case.status == ValidationStatus.PASSED
    (metric,) = [m for m in case.metrics if m.name == "operation-count"]
    assert metric.candidate < metric.baseline
    assert metric.baseline > metric.candidate


# Metric predicate enforcement
def test_violated_metric_predicate_is_invariant_failure(tmp_path):
    good = _good_input(tmp_path)
    result = validate(
        _request([good], metrics=(MetricSpec("operation-count",
                                             "decreasing"),)))
    case = result.cases[0]
    (metric,) = case.metrics
    if metric.baseline == metric.candidate:
        assert not metric.satisfied
        assert case.status == ValidationStatus.INVARIANT_FAILURE
        assert result.status == ValidationStatus.INVARIANT_FAILURE
    else:
        assert metric.satisfied


# Metric/objective split: gating vs informational, and no scalar reward
def _simplify_request(good, gating):
    """A candidate that changes the operation count, against an ``unchanged``
    predicate: the predicate is violated by construction."""
    return _request([good],
                    pipeline=PipelineSpec(prepare=_PREPARE,
                                          candidate=_QUAKE_SIMPLIFY),
                    metrics=(MetricSpec("operation-count",
                                        "unchanged",
                                        gating=gating),))


def test_informational_metric_reports_outcome_but_does_not_gate(tmp_path):
    good = _good_input(tmp_path)
    result = validate(_simplify_request(good, gating=False))
    case = result.cases[0]
    (metric,) = [m for m in case.metrics if m.name == "operation-count"]
    assert not metric.satisfied
    assert metric.gating is False
    assert case.status == ValidationStatus.PASSED
    assert result.status == ValidationStatus.PASSED


def test_gating_metric_violation_fails_the_case(tmp_path):
    good = _good_input(tmp_path)
    result = validate(_simplify_request(good, gating=True))
    case = result.cases[0]
    (metric,) = [m for m in case.metrics if m.name == "operation-count"]
    assert not metric.satisfied
    assert metric.gating is True
    assert case.status == ValidationStatus.INVARIANT_FAILURE


# Derived metrics.
_DERIVED_METRICS = ("single-qubit-count", "three-plus-qubit-count",
                    "qubit-count", "rotation-count")


def _metrics_by_name(tmp_path, names):
    result = validate(
        _request([_good_input(tmp_path)],
                 metrics=tuple(MetricSpec(n, "any") for n in names)))
    case = result.cases[0]
    assert case.status == ValidationStatus.PASSED
    return {m.name: m for m in case.metrics}


def test_declared_metrics_are_all_measurable(tmp_path):
    """Every metric capabilities() advertises can actually be counted.

    The loop builds its metric list straight from capabilities(), so a name
    advertised there but unhandled in _metric_value would fail every run.
    """
    names = [n for n in capabilities().metrics if not n.endswith("<name>")]
    assert set(_DERIVED_METRICS) <= set(names)
    metrics = _metrics_by_name(tmp_path, names)
    assert set(metrics) == set(names)


def test_arity_metrics_partition_the_operation_count(tmp_path):
    """single + multi == total, and three-plus is multi minus two."""
    m = _metrics_by_name(
        tmp_path, ("operation-count", "single-qubit-count", "two-qubit-count",
                   "multi-qubit-count", "three-plus-qubit-count"))
    for side in ("baseline", "candidate"):
        val = lambda name: getattr(m[name], side)
        assert val("single-qubit-count") + val("multi-qubit-count") == \
               val("operation-count")
        assert val("three-plus-qubit-count") == \
               val("multi-qubit-count") - val("two-qubit-count")


def test_rotation_count_counts_only_rotation_bearing_gates(tmp_path):
    """rotation-count is bounded by the op count and agrees with per-gate sums.

    It is a name-membership test, so the invariant that matters is that it never
    exceeds the total and never counts a gate that carries no angle.
    """
    names = ("operation-count", "rotation-count", "gate:rz", "gate:h",
             "gate:cx")
    m = _metrics_by_name(tmp_path, names)
    for side in ("baseline", "candidate"):
        val = lambda name: getattr(m[name], side)
        assert 0 <= val("rotation-count") <= val("operation-count")
        # rz carries an angle and is counted; h and cx do not and are not.
        assert val("rotation-count") >= val("gate:rz")
        assert val("rotation-count") <= \
               val("operation-count") - val("gate:h") - val("gate:cx")


def test_qubit_count_is_the_allocated_width(tmp_path):
    m = _metrics_by_name(tmp_path, ("qubit-count", "operation-count"))
    assert m["qubit-count"].baseline > 0
    # A pipeline that only canonicalizes cannot widen the register.
    assert m["qubit-count"].candidate == m["qubit-count"].baseline


def test_unknown_metric_is_invalid_request(tmp_path):
    result = validate(
        _request([_good_input(tmp_path)],
                 metrics=(MetricSpec("not-a-metric", "any"),)))
    assert result.status == ValidationStatus.INVALID_REQUEST


def _all_keys(obj):
    """Every mapping key anywhere in a nested dict/list structure."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _all_keys(value)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _all_keys(item)


def test_core_never_emits_a_scalar_reward(tmp_path):
    result = validate(_request([_good_input(tmp_path)]))
    keys = {k.lower() for k in _all_keys(result_to_dict(result))}
    for banned in ("reward", "score", "scalar", "objective", "fitness"):
        assert banned not in keys


# Request validation -> INVALID_REQUEST
def test_missing_input_file_is_invalid_request():
    result = validate(_request([Path("/no/such/file.qke")]))
    assert result.status == ValidationStatus.INVALID_REQUEST


def test_unknown_oracle_is_invalid_request(tmp_path):
    result = validate(
        _request([_good_input(tmp_path)], oracle=OracleSpec(kind="bogus")))
    assert result.status == ValidationStatus.INVALID_REQUEST


def test_bad_pipeline_string_is_invalid_request(tmp_path):
    result = validate(
        _request([_good_input(tmp_path)],
                 pipeline=PipelineSpec(candidate="not-a-real-pipeline(((")))
    assert result.status == ValidationStatus.INVALID_REQUEST


def test_empty_inputs_is_invalid_request():
    result = validate(_request([]))
    assert result.status == ValidationStatus.INVALID_REQUEST


def test_malformed_input_ir_is_invalid_request():
    result = validate(_request([_INPUTS / "invalid_ir.qke"]))
    assert result.status == ValidationStatus.INVALID_REQUEST
    case = result.cases[0]
    assert case.status == ValidationStatus.INVALID_REQUEST
    assert any("failed to parse or verify" in m or "failed verification" in m
               for m in case.messages)
    assert not case.strict_equal
    assert not case.equal_up_to_global_phase


# JSON serialization round-trip
def test_result_json_round_trips(tmp_path):
    result = validate(_request([_good_input(tmp_path)]))
    payload = result_to_dict(result)
    text = json.dumps(payload, sort_keys=True)
    back = json.loads(text)
    assert back["status"] == result.status
    assert len(back["cases"]) == len(result.cases)


# Aggregate status is the most severe case
def test_aggregate_status_is_worst_case(tmp_path):
    good = _good_input(tmp_path)
    bad = _INPUTS / "measurement.qke"
    result = validate(_request([good, bad]))
    assert result.status == ValidationStatus.UNSUPPORTED_DOMAIN
    statuses = {c.status for c in result.cases}
    assert ValidationStatus.PASSED in statuses
    assert ValidationStatus.UNSUPPORTED_DOMAIN in statuses


# Capabilities / oracle-tier contract
def test_capabilities_accepts_only_exact_tiers():
    caps = capabilities()
    assert caps.assurance_tiers == (ASSURANCE_TIER_EXACT_UNITARY,
                                    ASSURANCE_TIER_EXACT_CLIFFORD_SIM)


def test_capabilities_advertise_first_class_invariants():
    caps = capabilities()
    assert set(caps.invariants) == set(INVARIANT_KINDS)


def test_oracle_roadmap_lists_only_supported_oracles():
    caps = capabilities()
    roadmap = {o.kind: o for o in caps.oracle_roadmap}
    # Every listed oracle is executable, so the roadmap kinds match `oracles`.
    assert set(roadmap) == set(caps.oracles) == {
        "strict-unitary", "up-to-global-phase", CLIFFORD_TABLEAU_ORACLE_KIND
    }
    for descriptor in caps.oracle_roadmap:
        assert descriptor.tier in caps.assurance_tiers
    assert roadmap[CLIFFORD_TABLEAU_ORACLE_KIND].tier == (
        ASSURANCE_TIER_EXACT_CLIFFORD_SIM)


def test_oracle_roadmap_serializes():
    caps = capabilities()
    payload = dataclasses.asdict(caps)
    text = json.dumps(payload, sort_keys=True)
    back = json.loads(text)
    assert len(back["oracle_roadmap"]) == len(ORACLE_ROADMAP)
    for entry in back["oracle_roadmap"]:
        assert {"kind", "tier", "method", "note"} == set(entry.keys())


# Ancilla-introducing candidates
_MULTICONTROL = "builtin.module(func.func(multicontrol-decomposition))"

# A four-control X that allocates its own qubits, so `multicontrol-decomposition`
# gives its two ancillas the lowest indices.
_MCX = """
func.func @mcx() {
  %v = quake.alloca !quake.veq<5>
  %c0 = quake.extract_ref %v[0] : (!quake.veq<5>) -> !quake.ref
  %c1 = quake.extract_ref %v[1] : (!quake.veq<5>) -> !quake.ref
  %c2 = quake.extract_ref %v[2] : (!quake.veq<5>) -> !quake.ref
  %c3 = quake.extract_ref %v[3] : (!quake.veq<5>) -> !quake.ref
  %t = quake.extract_ref %v[4] : (!quake.veq<5>) -> !quake.ref
  quake.x [%c0, %c1, %c2, %c3] %t : (!quake.ref, !quake.ref, !quake.ref, !quake.ref, !quake.ref) -> ()
  return
}
"""

# The same three-control X written two ways: with a marked ancilla that is
# uncomputed, and with the uncompute dropped so the ancilla stays entangled.
_CCCX = """
func.func @ccc_x() {
  %v = quake.alloca !quake.veq<4>
  %c0 = quake.extract_ref %v[0] : (!quake.veq<4>) -> !quake.ref
  %c1 = quake.extract_ref %v[1] : (!quake.veq<4>) -> !quake.ref
  %c2 = quake.extract_ref %v[2] : (!quake.veq<4>) -> !quake.ref
  %t = quake.extract_ref %v[3] : (!quake.veq<4>) -> !quake.ref
  quake.x [%c0, %c1, %c2] %t : (!quake.ref, !quake.ref, !quake.ref, !quake.ref) -> ()
  return
}
"""

_CCCX_CLEAN_ANCILLA = """
func.func @ccc_x() {
  %c0 = quake.alloca !quake.ref
  %anc = quake.alloca !quake.ref {quake.ancilla}
  %c1 = quake.alloca !quake.ref
  %c2 = quake.alloca !quake.ref
  %t = quake.alloca !quake.ref
  quake.x [%c0, %c1] %anc : (!quake.ref, !quake.ref, !quake.ref) -> ()
  quake.x [%c2, %anc] %t : (!quake.ref, !quake.ref, !quake.ref) -> ()
  quake.x [%c0, %c1] %anc : (!quake.ref, !quake.ref, !quake.ref) -> ()
  return
}
"""

_CCCX_DIRTY_ANCILLA = """
func.func @ccc_x() {
  %c0 = quake.alloca !quake.ref
  %anc = quake.alloca !quake.ref {quake.ancilla}
  %c1 = quake.alloca !quake.ref
  %c2 = quake.alloca !quake.ref
  %t = quake.alloca !quake.ref
  quake.x [%c0, %c1] %anc : (!quake.ref, !quake.ref, !quake.ref) -> ()
  quake.x [%c2, %anc] %t : (!quake.ref, !quake.ref, !quake.ref) -> ()
  return
}
"""


def _decide(baseline_text, candidate_text) -> OracleDecision:
    ctx = _make_context()
    return DenseUnitaryOracle(kind="up-to-global-phase").decide(
        Module.parse(baseline_text, ctx), Module.parse(candidate_text, ctx),
        None)


def test_ancilla_introducing_candidate_is_certified(tmp_path):
    """A pass that adds ancillas is certifiable, not a dimension mismatch."""
    result = validate(
        _request([_write(tmp_path, "mcx.qke", _MCX)],
                 pipeline=PipelineSpec(candidate=_MULTICONTROL),
                 metrics=(MetricSpec("qubit-count", "any", gating=False),)))
    assert result.status == ValidationStatus.PASSED
    case = result.cases[0]
    assert case.equal_up_to_global_phase
    # The ancillas are real. The candidate is wider than the baseline.
    qubits = {m.name: m for m in case.metrics}["qubit-count"]
    assert qubits.candidate > qubits.baseline


def test_ancilla_verdict_names_the_weaker_guarantee(tmp_path):
    """An ancilla verdict is a claim about ancillas starting in |0>, and says so."""
    decision = _decide(_CCCX, _CCCX_CLEAN_ANCILLA)
    assert decision.computed and decision.equivalent
    assert decision.guarantee == GUARANTEE_CLEAN_ANCILLA


def test_ancilla_free_verdict_is_exact(tmp_path):
    result = validate(_request([_good_input(tmp_path)]))
    assert result.cases[0].guarantee == GUARANTEE_EXACT


def test_dirty_ancilla_is_a_verdict_not_an_infrastructure_failure():
    """A candidate that leaves its ancilla entangled is wrong, not unbuildable."""
    decision = _decide(_CCCX, _CCCX_DIRTY_ANCILLA)
    assert decision.supported
    assert decision.computed
    assert not decision.equivalent
    assert "ancilla-not-restored" in decision.detail


def test_preflight_reports_the_ancilla_share_of_the_qubit_bound():
    """Ancillas count against the flat bound."""
    ctx = _make_context()
    status = cudaq_runtime.preflight_bounded_unitary(
        Module.parse(_CCCX_CLEAN_ANCILLA, ctx), DEFAULT_EXACT_QUBIT_BOUND)
    assert status["supported"]
    assert status["max_qubits"] == 5
    assert status["max_ancilla_qubits"] == 1


# Tableau tensor mode: a Clifford kernel that took on ancillas
_GHZ3 = """
func.func @ghz() {
  %v = quake.alloca !quake.veq<3>
  %q0 = quake.extract_ref %v[0] : (!quake.veq<3>) -> !quake.ref
  %q1 = quake.extract_ref %v[1] : (!quake.veq<3>) -> !quake.ref
  %q2 = quake.extract_ref %v[2] : (!quake.veq<3>) -> !quake.ref
  quake.h %q0 : (!quake.ref) -> ()
  quake.x [%q0] %q1 : (!quake.ref, !quake.ref) -> ()
  quake.x [%q1] %q2 : (!quake.ref, !quake.ref) -> ()
  return
}
"""

# The same circuit plus a marked ancilla at the *lowest* index, which is where
# an allocating pass actually puts one.
_GHZ3_WITH_ANCILLA = """
func.func @ghz() {
  %anc = quake.alloca !quake.ref {quake.ancilla}
  %v = quake.alloca !quake.veq<3>
  %q0 = quake.extract_ref %v[0] : (!quake.veq<3>) -> !quake.ref
  %q1 = quake.extract_ref %v[1] : (!quake.veq<3>) -> !quake.ref
  %q2 = quake.extract_ref %v[2] : (!quake.veq<3>) -> !quake.ref
  quake.h %q0 : (!quake.ref) -> ()
  quake.x [%q0] %q1 : (!quake.ref, !quake.ref) -> ()
  quake.x [%q1] %q2 : (!quake.ref, !quake.ref) -> ()
  return
}
"""

# The ancilla is entangled with the system and left that way, so the candidate
# is not the baseline tensored with anything.
_GHZ3_ANCILLA_USED = _GHZ3_WITH_ANCILLA.replace(
    "  return",
    "  quake.x [%q0] %anc : (!quake.ref, !quake.ref) -> ()\n  return")

# Touched but returned: still the baseline tensored with the identity.
_GHZ3_ANCILLA_RESTORED = _GHZ3_WITH_ANCILLA.replace(
    "  return", "  quake.x [%q0] %anc : (!quake.ref, !quake.ref) -> ()\n"
    "  quake.x [%q0] %anc : (!quake.ref, !quake.ref) -> ()\n  return")


def _decide_clifford(baseline_text, candidate_text) -> OracleDecision:
    ctx = _make_context()
    return CliffordTableauOracle().decide(Module.parse(baseline_text, ctx),
                                          Module.parse(candidate_text, ctx),
                                          None)


def test_tableau_certifies_a_kernel_padded_with_an_unused_ancilla():
    decision = _decide_clifford(_GHZ3, _GHZ3_WITH_ANCILLA)
    assert decision.computed and decision.equivalent
    # Untouched is untouched whatever the ancilla arrives in, which is a
    # stronger claim than the dense oracle's clean-ancilla projection.
    assert decision.guarantee == GUARANTEE_BORROWED_ANCILLA


def test_tableau_certifies_an_ancilla_that_is_used_and_returned():
    decision = _decide_clifford(_GHZ3, _GHZ3_ANCILLA_RESTORED)
    assert decision.computed and decision.equivalent


def test_tableau_rejects_a_kernel_whose_ancilla_is_genuinely_used():
    decision = _decide_clifford(_GHZ3, _GHZ3_ANCILLA_USED)
    assert decision.computed
    assert not decision.equivalent


def test_tableau_verdict_without_ancillas_is_exact():
    decision = _decide_clifford(_GHZ3, _GHZ3)
    assert decision.equivalent
    assert decision.guarantee == GUARANTEE_EXACT


def test_tableau_fails_closed_on_an_unmarked_extra_qubit():
    """Widening without the marker is indistinguishable from a relabeling."""
    unmarked = _GHZ3_WITH_ANCILLA.replace(" {quake.ancilla}", "")
    decision = _decide_clifford(_GHZ3, unmarked)
    assert decision.computed
    assert not decision.equivalent


# The depth-vs-ancilla frontier: one input, two decomposition strategies
_C5X = """
func.func @c5x() {
  %v = quake.alloca !quake.veq<6>
  %c0 = quake.extract_ref %v[0] : (!quake.veq<6>) -> !quake.ref
  %c1 = quake.extract_ref %v[1] : (!quake.veq<6>) -> !quake.ref
  %c2 = quake.extract_ref %v[2] : (!quake.veq<6>) -> !quake.ref
  %c3 = quake.extract_ref %v[3] : (!quake.veq<6>) -> !quake.ref
  %c4 = quake.extract_ref %v[4] : (!quake.veq<6>) -> !quake.ref
  %t = quake.extract_ref %v[5] : (!quake.veq<6>) -> !quake.ref
  quake.x [%c0, %c1, %c2, %c3, %c4] %t : (!quake.ref, !quake.ref, !quake.ref, !quake.ref, !quake.ref, !quake.ref) -> ()
  return
}
"""


def _decomposition_metrics(tmp_path, strategy):
    """Validate the same kernel under one decomposition strategy."""
    path = _write(tmp_path, f"c5x_{strategy}.qke", _C5X)
    result = validate(
        _request([path],
                 pipeline=PipelineSpec(
                     candidate="builtin.module(func.func("
                     f"multicontrol-decomposition{{strategy={strategy}}}))"),
                 metrics=(MetricSpec("qubit-count", "any", gating=False),
                          MetricSpec("operation-count", "any", gating=False))))
    assert result.status == ValidationStatus.PASSED
    case = result.cases[0]
    assert case.guarantee == GUARANTEE_CLEAN_ANCILLA
    return {m.name: m for m in case.metrics}


def test_decomposition_strategies_are_both_certified(tmp_path):
    """Both constructions of the same operation certify against the original."""
    for strategy in ("v-ladder", "barenco"):
        metrics = _decomposition_metrics(tmp_path, strategy)
        assert metrics["qubit-count"].candidate > metrics[
            "qubit-count"].baseline


def test_decomposition_strategies_trade_qubits_against_gates(tmp_path):
    """The frontier the agent loop is meant to search: neither strategy wins on
    both axes, so the choice is an objective's to make, not the core's."""
    ladder = _decomposition_metrics(tmp_path, "v-ladder")
    barenco = _decomposition_metrics(tmp_path, "barenco")
    assert barenco["qubit-count"].candidate < ladder["qubit-count"].candidate
    assert barenco["operation-count"].candidate > ladder[
        "operation-count"].candidate
