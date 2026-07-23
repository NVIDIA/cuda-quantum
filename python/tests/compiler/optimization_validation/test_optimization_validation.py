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
    ASSURANCE_TIER_ADVISORY,
    ASSURANCE_TIER_EXACT_CLIFFORD_SIM,
    ASSURANCE_TIER_EXACT_DENSITY_SIM,
    ASSURANCE_TIER_EXACT_UNITARY,
    INVARIANT_KINDS,
    ORACLE_ROADMAP,
    MetricSpec,
    Oracle,
    OracleDecision,
    OracleSpec,
    PipelineSpec,
    ValidationRequest,
    ValidationStatus,
    capabilities,
    result_to_dict,
    validate,
)

_INPUTS = Path(__file__).parent / "Inputs"

# Reused pipelines: memtoreg normalizes to wire form, phase-folding is a real,
# semantics-preserving candidate to exercise the happy path against.
_PREPARE = "builtin.module(func.func(memtoreg))"
_PHASE_FOLDING = "builtin.module(func.func(phase-folding))"


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
        seed=184467,
        preset="quick",
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
                     candidate="builtin.module(func.func(canonicalize))"),
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


# Request validation -> INVALID_REQUEST
def test_missing_input_file_is_invalid_request():
    result = validate(_request([Path("/no/such/file.qke")]))
    assert result.status == ValidationStatus.INVALID_REQUEST


def test_unknown_oracle_is_invalid_request(tmp_path):
    result = validate(
        _request([_good_input(tmp_path)], oracle=OracleSpec(kind="bogus")))
    assert result.status == ValidationStatus.INVALID_REQUEST


def test_unknown_preset_is_invalid_request(tmp_path):
    result = validate(_request([_good_input(tmp_path)], preset="deep"))
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
    assert back["result_schema_version"] == result.result_schema_version
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
def test_capabilities_accepts_only_exact_tier():
    caps = capabilities()
    assert caps.assurance_tiers == (ASSURANCE_TIER_EXACT_UNITARY,)
    assert caps.capability_schema_version == 4


def test_capabilities_advertise_first_class_invariants():
    caps = capabilities()
    assert set(caps.invariants) == set(INVARIANT_KINDS)


def test_oracle_roadmap_is_machine_readable_and_complete():
    caps = capabilities()
    roadmap = {o.kind: o for o in caps.oracle_roadmap}
    supported = {o.kind for o in caps.oracle_roadmap if o.status == "supported"}
    assert supported == set(
        caps.oracles) == {"strict-unitary", "up-to-global-phase"}
    for kind in supported:
        assert roadmap[kind].tier == ASSURANCE_TIER_EXACT_UNITARY

    assert roadmap["clifford-tableau"].tier == ASSURANCE_TIER_EXACT_CLIFFORD_SIM
    assert roadmap["clifford-tableau"].status == "deferred"
    assert roadmap["density-matrix"].tier == ASSURANCE_TIER_EXACT_DENSITY_SIM
    assert roadmap["density-matrix"].status == "deferred"
    assert roadmap["statevector-expectation"].tier == ASSURANCE_TIER_ADVISORY
    assert roadmap["statevector-expectation"].status == "deferred"


def test_oracle_roadmap_serializes():
    caps = capabilities()
    payload = dataclasses.asdict(caps)
    text = json.dumps(payload, sort_keys=True)
    back = json.loads(text)
    assert len(back["oracle_roadmap"]) == len(ORACLE_ROADMAP)
    for entry in back["oracle_roadmap"]:
        assert {"kind", "tier", "status", "method", "note"} <= set(entry.keys())
