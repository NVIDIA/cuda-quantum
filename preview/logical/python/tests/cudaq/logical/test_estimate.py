# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
import json
import math
import threading

import numpy as np
import pytest

import cudaq.logical as qlx
import cudaq.mlir.ir as mlir_ir
import cudaq.logical.architectures.surface as surface
from cudaq.logical.analysis import Tier
from cudaq.logical.architectures import pinnacle
from cudaq.logical.estimate.types import ScheduleEstimate, ScheduleTermination

TinyCode = qlx.codes.Steane


@qlx.gadget(implements=qlx.logical.h)
def encoded_h(block: qlx.patch[TinyCode]) -> qlx.patch[TinyCode]:
    return qlx.h(block.data)


@qlx.protocol(implements=qlx.logical.idle)
def four_h(block: qlx.patch[TinyCode]) -> qlx.patch[TinyCode]:
    return qlx.ops.repeat(4,
                          carries=(block,),
                          body=lambda value: encoded_h(value))


@qlx.gadget(implements=qlx.logical.idle)
def memory(block: qlx.patch[TinyCode]) -> qlx.patch[TinyCode]:
    block, _ = qlx.extract_syndrome(block, record="r0")
    block, _ = qlx.extract_syndrome(block, record="r1")
    return block


def test_callable_estimate_namespace_preserves_folded_logical_counts():

    @qlx.program
    def folded() -> bool:
        q = qlx.prepare_zero()
        q, = qlx.ops.repeat(5,
                            carries=(q,),
                            body=lambda i, value: (qlx.h(value),))
        return qlx.measure_z(q)

    build = qlx.compile(folded)
    profile = qlx.estimate(build, tier=Tier.LOGICAL)
    # The body remains folded in IR, while its exact static multiplicity is
    # reflected in the estimate.
    assert profile.actions["qlx_standard_h"] == 5
    assert profile.instruments["qlx_standard_prepare_zero"] == 1
    assert profile.instruments["qlx_standard_measure_z"] == 1
    assert profile.logical_qubits_peak == 1
    assert profile.action_depth_upper_bound == 7
    assert profile.build_root == build.root.symbol
    assert profile.build_sha256 == build.content_sha256


def test_in_scope_estimate_results_project_to_plain_data():

    @qlx.program
    def portable() -> bool:
        return qlx.measure_z(qlx.prepare_zero())

    logical = qlx.estimate(portable, tier=Tier.LOGICAL)
    static = qlx.estimate(qlx.compile(four_h))
    analytical = qlx.estimate(
        memory,
        tier=Tier.ANALYTICAL,
        p_phys=1e-3,
        failure_budget=0.1,
    )

    for value in (logical, static, analytical):
        payload = value.to_dict()
        assert json.loads(json.dumps(payload)) == payload


def test_estimate_results_rehydrate_from_cudaq_annotations():

    @qlx.program
    def portable() -> bool:
        return qlx.measure_z(qlx.prepare_zero())

    logical = qlx.estimate(portable, tier=Tier.LOGICAL)
    static = qlx.estimate(qlx.compile(four_h))
    analytical = qlx.estimate(
        memory,
        tier=Tier.ANALYTICAL,
        p_phys=1e-3,
        failure_budget=0.1,
    )
    annotations = {
        Tier.LOGICAL.name: logical.to_dict(),
        Tier.STATIC.name: static.to_dict(),
        Tier.ANALYTICAL.name: analytical.to_dict(),
    }

    assert qlx.estimate.LogicalProfile.from_annotations(annotations) == logical
    assert qlx.estimate.FabricCounts.from_annotations(annotations) == static
    assert qlx.estimate.FabricEstimate.from_annotations(
        annotations) == analytical


def test_static_estimate_rejects_a_program_without_selected_p2_evidence():

    @qlx.program
    def portable() -> bool:
        return qlx.measure_z(qlx.prepare_zero())

    with pytest.raises(ValueError, match="selected P2"):
        qlx.estimate(qlx.compile(portable))


def test_callable_estimate_materializes_authoring_definitions_implicitly():

    @qlx.program
    def portable() -> bool:
        return qlx.measure_z(qlx.prepare_zero())

    logical = qlx.estimate(portable, tier=Tier.LOGICAL)
    static = qlx.estimate(four_h, tier=Tier.STATIC)
    analytical = qlx.estimate(
        memory,
        tier=Tier.ANALYTICAL,
        p_phys=1e-3,
        failure_budget=0.1,
    )

    assert logical.logical_qubits_peak == 1
    assert static.gadget_calls["encoded_h"] == 4
    assert analytical.distance == 3


def test_static_estimate_expands_folded_call_multiplicity_analytically():
    counts = qlx.estimate(qlx.compile(four_h))
    assert counts.source_stage == "p2"
    assert "protocol_network" in counts.source_facets
    assert counts.operation_counts["repeat"] == 1
    assert counts.operation_counts["h"] == 4
    assert counts.gadget_calls["encoded_h"] == 4
    assert counts.logical_qubits_peak == 1
    assert counts.total_operations == 4


def test_static_profile_counts_analysis_and_realization_separately():
    profile = qlx.gadgets.GadgetProfile(
        memory,
        success=(qlx.gadgets.SuccessPredicate(memory.record("r1.s0")),),
        name="memory_analysis",
    )
    counts = qlx.estimate(qlx.compile(profile))
    assert counts.success_count == 1
    assert counts.syndrome_rounds == 2
    assert counts.operation_counts["read_syndrome_ancillas"] == 2


def test_estimation_tiers_fail_closed_when_fidelity_inputs_are_missing():
    build = qlx.compile(four_h)
    estimate = qlx.estimate(
        build,
        tier=Tier.ANALYTICAL,
        p_phys=1e-3,
        failure_budget=qlx.analysis.FailureBudget(total=0.01),
        cycle_time=2.0,
    )
    assert estimate.counts.total_operations == 4
    assert estimate.distance == 3
    assert estimate.physical_qubits_peak == 13
    assert estimate.wallclock == 8.0
    assert estimate.logical_error == pytest.approx(1 - (1 - 1e-3)**4)
    assert estimate.budget_met

    with pytest.raises(qlx.errors.MissingEvidence, match="not established"):
        qlx.estimate(
            build,
            tier=Tier.ANALYTICAL,
            p_phys=1e-3,
            failure_budget=0.01,
            evidence_policy=qlx.analysis.EvidencePolicy.
            require_established_distance(),
        )

    @qlx.program
    def portable() -> bool:
        return qlx.measure_z(qlx.prepare_zero())

    with pytest.raises(ValueError, match="selected P2"):
        qlx.estimate(qlx.compile(portable))


def test_analytical_probability_aggregation_is_stable_at_small_error():

    @qlx.protocol
    def three_sites(block: qlx.patch[TinyCode]) -> qlx.patch[TinyCode]:
        block = encoded_h(block)
        block = encoded_h(block)
        return encoded_h(block)

    build = qlx.compile(three_sites)
    estimate = qlx.analysis.estimate(
        build,
        tier=Tier.ANALYTICAL,
        p_phys=1.0e-4,
        failure_budget=1.0e-20,
    )
    assert estimate.logical_error > 0.0
    assert not estimate.budget_met

    for invalid in (math.nan, math.inf):
        with pytest.raises(ValueError, match="cycle_time must be finite"):
            qlx.analysis.estimate(
                build,
                tier=Tier.ANALYTICAL,
                p_phys=1.0e-4,
                failure_budget=1.0e-2,
                cycle_time=invalid,
            )
    for invalid in (math.nan, math.inf):
        with pytest.raises(ValueError, match="scaling prefactor"):
            qlx.analysis.Scaling(prefactor=invalid)


@qlx.machine
class EstimateMachine:
    compute = qlx.architecture.Space(capacity=1)


estimate_qubits = qlx.architecture.ResourceClass("qubit", 2)
estimate_architecture = qlx.architecture.PhysicalMachine(
    "estimate_architecture",
    resource_classes={"qubits": estimate_qubits},
)


@qlx.physical(estimate_architecture)
def scheduled_experiment():
    q, = qlx.ops.acquire(estimate_architecture.qubits,
                         count=1,
                         kind=qlx.architecture.physical_qubit)
    q, = qlx.ops.load((q,), state="zero")
    q = qlx.ops.apply("h", q)
    return qlx.measure(q, basis=qlx.architecture.Basis.Z)


@qlx.physical(estimate_architecture)
def scheduled_experiment_same_metrics():
    q, = qlx.ops.acquire(estimate_architecture.qubits,
                         count=1,
                         kind=qlx.architecture.physical_qubit)
    q, = qlx.ops.load((q,), state="zero")
    q = qlx.ops.apply("h", q)
    return qlx.measure(q, basis=qlx.architecture.Basis.Z)


@qlx.program
def selected_schedule_experiment() -> bool:
    qubit = qlx.prepare_zero()
    qubit = qlx.h(qubit)
    return qlx.measure_z(qubit)


@qlx.program
def selected_schedule_experiment_same_metrics() -> bool:
    qubit = qlx.prepare_zero()
    qubit = qlx.h(qubit)
    return qlx.measure_z(qubit)


def _schedule_estimate_device():
    definitions = surface.definitions(3)
    builder = qlx.devices.DeviceBuilder("ScheduleEstimateDevice")
    logical = builder.logical.add_compute(capacity=1, name="compute")
    encoded = builder.qec.bind(logical, architecture=definitions.wsc())
    patches = builder.physical.add_resources(
        "surface_code_patch",
        4,
        name="compute_patches",
        granularity=qlx.architecture.ResourceGranularity.PATCH,
        footprint=definitions.square_patch_footprint,
        native_actions=qlx.architecture.physical_actions.clifford_set(),
        native_instruments=(
            qlx.architecture.physical_instruments.MX,
            qlx.architecture.physical_instruments.MZ,
        ),
    )
    builder.physical.bind(encoded, to=patches)
    for auxiliary in encoded.auxiliary_regions:
        builder.physical.bind(auxiliary, to=patches)
    builder.physical.set_operating_point(
        timing={
            "surface_cycle_ns": 1.0,
            "h_ns": 1.0,
            "measure_z_instrument_ns": 1.0,
        },
        calibration={
            "physical_error": 1.0e-3,
            "surface_scaling_prefactor": 0.1,
            "surface_threshold": 0.01,
        },
    )
    return builder.build()


ScheduleEstimateDevice = _schedule_estimate_device()
_SELECTED_SCHEDULE_BUILDS = {}
_SCHEDULE_ESTIMATE_OPTIONS = {"failure_budget": 0.1}


def _selected_schedule_build(definition=selected_schedule_experiment):
    key = definition.name
    if key not in _SELECTED_SCHEDULE_BUILDS:
        p2 = qlx.compile(
            definition,
            pipeline=qlx.compiler.pipelines.qec(),
            device=ScheduleEstimateDevice,
        )
        _SELECTED_SCHEDULE_BUILDS[key] = qlx.compile(
            p2,
            pipeline=qlx.compiler.pipelines.physical(),
            device=ScheduleEstimateDevice,
        )
    return _SELECTED_SCHEDULE_BUILDS[key]


def test_physical_parameters_default_to_the_device_operating_point():
    selected = qlx.compile(
        selected_schedule_experiment,
        pipeline=qlx.compiler.pipelines.qec(),
        device=ScheduleEstimateDevice,
    )

    inferred = qlx.estimate(
        selected,
        tier=Tier.ANALYTICAL,
        failure_budget=0.1,
    )
    overridden = qlx.estimate(
        selected,
        tier=Tier.ANALYTICAL,
        p_phys=2.0e-3,
        failure_budget=0.1,
        scaling=qlx.estimate.Scaling(prefactor=0.2, threshold=0.02),
        cycle_time=2.0e-9,
    )

    assert inferred.p_phys == pytest.approx(1.0e-3)
    assert inferred.cycle_time == pytest.approx(1.0e-9)
    assert overridden.p_phys == pytest.approx(2.0e-3)
    assert overridden.cycle_time == pytest.approx(2.0e-9)


def test_physical_error_is_required_without_a_calibrated_device():
    with pytest.raises(TypeError, match="selected device operating point"):
        qlx.estimate(
            memory,
            tier=Tier.ANALYTICAL,
            failure_budget=0.1,
        )


def test_schedule_tier_uses_explicit_p3_intervals_and_resources():
    result = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    assert result.event_count > 4
    assert result.event_counts["acquire"] == 1
    assert result.event_counts["prepare"] == 1
    assert result.event_counts["apply"] == 1
    assert result.event_counts["measure"] == 1
    assert result.makespan_ns == 3.0
    assert result.active_resource_time_ns == 3.0
    assert result.physical_resources == 1
    assert result.peak_concurrency == 1
    assert result.utilization == 1.0
    assert result.makespan_ns == result.expected_makespan_ns
    assert result.makespan_ns == result.maximum_makespan_ns
    assert (result.active_resource_time_ns ==
            result.expected_active_resource_time_ns ==
            result.maximum_active_resource_time_ns)
    assert (result.active_physical_qubit_time_ns ==
            result.expected_active_physical_qubit_time_ns ==
            result.maximum_active_physical_qubit_time_ns)
    assert result.utilization == result.expected_utilization
    assert result.utilization == result.maximum_utilization
    assert result.termination is ScheduleTermination.PROGRAM
    assert all(0.0 <= value <= 1.0 for value in (
        result.utilization,
        result.expected_utilization,
        result.maximum_utilization,
    ))


def test_schedule_estimate_retains_typed_immutable_input_provenance():
    first = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    other = qlx.estimate(
        _selected_schedule_build(selected_schedule_experiment_same_metrics),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )

    assert first.input_root == (
        "selected_schedule_experiment_placed_qec_physical")
    assert first.schedule_symbol == (
        "selected_schedule_experiment_placed_qec_physical_schedule")
    assert first.source_stage is qlx.stages.P3
    assert qlx.stages.PHYSICAL_SCHEDULE in first.source_facets
    assert first.tier is Tier.SCHEDULE
    assert first.device_identity == "ScheduleEstimateDevice"
    assert first.physical_model_identity == (
        "ScheduleEstimateDevicePhysicalMachine")
    assert first.operating_point_identity == (
        "ScheduleEstimateDevice_default_operating_point")
    assert first.lower_tier_identity.endswith("_analytical")
    assert first.termination is ScheduleTermination.PROGRAM
    assert first.assumptions
    with pytest.raises(AttributeError):
        first.input_root = "forged"

    assert first.makespan_ns == other.makespan_ns
    assert first.active_resource_time_ns == other.active_resource_time_ns
    assert first.physical_qubits == other.physical_qubits
    assert first.input_root != other.input_root
    assert first.schedule_symbol != other.schedule_symbol
    assert first != other


def test_schedule_estimate_is_opaque_to_public_construction():
    with pytest.raises(TypeError, match="cannot be constructed directly"):
        ScheduleEstimate()


def _native_payload(result):
    excluded = {
        "input_root",
        "schedule_symbol",
        "source_stage",
        "source_facets",
        "tier",
    }
    payload = {
        name: getattr(result, name)
        for name in result.__dataclass_fields__
        if name not in excluded
    }
    payload["event_counts"] = dict(payload["event_counts"])
    payload["assumptions"] = list(payload["assumptions"])
    payload["termination_semantics"] = payload.pop("termination").value
    return payload


def test_schedule_estimate_full_workload_policy_is_typed_and_native_authenticated(
        monkeypatch):
    from cudaq.logical._native import native

    program = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    payload = _native_payload(program)
    payload["termination_semantics"] = "full_workload"
    observed = []

    def full_workload_native(
        _module,
        _graph,
        _schedule,
        _lower_tier,
        full_workload,
    ):
        observed.append(full_workload)
        return json.dumps(payload)

    monkeypatch.setattr(
        native,
        "_schedule_verified_and_estimate_json",
        full_workload_native,
    )
    result = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        termination=ScheduleTermination.FULL_WORKLOAD,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )

    assert observed == [True]
    assert result.termination is ScheduleTermination.FULL_WORKLOAD
    assert result.exhaustion_probability == program.exhaustion_probability


def test_schedule_estimate_rejects_untyped_or_mismatched_termination_policy(
        monkeypatch):
    with pytest.raises(TypeError, match="ScheduleTermination"):
        qlx.estimate(
            _selected_schedule_build(),
            tier=Tier.SCHEDULE,
            termination="full_workload",
            **_SCHEDULE_ESTIMATE_OPTIONS,
        )

    from cudaq.logical._native import native

    program = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    payload = _native_payload(program)
    monkeypatch.setattr(
        native,
        "_schedule_verified_and_estimate_json",
        lambda *_args: json.dumps(payload),
    )
    with pytest.raises(
            qlx.errors.ScheduleConflict,
            match="termination policy differs",
    ):
        qlx.estimate(
            _selected_schedule_build(),
            tier=Tier.SCHEDULE,
            termination=ScheduleTermination.FULL_WORKLOAD,
            **_SCHEDULE_ESTIMATE_OPTIONS,
        )


def test_schedule_estimate_rejects_malformed_native_evidence(monkeypatch):
    from cudaq.logical._native import native

    authenticated = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    baseline = _native_payload(authenticated)

    def negative_event_count(data):
        data["event_count"] = -1

    def inconsistent_event_count(data):
        data["event_count"] += 1

    def negative_kind_count(data):
        data["event_counts"][next(iter(data["event_counts"]))] = -1

    def nonfinite_metric(data):
        data["makespan_ns"] = math.nan

    def invalid_metric_order(data):
        data["expected_makespan_ns"] = data["maximum_makespan_ns"] + 1.0

    def impossible_peak(data):
        data["peak_concurrency"] = data["physical_resources"] + 1

    def invalid_utilization(data):
        data["utilization"] = 1.01

    def inconsistent_utilization(data):
        data["utilization"] = 0.5

    def invalid_exhaustion(data):
        data["exhaustion_probability"] = -0.01

    mutations = (
        negative_event_count,
        inconsistent_event_count,
        negative_kind_count,
        nonfinite_metric,
        invalid_metric_order,
        impossible_peak,
        invalid_utilization,
        inconsistent_utilization,
        invalid_exhaustion,
    )
    for mutate in mutations:
        malformed = copy.deepcopy(baseline)
        mutate(malformed)
        monkeypatch.setattr(
            native,
            "_schedule_verified_and_estimate_json",
            lambda *_args, value=json.dumps(malformed): value,
        )
        with pytest.raises(
                qlx.errors.ScheduleConflict,
                match="invalid authenticated evidence",
        ):
            qlx.estimate(
                _selected_schedule_build(),
                tier=Tier.SCHEDULE,
                **_SCHEDULE_ESTIMATE_OPTIONS,
            )


def test_schedule_estimate_accepts_abort_conditioned_expected_below_first(
        monkeypatch):
    from cudaq.logical._native import native

    authenticated = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    payload = _native_payload(authenticated)
    payload["expected_makespan_ns"] = payload["makespan_ns"] / 2.0
    payload["expected_active_resource_time_ns"] = (
        payload["active_resource_time_ns"] / 2.0)
    payload["expected_active_physical_qubit_time_ns"] = (
        payload["active_physical_qubit_time_ns"] / 2.0)
    capacity = (payload["expected_makespan_ns"] * payload["physical_resources"])
    payload["expected_utilization"] = (
        0.0 if capacity == 0.0 else
        payload["expected_active_resource_time_ns"] / capacity)
    monkeypatch.setattr(
        native,
        "_schedule_verified_and_estimate_json",
        lambda *_args: json.dumps(payload),
    )

    result = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )

    assert result.expected_makespan_ns < result.makespan_ns
    assert (result.expected_active_resource_time_ns
            < result.active_resource_time_ns)


def test_schedule_estimate_rejects_mismatched_native_provenance(monkeypatch):
    from cudaq.logical._native import native

    authenticated = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    malformed = _native_payload(authenticated)
    malformed["physical_model_identity"] = "forged_machine"
    monkeypatch.setattr(
        native,
        "_schedule_verified_and_estimate_json",
        lambda *_args: json.dumps(malformed),
    )
    with pytest.raises(
            qlx.errors.ScheduleConflict,
            match="model provenance differs",
    ):
        qlx.estimate(
            _selected_schedule_build(),
            tier=Tier.SCHEDULE,
            **_SCHEDULE_ESTIMATE_OPTIONS,
        )


def test_native_schedule_estimate_diagnostics_are_context_scoped():
    from cudaq.logical._native import native

    published = qlx.compiler.schedule(scheduled_experiment)
    module = published.build._module

    def missing(symbol, barrier):
        barrier.wait(timeout=5.0)
        try:
            native.estimate_schedule_json(module, symbol)
        except RuntimeError as error:
            return str(error)
        raise AssertionError("missing schedule symbol unexpectedly resolved")

    symbols = ("missing_schedule_left", "missing_schedule_right")
    with ThreadPoolExecutor(max_workers=2) as executor:
        for _ in range(32):
            barrier = threading.Barrier(2, timeout=5.0)
            futures = tuple(
                executor.submit(missing, symbol, barrier) for symbol in symbols)
            messages = tuple(future.result(timeout=10.0) for future in futures)
            for index, symbol in enumerate(symbols):
                assert f"@{symbol}" in messages[index]
                assert f"@{symbols[1 - index]}" not in messages[index]


def test_end_to_end_schedule_estimate_consumes_native_rows_without_reparse(
    monkeypatch,):
    from cudaq.logical._native import native
    from cudaq.logical.compiler import Build

    def reject_standalone_parse(*_args, **_kwargs):
        raise AssertionError("end-to-end estimate reparsed phys.schedule rows")

    monkeypatch.setattr(
        native,
        "estimate_schedule_json",
        reject_standalone_parse,
    )
    monkeypatch.setattr(
        Build,
        "_parse_schedule_entry",
        staticmethod(reject_standalone_parse),
    )
    fused = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    assert fused.event_count > 4
    assert fused.makespan_ns > 0.0


def test_public_schedule_remains_available_beside_fused_native_estimation():
    estimate = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    published = qlx.compiler.schedule(_selected_schedule_build())
    assert estimate.event_count == len(published.entries)


def test_fused_and_published_schedule_estimates_are_identical():
    fused = qlx.estimate(
        _selected_schedule_build(),
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    published = qlx.compiler.schedule(_selected_schedule_build())
    standalone = qlx.estimate(
        published,
        tier=Tier.SCHEDULE,
        **_SCHEDULE_ESTIMATE_OPTIONS,
    )
    assert fused == standalone


@pytest.mark.parametrize(
    "name,value",
    (
        ("device", object()),
        ("strategy", qlx.compiler.scheduling.greedy_asap),
        ("objective", "makespan"),
    ),
)
def test_standalone_schedule_rejects_compilation_overrides(name, value):
    published = qlx.compiler.schedule(_selected_schedule_build())
    with pytest.raises(qlx.errors.ScheduleConflict,
                       match=rf"{name}= cannot override.*PhysicalSchedule"):
        qlx.estimate(
            published,
            tier=Tier.SCHEDULE,
            **_SCHEDULE_ESTIMATE_OPTIONS,
            **{name: value},
        )


def test_standalone_schedule_estimate_rejects_tampered_portable_claims():
    published = qlx.compiler.schedule(_selected_schedule_build())
    module = published.build._module
    schedule_op = next(operation for operation in module.body.operations
                       if operation.name == "phys.schedule")
    with module.context, mlir_ir.Location.unknown():
        schedule_op.attributes["provider"] = mlir_ir.StringAttr.get(
            "tampered", context=module.context)

    with pytest.raises(
            qlx.errors.ScheduleConflict,
            match="native Tier-3 schedule estimation failed",
    ):
        qlx.estimate(
            published,
            tier=Tier.SCHEDULE,
            **_SCHEDULE_ESTIMATE_OPTIONS,
        )


def test_schedule_tier_rejects_detached_zero_schedule_forgery():
    schedule = qlx.compiler.schedule(scheduled_experiment)
    forged = copy.copy(schedule)
    object.__setattr__(forged, "entries", ())
    object.__setattr__(forged, "makespan_ns", 0.0)

    with pytest.raises(qlx.errors.ScheduleConflict, match="no longer matches"):
        qlx.analysis.estimate(
            forged,
            tier=Tier.SCHEDULE,
            p_phys=1.0e-3,
            failure_budget=0.1,
        )


def test_schedule_tier_rejects_p2_instead_of_reconstructing_tasks():
    selected = qlx.compile(four_h)

    assert selected.stage == qlx.stages.P2
    with pytest.raises(ValueError,
                       match="never reconstructs.*directly from P2"):
        qlx.estimate(
            selected,
            tier=Tier.SCHEDULE,
            p_phys=1.0e-3,
            failure_budget=0.1,
        )


def test_estimation_tiers_are_reproducible_from_replayed_builds():

    @qlx.program
    def portable() -> bool:
        return qlx.measure_z(qlx.prepare_zero())

    logical = qlx.compile(portable)
    logical_replay = qlx.compiler.Build.replay(logical.serialize())
    assert qlx.estimate(logical,
                        tier=Tier.LOGICAL) == qlx.estimate(logical_replay,
                                                           tier=Tier.LOGICAL)

    selected = qlx.compile(four_h)
    selected_replay = qlx.compiler.Build.replay(selected.serialize())
    assert qlx.estimate(selected,
                        tier=Tier.STATIC) == qlx.estimate(selected_replay,
                                                          tier=Tier.STATIC)
    analytical_options = {
        "tier": Tier.ANALYTICAL,
        "p_phys": 1e-3,
        "failure_budget": 0.1,
        "cycle_time": 2.0,
    }
    assert qlx.estimate(selected, **analytical_options) == qlx.estimate(
        selected_replay, **analytical_options)

    physical = qlx.compile(scheduled_experiment)
    physical_replay = qlx.compiler.Build.replay(physical.serialize())
    for candidate in (physical, physical_replay):
        with pytest.raises(
                qlx.analysis.MissingEvidence,
                match="retaining its selected P2 source protocol",
        ):
            qlx.estimate(
                candidate,
                tier=Tier.SCHEDULE,
                p_phys=1.0e-3,
                failure_budget=0.1,
            )
