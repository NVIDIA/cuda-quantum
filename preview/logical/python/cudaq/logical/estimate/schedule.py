# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Native Tier-3 schedule estimation over an immutable P3 artifact."""

from __future__ import annotations

import json
import math
import os
import time

from ..compiler import PhysicalSchedule, schedule
from ..errors import MissingEvidence, ScheduleConflict
from ._parameters import resolve_physical_parameters
from .types import (
    ScheduleEstimate,
    EvidencePolicy,
    FailureBudget,
    ScheduleTermination,
    Tier,
    _SCHEDULE_ESTIMATE_CREATION_TOKEN,
)


def _text(attribute) -> str:
    value = getattr(attribute, "value", None)
    return str(value if value is not None else attribute).strip('"').lstrip("@")


def _symbol(operation) -> str | None:
    try:
        return _text(operation.attributes["sym_name"])
    except KeyError:
        return None


def _walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _unique_result_symbol(module, base: str) -> str:
    symbols = {
        symbol for operation in module.body.operations
        if (symbol := _symbol(operation.operation)) is not None
    }
    result = base
    while result in symbols:
        result += "_"
    return result


def _lower_tier_preparer(*, p_phys, failure_budget, scaling, cycle_time,
                         evidence_policy):

    def prepare(module, graph_symbol: str, schedule_symbol: str) -> str:
        profile = os.getenv("QLX_PROFILE_SCHEDULE_ESTIMATE") is not None
        preparation_started = time.perf_counter()
        symbols = {
            symbol: operation.operation
            for operation in module.body.operations
            if (symbol := _symbol(operation.operation)) is not None
        }
        graph = symbols.get(graph_symbol)
        if graph is None or graph.name != "phys.graph":
            raise MissingEvidence(
                "Tier-3 estimation requires the scheduled P3 graph closure")
        unresolved_supply = tuple(
            operation for operation in _walk(graph)
            if operation.name == "phys.resource_request" and
            ("external" in operation.attributes or "provider" not in operation.
             attributes or "physical_binding" not in operation.attributes))
        if unresolved_supply:
            raise MissingEvidence(
                "unresolved external resource requests prevent Tier-3 "
                "estimation")
        source_attribute = graph.attributes.get("source_protocol")
        if source_attribute is None:
            raise MissingEvidence(
                "Tier-3 estimation requires a P3 graph retaining its selected "
                "P2 source protocol")
        source_protocol = _text(source_attribute)
        source = symbols.get(source_protocol)
        if source is None or source.name not in {
                "fabric.gadget", "fabric.protocol"
        }:
            raise MissingEvidence(
                "Tier-3 estimation source_protocol must resolve to selected P2")

        architecture = graph.attributes.get("architecture")
        operating_point = graph.attributes.get("operating_point")
        devices = [
            operation for operation in symbols.values()
            if operation.name == "qlx.device" and
            operation.attributes.get("physical") == architecture and
            operation.attributes.get("operating_point") == operating_point
        ]
        if len(devices) != 1:
            raise MissingEvidence(
                "Tier-3 estimation requires one selected device whose physical "
                "machine and operating point match the scheduled P3 graph")
        device_symbol = _symbol(devices[0])
        if device_symbol is None:
            raise MissingEvidence(
                "Tier-3 estimation selected device has no stable symbol")

        static_symbol = _unique_result_symbol(module,
                                              f"{schedule_symbol}_static")
        analytical_symbol = _unique_result_symbol(
            module, f"{schedule_symbol}_analytical")
        from cudaq.logical._native import native

        native_started = time.perf_counter()
        native._materialize_verified_analytical_lower_tier(
            module,
            source_protocol,
            device_symbol,
            static_symbol,
            analytical_symbol,
            p_phys,
            failure_budget.total,
            cycle_time,
            scaling.prefactor,
            scaling.threshold,
            evidence_policy.require_established,
        )
        if profile:
            print(
                "phys-estimate-schedule lower-tier native closure "
                f"{time.perf_counter() - native_started:.6f}s",
                flush=True,
            )
            print(
                "phys-estimate-schedule lower-tier total "
                f"{time.perf_counter() - preparation_started:.6f}s",
                flush=True,
            )
        return analytical_symbol

    return prepare


def scheduled(
    root,
    *,
    p_phys=None,
    failure_budget,
    scaling=None,
    cycle_time=None,
    evidence_policy=None,
    device=None,
    strategy=None,
    objective=None,
    termination=ScheduleTermination.PROGRAM,
):
    """Return one native Tier-3 estimate over a verified P3 schedule.

    Tier 3 always refines an exact native Tier-2 analytical result from the
    same retained P2 program and selected device. For an unscheduled input,
    native scheduling independently verifies and estimates transient typed
    claims without publishing portable schedule rows. An explicitly supplied
    :class:`PhysicalSchedule` remains the immutable authority for standalone
    estimation. Canonical textual MLIR is neither produced nor reparsed.

    Physical parameters default to the selected device's operating point.
    Explicit values override those defaults for sensitivity studies.
    """

    if isinstance(root, PhysicalSchedule):
        overrides = tuple(name for name, value in (
            ("device", device),
            ("strategy", strategy),
            ("objective", objective),
        ) if value is not None)
        if overrides:
            rendered = ", ".join(f"{name}=" for name in overrides)
            raise ScheduleConflict(
                f"{rendered} cannot override an existing immutable "
                "PhysicalSchedule")

    p_phys, scaling, cycle_time = resolve_physical_parameters(
        root,
        device=device,
        p_phys=p_phys,
        scaling=scaling,
        cycle_time=cycle_time,
    )
    if not isinstance(failure_budget, FailureBudget):
        failure_budget = FailureBudget(float(failure_budget))
    if not isinstance(termination, ScheduleTermination):
        raise TypeError("termination must be a ScheduleTermination value")
    evidence_policy = evidence_policy or EvidencePolicy()
    p_phys = float(p_phys)
    cycle_time = float(cycle_time)
    if not math.isfinite(p_phys) or not 0.0 <= p_phys <= 1.0:
        raise ValueError("p_phys must lie in [0, 1]")
    if not math.isfinite(cycle_time) or cycle_time <= 0.0:
        raise ValueError("cycle_time must be finite and positive")
    prepare_lower_tier = _lower_tier_preparer(
        p_phys=p_phys,
        failure_budget=failure_budget,
        scaling=scaling,
        cycle_time=cycle_time,
        evidence_policy=evidence_policy,
    )

    if isinstance(root, PhysicalSchedule):
        physical_schedule = root.canonical()
    else:
        kwargs = {
            "device":
                device,
            "_estimate":
                prepare_lower_tier,
            "_estimate_full_workload":
                termination is ScheduleTermination.FULL_WORKLOAD,
        }
        if strategy is not None:
            kwargs["strategy"] = strategy
        if objective is not None:
            kwargs["objective"] = objective
        physical_schedule = schedule(root, **kwargs)

    profile = os.getenv("QLX_PROFILE_SCHEDULE_ESTIMATE") is not None
    from cudaq.logical._native import native

    started = time.perf_counter()
    expected_lower_tier = getattr(physical_schedule, "_lower_tier_symbol", None)
    try:
        payload = getattr(physical_schedule, "payload", None)
        if payload is None:
            module = physical_schedule.build._fresh_module()
            lower_tier = prepare_lower_tier(
                module,
                physical_schedule._graph_symbol,
                physical_schedule._schedule_symbol,
            )
            expected_lower_tier = lower_tier
            payload = native.estimate_schedule_json(
                module,
                physical_schedule._schedule_symbol,
                lower_tier,
                termination is ScheduleTermination.FULL_WORKLOAD,
            )
        data = json.loads(payload)
    except MissingEvidence:
        raise
    except (RuntimeError, ValueError, json.JSONDecodeError) as error:
        message = str(error)
        if "missing evidence:" in message:
            detail = message.partition("missing evidence:")[2].splitlines()[0]
            raise MissingEvidence(detail.strip()) from error
        raise ScheduleConflict(
            "native Tier-3 schedule estimation failed for "
            f"@{physical_schedule._schedule_symbol}") from error
    if profile:
        print(
            "phys-estimate-schedule python-native-pass "
            f"{time.perf_counter() - started:.6f}s",
            flush=True,
        )

    if not isinstance(data, dict):
        raise ScheduleConflict(
            "native Tier-3 schedule estimation returned a non-object payload")
    try:
        native_termination = ScheduleTermination(data["termination_semantics"])
        physical_model_identity = data["physical_model_identity"]
        operating_point_identity = data.get("operating_point_identity")
        lower_tier_identity = data["lower_tier_identity"]
    except (KeyError, TypeError) as error:
        raise ScheduleConflict(
            "native Tier-3 schedule estimation returned incomplete provenance"
        ) from error
    except ValueError as error:
        raise ScheduleConflict(
            "native Tier-3 schedule estimation returned an unknown "
            "termination policy") from error
    if native_termination is not termination:
        raise ScheduleConflict(
            "native Tier-3 estimate termination policy differs from the "
            "requested policy")
    if (physical_model_identity != physical_schedule._machine_symbol or
            operating_point_identity
            != physical_schedule._operating_point_symbol):
        raise ScheduleConflict(
            "native Tier-3 estimate model provenance differs from its schedule")
    if lower_tier_identity != expected_lower_tier:
        raise ScheduleConflict(
            "native Tier-3 estimate lower-tier provenance differs from its "
            "analytical input")
    try:
        return ScheduleEstimate._create(
            event_count=data["event_count"],
            event_counts=data["event_counts"],
            makespan_ns=data["makespan_ns"],
            expected_makespan_ns=data["expected_makespan_ns"],
            maximum_makespan_ns=data["maximum_makespan_ns"],
            active_resource_time_ns=data["active_resource_time_ns"],
            expected_active_resource_time_ns=data[
                "expected_active_resource_time_ns"],
            maximum_active_resource_time_ns=data[
                "maximum_active_resource_time_ns"],
            active_physical_qubit_time_ns=data["active_physical_qubit_time_ns"],
            expected_active_physical_qubit_time_ns=data[
                "expected_active_physical_qubit_time_ns"],
            maximum_active_physical_qubit_time_ns=data[
                "maximum_active_physical_qubit_time_ns"],
            physical_resources=data["physical_resources"],
            physical_qubits=data["physical_qubits"],
            peak_concurrency=data["peak_concurrency"],
            peak_active_physical_qubits=data["peak_active_physical_qubits"],
            utilization=data["utilization"],
            expected_utilization=data["expected_utilization"],
            maximum_utilization=data["maximum_utilization"],
            exhaustion_probability=data["exhaustion_probability"],
            termination=native_termination,
            bottleneck=data["bottleneck"],
            assumptions=data["assumptions"],
            input_root=physical_schedule._requested_root_symbol,
            schedule_symbol=physical_schedule._schedule_symbol,
            source_stage=physical_schedule.stage,
            source_facets=physical_schedule.facets,
            tier=Tier.SCHEDULE,
            device_identity=data.get("device_identity"),
            physical_model_identity=physical_model_identity,
            operating_point_identity=operating_point_identity,
            lower_tier_identity=lower_tier_identity,
            _token=_SCHEDULE_ESTIMATE_CREATION_TOKEN,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ScheduleConflict(
            "native Tier-3 schedule estimation returned invalid authenticated "
            "evidence") from error
