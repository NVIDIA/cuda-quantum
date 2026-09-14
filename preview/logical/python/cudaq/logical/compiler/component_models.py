# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Characterize compact synchronous and transport P3 models."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math

from cudaq.logical.analysis.evidence import computation
from cudaq.logical.devices.component_models import (
    InitiationIntervalSemantics,
    PhysicalResourceClaim,
    SpacetimePhase,
    SpacetimePlanModel,
    TransportModel,
    _spacetime_characterization_from_verified_schedule,
    _transport_characterization_from_verified_schedule,
)
from cudaq.logical.devices.definition import QECChannelRealization
from cudaq.logical.protocols.definition import ProtocolDefinition

from .build import _attr_text, _walk_operation
from .component_identity import (
    channel_binding_sha256,
    channel_identity_sha256,
    logical_channel_sha256,
    physical_architecture_sha256,
    protocol_contract,
    selected_protocol_closure_sha256,
    selected_protocol_sha256,
    spacetime_model_sha256,
    timing_profile,
    transport_model_sha256,
)
from .factory import _integer, _number, _source_code_distances, _symbol_leaf
from .schedule import PhysicalSchedule


def _schedule_sha256(schedule: PhysicalSchedule, build_sha256: str) -> str:
    payload = {
        "schema": "qlx.component-model-characterization.schedule/v1",
        "build_sha256": build_sha256,
        "entries": [asdict(entry) for entry in schedule.entries],
        "strategy": schedule.strategy.name,
        "strategy_domain": schedule.strategy_domain,
        "provider": schedule.provider,
        "provider_version": schedule.provider_version,
        "constraint_profile": schedule.constraint_profile,
        "constraints": schedule.constraints,
        "timing_profile": schedule.timing_profile,
        "tie_break": schedule.tie_break,
        "makespan_ns": schedule.makespan_ns,
        "requested_root": schedule._requested_root_symbol,
        "graph": schedule._graph_symbol,
        "machine": schedule._machine_symbol,
        "operating_point": schedule._operating_point_symbol,
    }
    encoded = json.dumps(payload, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return sha256(encoded).hexdigest()


def _find_symbol(module, name: str, kind: str):
    matches = [
        operation.operation
        for operation in module.body.operations
        if operation.operation.name == kind and
        _attr_text(operation.operation.attributes["sym_name"]) == name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"component characterization requires exactly one {kind} @{name}")
    return matches[0]


def _cycle_ns(device) -> float:
    if device is None or device.operating_point is None:
        raise ValueError(
            "component characterization requires the exact source Device; "
            "pass device= when characterizing a replayed schedule")
    timing = device.operating_point.timing
    raw = timing.get("surface_cycle_ns", timing.get("cycle_ns"))
    try:
        value = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "component characterization requires a finite positive cycle"
        ) from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            "component characterization requires a finite positive cycle")
    return value


def _source_device(schedule, device):
    retained = getattr(schedule.build, "_device", None)
    if device is None:
        device = retained
    elif retained is not None and device is not retained and device != retained:
        raise ValueError(
            "component characterization device differs from the scheduled source"
        )
    if device is None or device.physical is None:
        raise ValueError(
            "component characterization requires the exact physical Device")
    return device


def _resource_by_name(device):
    return {
        resource.name: resource for resource in device.physical.resource_classes
    }


def _factory_by_source(device):
    result = {}
    for binding in device.qec_to_physical:
        model = binding.factory_model
        if model is None:
            continue
        source = (model.characterization.source_provider
                  if model.characterization is not None else
                  binding.qec_region.name)
        result.setdefault(source, []).append(model)
    return result


def _plan_for_protocol(module, graph, protocol: ProtocolDefinition):
    plans = {
        _attr_text(candidate.attributes["sym_name"]): candidate
        for candidate in (view.operation for view in module.body.operations)
        if candidate.name == "phys.spacetime_plan" and
        "recurrence_resource_kind" not in candidate.attributes
    }
    selected = set()
    for operation in _walk_operation(graph):
        if operation.name != "phys.spacetime_call":
            continue
        actual = operation.attributes.get("source_protocol")
        if actual is not None and _attr_text(actual) != protocol.name:
            continue
        plan = plans.get(_attr_text(operation.attributes["plan"]))
        if plan is None:
            raise ValueError(
                "spacetime characterization call has an unresolved plan")
        if actual is None and _attr_text(
                plan.attributes["source_protocol"]) != protocol.name:
            continue
        selected.add(_attr_text(plan.attributes["sym_name"]))
    if not selected:
        for name, plan in plans.items():
            if _attr_text(plan.attributes["source_protocol"]) == protocol.name:
                selected.add(name)
    if len(selected) != 1:
        raise ValueError(
            "spacetime characterization requires exactly one invoked compact "
            f"plan for protocol @{protocol.name}, found {len(selected)}")
    return plans[next(iter(selected))]


def _phase_models(module, device, phase):
    references = tuple(
        _attr_text(value) for value in phase.attributes["factory_models"])
    if not references:
        return ()
    factory_ops = {
        _attr_text(candidate.attributes["sym_name"]): candidate
        for candidate in (view.operation for view in module.body.operations)
        if candidate.name == "phys.factory_model"
    }
    by_source = _factory_by_source(device)
    result = []
    for reference in references:
        operation = factory_ops.get(reference)
        if operation is None:
            raise ValueError(
                f"spacetime phase references unresolved factory model @{reference}"
            )
        source = (_attr_text(operation.attributes["source_provider"])
                  if "source_provider" in operation.attributes else
                  _symbol_leaf(operation.attributes["region"]))
        matches = tuple(by_source.get(source, ()))
        if len(matches) != 1:
            raise ValueError(
                "spacetime characterization cannot rebind an ambiguous "
                f"factory dependency {source!r}")
        result.append(matches[0])
    return tuple(result)


def spacetime_plan_model(
    schedule: PhysicalSchedule,
    *,
    protocol: ProtocolDefinition,
    device=None,
) -> SpacetimePlanModel:
    """Derive a compact synchronous model from one verified spacetime plan."""

    if not isinstance(schedule, PhysicalSchedule):
        raise TypeError(
            "cudaq.logical.compiler.spacetime_plan_model requires a "
            "PhysicalSchedule")
    if not isinstance(protocol, ProtocolDefinition):
        raise TypeError(
            "spacetime_plan_model protocol= requires a ProtocolDefinition")
    schedule = schedule.canonical()
    if schedule.profile != "p3" or schedule._operating_point_symbol is None:
        raise ValueError(
            "spacetime characterization requires a scheduled P3 graph with "
            "an operating point")
    device = _source_device(schedule, device)
    cycle_ns = _cycle_ns(device)
    module = schedule.build._fresh_module()
    graph = _find_symbol(module, schedule._graph_symbol, "phys.graph")
    plan = _plan_for_protocol(module, graph, protocol)
    selected_protocol = _find_symbol(module, protocol.name, "fabric.protocol")
    if (_attr_text(plan.attributes["architecture"]) != schedule._machine_symbol
            or _attr_text(plan.attributes["operating_point"])
            != schedule._operating_point_symbol):
        raise ValueError(
            "spacetime plan differs from the scheduled architecture or point")
    latency_ns = _number(plan.attributes.get("forwarding_latency_ns"))
    interval_ns = _number(plan.attributes.get("initiation_interval_ns"))
    if any(not math.isfinite(value) or value <= 0.0
           for value in (latency_ns, interval_ns)):
        raise ValueError("spacetime plan has invalid latency or interval")
    resources = _resource_by_name(device)
    phases = []
    for operation in _walk_operation(plan):
        if operation.name != "phys.spacetime_phase":
            continue
        claims = []
        resource_claims = operation.attributes.get("resource_claims")
        if resource_claims is not None:
            for raw in resource_claims:
                reference = raw["resource_class"]
                name = _symbol_leaf(reference)
                resource = resources.get(name)
                if resource is None:
                    raise ValueError(
                        f"spacetime plan claims foreign resource class @{name}")
                claims.append(
                    PhysicalResourceClaim(
                        resource,
                        offset=_integer(raw["offset"]),
                        count=_integer(raw["count"]),
                    ))
        else:
            for reference in operation.attributes["resource_classes"]:
                name = _symbol_leaf(reference)
                resource = resources.get(name)
                if resource is None:
                    raise ValueError(
                        f"spacetime plan claims foreign resource class @{name}")
                claims.append(PhysicalResourceClaim(resource))
        phases.append(
            SpacetimePhase(
                _attr_text(operation.attributes["sym_name"]),
                steps=_integer(operation.attributes["steps"]),
                step_duration_cycles=(
                    _number(operation.attributes["step_duration_ns"]) /
                    cycle_ns),
                resources=tuple(claims),
                factories=_phase_models(module, device, operation),
                after=tuple(
                    _attr_text(value)
                    for value in operation.attributes["after"]),
            ))
    if not phases:
        raise ValueError("spacetime characterization found no callable phases")
    if "policy" in plan.attributes:
        policy = _attr_text(plan.attributes["policy"])
    elif (_attr_text(plan.attributes["provider"]) == "qlx.compiler.spacetime"):
        # The registered built-in callable derivations are deterministic and
        # their provider verifier authenticates the exact source plan.  This is
        # component-local provider evidence, unlike scanning unrelated graph
        # selections.
        policy = "guaranteed"
    else:
        raise ValueError(
            "spacetime characterization requires component-local policy evidence"
        )
    if policy not in {"guaranteed", "single_shot"}:
        raise ValueError("spacetime plan has unsupported policy evidence")
    semantics = (InitiationIntervalSemantics.BACKPRESSURED if interval_ns
                 > latency_ns else InitiationIntervalSemantics.PIPELINED)
    distances = _source_code_distances(module, plan)
    placeholder = SpacetimePlanModel(
        protocol=protocol,
        latency_cycles=latency_ns / cycle_ns,
        initiation_interval_cycles=interval_ns / cycle_ns,
        phases=tuple(phases),
        code_distances=distances,
        evidence=computation("qlx.spacetime-plan-characterization/pending"),
        interval_semantics=semantics,
        policy=policy,
    )
    build_sha256 = schedule.build.content_sha256.removeprefix("sha256:")
    schedule_sha256 = _schedule_sha256(schedule, build_sha256)
    model_sha256 = spacetime_model_sha256(placeholder)
    contract = protocol_contract(protocol)
    characterization = _spacetime_characterization_from_verified_schedule(
        source_protocol=protocol.name,
        source_protocol_sha256=contract["source_sha256"],
        selected_protocol_sha256=selected_protocol_closure_sha256(
            selected_protocol),
        objective_sha256=contract["objective_sha256"],
        boundary_sha256=contract["boundary_sha256"],
        architecture=schedule._machine_symbol,
        architecture_sha256=physical_architecture_sha256(device.physical),
        operating_point=device.operating_point.name,
        timing_source=device.operating_point.timing_source,
        timing_profile=timing_profile(device.operating_point),
        code_distances=distances,
        latency_cycles=placeholder.latency_cycles,
        initiation_interval_cycles=placeholder.initiation_interval_cycles,
        provider=_attr_text(plan.attributes["provider"]),
        provider_version=_attr_text(plan.attributes["provider_version"]),
        derivation=_attr_text(plan.attributes["derivation"]),
        derivation_version=_integer(plan.attributes["derivation_version"]),
        model_sha256=model_sha256,
        build_sha256=build_sha256,
        schedule_sha256=schedule_sha256,
    )
    return SpacetimePlanModel(
        protocol=protocol,
        latency_cycles=placeholder.latency_cycles,
        initiation_interval_cycles=placeholder.initiation_interval_cycles,
        phases=placeholder.phases,
        code_distances=distances,
        evidence=computation(
            f"qlx.spacetime-plan-characterization/v1:{build_sha256}:"
            f"{schedule_sha256}:{model_sha256}"),
        interval_semantics=semantics,
        policy=policy,
        characterization=characterization,
    )


def transport_model(
    schedule: PhysicalSchedule,
    *,
    channel: QECChannelRealization,
    device=None,
) -> TransportModel:
    """Derive a compact channel model from verified detailed P3 transfers."""

    if not isinstance(schedule, PhysicalSchedule):
        raise TypeError("cudaq.logical.compiler.transport_model requires a "
                        "PhysicalSchedule")
    if not isinstance(channel, QECChannelRealization):
        raise TypeError(
            "transport_model channel= requires a QECChannelRealization")
    if channel.protocol is None:
        raise ValueError(
            "transport characterization requires a selected typed protocol")
    schedule = schedule.canonical()
    if schedule.profile != "p3" or schedule._operating_point_symbol is None:
        raise ValueError(
            "transport characterization requires a scheduled P3 graph with "
            "an operating point")
    device = _source_device(schedule, device)
    matches = tuple(binding for binding in device.qec_channels_to_physical
                    if binding.qec_channel is channel)
    if len(matches) != 1:
        raise ValueError(
            "transport characterization channel requires one exact P3 binding")
    binding = matches[0]
    module = schedule.build._fresh_module()
    graph = _find_symbol(module, schedule._graph_symbol, "phys.graph")
    route = f"{channel.name}_physical"
    operations = []
    for operation in _walk_operation(graph):
        if operation.name != "phys.transport_resource":
            continue
        selected = operation.attributes.get("route")
        if selected is None or _symbol_leaf(selected) != route:
            continue
        if (_attr_text(operation.attributes["source"])
                != (channel.logical_channel.source.region.name
                    if hasattr(channel.logical_channel.source, "region") and
                    channel.logical_channel.source.region is not None else
                    channel.logical_channel.source.name) or
                _attr_text(
                    operation.attributes["protocol"]) != channel.protocol.name):
            raise ValueError(
                "transport graph endpoint or protocol differs from the "
                "selected channel")
        operations.append(operation)
    if not operations:
        raise ValueError(
            f"transport characterization found no transfer on @{channel.name}")
    selected_protocol = _find_symbol(module, channel.protocol.name,
                                     "fabric.protocol")
    entries = {entry.event_id: entry for entry in schedule.entries}
    transfer_entries = []
    for operation in operations:
        event = _attr_text(operation.attributes["event_id"])
        entry = entries.get(event)
        if entry is None or entry.kind != "transport_resource":
            raise ValueError(
                f"transport event {event!r} is absent from the verified schedule"
            )
        transfer_entries.append(entry)
    if not binding.transport_claims or binding.endpoint_occupancy is None:
        raise ValueError(
            "transport characterization requires detailed P3 route claims and "
            "endpoint occupancy on its physical channel binding")
    point_timing = device.operating_point.timing
    try:
        latency_ns = float(point_timing["transport_resource_ns"])
        interval_ns = float(
            point_timing["transport_resource_initiation_interval_ns"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "transport characterization requires calibrated "
            "transport_resource_ns and "
            "transport_resource_initiation_interval_ns timing facts") from error
    if any(not math.isfinite(value) or value <= 0.0
           for value in (latency_ns, interval_ns)):
        raise ValueError(
            "transport characterization timing facts must be finite and positive"
        )
    durations = {entry.duration_ns for entry in transfer_entries}
    if durations != {latency_ns}:
        raise ValueError(
            "transport schedule does not retain its calibrated P3 latency")
    cycle_ns = _cycle_ns(device)
    claims = binding.transport_claims
    occupancy = binding.endpoint_occupancy
    # This model describes the selected route operation itself. Unrelated
    # factory or callable selections elsewhere in the representative graph do
    # not change the channel's deterministic reservation contract.
    policy = "guaranteed"

    def selected_members(claim, occurrence):
        lanes = claim.count // claim.units
        if lanes <= 0:
            raise ValueError(
                "detailed transport claim has no complete acquisition lane")
        begin = claim.offset + occurrence % lanes * claim.units
        return {
            f"{claim.resource_class.name}[{index}]"
            for index in range(begin, begin + claim.units)
        }

    def selected_endpoint(name, capacity, units, occurrence):
        lanes = capacity // units
        if lanes <= 0:
            raise ValueError(
                "detailed transport endpoint occupancy exceeds its port capacity"
            )
        begin = occurrence % lanes * units
        return {
            f"control:transport-port:{name}:{index}"
            for index in range(begin, begin + units)
        }

    for occurrence, entry in enumerate(transfer_entries):
        expected = set()
        for claim in claims:
            expected.update(selected_members(claim, occurrence))
        expected.update(
            selected_endpoint(channel.source.name, channel.source.concurrency,
                              occupancy[0], occurrence))
        expected.update(
            selected_endpoint(channel.destination.name,
                              channel.destination.concurrency, occupancy[1],
                              occurrence))
        expected.add(f"control:transport-channel:{channel.name}:"
                     f"{occurrence % channel.concurrency}")
        expected.add(f"control:transport-init:{channel.name}_physical")
        if set(entry.resources) != expected:
            raise ValueError(
                "transport schedule resource reservations differ from the "
                "detailed P3 route claims")
    semantics = (InitiationIntervalSemantics.BACKPRESSURED if interval_ns
                 > latency_ns else InitiationIntervalSemantics.PIPELINED)
    placeholder = TransportModel(
        latency_cycles=latency_ns / cycle_ns,
        initiation_interval_cycles=interval_ns / cycle_ns,
        resources=claims,
        endpoint_occupancy=occupancy,
        evidence=computation("qlx.transport-characterization/pending"),
        interval_semantics=semantics,
        policy=policy,
    )
    build_sha256 = schedule.build.content_sha256.removeprefix("sha256:")
    schedule_sha256 = _schedule_sha256(schedule, build_sha256)
    model_sha256 = transport_model_sha256(placeholder)
    characterization = _transport_characterization_from_verified_schedule(
        channel=channel.logical_channel.name,
        channel_sha256=logical_channel_sha256(channel),
        realization_sha256=channel_identity_sha256(channel),
        protocol_sha256=protocol_contract(channel.protocol)["source_sha256"],
        selected_protocol_sha256=selected_protocol_sha256(selected_protocol),
        binding_sha256=channel_binding_sha256(binding),
        architecture=schedule._machine_symbol,
        architecture_sha256=physical_architecture_sha256(device.physical),
        operating_point=device.operating_point.name,
        timing_source=device.operating_point.timing_source,
        timing_profile=timing_profile(device.operating_point),
        latency_cycles=placeholder.latency_cycles,
        initiation_interval_cycles=placeholder.initiation_interval_cycles,
        model_sha256=model_sha256,
        provider="qlx.compiler.transport",
        provider_version="1",
        build_sha256=build_sha256,
        schedule_sha256=schedule_sha256,
        transfer_events=tuple(entry.event_id for entry in transfer_entries),
    )
    return TransportModel(
        latency_cycles=placeholder.latency_cycles,
        initiation_interval_cycles=placeholder.initiation_interval_cycles,
        resources=claims,
        endpoint_occupancy=placeholder.endpoint_occupancy,
        evidence=computation(
            f"qlx.transport-characterization/v1:{build_sha256}:"
            f"{schedule_sha256}:{model_sha256}"),
        interval_semantics=semantics,
        policy=policy,
        characterization=characterization,
    )


__all__ = ["spacetime_plan_model", "transport_model"]
