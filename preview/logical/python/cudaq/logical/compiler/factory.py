# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Characterize compact factory models from verified physical schedules."""

from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
import math

from cudaq.logical.analysis.evidence import computation
from cudaq.logical.devices.definition import (
    FactoryModel,
    _factory_characterization_from_verified_schedule,
)
from cudaq.logical.std import ResourceKind

from .build import _attr_text, _walk_operation
from .schedule import PhysicalSchedule


def _number(attribute) -> float:
    value = getattr(attribute, "value", attribute)
    return float(str(value).strip('"'))


def _integer(attribute) -> int:
    """Read an integer attribute without a lossy float intermediary."""

    value = getattr(attribute, "value", attribute)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return int(str(value).strip('"'))


def _symbol_leaf(attribute) -> str:
    """Return the leaf name of either a flat or nested symbol reference."""

    value = getattr(attribute, "value", None)
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("factory evidence contains an empty symbol path")
        return str(value[-1])
    return _attr_text(attribute)


def _schedule_sha256(schedule: PhysicalSchedule, build_sha256: str) -> str:
    payload = {
        "schema": "qlx.factory-characterization.schedule/v1",
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
        "optimization_status": schedule.optimization_status,
        "objective_value": schedule.objective_value,
        "makespan_ns": schedule.makespan_ns,
        "requested_root": schedule._requested_root_symbol,
        "graph": schedule._graph_symbol,
        "machine": schedule._machine_symbol,
        "operating_point": schedule._operating_point_symbol,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
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
            f"factory characterization requires exactly one {kind} @{name}")
    return matches[0]


def _source_code_distances(module, graph) -> tuple[int, ...]:
    """Return code distances reached by the selected factory definition."""

    symbols = {
        _attr_text(operation.operation.attributes["sym_name"]):
            operation.operation
        for operation in module.body.operations
        if "sym_name" in operation.operation.attributes
    }
    source = graph.attributes.get("source_protocol")
    if source is None:
        raise ValueError(
            "factory characterization P3 graph lacks its source protocol")
    pending = [_attr_text(source)]
    visited = set()
    code_names = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        definition = symbols.get(name)
        if definition is None or definition.name not in {
                "fabric.gadget", "fabric.protocol"
        }:
            raise ValueError(
                f"factory characterization has unresolved callable @{name}")
        for operation in _walk_operation(definition):
            if operation.name == "fabric.alloc":
                code_names.add(_attr_text(operation.attributes["code"]))
            elif operation.name == "fabric.call":
                pending.append(_attr_text(operation.attributes["callee"]))
    distances = set()
    for name in code_names:
        code = symbols.get(name)
        if (code is None or code.name != "fabric.code" or
                "distance" not in code.attributes):
            raise ValueError(
                f"factory characterization has unresolved code @{name}")
        distance = _integer(code.attributes["distance"])
        if distance <= 0:
            raise ValueError(
                f"factory characterization code @{name} has no positive "
                "distance")
        distances.add(distance)
    if not distances:
        raise ValueError(
            "factory characterization source uses no typed QEC code")
    return tuple(sorted(distances))


def _source_provider_evidence(module, graph, produces: ResourceKind):
    """Resolve the canonical factory semantics committed by the P1 stream."""

    source = graph.attributes.get("source_protocol")
    if source is None:
        raise ValueError(
            "factory characterization P3 graph lacks its source protocol")
    source_provider = _attr_text(source)
    candidates = set()
    for operation in _walk_operation(module.operation):
        if operation.name != "lvm.stream":
            continue
        resource = operation.attributes.get("produces")
        identity = operation.attributes.get("producer_identity")
        digest = operation.attributes.get("producer_semantics_sha256")
        if (resource is None or _attr_text(resource) != produces.name or
                identity is None or _attr_text(identity) != source_provider or
                digest is None):
            continue
        candidates.add((_attr_text(identity), _attr_text(digest)))
    if len(candidates) != 1:
        raise ValueError(
            "factory characterization requires exactly one retained backed "
            f"stream producing @{produces.name} with canonical producer "
            f"evidence, found {len(candidates)}")
    identity, digest = next(iter(candidates))
    if (not digest.startswith("sha256:") or len(digest) != 71 or
            any(character not in "0123456789abcdef"
                for character in digest.removeprefix("sha256:"))):
        raise ValueError(
            "factory characterization stream has a malformed canonical "
            "producer commitment")
    return identity, digest


def _operating_point_timing_profile(point, schedule: PhysicalSchedule):
    """Retain portable operating-point facts, not nested model cache keys."""

    timing = point.attributes.get("timing")
    if timing is None:
        return ()
    retained = []
    for name, value in schedule.timing_profile:
        source_name = "surface_cycle_ns" if name == "cycle_ns" else name
        if source_name in timing:
            retained.append((name, value))
    return tuple(retained)


def _factory_recurrence_plan(module, graph, produces, output_event):
    """Resolve compiler-authenticated recurring P3 schedule evidence."""

    source = graph.attributes.get("source_protocol")
    if source is None:
        raise ValueError(
            "factory characterization P3 graph lacks its source protocol")
    source_name = _attr_text(source)
    plans = []
    for operation in module.body.operations:
        candidate = operation.operation
        if candidate.name != "phys.spacetime_plan":
            continue
        recurrence_kind = candidate.attributes.get("recurrence_resource_kind")
        if (recurrence_kind is None or _attr_text(
                candidate.attributes["source_protocol"]) != source_name or
                _attr_text(recurrence_kind) != produces.name):
            continue
        plans.append(candidate)
    if not plans:
        return None, None
    if len(plans) != 1:
        raise ValueError(
            "factory characterization requires at most one periodic P3 "
            f"plan for @{produces.name}, found {len(plans)}")
    plan = plans[0]
    if ("recurrence_output_event" not in plan.attributes or _attr_text(
            plan.attributes["recurrence_output_event"]) != output_event):
        raise ValueError(
            "factory periodic evidence does not identify the materialized "
            "physical output event")
    if (_attr_text(plan.attributes["architecture"]) != _attr_text(
            graph.attributes["architecture"]) or
            _attr_text(plan.attributes["operating_point"]) != _attr_text(
                graph.attributes["operating_point"])):
        raise ValueError(
            "factory periodic evidence does not use the source P3 graph "
            "architecture and operating point")

    outputs = []
    for event in _walk_operation(plan):
        if event.name != "phys.spacetime_event":
            continue
        if ("output_resource_kind" not in event.attributes or _attr_text(
                event.attributes["output_resource_kind"]) != produces.name):
            continue
        iteration = _integer(event.attributes["iteration"])
        start = _number(event.attributes["start_ns"])
        duration = _number(event.attributes["duration_ns"])
        finish = start + duration
        if (iteration < 0 or not math.isfinite(finish) or duration <= 0.0):
            raise ValueError(
                "factory recurring output event has invalid schedule timing")
        outputs.append((iteration, finish))
    outputs.sort()
    if len(outputs) < 3:
        raise ValueError(
            "factory recurring P3 schedule must contain at least three typed "
            "outputs")
    if tuple(iteration for iteration, _ in outputs) != tuple(
            range(outputs[0][0], outputs[0][0] + len(outputs))):
        raise ValueError(
            "factory recurring outputs must have consecutive iterations")
    intervals = tuple(outputs[index][1] - outputs[index - 1][1]
                      for index in range(1, len(outputs)))
    interval_ns = intervals[0]
    if (not math.isfinite(interval_ns) or interval_ns <= 0.0 or
            any(not math.isclose(
                value, interval_ns, rel_tol=1.0e-12, abs_tol=1.0e-9)
                for value in intervals[1:])):
        raise ValueError(
            "factory recurring P3 schedule does not prove one stable output "
            "cadence")
    return plan, interval_ns


def _summarize_physical_footprint(classes, whole_classes, concrete_members):
    """Count whole classes symbolically and deduplicate concrete members."""

    if not whole_classes and not concrete_members:
        raise ValueError(
            "factory characterization schedule uses no physical resources")
    unknown = sorted(
        set(name for name in whole_classes if name not in classes) |
        set(name for name, _ in concrete_members if name not in classes))
    if unknown:
        raise ValueError(
            "factory characterization has unresolved resource classes: "
            f"{unknown!r}")
    unit_kinds = {classes[name][2] for name in whole_classes
                 } | {classes[name][2] for name, _ in concrete_members}
    if len(unit_kinds) != 1:
        raise ValueError(
            "factory characterization requires one common physical base unit")
    maximum = 2**63 - 1
    physical_units = 0
    for name in whole_classes:
        count, units, _ = classes[name]
        if count and units > maximum // count:
            raise OverflowError(
                "factory characterization physical footprint exceeds signed "
                "64-bit range")
        contribution = count * units
        if contribution > maximum - physical_units:
            raise OverflowError(
                "factory characterization physical footprint exceeds signed "
                "64-bit range")
        physical_units += contribution
    for name, _ in concrete_members:
        if name in whole_classes:
            continue
        units = classes[name][1]
        if units > maximum - physical_units:
            raise OverflowError(
                "factory characterization physical footprint exceeds signed "
                "64-bit range")
        physical_units += units
    if physical_units <= 0:
        raise OverflowError(
            "factory characterization physical footprint exceeds signed "
            "64-bit range")
    return physical_units, next(iter(unit_kinds))


def _physical_footprint(
    module,
    machine,
    schedule: PhysicalSchedule,
    recurrence_plan=None,
) -> tuple[int, str]:
    classes = {}
    for operation in _walk_operation(machine):
        if operation.name != "phys.resource_class":
            continue
        name = _attr_text(operation.attributes["sym_name"])
        granularity = (_attr_text(operation.attributes["granularity"])
                       if "granularity" in operation.attributes else "carrier")
        if granularity == "patch":
            if ("physical_units" not in operation.attributes or
                    "physical_unit_kind" not in operation.attributes):
                raise ValueError(
                    f"patch resource class @{name} lacks a typed physical "
                    "footprint")
            units = _integer(operation.attributes["physical_units"])
            unit_kind = _attr_text(operation.attributes["physical_unit_kind"])
        else:
            units = 1
            unit_kind = _attr_text(operation.attributes["kind"])
        if units <= 0 or not unit_kind:
            raise ValueError(
                f"physical resource class @{name} has an invalid footprint")
        count = _integer(operation.attributes["count"])
        if count < 0:
            raise ValueError(
                f"physical resource class @{name} has negative capacity")
        classes[name] = (count, units, unit_kind)

    scheduled_resources = {
        resource for entry in schedule.entries for resource in entry.resources
    }
    machine_name = _attr_text(machine.attributes["sym_name"])
    whole_classes = set()
    concrete_members = set()

    def add_class(name):
        if name not in classes:
            raise ValueError(
                f"factory characterization has unresolved resource class "
                f"@{name}")
        whole_classes.add(name)

    if recurrence_plan is not None:
        for operation in _walk_operation(recurrence_plan):
            if operation.name != "phys.spacetime_event":
                continue
            resource = operation.attributes.get("resource_class")
            if resource is not None:
                add_class(_symbol_leaf(resource))

    factory_classes = {}
    for operation in module.body.operations:
        candidate = operation.operation
        if candidate.name == "phys.factory_model":
            factory_classes[_attr_text(
                candidate.attributes["sym_name"])] = (_attr_text(
                    candidate.attributes["physical_resource_class"]))
    binding_classes = {}
    for operation in _walk_operation(machine):
        if operation.name != "phys.qec_binding":
            continue
        binding_classes[_attr_text(operation.attributes["sym_name"])] = tuple(
            _attr_text(value) for value in operation.attributes["resources"])

    for resource in scheduled_resources:
        if resource.startswith("class:"):
            add_class(resource.removeprefix("class:"))
        elif resource.startswith("factory:"):
            name = resource.removeprefix("factory:")
            if name not in factory_classes:
                raise ValueError(
                    f"factory characterization has unresolved compact "
                    f"factory model @{name}")
            add_class(factory_classes[name])
        elif resource.startswith("binding:"):
            name = resource.removeprefix("binding:")
            if name not in binding_classes:
                raise ValueError(
                    f"factory characterization has unresolved physical "
                    f"binding @{name}")
            for resource_class in binding_classes[name]:
                add_class(resource_class)
    for operation in module.body.operations:
        candidate = operation.operation
        if candidate.name != "phys.resource":
            continue
        if (_attr_text(candidate.attributes["architecture"]) != machine_name):
            continue
        resource_class = _attr_text(candidate.attributes["resource_class"])
        index = _integer(candidate.attributes["index"])
        if f"{resource_class}[{index}]" not in scheduled_resources:
            continue
        concrete_members.add((resource_class, index))
    return _summarize_physical_footprint(
        classes,
        whole_classes,
        concrete_members,
    )


def factory_model(
    schedule: PhysicalSchedule,
    *,
    produces: ResourceKind,
) -> FactoryModel:
    """Derive a compact model from a verified P3 factory schedule.

    The selected typed output must be packed exactly once.  Its completion
    time is the empty-lane startup.  A registered periodic P3 derivation may
    prove a shorter steady-state output interval from the retained P2 circuit,
    physical pools, and selected timing.  Without that evidence the startup is
    retained as a conservative non-overlapped interval.  If the detailed
    schedule contains postselection, the model is explicitly ``single_shot``
    and is conditional on those selections accepting; no retry latency is
    invented.
    """

    if not isinstance(schedule, PhysicalSchedule):
        raise TypeError(
            "cudaq.logical.compiler.factory_model requires a PhysicalSchedule")
    if not isinstance(produces, ResourceKind):
        raise TypeError("factory_model produces= requires a ResourceKind")
    schedule = schedule.canonical()
    if schedule.profile != "p3" or schedule._operating_point_symbol is None:
        raise ValueError(
            "factory characterization requires a scheduled P3 graph with a "
            "selected operating point")

    module = schedule.build._fresh_module()
    graph = _find_symbol(module, schedule._graph_symbol, "phys.graph")
    machine = _find_symbol(module, schedule._machine_symbol, "phys.machine")
    point = _find_symbol(
        module,
        schedule._operating_point_symbol,
        "phys.operating_point",
    )

    packed = []
    selections = []
    for operation in _walk_operation(graph):
        if operation.name == "phys.pack_resource" and _attr_text(
                operation.attributes["resource_kind"]) == produces.name:
            packed.append(_attr_text(operation.attributes["event_id"]))
        elif operation.name == "event.selection":
            selections.append(_attr_text(operation.attributes["event_id"]))
    if len(packed) != 1:
        raise ValueError("factory characterization requires exactly one packed "
                         f"@{produces.name} output, found {len(packed)}")
    entries = {entry.event_id: entry for entry in schedule.entries}
    output = entries.get(packed[0])
    if output is None or output.kind != "pack_resource":
        raise ValueError(
            "factory output event is absent from the verified schedule")

    timing = point.attributes.get("timing")
    cycle = (timing["surface_cycle_ns"]
             if timing is not None and "surface_cycle_ns" in timing else None)
    cycle_ns = _number(cycle) if cycle is not None else 0.0
    if not math.isfinite(cycle_ns) or cycle_ns <= 0.0:
        raise ValueError(
            "factory characterization operating point requires finite "
            "positive surface_cycle_ns")
    startup_cycles = output.finish_ns / cycle_ns
    if not math.isfinite(startup_cycles) or startup_cycles <= 0.0:
        raise ValueError("factory output has no finite positive startup time")

    recurrence, recurrence_interval_ns = _factory_recurrence_plan(
        module, graph, produces, packed[0])
    output_interval_cycles = (recurrence_interval_ns /
                              cycle_ns if recurrence_interval_ns is not None
                              else startup_cycles)
    if (not math.isfinite(output_interval_cycles) or
            output_interval_cycles <= 0.0 or
            output_interval_cycles > startup_cycles):
        raise ValueError(
            "factory periodic interval must be finite, positive, and no "
            "longer than the materialized startup")

    physical_units, unit_kind = _physical_footprint(module, machine, schedule,
                                                    recurrence)
    source_provider, source_provider_sha256 = _source_provider_evidence(
        module, graph, produces)
    build_sha256 = schedule.build.content_sha256.removeprefix("sha256:")
    schedule_sha256 = _schedule_sha256(schedule, build_sha256)
    characterization = _factory_characterization_from_verified_schedule(
        resource_kind=produces,
        source_provider=source_provider,
        source_provider_sha256=source_provider_sha256,
        startup_cycles=startup_cycles,
        output_interval_cycles=output_interval_cycles,
        build_sha256=build_sha256,
        schedule_sha256=schedule_sha256,
        operating_point=schedule._operating_point_symbol,
        output_events=tuple(packed),
        physical_units=physical_units,
        physical_unit_kind=unit_kind,
        code_distances=_source_code_distances(module, graph),
        timing_profile=_operating_point_timing_profile(point, schedule),
        timing_source=(_attr_text(point.attributes["timing_source"])
                       if "timing_source" in point.attributes else None),
        selection_events=tuple(selections),
    )
    return FactoryModel(
        startup_cycles=startup_cycles,
        output_interval_cycles=output_interval_cycles,
        evidence=computation("qlx.factory-characterization/v1:"
                             f"{build_sha256}:{schedule_sha256}"),
        policy="single_shot" if selections else "guaranteed",
        characterization=characterization,
    )


__all__ = ["factory_model"]
