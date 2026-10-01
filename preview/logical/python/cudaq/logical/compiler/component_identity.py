# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Canonical commitments for compact physical component models."""

from __future__ import annotations

from hashlib import sha256
import json

from cudaq.logical.architecture.logical import Space, Stream
from cudaq.logical.architecture.physical_definition import PhysicalMachine
from cudaq.logical.devices.definition import (
    PhysicalOperatingPoint,
    QECChannelRealization,
    QECChannelToPhysicalBinding,
)
from cudaq.logical.protocols.definition import ProtocolDefinition

from .context import CompilationContext
from .protocol_identity import protocol_definition_sha256


def _canonical_json(payload) -> str:
    return json.dumps(payload,
                      sort_keys=True,
                      separators=(",", ":"),
                      ensure_ascii=True)


def _digest(payload) -> str:
    encoded = _canonical_json(payload).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def protocol_contract(definition: ProtocolDefinition):
    """Return canonical source, objective, and boundary commitments."""

    if not isinstance(definition, ProtocolDefinition):
        raise TypeError("protocol contract requires a ProtocolDefinition")
    arity = getattr(definition.implements, "arity", 0)
    payload_blocks = tuple(
        f"__qlx_component_block{index}" for index in range(arity))
    transaction = CompilationContext()
    if payload_blocks:
        transaction.bind_protocol_payload_blocks(definition, payload_blocks)
    handle = transaction.materialize(definition)
    operation = transaction.find_symbol(handle.symbol, "fabric.protocol")
    if operation is None:
        raise ValueError(
            "protocol contract did not materialize fabric.protocol")
    function_type = str(operation.attributes["function_type"])
    objective = str(operation.attributes.get("objective", "none"))
    return {
        "name":
            definition.name,
        "source_sha256":
            protocol_definition_sha256(
                definition,
                payload_blocks=payload_blocks or None,
            ),
        "objective_sha256":
            _digest(("qlx.protocol-objective/v1", objective)),
        "boundary_sha256":
            _digest(("qlx.protocol-boundary/v1", function_type)),
        "function_type":
            function_type,
        "objective":
            objective,
    }


def selected_protocol_sha256(operation) -> str:
    """Commit the exact retained selected ``fabric.protocol`` operation."""

    if getattr(operation, "name", None) != "fabric.protocol":
        raise TypeError("selected protocol commitment requires fabric.protocol")
    # A detached clone gives Python and native replay the same generic local
    # SSA numbering, independent of the operation's parent state.
    clone = operation.clone()
    for name in ("component_source_sha256", "component_objective_sha256",
                 "component_boundary_sha256"):
        if name in clone.attributes:
            del clone.attributes[name]
    encoded = str(clone).encode("utf-8")
    return f"sha256:{sha256(encoded).hexdigest()}"


def _walk_operation(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk_operation(child.operation)


def selected_protocol_closure_sha256(operation) -> str:
    """Commit one selected protocol and its reachable P2 call closure."""

    if getattr(operation, "name", None) != "fabric.protocol":
        raise TypeError("selected protocol commitment requires fabric.protocol")
    module = operation
    while module.parent is not None:
        module = module.parent
    if module.name != "builtin.module":
        raise ValueError("selected protocol has no enclosing builtin.module")

    def symbol_value(attribute):
        if attribute is None:
            return None
        raw = getattr(attribute, "value", attribute)
        if isinstance(raw, (tuple, list)):
            return str(raw[-1]).lstrip("@")
        return str(raw).strip('"').lstrip("@").split("::@")[-1]

    def symbol(current):
        return symbol_value(current.attributes.get("sym_name"))

    top_level = {}
    for view in module.regions[0].blocks[0].operations:
        current = view.operation
        name = symbol(current)
        if name is None:
            continue
        if name in top_level:
            raise ValueError(
                f"selected protocol closure contains duplicate symbol {name!r}")
        top_level[name] = current

    closure = {}
    pending = [operation]
    while pending:
        current = pending.pop()
        name = symbol(current)
        if name is None:
            raise ValueError(
                "selected protocol closure contains an unnamed definition")
        if name in closure:
            continue
        closure[name] = current
        for candidate in _walk_operation(current):
            if candidate.name != "fabric.call":
                continue
            callee = symbol_value(candidate.attributes["callee"])
            target = top_level.get(callee)
            if target is None:
                raise ValueError(
                    f"selected protocol closure contains unresolved callee @{callee}"
                )
            pending.append(target)

    payload = bytearray(b"qlx.selected-protocol-closure/v1\n")

    def append_field(value: str) -> None:
        encoded = value.encode("utf-8")
        payload.extend(str(len(encoded)).encode("ascii"))
        payload.extend(b":")
        payload.extend(encoded)
        payload.extend(b"\n")

    for name in sorted(closure):
        current = closure[name]
        clone = current.clone()
        if clone.name == "fabric.protocol":
            for attribute in (
                    "component_source_sha256",
                    "component_objective_sha256",
                    "component_boundary_sha256",
            ):
                if attribute in clone.attributes:
                    del clone.attributes[attribute]
        append_field(name)
        append_field(current.name)
        append_field(str(clone))
    return f"sha256:{sha256(payload).hexdigest()}"


def physical_architecture_sha256(machine: PhysicalMachine) -> str:
    """Commit one exact typed physical-machine declaration."""

    if not isinstance(machine, PhysicalMachine):
        raise TypeError(
            "physical architecture commitment requires a PhysicalMachine")
    transaction = CompilationContext()
    handle = transaction.materialize(machine)
    operation = transaction.find_symbol(handle.symbol, "phys.machine")
    if operation is None:
        raise ValueError(
            "physical architecture did not materialize phys.machine")
    return _digest(("qlx.physical-architecture/v1", str(transaction.module)))


def timing_profile(point: PhysicalOperatingPoint):
    if not isinstance(point, PhysicalOperatingPoint):
        raise TypeError("timing profile requires a PhysicalOperatingPoint")
    return tuple(
        sorted((str(name), float(value))
               for name, value in point.timing.items()
               if name == "cycle_ns" or name.endswith("_ns")))


def endpoint_identity(endpoint) -> tuple:
    if isinstance(endpoint, Space):
        return "space", endpoint.name
    if isinstance(endpoint, Stream):
        return "stream", endpoint.name, endpoint.produces.name
    return type(endpoint).__name__, getattr(endpoint, "name", None)


def channel_identity_sha256(channel: QECChannelRealization) -> str:
    if not isinstance(channel, QECChannelRealization):
        raise TypeError("channel commitment requires QECChannelRealization")
    logical = channel.logical_channel

    def region_identity(region):
        transaction = CompilationContext()
        transaction.materialize(region.encoding)
        return (
            region.name,
            _digest(("qlx.qec-region-encoding/v1", str(transaction.module))),
            region.block_capacity,
            region.packing,
            region.role,
        )

    payload = (
        "qlx.qec-channel-realization/v2",
        channel.name,
        logical.name,
        endpoint_identity(logical.source),
        endpoint_identity(logical.destination),
        str(logical.direction),
        tuple(value.key for value in logical.capabilities),
        logical.concurrency,
        (
            channel.source.name,
            region_identity(channel.source.region),
            channel.source.slot,
            tuple(value.key for value in channel.source.capabilities),
            channel.source.concurrency,
            channel.source.provider,
        ),
        (
            channel.destination.name,
            region_identity(channel.destination.region),
            channel.destination.slot,
            tuple(value.key for value in channel.destination.capabilities),
            channel.destination.concurrency,
            channel.destination.provider,
        ),
        tuple(value.key for value in channel.capabilities),
        channel.concurrency,
        channel.provider,
    )
    return _digest(payload)


def logical_channel_sha256(channel: QECChannelRealization) -> str:
    if not isinstance(channel, QECChannelRealization):
        raise TypeError(
            "logical channel commitment requires QECChannelRealization")
    logical = channel.logical_channel
    return _digest((
        "qlx.logical-channel/v1",
        logical.name,
        endpoint_identity(logical.source),
        endpoint_identity(logical.destination),
        str(logical.direction),
        tuple(value.key for value in logical.capabilities),
        logical.concurrency,
    ))


def channel_binding_sha256(binding: QECChannelToPhysicalBinding) -> str:
    if not isinstance(binding, QECChannelToPhysicalBinding):
        raise TypeError(
            "binding commitment requires QECChannelToPhysicalBinding")
    return _digest((
        "qlx.qec-channel-physical-binding/v2",
        channel_identity_sha256(binding.qec_channel),
        tuple((resource.name, resource.kind, resource.count,
               resource.granularity.value,
               None if resource.footprint is None else (
                   resource.footprint.unit_kind,
                   resource.footprint.units,
                   resource.footprint.evidence,
               )) for resource in binding.resources),
        tuple(
            resource_claim_payload(claim)
            for claim in binding.transport_claims),
        binding.endpoint_occupancy,
    ))


def resource_claim_payload(claim):
    resource = claim.resource_class
    return (
        resource.name,
        resource.kind,
        resource.count,
        resource.granularity.value,
        claim.offset,
        claim.count,
        claim.units,
    )


def _resource_claim_commitment(claim):
    resource = claim.resource_class
    return {
        "resource": resource.name,
        "kind": resource.kind,
        "class_count": resource.count,
        "granularity": resource.granularity.value,
        "offset": claim.offset,
        "count": claim.count,
        "units": claim.units,
    }


def spacetime_model_payload(model):
    return {
        "schema":
            "qlx.spacetime-plan-model/v2",
        "protocol":
            protocol_contract(model.protocol),
        "latency_cycles":
            model.latency_cycles,
        "initiation_interval_cycles":
            model.initiation_interval_cycles,
        "interval_semantics":
            model.interval_semantics.value,
        "policy":
            model.policy,
        "code_distances":
            list(model.code_distances),
        "phases": [{
            "name": phase.name,
            "steps": phase.steps,
            "step_duration_cycles": phase.step_duration_cycles,
            "resources": [
                _resource_claim_commitment(value) for value in phase.resources
            ],
            "factories": [{
                "startup_cycles": factory.startup_cycles,
                "output_interval_cycles": factory.output_interval_cycles,
                "policy": factory.policy,
                "evidence": str(factory.evidence),
            } for factory in phase.factories],
            "after": list(phase.after),
        } for phase in model.phases],
    }


def spacetime_model_commitment(model) -> str:
    return _canonical_json(spacetime_model_payload(model))


def spacetime_model_sha256(model) -> str:
    return sha256(spacetime_model_commitment(model).encode("utf-8")).hexdigest()


def transport_model_payload(model):
    return {
        "schema": "qlx.transport-model/v2",
        "latency_cycles": model.latency_cycles,
        "initiation_interval_cycles": model.initiation_interval_cycles,
        "interval_semantics": model.interval_semantics.value,
        "policy": model.policy,
        "resources": [
            _resource_claim_commitment(value) for value in model.resources
        ],
        "endpoint_occupancy": list(model.endpoint_occupancy),
    }


def transport_model_commitment(model) -> str:
    return _canonical_json(transport_model_payload(model))


def transport_model_sha256(model) -> str:
    return sha256(transport_model_commitment(model).encode("utf-8")).hexdigest()


__all__ = [
    "channel_binding_sha256",
    "channel_identity_sha256",
    "logical_channel_sha256",
    "physical_architecture_sha256",
    "protocol_contract",
    "resource_claim_payload",
    "selected_protocol_closure_sha256",
    "selected_protocol_sha256",
    "spacetime_model_commitment",
    "spacetime_model_payload",
    "spacetime_model_sha256",
    "timing_profile",
    "transport_model_commitment",
    "transport_model_payload",
    "transport_model_sha256",
]
