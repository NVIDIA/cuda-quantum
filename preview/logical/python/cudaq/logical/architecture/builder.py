# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import json

import cudaq.mlir.ir as mlir_ir

from cudaq.logical.programs.context import (
    pop_trace,
    push_trace,
)
from cudaq.logical.types.values import (
    PhysicalRecord,
    PhysicalState,
)
from cudaq.logical.programs.definition import DefinitionHandle
from cudaq.logical.devices.definition import Device
from cudaq.logical.architecture.physical_definition import (
    PhysicalAction,
    PhysicalInstrument,
    PhysicalDefinition,
    PhysicalMachine,
    ResourceClass,
    physical_qubit,
)
from ..architecture.physical_instruments import MX, MZ


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_.$-" else "_" for c in name)


def _i64(context, value):
    return mlir_ir.IntegerAttr.get(
        mlir_ir.IntegerType.get_signless(64, context=context), value)


def _dict(context, values):
    if not values:
        return None
    return mlir_ir.DictAttr.get(
        {
            str(key): mlir_ir.StringAttr.get(str(value), context=context)
            for key, value in values.items()
        },
        context=context,
    )


def _structured_dict(context, values):
    """Encode a versioned process as deterministic JSON inside P3 MLIR."""

    return mlir_ir.StringAttr.get(
        json.dumps(values, sort_keys=True, separators=(",", ":")),
        context=context,
    )


def _strings(context, values):
    return mlir_ir.ArrayAttr.get(
        [
            mlir_ir.StringAttr.get(str(value), context=context)
            for value in values
        ],
        context=context,
    )


def _physical_capability_key(value):
    """Return the stable IR key of a typed or compatibility capability."""

    return getattr(value, "key", value)


def _resource_footprint_attrs(context, resource):
    attrs = {
        "granularity":
            mlir_ir.StringAttr.get(
                resource.granularity.value,
                context=context,
            )
    }
    if resource.footprint is not None:
        attrs.update({
            "physical_unit_kind":
                mlir_ir.StringAttr.get(
                    resource.footprint.unit_kind,
                    context=context,
                ),
            "physical_units":
                _i64(context, resource.footprint.units),
            "footprint_evidence":
                mlir_ir.StringAttr.get(
                    resource.footprint.evidence,
                    context=context,
                ),
        })
    return attrs


def _f64(context, value):
    with mlir_ir.Location.unknown(context):
        return mlir_ir.FloatAttr.get(mlir_ir.F64Type.get(context=context),
                                     float(value))


def _binding_entry(context, **values):
    attributes = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, dict):
            attributes[key] = mlir_ir.DictAttr.get(
                {
                    str(name): _f64(context, timing)
                    for name, timing in value.items()
                },
                context=context,
            )
        elif isinstance(value, (tuple, list)):
            if all(
                    isinstance(item, int) and not isinstance(item, bool)
                    for item in value):
                attributes[key] = mlir_ir.DenseI64ArrayAttr.get(value, context)
            else:
                attributes[key] = _strings(context, value)
        elif isinstance(value, int):
            attributes[key] = _i64(context, value)
        else:
            attributes[key] = mlir_ir.StringAttr.get(str(value),
                                                     context=context)
    return mlir_ir.DictAttr.get(attributes, context=context)


def _materialize_layered_device(transaction, device: Device):
    """Materialize the four stage machines and their typed refinements."""

    from cudaq.logical.compiler.protocol_identity import (
        factory_protocol_semantics_sha256,
        protocol_definition_sha256,
    )

    existing = transaction.lookup(device)
    if existing is not None:
        return existing
    context = transaction.context
    location = transaction.location

    def append(name, attrs, *, regions=0):
        with location:
            operation = mlir_ir.Operation.create(
                name,
                attributes=attrs,
                regions=regions,
                loc=location,
            )
            transaction.module.body.append(operation)
        return operation

    def insert(ip, name, attrs):
        with location:
            operation = mlir_ir.Operation.create(name,
                                                 attributes=attrs,
                                                 loc=location)
            ip.insert(operation)
        return operation

    existing_domain = transaction.find_symbol(device.logical.name, "lvm.domain")
    if existing_domain is not None:
        logical_symbol = device.logical.name
    else:
        logical_symbol = transaction.unique_symbol(device.logical.name)
        transaction.bind_resource_streams(
            device,
            logical_symbol=logical_symbol,
        )
        domain = append(
            "lvm.domain",
            {
                "sym_name":
                    mlir_ir.StringAttr.get(logical_symbol, context=context)
            },
            regions=1,
        )
        domain_ip = mlir_ir.InsertionPoint(domain.regions[0].blocks.append())
        for space in device.logical.spaces:
            attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(space.name, context=context),
                "capabilities":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.Attribute.parse(
                                f'#lvm.capability<"{item.key}">',
                                context=context,
                            ) for item in space.capabilities
                        ],
                        context=context,
                    ),
            }
            if space.capacity is not None:
                attrs["capacity"] = _i64(context, space.capacity)
            if space.tags:
                attrs["tags"] = _strings(context, space.tags)
            insert(domain_ip, "lvm.space", attrs)
        for stream in device.logical.streams:
            attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(stream.name, context=context),
                "produces":
                    mlir_ir.FlatSymbolRefAttr.get(
                        getattr(stream.produces, "name", str(stream.produces)),
                        context=context,
                    ),
            }
            if stream.buffer_size is not None:
                attrs["capacity"] = _i64(context, stream.buffer_size)
            if stream.produced_by is not None:
                attrs["produced_by"] = mlir_ir.FlatSymbolRefAttr.get(
                    transaction.materialize(stream.produced_by).symbol,
                    context=context,
                )
                attrs["produced_by_sha256"] = mlir_ir.StringAttr.get(
                    protocol_definition_sha256(stream.produced_by),
                    context=context,
                )
                attrs["producer_identity"] = mlir_ir.StringAttr.get(
                    stream.produced_by.name,
                    context=context,
                )
                attrs["producer_semantics_sha256"] = mlir_ir.StringAttr.get(
                    factory_protocol_semantics_sha256(stream.produced_by),
                    context=context,
                )
            if stream.transfer is not None:
                attrs["transfer"] = mlir_ir.FlatSymbolRefAttr.get(
                    transaction.materialize(stream.transfer).symbol,
                    context=context,
                )
                attrs["transfer_sha256"] = mlir_ir.StringAttr.get(
                    protocol_definition_sha256(stream.transfer),
                    context=context,
                )
            if stream.region is not None:
                attrs["backing_region"] = mlir_ir.FlatSymbolRefAttr.get(
                    stream.region.name, context=context)
            if stream.external:
                attrs["external"] = mlir_ir.UnitAttr.get(context=context)
            insert(domain_ip, "lvm.stream", attrs)
        for channel in device.logical.channels:
            attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(channel.name, context=context),
                "from":
                    mlir_ir.FlatSymbolRefAttr.get(channel.source.name,
                                                  context=context),
                "to":
                    mlir_ir.FlatSymbolRefAttr.get(channel.destination.name,
                                                  context=context),
                "capabilities":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.Attribute.parse(
                                f'#lvm.capability<"{item.key}">',
                                context=context,
                            ) for item in channel.capabilities
                        ],
                        context=context,
                    ),
                "direction":
                    mlir_ir.StringAttr.get(str(channel.direction),
                                           context=context),
            }
            if channel.capacity is not None:
                attrs["capacity"] = _i64(context, channel.capacity)
            insert(domain_ip, "lvm.channel", attrs)

    transaction.bind_resource_streams(device, logical_symbol=logical_symbol)

    qec_symbol = None
    logical_to_qec_symbol = None
    if device.qec is not None:
        qec_symbol = transaction.unique_symbol(device.qec.name)
        qec_machine = append(
            "fabric.machine",
            {"sym_name": mlir_ir.StringAttr.get(qec_symbol, context=context)},
            regions=1,
        )
        qec_ip = mlir_ir.InsertionPoint(qec_machine.regions[0].blocks.append())
        logical_by_qec = {
            binding.qec_region.name: binding.logical_region
            for binding in device.logical_to_qec
        }
        for region in device.qec.regions:
            logical_region = logical_by_qec.get(region.name)
            role = region.role
            if role is None:
                if logical_region is None:
                    raise ValueError(
                        f"internal QEC region @{region.name} requires an explicit role"
                    )
                capability_names = {
                    item.key for item in logical_region.capabilities
                }
                role = "compute"
                if any("memory" in item for item in capability_names):
                    role = "memory"
                elif any("factory" in item for item in capability_names):
                    role = "factory"
            code = transaction.materialize(region.encoding.code)
            encoding = transaction.materialize(region.encoding)
            epoch = transaction.materialize(region.encoding.initial_epoch)
            region_attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(region.name, context=context),
                "code":
                    mlir_ir.FlatSymbolRefAttr.get(code.symbol, context=context),
                "encoding":
                    mlir_ir.FlatSymbolRefAttr.get(encoding.symbol,
                                                  context=context),
                "epoch":
                    mlir_ir.FlatSymbolRefAttr.get(epoch.symbol,
                                                  context=context),
                "role":
                    mlir_ir.Attribute.parse(f"#fabric.role<{role}>",
                                            context=context),
                "floorplan":
                    mlir_ir.Attribute.parse(
                        "#fabric.floorplan<linear, "
                        f"[{region.block_capacity}]>",
                        context=context,
                    ),
                "block_capacity":
                    _i64(context, region.block_capacity),
                "packing":
                    mlir_ir.StringAttr.get(region.packing, context=context),
            }
            insert(
                qec_ip,
                "fabric.region",
                region_attrs,
            )
        for channel in device.qec.channels:
            with context:
                logical_channel = mlir_ir.SymbolRefAttr.get(
                    [logical_symbol, channel.logical_channel.name],
                    context=context,
                )
            channel_attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(channel.name, context=context),
                "region_a":
                    mlir_ir.FlatSymbolRefAttr.get(channel.source.region.name,
                                                  context=context),
                "port_a":
                    _i64(context, channel.source.slot),
                "region_b":
                    mlir_ir.FlatSymbolRefAttr.get(
                        channel.destination.region.name, context=context),
                "port_b":
                    _i64(context, channel.destination.slot),
                "logical_channel":
                    logical_channel,
                "port_a_name":
                    mlir_ir.StringAttr.get(channel.source.name,
                                           context=context),
                "port_b_name":
                    mlir_ir.StringAttr.get(channel.destination.name,
                                           context=context),
                "port_a_concurrency":
                    _i64(context, channel.source.concurrency),
                "port_b_concurrency":
                    _i64(context, channel.destination.concurrency),
                "port_a_capabilities":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.Attribute.parse(
                                f'#lvm.capability<"{item.key}">',
                                context=context,
                            ) for item in channel.source.capabilities
                        ],
                        context=context,
                    ),
                "port_b_capabilities":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.Attribute.parse(
                                f'#lvm.capability<"{item.key}">',
                                context=context,
                            ) for item in channel.destination.capabilities
                        ],
                        context=context,
                    ),
                "port_a_provider":
                    mlir_ir.StringAttr.get(channel.source.provider,
                                           context=context),
                "port_b_provider":
                    mlir_ir.StringAttr.get(channel.destination.provider,
                                           context=context),
                "capabilities":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.Attribute.parse(
                                f'#lvm.capability<"{item.key}">',
                                context=context,
                            ) for item in channel.capabilities
                        ],
                        context=context,
                    ),
                "direction":
                    mlir_ir.StringAttr.get(str(
                        channel.logical_channel.direction),
                                           context=context),
                "concurrency":
                    _i64(context, channel.concurrency),
                "provider":
                    mlir_ir.StringAttr.get(channel.provider, context=context),
            }
            source_metadata = _dict(context, channel.source.metadata)
            if source_metadata is not None:
                channel_attrs["port_a_metadata"] = source_metadata
            destination_metadata = _dict(context, channel.destination.metadata)
            if destination_metadata is not None:
                channel_attrs["port_b_metadata"] = destination_metadata
            metadata = _dict(context, channel.metadata)
            if metadata is not None:
                channel_attrs["metadata"] = metadata
            if channel.protocol is not None:
                channel_attrs["protocol"] = mlir_ir.FlatSymbolRefAttr.get(
                    transaction.materialize(channel.protocol).symbol,
                    context=context,
                )
            insert(qec_ip, "fabric.interconnect", channel_attrs)
        logical_to_qec_symbol = transaction.unique_symbol(
            f"{device.name}_logical_to_qec")
        append(
            "qlx.logical_to_qec",
            {
                "sym_name":
                    mlir_ir.StringAttr.get(logical_to_qec_symbol,
                                           context=context),
                "logical":
                    mlir_ir.FlatSymbolRefAttr.get(logical_symbol,
                                                  context=context),
                "qec":
                    mlir_ir.FlatSymbolRefAttr.get(qec_symbol, context=context),
                "entries":
                    mlir_ir.ArrayAttr.get(
                        [
                            _binding_entry(
                                context,
                                logical=binding.logical_region.name,
                                qec=binding.qec_region.name,
                                packing=binding.packing,
                                logical_capacity=binding.logical_region.
                                capacity,
                                block_capacity=binding.qec_region.
                                block_capacity,
                            ) for binding in device.logical_to_qec
                        ],
                        context=context,
                    ),
            },
        )

    physical_symbol = None
    qec_to_physical_symbol = None
    operating_point_symbol = None
    if device.physical is not None:
        physical_symbol = transaction.unique_symbol(device.physical.name)
        attrs = {
            "sym_name": mlir_ir.StringAttr.get(physical_symbol, context=context)
        }
        metadata = _dict(context, device.physical.metadata)
        if metadata is not None:
            attrs["metadata"] = metadata
        physical_machine = append("phys.machine", attrs, regions=1)
        physical_ip = mlir_ir.InsertionPoint(
            physical_machine.regions[0].blocks.append())
        for resource in device.physical.resource_classes:
            actions = [(transaction.materialize(action).symbol if isinstance(
                action, PhysicalAction) else action)
                       for action in resource.native_actions]
            instruments = [
                transaction.materialize(instrument).symbol
                for instrument in resource.native_instruments
            ]
            insert(
                physical_ip,
                "phys.resource_class",
                {
                    "sym_name":
                        mlir_ir.StringAttr.get(resource.name, context=context),
                    "kind":
                        mlir_ir.StringAttr.get(resource.kind, context=context),
                    "count":
                        _i64(context, resource.count),
                    "native_actions":
                        mlir_ir.ArrayAttr.get(
                            [
                                mlir_ir.FlatSymbolRefAttr.get(action,
                                                              context=context)
                                for action in actions
                            ],
                            context=context,
                        ),
                    "native_instruments":
                        mlir_ir.ArrayAttr.get(
                            [
                                mlir_ir.FlatSymbolRefAttr.get(instrument,
                                                              context=context)
                                for instrument in instruments
                            ],
                            context=context,
                        ),
                    "capabilities":
                        _strings(
                            context,
                            tuple(
                                map(_physical_capability_key,
                                    resource.capabilities)) +
                            tuple(binding.capability.key
                                  for binding in resource.capability_bindings),
                        ),
                    **_resource_footprint_attrs(context, resource),
                },
            )
        for topology in device.physical.topologies:
            topology_attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(topology.name, context=context),
                "kind":
                    mlir_ir.StringAttr.get(topology.kind, context=context),
            }
            parameters = _dict(context, topology.parameters)
            if parameters is not None:
                topology_attrs["parameters"] = parameters
            if topology.strict or topology.coordinates is not None:
                topology_attrs["num_nodes"] = _i64(context, len(topology.nodes))
            if topology.strict:
                topology_attrs["strict"] = mlir_ir.UnitAttr.get(context=context)
            if topology.edges:
                topology_attrs["edges"] = mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.DenseI64ArrayAttr.get(
                            (edge.source, edge.target), context)
                        for edge in topology.edges
                    ],
                    context=context,
                )
            if topology.coordinates is not None:
                topology_attrs["coordinates"] = mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.DenseI64ArrayAttr.get(
                            topology.coordinates[index], context)
                        for index in topology.nodes
                    ],
                    context=context,
                )
            insert(physical_ip, "phys.topology", topology_attrs)
        for binding in device.qec_to_physical:
            if binding.patch_topology is None:
                continue
            patch = binding.patch_topology
            insert(
                physical_ip,
                "phys.patch_topology",
                {
                    "sym_name":
                        mlir_ir.StringAttr.get(f"{binding.name}_patch_topology",
                                               context=context),
                    "capacity":
                        _i64(context, patch.capacity),
                    "carrier_groups":
                        mlir_ir.ArrayAttr.get(
                            [
                                mlir_ir.DenseI64ArrayAttr.get(group, context)
                                for group in patch.carrier_groups
                            ],
                            context=context,
                        ),
                    "categories":
                        _strings(
                            context,
                            [
                                item.name if item is not None else ""
                                for item in patch.categories
                            ],
                        ),
                    "edges":
                        mlir_ir.ArrayAttr.get(
                            [
                                mlir_ir.DenseI64ArrayAttr.get(edge, context)
                                for edge in patch.edges
                            ],
                            context=context,
                        ),
                    "carrier_topology":
                        mlir_ir.FlatSymbolRefAttr.get(binding.topology.name,
                                                      context=context),
                    "resource_class":
                        mlir_ir.FlatSymbolRefAttr.get(binding.resources[0].name,
                                                      context=context),
                },
            )
        for binding in device.qec_to_physical:
            with context:
                qec_region = mlir_ir.SymbolRefAttr.get(
                    [qec_symbol, binding.qec_region.name],
                    context=context,
                )
            binding_attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(binding.name, context=context),
                "qec_region":
                    qec_region,
                "resources":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.FlatSymbolRefAttr.get(resource.name,
                                                          context=context)
                            for resource in binding.resources
                        ],
                        context=context,
                    ),
            }
            if binding.topology is not None:
                binding_attrs["topology"] = mlir_ir.FlatSymbolRefAttr.get(
                    binding.topology.name, context=context)
            if binding.patch_topology is not None:
                binding_attrs["patch_topology"] = (
                    mlir_ir.FlatSymbolRefAttr.get(
                        f"{binding.name}_patch_topology", context=context))
            insert(physical_ip, "phys.qec_binding", binding_attrs)
        for binding in device.qec_channels_to_physical:
            with context:
                qec_channel = mlir_ir.SymbolRefAttr.get(
                    [qec_symbol, binding.qec_channel.name],
                    context=context,
                )
                transport_claims = mlir_ir.ArrayAttr.get([
                    mlir_ir.DictAttr.get(
                        {
                            "resource_class":
                                mlir_ir.FlatSymbolRefAttr.get(
                                    claim.resource_class.name, context=context),
                            "offset":
                                mlir_ir.IntegerAttr.get(
                                    mlir_ir.IntegerType.get_signless(
                                        64, context=context), claim.offset),
                            "count":
                                mlir_ir.IntegerAttr.get(
                                    mlir_ir.IntegerType.get_signless(
                                        64, context=context), claim.count),
                            "units":
                                mlir_ir.IntegerAttr.get(
                                    mlir_ir.IntegerType.get_signless(
                                        64, context=context), claim.units),
                        },
                        context=context) for claim in binding.transport_claims
                ],
                                                         context=context)
            attributes = {
                "sym_name":
                    mlir_ir.StringAttr.get(
                        f"{binding.qec_channel.name}_physical",
                        context=context,
                    ),
                "qec_channel":
                    qec_channel,
                "resources":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.FlatSymbolRefAttr.get(resource.name,
                                                          context=context)
                            for resource in binding.resources
                        ],
                        context=context,
                    ),
            }
            if binding.transport_claims:
                attributes["transport_claims"] = transport_claims
                attributes["source_endpoint_occupancy"] = (
                    mlir_ir.IntegerAttr.get(
                        mlir_ir.IntegerType.get_signless(64, context=context),
                        binding.endpoint_occupancy[0]))
                attributes["destination_endpoint_occupancy"] = (
                    mlir_ir.IntegerAttr.get(
                        mlir_ir.IntegerType.get_signless(64, context=context),
                        binding.endpoint_occupancy[1]))
            insert(
                physical_ip,
                "phys.qec_channel_binding",
                attributes,
            )
        qec_to_physical_symbol = transaction.unique_symbol(
            f"{device.name}_qec_to_physical")
        append(
            "qlx.qec_to_physical",
            {
                "sym_name":
                    mlir_ir.StringAttr.get(qec_to_physical_symbol,
                                           context=context),
                "qec":
                    mlir_ir.FlatSymbolRefAttr.get(qec_symbol, context=context),
                "physical":
                    mlir_ir.FlatSymbolRefAttr.get(physical_symbol,
                                                  context=context),
                "entries":
                    mlir_ir.ArrayAttr.get(
                        [
                            _binding_entry(
                                context,
                                qec=binding.qec_region.name,
                                binding=binding.name,
                                resources=tuple(
                                    resource.name
                                    for resource in binding.resources),
                                topology=(None if binding.topology is None else
                                          binding.topology.name),
                                patch_topology=(
                                    None if binding.patch_topology is None else
                                    f"{binding.name}_patch_topology"),
                                factory_startup_cycles=(
                                    None if binding.factory_model is None else
                                    binding.factory_model.startup_cycles),
                                factory_output_interval_cycles=(
                                    None if binding.factory_model is None else
                                    binding.factory_model.output_interval_cycles
                                ),
                                factory_model_policy=(
                                    None if binding.factory_model is None else
                                    binding.factory_model.policy),
                                factory_model_evidence=(
                                    None if binding.factory_model is None else
                                    str(binding.factory_model.evidence)),
                                factory_source_build_sha256=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.build_sha256),
                                factory_source_provider=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.source_provider),
                                factory_source_provider_sha256=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.source_provider_sha256),
                                factory_source_startup_cycles=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.startup_cycles),
                                factory_source_output_interval_cycles=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.output_interval_cycles),
                                factory_source_schedule_sha256=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.schedule_sha256),
                                factory_source_operating_point=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.operating_point),
                                factory_source_timing_source=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.timing_source),
                                factory_source_timing_profile=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else dict(
                                        binding.factory_model.characterization.
                                        timing_profile)),
                                factory_source_output_events=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.output_events),
                                factory_source_selection_events=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.selection_events),
                                factory_source_physical_units=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.physical_units),
                                factory_source_physical_unit_kind=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.physical_unit_kind),
                                factory_source_code_distances=(
                                    None if binding.factory_model is None or
                                    binding.factory_model.characterization
                                    is None else binding.factory_model.
                                    characterization.code_distances),
                            )
                            for binding in device.qec_to_physical
                        ],
                        context=context,
                    ),
            },
        )
        if device.operating_point is not None:
            point = device.operating_point
            operating_point_symbol = transaction.unique_symbol(
                f"{device.name}_{point.name}_operating_point")
            point_attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(operating_point_symbol,
                                           context=context),
                "machine":
                    mlir_ir.FlatSymbolRefAttr.get(physical_symbol,
                                                  context=context),
            }
            for key in ("timing", "calibration", "costs"):
                value = _dict(context, getattr(point, key))
                if value is not None:
                    point_attrs[key] = value
            if point.timing_source is not None:
                point_attrs["timing_source"] = mlir_ir.StringAttr.get(
                    point.timing_source, context=context)
            if point.target_compatibility:
                point_attrs["target_compatibility"] = _strings(
                    context, point.target_compatibility)
            append("phys.operating_point", point_attrs)

    device_symbol = transaction.unique_symbol(device.name)
    attrs = {
        "sym_name":
            mlir_ir.StringAttr.get(device_symbol, context=context),
        "logical":
            mlir_ir.FlatSymbolRefAttr.get(logical_symbol, context=context),
    }
    resource_bindings = []
    for stream in device.logical.streams:
        if stream.produced_by is None:
            continue
        producer = transaction.materialize(stream.produced_by)
        with context:
            stream_ref = mlir_ir.SymbolRefAttr.get(
                [logical_symbol, stream.name], context=context)
        values = {
            "stream":
                stream_ref,
            "producer":
                mlir_ir.FlatSymbolRefAttr.get(producer.symbol, context=context),
        }
        if stream.region is not None:
            with context:
                values["factory"] = mlir_ir.SymbolRefAttr.get(
                    [logical_symbol, stream.region.name], context=context)
        if stream.transfer is not None:
            transfer = transaction.materialize(stream.transfer)
            values["transfer"] = mlir_ir.FlatSymbolRefAttr.get(transfer.symbol,
                                                               context=context)
        resource_bindings.append(mlir_ir.DictAttr.get(values, context=context))
    if resource_bindings:
        attrs["resource_bindings"] = mlir_ir.ArrayAttr.get(resource_bindings,
                                                           context=context)
    for key, value in (
        ("qec", qec_symbol),
        ("physical", physical_symbol),
        ("logical_to_qec", logical_to_qec_symbol),
        ("qec_to_physical", qec_to_physical_symbol),
        ("operating_point", operating_point_symbol),
    ):
        if value is not None:
            attrs[key] = mlir_ir.FlatSymbolRefAttr.get(value, context=context)
    metadata = _dict(context, device.metadata)
    if metadata is not None:
        attrs["metadata"] = metadata
    append("qlx.device", attrs)
    profile = device.layers[-1].value
    transaction.add_profile(profile)
    handle = DefinitionHandle(symbol=device_symbol,
                              kind="device",
                              profile=profile)
    transaction.bind(device, handle)
    return handle


def materialize_physical_action(transaction, action: PhysicalAction):
    existing = transaction.lookup(action)
    if existing is not None:
        return existing
    context = transaction.context
    symbol = transaction.unique_symbol(action.name)
    attrs = {
        "sym_name": mlir_ir.StringAttr.get(symbol, context=context),
        "arity": _i64(context, action.arity),
        "process": _structured_dict(context, action.process.to_dict()),
    }
    bindings = _dict(context, action.controller_bindings)
    if bindings is not None:
        attrs["controller_bindings"] = bindings
    for name in ("parameters",):
        values = getattr(action, name)
        if values:
            attrs[name] = _strings(context, values)
    if action.broadcast:
        attrs["broadcast"] = mlir_ir.UnitAttr.get(context=context)
    metadata = _dict(context, action.metadata)
    if metadata is not None:
        attrs["metadata"] = metadata
    with transaction.location:
        operation = mlir_ir.Operation.create("phys.action",
                                             attributes=attrs,
                                             loc=transaction.location)
        transaction.module.body.append(operation)
    handle = DefinitionHandle(symbol=symbol,
                              kind="physical_action",
                              profile="p3")
    transaction.bind(action, handle)
    transaction.add_profile("p3")
    return handle


def materialize_physical_instrument(transaction,
                                    instrument: PhysicalInstrument):
    existing = transaction.lookup(instrument)
    if existing is not None:
        return existing
    context = transaction.context
    symbol = transaction.unique_symbol(instrument.name)
    attrs = {
        "sym_name":
            mlir_ir.StringAttr.get(symbol, context=context),
        "kind":
            mlir_ir.StringAttr.get(instrument.operation, context=context),
        "record_schema":
            mlir_ir.StringAttr.get(instrument.record_schema, context=context),
        "process":
            _structured_dict(context, instrument.process.to_dict()),
    }
    bindings = _dict(context, instrument.controller_bindings)
    if bindings is not None:
        attrs["controller_bindings"] = bindings
    if instrument.arity is None:
        attrs["variadic"] = mlir_ir.UnitAttr.get(context=context)
    else:
        attrs["arity"] = _i64(context, instrument.arity)
    if instrument.preserves_inputs:
        attrs["preserves_inputs"] = mlir_ir.UnitAttr.get(context=context)
    for name in ("parameters",):
        values = getattr(instrument, name)
        if values:
            attrs[name] = _strings(context, values)
    metadata = _dict(context, instrument.metadata)
    if metadata is not None:
        attrs["metadata"] = metadata
    with transaction.location:
        operation = mlir_ir.Operation.create("phys.instrument",
                                             attributes=attrs,
                                             loc=transaction.location)
        transaction.module.body.append(operation)
    handle = DefinitionHandle(symbol=symbol,
                              kind="physical_instrument",
                              profile="p3")
    transaction.bind(instrument, handle)
    transaction.add_profile("p3")
    return handle


def materialize_physical_machine(transaction, machine: PhysicalMachine):
    """Materialize one standalone P3 machine declaration.

    A physical machine owns carrier resources and carrier topology only.
    QEC-region bindings belong to a Device's P2-to-P3 refinement and are
    therefore intentionally not emitted by this standalone materializer.
    """

    existing = transaction.lookup(machine)
    if existing is not None:
        return existing
    context = transaction.context
    location = transaction.location
    symbol = transaction.unique_symbol(machine.name)
    attrs = {
        "sym_name": mlir_ir.StringAttr.get(symbol, context=context),
    }
    metadata = _dict(context, machine.metadata)
    if metadata is not None:
        attrs["metadata"] = metadata
    with location:
        operation = mlir_ir.Operation.create(
            "phys.machine",
            attributes=attrs,
            regions=1,
            loc=location,
        )
        transaction.module.body.append(operation)
        block = operation.regions[0].blocks.append()
    insertion_point = mlir_ir.InsertionPoint(block)

    for resource in machine.resource_classes:
        actions = [(transaction.materialize(action).symbol if isinstance(
            action, PhysicalAction) else action)
                   for action in resource.native_actions]
        instruments = [
            transaction.materialize(instrument).symbol
            for instrument in resource.native_instruments
        ]
        insert_attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(resource.name, context=context),
            "kind":
                mlir_ir.StringAttr.get(resource.kind, context=context),
            "count":
                _i64(context, resource.count),
            "native_actions":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.FlatSymbolRefAttr.get(action, context=context)
                        for action in actions
                    ],
                    context=context,
                ),
            "native_instruments":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.FlatSymbolRefAttr.get(instrument,
                                                      context=context)
                        for instrument in instruments
                    ],
                    context=context,
                ),
            "capabilities":
                _strings(
                    context,
                    tuple(map(_physical_capability_key, resource.capabilities))
                    + tuple(binding.capability.key
                            for binding in resource.capability_bindings),
                ),
            **_resource_footprint_attrs(context, resource),
        }
        with location:
            insertion_point.insert(
                mlir_ir.Operation.create(
                    "phys.resource_class",
                    attributes=insert_attrs,
                    loc=location,
                ))

    for topology in machine.topologies:
        topology_attrs = {
            "sym_name": mlir_ir.StringAttr.get(topology.name, context=context),
            "kind": mlir_ir.StringAttr.get(topology.kind, context=context),
        }
        parameters = _dict(context, topology.parameters)
        if parameters is not None:
            topology_attrs["parameters"] = parameters
        if topology.strict or topology.coordinates is not None:
            topology_attrs["num_nodes"] = _i64(context, len(topology.nodes))
        if topology.strict:
            topology_attrs["strict"] = mlir_ir.UnitAttr.get(context=context)
        if topology.edges:
            topology_attrs["edges"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.DenseI64ArrayAttr.get(
                        (edge.source, edge.target), context)
                    for edge in topology.edges
                ],
                context=context,
            )
        if topology.coordinates is not None:
            topology_attrs["coordinates"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.DenseI64ArrayAttr.get(topology.coordinates[index],
                                                  context)
                    for index in topology.nodes
                ],
                context=context,
            )
        with location:
            insertion_point.insert(
                mlir_ir.Operation.create(
                    "phys.topology",
                    attributes=topology_attrs,
                    loc=location,
                ))

    handle = DefinitionHandle(symbol=symbol,
                              kind="physical_machine",
                              profile="p3")
    transaction.bind(machine, handle)
    transaction.add_profile("p3")
    return handle


def materialize_device(transaction, device: Device):
    return _materialize_layered_device(transaction, device)


def _flatten(values):
    if isinstance(values, (tuple, list)):
        result = []
        for value in values:
            result.extend(_flatten(value))
        return tuple(result)
    return (values,)


class PhysicalBuilder:
    """Direct Python tracer for one P3 physical event graph."""

    def __init__(self, transaction, definition: PhysicalDefinition) -> None:
        self.transaction = transaction
        self.definition = definition
        self.context = transaction.context
        self.location = transaction.location
        self.architecture = definition.architecture
        self.architecture_handle = transaction.materialize(self.architecture)
        self.symbol = transaction.unique_symbol(definition.name)
        self._states = []
        self._event = 0
        self._record = 0
        self._finished = False
        function_type = mlir_ir.FunctionType.get((), (), context=self.context)
        with self.context:
            type_attr = mlir_ir.TypeAttr.get(function_type)
        with self.location:
            self.operation = mlir_ir.Operation.create(
                "phys.graph",
                attributes={
                    "sym_name":
                        mlir_ir.StringAttr.get(self.symbol,
                                               context=self.context),
                    "architecture":
                        mlir_ir.FlatSymbolRefAttr.get(
                            self.architecture_handle.symbol,
                            context=self.context),
                    "function_type":
                        type_attr,
                },
                regions=1,
                loc=self.location,
            )
            transaction.module.body.append(self.operation)
            self.block = self.operation.regions[0].blocks.append()
        self.insertion_point = mlir_ir.InsertionPoint(self.block)

    def _emit(self, name, *, operands=(), results=(), attributes=None):
        with self.location:
            operation = mlir_ir.Operation.create(
                name,
                operands=list(operands),
                results=list(results),
                attributes=dict(attributes or {}),
                loc=self.location,
            )
            self.insertion_point.insert(operation)
        return operation

    def _event_id(self, prefix):
        value = f"{prefix}{self._event}"
        self._event += 1
        return value

    def _new_state(self, value, resource, resource_class=None):
        state = PhysicalState(
            value,
            owner=self,
            resource=resource,
            resource_class=resource_class,
            location=self.location,
        )
        self._states.append(state)
        return state

    def _consume(self, values, operation):
        states = _flatten(values)
        for state in states:
            if not isinstance(state, PhysicalState) or state.owner is not self:
                raise TypeError(
                    f"{operation} expects physical states from this graph")
            state._consume(operation)
        return states

    def _apply_resource_attrs(self, states):
        attrs = {
            "resources":
                mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.FlatSymbolRefAttr.get(state.resource,
                                                      context=self.context)
                        for state in states
                    ],
                    context=self.context,
                )
        }
        class_names = {
            state.resource_class.name if isinstance(
                state.resource_class, ResourceClass) else state.resource_class
            for state in states
        }
        if len(states) > 1 and len(
                class_names) == 1 and None not in class_names:
            resource_class_name = next(iter(class_names))
            if len(self.architecture.topologies) == 1:
                topology = self.architecture.topologies[0]
                if topology.strict:
                    attrs["topology"] = mlir_ir.FlatSymbolRefAttr.get(
                        topology.name, context=self.context)
        return attrs

    def acquire(self, resource_class, *, count, kind=physical_qubit, name=None):
        if not isinstance(resource_class, ResourceClass):
            raise TypeError(
                "cudaq.logical.acquire expects one architecture ResourceClass")
        if resource_class not in self.architecture.resource_classes:
            raise ValueError(
                "resource class does not belong to this architecture")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise TypeError("resource count must be a nonnegative int")
        if count > resource_class.count:
            raise ValueError("resource request exceeds architecture capacity")
        # ``kind`` describes the Python state proxy requested by the author;
        # the resource declaration itself must retain the architecture's
        # physical carrier kind (for example, ``atom`` rather than ``qubit``).
        kind_name = resource_class.kind
        symbols = []
        types = []
        for index in range(count):
            symbol = self.transaction.unique_symbol(
                f"{self.symbol}_{name or resource_class.name}_{index}")
            symbols.append(symbol)
            with self.location:
                declaration = mlir_ir.Operation.create(
                    "phys.resource",
                    attributes={
                        "sym_name":
                            mlir_ir.StringAttr.get(symbol,
                                                   context=self.context),
                        "kind":
                            mlir_ir.StringAttr.get(kind_name,
                                                   context=self.context),
                        "architecture":
                            mlir_ir.FlatSymbolRefAttr.get(
                                self.architecture.name, context=self.context),
                        "resource_class":
                            mlir_ir.FlatSymbolRefAttr.get(resource_class.name,
                                                          context=self.context),
                        "index":
                            _i64(self.context, index),
                    },
                    loc=self.location,
                )
                self.transaction.module.body.append(declaration)
            types.append(
                mlir_ir.Type.parse(f"!phys.state<@{symbol}>",
                                   context=self.context))
        operation = self._emit(
            "phys.acquire",
            results=types,
            attributes={
                "resources":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.FlatSymbolRefAttr.get(symbol,
                                                          context=self.context)
                            for symbol in symbols
                        ],
                        context=self.context,
                    ),
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("acquire"),
                                           context=self.context),
            },
        )
        return tuple(
            self._new_state(value, symbol, resource_class)
            for value, symbol in zip(operation.results, symbols))

    def load(self, values, *, state="loaded"):
        inputs = self._consume(values, "phys.prepare")
        operation = self._emit(
            "phys.prepare",
            operands=[value.mlir_value for value in inputs],
            results=[value.type for value in inputs],
            attributes={
                "state":
                    mlir_ir.StringAttr.get(str(state), context=self.context),
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("prepare"),
                                           context=self.context),
            },
        )
        return tuple(
            self._new_state(result, value.resource, value.resource_class)
            for result, value in zip(operation.results, inputs))

    def reset(self, values, *, state="zero"):
        inputs = self._consume(values, "phys.reset")
        operation = self._emit(
            "phys.reset",
            operands=[value.mlir_value for value in inputs],
            results=[value.type for value in inputs],
            attributes={
                "state":
                    mlir_ir.StringAttr.get(str(state), context=self.context),
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("reset"),
                                           context=self.context),
            },
        )
        return tuple(
            self._new_state(result, value.resource, value.resource_class)
            for result, value in zip(operation.results, inputs))

    def move(self, values, *, via, trajectory=None):
        inputs = self._consume(values, "phys.move")
        route = getattr(via, "name", str(via))
        attrs = {
            "route":
                mlir_ir.FlatSymbolRefAttr.get(route, context=self.context),
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("move"),
                                       context=self.context),
        }
        if trajectory is not None:
            attrs["trajectory"] = mlir_ir.StringAttr.get(str(
                getattr(trajectory, "name", trajectory)),
                                                         context=self.context)
        operation = self._emit(
            "phys.move",
            operands=[value.mlir_value for value in inputs],
            results=[value.type for value in inputs],
            attributes=attrs,
        )
        return tuple(
            self._new_state(result, value.resource, value.resource_class)
            for result, value in zip(operation.results, inputs))

    def apply_definition(self, action, values, parameters):
        if isinstance(action, PhysicalAction):
            return self._apply_typed_action(action, values, parameters)
        groups = [tuple(_flatten(value)) for value in values]
        inputs = self._consume(values, "phys.apply")
        action_name = getattr(action, "name", str(action))
        attrs = {
            "action":
                mlir_ir.FlatSymbolRefAttr.get(action_name,
                                              context=self.context),
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("apply"),
                                       context=self.context),
        }
        attrs.update(self._apply_resource_attrs(inputs))
        if parameters:
            attrs["reservations"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(f"{key}={value}",
                                           context=self.context)
                    for key, value in sorted(parameters.items())
                ],
                context=self.context,
            )
        operation = self._emit(
            "phys.apply",
            operands=[value.mlir_value for value in inputs],
            results=[value.type for value in inputs],
            attributes=attrs,
        )
        successors = [
            self._new_state(result, value.resource, value.resource_class)
            for result, value in zip(operation.results, inputs)
        ]
        shaped = []
        offset = 0
        for group in groups:
            part = tuple(successors[offset:offset + len(group)])
            offset += len(group)
            shaped.append(part[0] if len(part) == 1 else part)
        return shaped[0] if len(shaped) == 1 else tuple(shaped)

    @staticmethod
    def _native_action_names(resource_class):
        return {
            item.name if isinstance(item, PhysicalAction) else item
            for item in resource_class.native_actions
        }

    def _apply_action_once(self, action, symbol, states, parameters):
        states = tuple(states)
        if len(states) != action.arity:
            raise ValueError(
                f"physical action {action.name!r} has arity {action.arity}, "
                f"got {len(states)} carriers")
        for state in states:
            resource_class = state.resource_class
            if resource_class is None or action.name not in self._native_action_names(
                    resource_class):
                name = getattr(resource_class, "name", None)
                raise ValueError(
                    f"resource class {name!r} does not advertise physical "
                    f"action {action.name!r}")
        inputs = self._consume(states, "phys.apply")
        attrs = {
            "action":
                mlir_ir.FlatSymbolRefAttr.get(symbol, context=self.context),
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("apply"),
                                       context=self.context),
        }
        attrs.update(self._apply_resource_attrs(inputs))
        if parameters:
            attrs["reservations"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(f"{key}={value}",
                                           context=self.context)
                    for key, value in sorted(parameters.items())
                ],
                context=self.context,
            )
        operation = self._emit(
            "phys.apply",
            operands=[value.mlir_value for value in inputs],
            results=[value.type for value in inputs],
            attributes=attrs,
        )
        return tuple(
            self._new_state(result, value.resource, value.resource_class)
            for result, value in zip(operation.results, inputs))

    def _apply_typed_action(self, action, values, parameters):
        parameters = dict(parameters)
        pairs = parameters.pop("pairs", None)
        declared = set(action.parameters)
        special = {"schedule"}
        unknown = set(parameters) - declared - special
        if unknown:
            raise TypeError(
                f"physical action {action.name!r} has no parameter(s) "
                f"{sorted(unknown)!r}")
        groups = [list(_flatten(value)) for value in values]
        if len(groups) != action.arity:
            raise ValueError(
                f"physical action {action.name!r} expects {action.arity} "
                f"carrier operand group(s), got {len(groups)}")
        symbol = self.transaction.materialize(action).symbol

        if action.broadcast:
            if pairs is not None:
                raise TypeError(
                    "broadcast physical actions do not accept pairs=")
            if len(groups) != 1 or not groups[0]:
                raise ValueError(
                    "broadcast physical actions require one nonempty carrier collection"
                )
            states = tuple(groups[0])
            for state in states:
                resource_class = state.resource_class
                if (resource_class is None or action.name
                        not in self._native_action_names(resource_class)):
                    raise ValueError(
                        f"resource class {getattr(resource_class, 'name', None)!r} "
                        f"does not advertise broadcast action {action.name!r}")
            inputs = self._consume(states, "phys.apply")
            attrs = {
                "action":
                    mlir_ir.FlatSymbolRefAttr.get(symbol, context=self.context),
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("apply"),
                                           context=self.context),
            }
            if parameters:
                attrs["reservations"] = mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(f"{key}={value}",
                                               context=self.context)
                        for key, value in sorted(parameters.items())
                    ],
                    context=self.context,
                )
            operation = self._emit(
                "phys.apply",
                operands=[state.mlir_value for state in inputs],
                results=[state.type for state in inputs],
                attributes=attrs,
            )
            successors = tuple(
                self._new_state(result, state.resource, state.resource_class)
                for result, state in zip(operation.results, inputs))
            return successors[0] if len(successors) == 1 else successors

        if pairs is not None:
            if action.arity != 2:
                raise TypeError(
                    "pairs= is supported only for binary physical actions")
            relation = tuple(pairs)
            for edge in relation:
                if not isinstance(edge, (tuple, list)) or len(edge) != 2:
                    raise TypeError(
                        "pairs= entries must contain two indices or states")
                left, right = edge
                if isinstance(left, int) and not isinstance(left, bool):
                    i, j = left, right
                else:
                    try:
                        i = groups[0].index(left)
                        j = groups[1].index(right)
                    except ValueError as exc:
                        raise ValueError(
                            "pairs= references a state outside its operand group"
                        ) from exc
                if not isinstance(j, int) or isinstance(j, bool):
                    raise TypeError("pairs= index entries must be Python ints")
                if not (0 <= i < len(groups[0]) and 0 <= j < len(groups[1])):
                    raise IndexError(
                        "pairs= index is outside its carrier group")
                groups[0][i], groups[1][j] = self._apply_action_once(
                    action,
                    symbol,
                    (groups[0][i], groups[1][j]),
                    parameters,
                )
        else:
            widths = {len(group) for group in groups}
            if len(widths) != 1:
                raise ValueError(
                    "physical action operand collections require equal widths; "
                    "use pairs= for an explicit interaction relation")
            width = widths.pop()
            for index in range(width):
                updated = self._apply_action_once(
                    action,
                    symbol,
                    tuple(group[index] for group in groups),
                    parameters,
                )
                for group, successor in zip(groups, updated):
                    group[index] = successor

        shaped = [
            group[0] if len(group) == 1 else tuple(group) for group in groups
        ]
        return shaped[0] if len(shaped) == 1 else tuple(shaped)

    def measure(self, value, *, basis="z", destructive=True, record=None):
        (state,) = self._consume((value,), "phys.measure")
        basis = str(basis).lower()
        try:
            instrument = {"x": MX, "z": MZ}[basis]
        except KeyError as exc:
            raise ValueError(
                "physical measurement basis must be 'x' or 'z'") from exc
        measurement = self.transaction.materialize(instrument).symbol
        record = record or f"record{self._record}"
        self._record += 1
        record_type = mlir_ir.Type.parse("!phys.record<@bit>",
                                         context=self.context)
        results = [record_type] if destructive else [state.type, record_type]
        attrs = {
            "measurement":
                mlir_ir.FlatSymbolRefAttr.get(measurement,
                                              context=self.context),
            "record_id":
                mlir_ir.StringAttr.get(record, context=self.context),
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("measure"),
                                       context=self.context),
        }
        if destructive:
            attrs["destructive"] = mlir_ir.UnitAttr.get(context=self.context)
        operation = self._emit(
            "phys.measure",
            operands=[state.mlir_value],
            results=results,
            attributes=attrs,
        )
        physical_record = PhysicalRecord(
            operation.results[-1],
            owner=self,
            record=record,
            producer=attrs["event_id"].value,
            location=self.location,
        )
        if destructive:
            return physical_record
        return self._new_state(operation.results[0], state.resource,
                               state.resource_class), physical_record

    def delay(self, values, *, duration_ns):
        inputs = self._consume(values, "phys.delay")
        f64 = mlir_ir.F64Type.get(context=self.context)
        with self.location:
            duration = mlir_ir.FloatAttr.get(f64, float(duration_ns))
        operation = self._emit(
            "phys.delay",
            operands=[value.mlir_value for value in inputs],
            results=[value.type for value in inputs],
            attributes={
                "duration_ns":
                    duration,
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("delay"),
                                           context=self.context),
            },
        )
        return tuple(
            self._new_state(result, value.resource, value.resource_class)
            for result, value in zip(operation.results, inputs))

    def barrier(self, values=(), *, domains=()):
        inputs = self._consume(values, "phys.barrier") if values else ()
        attrs = {
            "event_id":
                mlir_ir.StringAttr.get(self._event_id("barrier"),
                                       context=self.context)
        }
        if domains:
            attrs["domains"] = _strings(self.context, domains)
        operation = self._emit(
            "phys.barrier",
            operands=[value.mlir_value for value in inputs],
            results=[value.type for value in inputs],
            attributes=attrs,
        )
        return tuple(
            self._new_state(result, value.resource, value.resource_class)
            for result, value in zip(operation.results, inputs))

    def fence(self, effects):
        effects = tuple(map(str, effects))
        self._emit(
            "event.fence",
            attributes={
                "effects":
                    _strings(self.context, effects),
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("fence"),
                                           context=self.context),
            },
        )

    def release(self, values):
        inputs = self._consume(values, "phys.release")
        self._emit(
            "phys.release",
            operands=[value.mlir_value for value in inputs],
            attributes={
                "event_id":
                    mlir_ir.StringAttr.get(self._event_id("release"),
                                           context=self.context)
            },
        )

    def trace(self):
        token = push_trace(self)
        try:
            returned = self.definition.provider()
        finally:
            pop_trace(token)
        self.finish(returned)

    def finish(self, returned=None):
        if self._finished:
            raise RuntimeError("PhysicalBuilder root is already finished")
        self._finished = True
        values = _flatten(returned) if returned is not None else ()
        operands = []
        for value in values:
            if isinstance(value, PhysicalState):
                if value.owner is not self:
                    raise TypeError("physical graph returned a foreign state")
                value._consume("phys.return")
            elif not isinstance(value,
                                PhysicalRecord) or value.owner is not self:
                raise TypeError(
                    "physical graph returns states and records only")
            operands.append(value.mlir_value)
        self._emit("phys.return", operands=operands)
        if any(state.is_live for state in self._states):
            raise RuntimeError(
                "physical graph leaves acquired state ownership live")
        function_type = mlir_ir.FunctionType.get(
            (), tuple(value.type for value in operands), context=self.context)
        with self.context:
            self.operation.attributes["function_type"] = mlir_ir.TypeAttr.get(
                function_type)
