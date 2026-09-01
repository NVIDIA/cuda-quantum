# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from .. import ir as mlir_ir

from ..programs.definition import DefinitionHandle
from ..devices.definition import Device


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


def _strings(context, values):
    return mlir_ir.ArrayAttr.get(
        [
            mlir_ir.StringAttr.get(str(value), context=context)
            for value in values
        ],
        context=context,
    )


def _binding_entry(context, **values):
    attributes = {}
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            attributes[key] = _strings(context, value)
        elif isinstance(value, int):
            attributes[key] = _i64(context, value)
        else:
            attributes[key] = mlir_ir.StringAttr.get(str(value),
                                                     context=context)
    return mlir_ir.DictAttr.get(attributes, context=context)


def _materialize_layered_device(transaction, device: Device):
    """Materialize the P1/P2 machines and their typed refinement."""

    from ..compiler.protocol_identity import protocol_definition_sha256

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
        for channel in device.logical._channels:
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
        ("logical_to_qec", logical_to_qec_symbol),
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


def materialize_device(transaction, device: Device):
    return _materialize_layered_device(transaction, device)
