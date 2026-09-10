# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import cudaq.mlir.ir as mlir_ir

from cudaq.logical.architecture.constraints import (
    Colocate,
    LocalPlacement,
    PlacementBinding,
    PlacementWitness,
    DistributedPlacement,
    TopologicalPlacement,
    TrajectoryPlacement,
)
from cudaq.logical.programs.definition import DefinitionHandle
from cudaq.logical.architecture.logical import LogicalMachine
from .context import CompilationContext
from .build import Build, EvidenceRecord, _placement_witness_sha256
from .pipeline import pipelines
from .placement_allocator import _Allocator
from .protocol_identity import (
    factory_protocol_semantics_sha256,
    protocol_definition_sha256,
)


def _walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


def _find_symbol(module, name: str, operation_name: str):
    for operation in _walk(module.operation):
        if operation.name != operation_name:
            continue
        attr = operation.attributes["sym_name"]
        if str(getattr(attr, "value", attr)).strip('"') == name:
            return operation
    raise KeyError(f"missing {operation_name} @{name}")


def _type_attr_value(operation, name: str):
    return mlir_ir.TypeAttr(operation.attributes[name]).value


def _symbol_name(attribute) -> str:
    raw = getattr(attribute, "value", attribute)
    if isinstance(raw, (tuple, list)):
        return str(raw[-1])
    value = str(raw).strip('"')
    if value.startswith("@"):
        value = value[1:]
    return value.split("::@")[-1]


def _as_machine(device) -> LogicalMachine:
    if isinstance(device, LogicalMachine):
        return device
    logical = getattr(device, "logical", None)
    if isinstance(logical, LogicalMachine):
        return logical
    raise TypeError(
        "placement requires a LogicalMachine or a Device with .logical")


class _P0ToP1:

    def __init__(self, build: Build, machine: LogicalMachine, constraints,
                 objective) -> None:
        self.source = build
        self.machine = machine
        self.constraints = constraints
        self.objective = str(objective or "first_fit")
        self.transaction = CompilationContext.replay(build)
        self.module = self.transaction.module
        self.context = self.transaction.context
        self.location = self.transaction.location
        self._programs = {
            _symbol_name(view.operation.attributes["sym_name"]): view.operation
            for view in self.module.body.operations
            if view.operation.name == "qlx.program"
        }
        try:
            self.program = self._programs[build.root.symbol]
        except KeyError as error:
            raise KeyError(
                f"missing qlx.program @{build.root.symbol}") from error
        self.function_type = _type_attr_value(self.program, "function_type")
        self.allocator = _Allocator(machine, constraints)
        self.domain_symbol = self._safe(machine.name)
        self.value_map = {}
        self.placement_map = {}
        self.bindings: list[PlacementBinding] = []
        self._space_by_placement = {}
        self._placement_count = 0
        self._site_count = 0
        self._stream_refs = {}
        self._structured_depth = 0
        self._call_stack = []
        self._call_instance_stack = []
        self._call_instance_count = 0
        self._p0_origin_cache = {}
        self._p0_origin_call_stack = []
        self._index_colocation_constraints()
        self._index_explicit_bindings()

    @staticmethod
    def _value_key(reference):
        identity = (reference.allocation
                    if reference.allocation is not None else reference.group)
        return identity, reference.path[0]

    def _index_colocation_constraints(self):
        groups = []
        for constraint in self.constraints:
            if not isinstance(constraint, Colocate):
                continue
            keys = set()
            for reference in constraint.values:
                if reference.program != self.source.root.symbol:
                    raise ValueError(
                        "cudaq.logical.colocate references a different P0 build"
                    )
                if len(reference.path) != 1:
                    raise ValueError(
                        "cudaq.logical.colocate currently requires top-level value groups"
                    )
                keys.add(self._value_key(reference))
            if len(keys) < 2:
                continue
            overlaps = [
                index for index, group in enumerate(groups) if group & keys
            ]
            if overlaps:
                merged = set(keys)
                for index in reversed(overlaps):
                    merged.update(groups.pop(index))
                groups.append(merged)
            else:
                groups.append(keys)

        self._colocate_for_value = {}
        self._colocate_demand = {}
        self._colocate_members = {}
        self._colocate_space = {}
        for group_index, keys in enumerate(groups):
            for key in keys:
                self._colocate_for_value[key] = group_index
            self._colocate_demand[group_index] = len(keys)
            self._colocate_members[group_index] = frozenset(keys)

    def _index_explicit_bindings(self):
        self._binding_for_value = {}
        descriptor_types = (
            LocalPlacement,
            DistributedPlacement,
            TrajectoryPlacement,
            TopologicalPlacement,
        )
        for descriptor in self.constraints:
            if not isinstance(descriptor, descriptor_types):
                continue
            reference = descriptor.value
            if reference.program != self.source.root.symbol:
                raise ValueError(
                    "placement descriptor references a different P0 build")
            key = self._value_key(reference)
            if key in self._binding_for_value:
                raise ValueError(
                    f"logical value {key} has several exact bindings")
            if key in self._colocate_for_value:
                raise ValueError(
                    "exact nonlocal placement cannot currently overlap cudaq.logical.colocate"
                )
            self._binding_for_value[key] = descriptor

    def _take_for_value(self, source_allocation, source_index):
        key = (source_allocation, source_index)
        group = self._colocate_for_value.get(key)
        if group is None:
            return self.allocator.take(value_keys=(key,))
        selected = self._colocate_space.get(group)
        if selected is None:
            space, slot = self.allocator.take(
                minimum_remaining=self._colocate_demand[group],
                value_keys=self._colocate_members[group],
            )
            self._colocate_space[group] = space
            return space, slot
        return self.allocator.take(
            space=selected,
            value_keys=(key,),
        )

    def _record_colocated_space(self, source_allocation, source_index, space):
        group = self._colocate_for_value.get((source_allocation, source_index))
        if group is None:
            return
        selected = self._colocate_space.setdefault(group, space)
        if selected.name != space.name:
            raise ValueError(
                "cudaq.logical.colocate conflicts with an existing explicit placement"
            )

    @staticmethod
    def _safe(name: str) -> str:
        return "".join(
            character if character.isalnum() or character in "_.$-" else "_"
            for character in name)

    def run(self) -> tuple[Any, PlacementWitness, str]:
        self._emit_domain()
        self._emit_kernel()
        profiles = [
            str(getattr(value, "value", value)).strip('"')
            for value in self.module.operation.attributes["qlx.profiles"]
        ]
        if "p1" not in profiles:
            profiles.append("p1")
        self.module.operation.attributes[
            "qlx.profiles"] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(profile, context=self.context)
                    for profile in profiles
                ],
                context=self.context,
            )
        witness = PlacementWitness(
            machine=self.domain_symbol,
            input_p0=self.source.root.symbol,
            bindings=tuple(self.bindings),
            relaxed_preferences=tuple(self.allocator.relaxed_preferences),
            objective=self.objective,
        )
        self.kernel.attributes["placement_witness_sha256"] = (
            mlir_ir.StringAttr.get(_placement_witness_sha256(witness),
                                   context=self.context))
        return self.module, witness, f"{self.source.root.symbol}_placed"

    def _emit_domain(self) -> None:
        with self.location:
            self.domain = mlir_ir.Operation.create(
                "lvm.domain",
                attributes={
                    "sym_name":
                        mlir_ir.StringAttr.get(self.domain_symbol,
                                               context=self.context),
                    "qlx.profile":
                        mlir_ir.StringAttr.get("p1", context=self.context),
                    "qlx.stage":
                        mlir_ir.StringAttr.get("p1", context=self.context),
                },
                regions=1,
                loc=self.location,
            )
            self.module.body.append(self.domain)
            self.domain_block = self.domain.regions[0].blocks.append()
        ip = mlir_ir.InsertionPoint(self.domain_block)
        for space in self.machine.spaces:
            attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(space.name, context=self.context),
                "capabilities":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.Attribute.parse(
                                f'#lvm.capability<"{item.key}">',
                                context=self.context)
                            for item in space.capabilities
                        ],
                        context=self.context,
                    ),
            }
            if space.capacity is not None:
                attrs["capacity"] = self._i64(space.capacity)
            if space.tags:
                attrs["tags"] = mlir_ir.ArrayAttr.get(
                    [
                        mlir_ir.StringAttr.get(tag, context=self.context)
                        for tag in space.tags
                    ],
                    context=self.context,
                )
            self._insert(ip, "lvm.space", attributes=attrs)
        for stream in self.machine.streams:
            attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(stream.name, context=self.context),
                "produces":
                    mlir_ir.FlatSymbolRefAttr.get(
                        getattr(stream.produces, "name", str(stream.produces)),
                        context=self.context,
                    ),
            }
            # The lvm.stream `capacity` attribute is the in-flight buffer depth,
            # spelled `buffer_size` on the Python authoring surface.
            if stream.buffer_size is not None:
                attrs["capacity"] = self._i64(stream.buffer_size)
            if stream.produced_by is not None:
                attrs["produced_by"] = mlir_ir.FlatSymbolRefAttr.get(
                    stream.produced_by.name,
                    context=self.context,
                )
                attrs["produced_by_sha256"] = mlir_ir.StringAttr.get(
                    protocol_definition_sha256(stream.produced_by),
                    context=self.context,
                )
                attrs["producer_identity"] = mlir_ir.StringAttr.get(
                    stream.produced_by.name,
                    context=self.context,
                )
                attrs["producer_semantics_sha256"] = mlir_ir.StringAttr.get(
                    factory_protocol_semantics_sha256(stream.produced_by),
                    context=self.context,
                )
            if stream.transfer is not None:
                attrs["transfer"] = mlir_ir.FlatSymbolRefAttr.get(
                    stream.transfer.name,
                    context=self.context,
                )
                attrs["transfer_sha256"] = mlir_ir.StringAttr.get(
                    protocol_definition_sha256(stream.transfer),
                    context=self.context,
                )
            if stream.region is not None:
                attrs["backing_region"] = mlir_ir.FlatSymbolRefAttr.get(
                    stream.region.name, context=self.context)
            if stream.external:
                attrs["external"] = mlir_ir.UnitAttr.get(context=self.context)
            self._insert(ip, "lvm.stream", attributes=attrs)
        for channel in self.machine.channels:
            attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(channel.name, context=self.context),
                "from":
                    mlir_ir.FlatSymbolRefAttr.get(channel.source.name,
                                                  context=self.context),
                "to":
                    mlir_ir.FlatSymbolRefAttr.get(channel.destination.name,
                                                  context=self.context),
                "capabilities":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.Attribute.parse(
                                f'#lvm.capability<"{item.key}">',
                                context=self.context)
                            for item in channel.capabilities
                        ],
                        context=self.context,
                    ),
                "direction":
                    mlir_ir.StringAttr.get(channel.direction,
                                           context=self.context),
            }
            if channel.capacity is not None:
                attrs["capacity"] = self._i64(channel.capacity)
            self._insert(ip, "lvm.channel", attributes=attrs)
        self.domain_ip = ip

    def _i64(self, value: int):
        return mlir_ir.IntegerAttr.get(
            mlir_ir.IntegerType.get_signless(64, context=self.context), value)

    def _insert(self,
                ip,
                name,
                *,
                operands=(),
                results=(),
                attributes=None,
                regions=0):
        with self.location:
            operation = mlir_ir.Operation.create(
                name,
                operands=list(operands),
                results=list(results),
                attributes=dict(attributes or {}),
                regions=regions,
                loc=self.location,
            )
            ip.insert(operation)
        return operation

    def _new_placement(
        self,
        *,
        source_allocation=None,
        source_group=None,
        source_index=None,
    ) -> tuple[Any, str]:
        source_identity = (source_allocation
                           if source_allocation is not None else source_group)
        key = (source_identity, source_index)
        descriptor = self._binding_for_value.get(key)
        if isinstance(descriptor, LocalPlacement):
            space, slot = self.allocator.take(
                space=descriptor.space,
                slot=descriptor.slot,
                value_keys=(key,),
            )
        elif isinstance(descriptor, DistributedPlacement):
            space, slot = self.allocator.take(
                space=descriptor.spaces[0],
                value_keys=(key,),
            )
        elif isinstance(descriptor, TrajectoryPlacement):
            space, slot = self.allocator.take(
                space=descriptor.segments[0],
                value_keys=(key,),
            )
        elif isinstance(descriptor, TopologicalPlacement):
            space, slot = self.allocator.take(
                space=descriptor.space,
                value_keys=(key,),
            )
        else:
            space, slot = self._take_for_value(
                source_identity,
                source_index,
            )
        symbol = f"p{self._placement_count}"
        self._placement_count += 1
        nonlocal_binding = isinstance(
            descriptor,
            (DistributedPlacement, TrajectoryPlacement, TopologicalPlacement),
        )
        explicit_placement = nonlocal_binding
        if explicit_placement:
            attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(symbol, context=self.context),
                "space":
                    mlir_ir.FlatSymbolRefAttr.get(space.name,
                                                  context=self.context),
                "slot":
                    self._i64(slot),
            }
            if nonlocal_binding:
                attrs["binding"] = self._binding_attr(descriptor, space, slot)
            if source_allocation is not None:
                attrs["source_value"] = mlir_ir.ArrayAttr.get(
                    [self._i64(source_allocation),
                     self._i64(source_index)],
                    context=self.context,
                )
            self._insert(
                self.domain_ip,
                "lvm.placement",
                attributes=attrs,
            )
        with self.context:
            reference = mlir_ir.SymbolRefAttr.get(
                [
                    self.domain_symbol,
                    symbol if explicit_placement else space.name
                ],
                context=self.context,
            )
        binding = PlacementBinding(
            symbol,
            space.name,
            slot,
            source_allocation=source_allocation,
            source_group=source_group,
            source_path=(() if source_index is None else (source_index,)),
            binding_kind=self._binding_kind(descriptor),
            binding_data=self._binding_data(descriptor),
        )
        self.bindings.append(binding)
        self._space_by_placement[binding.placement] = binding.space
        return reference, symbol

    @staticmethod
    def _binding_kind(descriptor):
        if isinstance(descriptor, DistributedPlacement):
            return "distributed"
        if isinstance(descriptor, TrajectoryPlacement):
            return "trajectory"
        if isinstance(descriptor, TopologicalPlacement):
            return "topological_record"
        return "local"

    @staticmethod
    def _binding_data(descriptor):
        if isinstance(descriptor, LocalPlacement):
            return (("witness",
                     descriptor.witness),) if descriptor.witness else ()
        if isinstance(descriptor, DistributedPlacement):
            return (
                ("spaces", tuple(space.name for space in descriptor.spaces)),
                ("support_views", descriptor.support_views),
                ("ownership_witness", descriptor.ownership_witness),
                ("link_obligations", descriptor.link_obligations),
            )
        if isinstance(descriptor, TrajectoryPlacement):
            return (
                ("segments",
                 tuple(space.name for space in descriptor.segments)),
                ("transition_events", descriptor.transition_events),
                ("continuity_witness", descriptor.continuity_witness),
            )
        if isinstance(descriptor, TopologicalPlacement):
            return (
                ("record", descriptor.record),
                ("frontier", descriptor.frontier),
                ("support_witness", descriptor.support_witness),
                ("observable_witness", descriptor.observable_witness),
            )
        return ()

    def _binding_attr(self, descriptor, space, slot):
        values = {
            "kind":
                mlir_ir.StringAttr.get(self._binding_kind(descriptor),
                                       context=self.context)
        }
        if descriptor is None or isinstance(descriptor, LocalPlacement):
            values.update({
                "space":
                    mlir_ir.FlatSymbolRefAttr.get(space.name,
                                                  context=self.context),
                "slot":
                    self._i64(slot),
            })
            if isinstance(descriptor, LocalPlacement) and descriptor.witness:
                values["witness"] = mlir_ir.StringAttr.get(descriptor.witness,
                                                           context=self.context)
        elif isinstance(descriptor, DistributedPlacement):
            values.update({
                "spaces":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.FlatSymbolRefAttr.get(item.name,
                                                          context=self.context)
                            for item in descriptor.spaces
                        ],
                        context=self.context,
                    ),
                "support_views":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.StringAttr.get(item, context=self.context)
                            for item in descriptor.support_views
                        ],
                        context=self.context,
                    ),
                "ownership_witness":
                    mlir_ir.StringAttr.get(descriptor.ownership_witness,
                                           context=self.context),
                "link_obligations":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.StringAttr.get(item, context=self.context)
                            for item in descriptor.link_obligations
                        ],
                        context=self.context,
                    ),
            })
        elif isinstance(descriptor, TrajectoryPlacement):
            values.update({
                "segments":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.FlatSymbolRefAttr.get(item.name,
                                                          context=self.context)
                            for item in descriptor.segments
                        ],
                        context=self.context,
                    ),
                "transition_events":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.StringAttr.get(item, context=self.context)
                            for item in descriptor.transition_events
                        ],
                        context=self.context,
                    ),
                "continuity_witness":
                    mlir_ir.StringAttr.get(descriptor.continuity_witness,
                                           context=self.context),
            })
        elif isinstance(descriptor, TopologicalPlacement):
            values.update({
                "record":
                    mlir_ir.StringAttr.get(descriptor.record,
                                           context=self.context),
                "frontier":
                    mlir_ir.ArrayAttr.get(
                        [
                            mlir_ir.StringAttr.get(item, context=self.context)
                            for item in descriptor.frontier
                        ],
                        context=self.context,
                    ),
                "support_witness":
                    mlir_ir.StringAttr.get(descriptor.support_witness,
                                           context=self.context),
                "observable_witness":
                    mlir_ir.StringAttr.get(descriptor.observable_witness,
                                           context=self.context),
            })
        return mlir_ir.DictAttr.get(values, context=self.context)

    def _next_site(self):
        site = self._site_count
        self._site_count += 1
        return self._i64(site)

    def _placed_type(self, placement) -> Any:
        return mlir_ir.Type.parse(f"!lvm.logical_qubit<{placement}>",
                                  context=self.context)

    @staticmethod
    def _quoted_parameter(type_, marker):
        text = str(type_)
        start = text.find(marker)
        if start < 0:
            return None
        start += len(marker)
        end = text.find('"', start)
        return text[start:end]

    def _stream_for(self, kind: str):
        cached = self._stream_refs.get(kind)
        if cached is not None:
            return cached
        candidates = tuple(candidate for candidate in self.machine.streams
                           if getattr(candidate.produces, "name",
                                      str(candidate.produces)) == kind)
        if not candidates:
            raise ValueError(
                f"machine @{self.machine.name} has no stream producing {kind!r}"
            )
        if len(candidates) != 1:
            raise ValueError(
                f"machine @{self.machine.name} has ambiguous streams producing "
                f"{kind!r}: {tuple(stream.name for stream in candidates)!r}")
        stream = candidates[0]
        with self.context:
            reference = mlir_ir.SymbolRefAttr.get(
                [self.domain_symbol, stream.name], context=self.context)
        self._stream_refs[kind] = reference
        return reference

    def _bound_resource_type(self, kind: str, stream):
        return mlir_ir.Type.parse(f'!lvm.logical_resource<"{kind}", {stream}>',
                                  context=self.context)

    def _bound_event_type(self, kind: str, stream):
        payload = self._bound_resource_type(kind, stream)
        return mlir_ir.Type.parse(
            f'!event.handle<{payload}, "linear", {stream}>',
            context=self.context,
        )

    def _placed_auxiliary_type(self, type_):
        text = str(type_)
        if text.startswith("!qlx.logical_frame<"):
            domain = self._quoted_parameter(type_, '!qlx.logical_frame<"')
            return mlir_ir.Type.parse(f'!lvm.logical_frame<"{domain}">',
                                      context=self.context)
        kind = self._quoted_parameter(type_,
                                      '!event.handle<!qlx.logical_resource<"')
        if kind is not None:
            return self._bound_event_type(kind, self._stream_for(kind))
        kind = self._quoted_parameter(type_, '!qlx.logical_resource<"')
        if kind is not None:
            return self._bound_resource_type(kind, self._stream_for(kind))
        return type_

    def _emit_kernel(self) -> None:
        input_types = []
        source_block = self.program.regions[0].blocks[0]
        for argument_index, argument in enumerate(source_block.arguments):
            if str(argument.type) == "!qlx.logical_qubit":
                placement, _ = self._new_placement(
                    source_group="argument",
                    source_index=argument_index,
                )
                placed_type = self._placed_type(placement)
                input_types.append(placed_type)
                self.placement_map[argument] = placement
            else:
                input_types.append(self._placed_auxiliary_type(argument.type))
        result_types = [
            self._placed_auxiliary_type(type_)
            for type_ in self.function_type.results
        ]
        # Quantum returns are assigned from the converted lvm.return values;
        # the common algorithmic entry points currently return classical data.
        function_type = mlir_ir.FunctionType.get(input_types,
                                                 result_types,
                                                 context=self.context)
        with self.context:
            function_type_attr = mlir_ir.TypeAttr.get(function_type)
        kernel_symbol = f"{self.source.root.symbol}_placed"
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(kernel_symbol, context=self.context),
            "domain":
                mlir_ir.FlatSymbolRefAttr.get(self.domain_symbol,
                                              context=self.context),
            "function_type":
                function_type_attr,
            "input_p0":
                mlir_ir.FlatSymbolRefAttr.get(self.source.root.symbol,
                                              context=self.context),
            "qlx.profile":
                mlir_ir.StringAttr.get("p1", context=self.context),
            "qlx.stage":
                mlir_ir.StringAttr.get("p1", context=self.context),
        }
        if "specialization" in self.program.attributes:
            attrs["specialization"] = self.program.attributes["specialization"]
        if "estimate_only" in self.program.attributes:
            attrs["estimate_only"] = self.program.attributes["estimate_only"]
        with self.location:
            kernel = mlir_ir.Operation.create("lvm.kernel",
                                              attributes=attrs,
                                              regions=1,
                                              loc=self.location)
            self.module.body.append(kernel)
            block = kernel.regions[0].blocks.append(*input_types)
        self.kernel = kernel
        for source, target in zip(source_block.arguments, block.arguments):
            self.value_map[source] = target
        self.kernel_ip = mlir_ir.InsertionPoint(block)
        for child in source_block.operations:
            self._convert(child.operation)

    def _mapped(self, value):
        try:
            return self.value_map[value]
        except KeyError as exc:
            raise NotImplementedError(
                f"P0 value {value} has no P1 mapping") from exc

    def _quantum_inputs(self, operation):
        return [
            value for value in operation.operands
            if str(value.type) == "!qlx.logical_qubit"
        ]

    def _placements_for(self, quantum_inputs):
        return [self.placement_map[value] for value in quantum_inputs]

    def _copy_parameters(self, operation, attrs):
        if "parameters" in operation.attributes:
            attrs["parameters"] = operation.attributes["parameters"]

    @staticmethod
    def _parameter_int(operation, name, default=0):
        try:
            value = operation.attributes["parameters"][name]
        except (KeyError, IndexError, TypeError):
            return default
        return int(getattr(value, "value", value))

    def _space_name(self, placement):
        name = _symbol_name(placement)
        return self._space_by_placement.get(name, name)

    def _remote_observable_channel(self, placements):
        spaces = tuple(
            dict.fromkeys(self._space_name(item) for item in placements))
        if len(spaces) != 2:
            raise NotImplementedError(
                "remote observable lowering currently requires exactly two "
                "logical regions")
        connected = tuple(
            channel for channel in self.machine.channels if {
                getattr(channel.source, "name", None),
                getattr(channel.destination, "name", None),
            } == set(spaces))
        if not connected:
            raise ValueError(
                "cross-region MPP has no logical channel connecting "
                f"@{spaces[0]} and @{spaces[1]}")
        capability_key = "qlx.machine/observable_remote"
        capable = tuple(channel for channel in connected if any(
            item.key == capability_key for item in channel.capabilities))
        if not capable:
            raise ValueError("cross-region MPP requires a channel advertising "
                             f"{capability_key!r}")
        directed = tuple(channel for channel in capable
                         if str(channel.direction) == "bidirectional")
        if not directed:
            raise ValueError(
                "symmetric cross-region MPP requires a bidirectional "
                "observable channel")
        available = tuple(channel for channel in directed
                          if channel.capacity is None or channel.capacity >= 1)
        if not available:
            raise ValueError(
                "cross-region MPP channel has no available capacity")
        if len(available) != 1:
            names = ", ".join(
                f"@{channel.name}"
                for channel in sorted(available, key=lambda item: item.name))
            raise ValueError(
                "cross-region MPP has several eligible observable channels; "
                f"selection is ambiguous: {names}")
        return available[0], spaces

    def _infer_placement(self, value):
        known = self.placement_map.get(value)
        if known is not None:
            return known
        owner = value.owner
        name = getattr(owner, "name", "")
        if name in {"qlx.apply", "qlx.instrument", "qlx.idle"}:
            quantum_results = [
                result for result in owner.results
                if str(result.type) == "!qlx.logical_qubit"
            ]
            quantum_inputs = [
                operand for operand in owner.operands
                if str(operand.type) == "!qlx.logical_qubit"
            ]
            index = quantum_results.index(value)
            return self._infer_placement(quantum_inputs[index])
        if name == "qlx.consume_resource":
            index = list(owner.results).index(value)
            return self._infer_placement(list(owner.operands)[index + 1])
        if name == "qlx.call":
            index = list(owner.results).index(value)
            return self._infer_call_result_placement(owner, index)
        if name == "cflow.if":
            index = list(owner.results).index(value)
            yielded = [
                region.blocks[0].operations[-1].operation.operands[index]
                for region in owner.regions
            ]
            placements = [self._infer_placement(item) for item in yielded]
            if any(item != placements[0] for item in placements[1:]):
                raise ValueError("branch joins cannot change logical placement")
            return placements[0]
        if name in {"cflow.repeat", "cflow.while"}:
            index = list(owner.results).index(value)
            return self._infer_placement(list(owner.operands)[index])
        raise NotImplementedError(f"cannot infer placement for {value}")

    def _callee_program(self, operation):
        callee = _symbol_name(operation.attributes["callee"])
        try:
            return self._programs[callee]
        except KeyError as error:
            raise KeyError(f"missing qlx.program @{callee}") from error

    def _infer_call_result_placement(self, operation, result_index):
        callee = self._callee_program(operation)
        origin = self._callee_result_origin(callee, result_index)
        if origin is None:
            raise NotImplementedError(
                "placed helper calls do not yet support returning a "
                "helper-local logical allocation")
        return self._infer_placement(operation.operands[origin])

    def _callee_result_origin(self, callee, result_index):
        callee_name = _symbol_name(callee.attributes["sym_name"])
        if callee_name in self._p0_origin_call_stack:
            chain = " -> ".join((*self._p0_origin_call_stack, callee_name))
            raise ValueError(
                f"recursive P0 helper calls are unsupported: {chain}")
        self._p0_origin_call_stack.append(callee_name)
        block = callee.regions[0].blocks[0]
        try:
            for index, argument in enumerate(block.arguments):
                self._p0_origin_cache.setdefault(argument, index)
            returned = block.operations[-1].operation
            if returned.name != "qlx.return":
                raise ValueError(
                    f"P0 helper @{callee_name} does not end in qlx.return")
            return self._p0_input_origin(returned.operands[result_index], set())
        finally:
            self._p0_origin_call_stack.pop()

    def _p0_input_origin(self, value, active):
        # Logical owner chains are routinely thousands of SSA edges long in
        # specialized arithmetic helpers.  Follow the single-predecessor path
        # iteratively and path-compress it into the existing cache; recurse
        # only across the genuinely branching cflow.if case.
        trail = []
        try:
            current = value
            while current not in self._p0_origin_cache:
                if current in active:
                    raise ValueError(
                        "P0 helper logical-owner lineage is cyclic")
                active.add(current)
                trail.append(current)
                owner = current.owner
                name = getattr(owner, "name", "")
                if name in {"qlx.apply", "qlx.instrument", "qlx.idle"}:
                    quantum_results = tuple(
                        result for result in owner.results
                        if str(result.type) == "!qlx.logical_qubit")
                    quantum_inputs = tuple(
                        operand for operand in owner.operands
                        if str(operand.type) == "!qlx.logical_qubit")
                    current = quantum_inputs[quantum_results.index(current)]
                    continue
                if name == "qlx.consume_resource":
                    index = tuple(owner.results).index(current)
                    current = tuple(owner.operands)[index + 1]
                    continue
                if name == "qlx.call":
                    result_index = tuple(owner.results).index(current)
                    callee = self._callee_program(owner)
                    input_index = self._callee_result_origin(
                        callee, result_index)
                    if input_index is None:
                        origin = None
                        break
                    current = tuple(owner.operands)[input_index]
                    continue
                if name == "cflow.if":
                    index = tuple(owner.results).index(current)
                    yielded = tuple(region.blocks[0].operations[-1].operation.
                                    operands[index] for region in owner.regions)
                    origins = tuple(
                        self._p0_input_origin(item, active) for item in yielded)
                    if any(item != origins[0] for item in origins[1:]):
                        raise ValueError(
                            "P0 helper branch joins change logical-owner "
                            "origin")
                    origin = origins[0]
                    break
                if name in {"cflow.repeat", "cflow.while"}:
                    index = tuple(owner.results).index(current)
                    current = tuple(owner.operands)[index]
                    continue
                if name == "qlx.prepare":
                    origin = None
                    break
                raise NotImplementedError(
                    f"cannot derive P0 helper owner origin for {current}")
            else:
                origin = self._p0_origin_cache[current]
            for traced in trail:
                self._p0_origin_cache[traced] = origin
            return origin
        finally:
            for traced in reversed(trail):
                active.remove(traced)

    def _infer_p0_value_placement(self, value, environment, active):
        known = environment.get(value)
        if known is not None:
            return known
        known = self.placement_map.get(value)
        if known is not None:
            return known
        if value in active:
            raise ValueError("P0 helper logical-owner lineage is cyclic")
        active.add(value)
        try:
            owner = value.owner
            name = getattr(owner, "name", "")
            if name in {"qlx.apply", "qlx.instrument", "qlx.idle"}:
                quantum_results = tuple(
                    result for result in owner.results
                    if str(result.type) == "!qlx.logical_qubit")
                quantum_inputs = tuple(
                    operand for operand in owner.operands
                    if str(operand.type) == "!qlx.logical_qubit")
                return self._infer_p0_value_placement(
                    quantum_inputs[quantum_results.index(value)],
                    environment,
                    active,
                )
            if name == "qlx.consume_resource":
                index = tuple(owner.results).index(value)
                return self._infer_p0_value_placement(
                    tuple(owner.operands)[index + 1], environment, active)
            if name == "qlx.call":
                result_index = tuple(owner.results).index(value)
                callee = self._callee_program(owner)
                input_index = self._callee_result_origin(callee, result_index)
                if input_index is None:
                    raise NotImplementedError(
                        "placed helper calls do not yet support returning a "
                        "helper-local logical allocation")
                return self._infer_p0_value_placement(
                    tuple(owner.operands)[input_index], environment, active)
            if name == "cflow.if":
                index = tuple(owner.results).index(value)
                yielded = tuple(
                    region.blocks[0].operations[-1].operation.operands[index]
                    for region in owner.regions)
                placements = tuple(
                    self._infer_p0_value_placement(item, environment, active)
                    for item in yielded)
                if any(item != placements[0] for item in placements[1:]):
                    raise ValueError(
                        "P0 helper branch joins cannot change placement")
                return placements[0]
            if name in {"cflow.repeat", "cflow.while"}:
                index = tuple(owner.results).index(value)
                return self._infer_p0_value_placement(
                    tuple(owner.operands)[index], environment, active)
            if name == "qlx.prepare":
                raise NotImplementedError(
                    "placed helper calls do not yet support helper-local "
                    "logical allocation")
            raise NotImplementedError(
                f"cannot infer P0 helper placement for {value}")
        finally:
            active.remove(value)

    def _convert_call(self, operation):
        callee = self._callee_program(operation)
        callee_name = _symbol_name(operation.attributes["callee"])
        if callee_name in self._call_stack:
            chain = " -> ".join((*self._call_stack, callee_name))
            raise ValueError(
                f"recursive P0 helper calls are unsupported: {chain}")

        source_block = callee.regions[0].blocks[0]
        mapped_inputs = tuple(
            self._mapped(value) for value in operation.operands)
        result_types = []
        result_placements = []
        for index, result in enumerate(operation.results):
            if str(result.type) == "!qlx.logical_qubit":
                placement = self._infer_call_result_placement(operation, index)
                result_types.append(self._placed_type(placement))
                result_placements.append(placement)
            else:
                result_types.append(self._placed_auxiliary_type(result.type))
                result_placements.append(None)

        scope = self._call_instance_count
        self._call_instance_count += 1
        placed = self._insert(
            self.kernel_ip,
            "lvm.call",
            operands=mapped_inputs,
            results=result_types,
            attributes={
                "callee": operation.attributes["callee"],
                "scope": self._i64(scope),
            },
            regions=1,
        )
        instance = f"{callee_name}:{scope}"
        with self.location:
            target_block = placed.regions[0].blocks.append(
                *(value.type for value in mapped_inputs))

        outer_values = self.value_map
        outer_placements = self.placement_map
        outer_ip = self.kernel_ip
        self.value_map = dict(outer_values)
        self.placement_map = dict(outer_placements)
        for source, target, operand in zip(source_block.arguments,
                                           target_block.arguments,
                                           operation.operands):
            self.value_map[source] = target
            if str(source.type) == "!qlx.logical_qubit":
                self.placement_map[source] = self._infer_placement(operand)
        self.kernel_ip = mlir_ir.InsertionPoint(target_block)
        self._call_stack.append(callee_name)
        self._call_instance_stack.append(instance)
        try:
            for child in source_block.operations:
                self._convert(child.operation)
        finally:
            self._call_instance_stack.pop()
            self._call_stack.pop()
            self.kernel_ip = outer_ip
            self.value_map = outer_values
            self.placement_map = outer_placements

        for old, result, placement in zip(operation.results, placed.results,
                                          result_placements):
            self.value_map[old] = result
            if placement is not None:
                self.placement_map[old] = placement

    def _placed_result_type(self, value):
        if str(value.type) == "!qlx.logical_qubit":
            return self._placed_type(self._infer_placement(value))
        return self._placed_auxiliary_type(value.type)

    def _convert_region(self, source_region, target_region, *, init_values=()):
        source_block = source_region.blocks[0]
        with self.location:
            target_block = target_region.blocks.append(
                *(self._mapped(value).type for value in init_values))
        for source, target, init in zip(source_block.arguments,
                                        target_block.arguments, init_values):
            self.value_map[source] = target
            if str(source.type) == "!qlx.logical_qubit":
                self.placement_map[source] = self._infer_placement(init)
        saved_ip = self.kernel_ip
        self.kernel_ip = mlir_ir.InsertionPoint(target_block)
        self._structured_depth += 1
        try:
            for child in source_block.operations:
                nested = child.operation
                if nested.name == "cflow.yield":
                    self._insert(
                        self.kernel_ip,
                        "cflow.yield",
                        operands=[
                            self._mapped(value) for value in nested.operands
                        ],
                    )
                elif nested.name == "cflow.while_condition":
                    self._insert(
                        self.kernel_ip,
                        "cflow.while_condition",
                        operands=[
                            self._mapped(value) for value in nested.operands
                        ],
                    )
                else:
                    self._convert(nested)
        finally:
            self._structured_depth -= 1
            self.kernel_ip = saved_ip
        return target_block

    def _convert_event_take_region(self, source_region, target_region, *,
                                   alternative_type, carries):
        source_block = source_region.blocks[0]
        argument_types = [
            alternative_type,
            *(self._mapped(value).type for value in carries),
        ]
        with self.location:
            target_block = target_region.blocks.append(*argument_types)
        for index, (source, target) in enumerate(
                zip(source_block.arguments, target_block.arguments)):
            self.value_map[source] = target
            if index and str(source.type) == "!qlx.logical_qubit":
                self.placement_map[source] = self._infer_placement(
                    carries[index - 1])
        saved_ip = self.kernel_ip
        self.kernel_ip = mlir_ir.InsertionPoint(target_block)
        self._structured_depth += 1
        try:
            for child in source_block.operations:
                nested = child.operation
                if nested.name == "event.yield":
                    self._insert(
                        self.kernel_ip,
                        "event.yield",
                        operands=[
                            self._mapped(value) for value in nested.operands
                        ],
                    )
                else:
                    self._convert(nested)
        finally:
            self._structured_depth -= 1
            self.kernel_ip = saved_ip

    def _convert(self, operation):
        name = operation.name
        if name == "qlx.return":
            self._insert(
                self.kernel_ip,
                "lvm.yield" if self._call_stack else "lvm.return",
                operands=[self._mapped(value) for value in operation.operands],
            )
            return
        if name == "qlx.call":
            self._convert_call(operation)
            return
        if name == "arith.constant":
            new = self._insert(
                self.kernel_ip,
                name,
                results=[result.type for result in operation.results],
                attributes=dict(operation.attributes),
            )
            for old, result in zip(operation.results, new.results):
                self.value_map[old] = result
            return
        if name == "qlx.xor":
            new = self._insert(
                self.kernel_ip,
                "lvm.xor",
                operands=[self._mapped(value) for value in operation.operands],
                results=[operation.result.type],
            )
            self.value_map[operation.result] = new.result
            return
        if name == "qlx.prepare":
            allocation = (int(operation.attributes["allocation"])
                          if "allocation" in operation.attributes else None)
            value_index = (int(operation.attributes["value_index"])
                           if "value_index" in operation.attributes else None)
            if self._call_instance_stack:
                source_group = ("call/" + "/".join(self._call_instance_stack) +
                                (f"/allocation:{allocation}"
                                 if allocation is not None else "/allocation"))
                source_allocation = None
            else:
                source_group = (self.source.values[allocation].name
                                if allocation is not None else None)
                source_allocation = allocation
            placement, placement_name = self._new_placement(
                source_allocation=source_allocation,
                source_group=source_group,
                source_index=value_index,
            )
            binding = self.bindings[-1]
            new = self._insert(
                self.kernel_ip,
                "lvm.prepare",
                results=[self._placed_type(placement)],
                attributes={
                    "state":
                        operation.attributes["state"],
                    "at":
                        placement,
                    "site":
                        self._next_site(),
                    "placement_owner":
                        mlir_ir.StringAttr.get(placement_name,
                                               context=self.context),
                    "placement_slot":
                        self._i64(binding.slot),
                    "source_allocation":
                        self._i64(-1 if source_allocation is
                                  None else source_allocation),
                    "source_group":
                        mlir_ir.StringAttr.get(
                            "" if source_group is None else source_group,
                            context=self.context),
                    "source_path":
                        mlir_ir.DenseI64ArrayAttr.get(binding.source_path,
                                                      context=self.context),
                },
            )
            self.value_map[operation.result] = new.result
            self.placement_map[operation.result] = placement
            return
        if name == "qlx.measure":
            placement = self.placement_map[operation.operands[0]]
            new = self._insert(
                self.kernel_ip,
                "lvm.measure",
                operands=[self._mapped(operation.operands[0])],
                results=[operation.result.type],
                attributes={
                    "basis": operation.attributes["basis"],
                    "at": placement,
                    "site": self._next_site(),
                },
            )
            self.value_map[operation.result] = new.result
            return
        if name == "qlx.instrument":
            quantum_inputs = self._quantum_inputs(operation)
            objective = operation.attributes["instrument"]
            placements = self._placements_for(quantum_inputs)
            result_types = []
            quantum_index = 0
            for result in operation.results:
                if str(result.type) == "!qlx.logical_qubit":
                    result_types.append(
                        self._placed_type(placements[quantum_index]))
                    quantum_index += 1
                else:
                    result_types.append(result.type)
            spaces = tuple(
                dict.fromkeys(self._space_name(item) for item in placements))
            remote = len(spaces) > 1
            objective_text = str(objective)
            if remote and objective_text not in {
                    "#qlx.instrument<mpp>",
                    '#qlx.instrument<"mpp">',
            }:
                raise NotImplementedError(
                    "cross-region logical instruments require an explicit "
                    "communication lowering; this slice supports uniform "
                    "two-body X or Z MPP")
            if remote and self._structured_depth:
                raise NotImplementedError(
                    "cross-region communication inside structured control is "
                    "not supported by this lowering slice")
            x_mask = self._parameter_int(operation, "x_mask")
            z_mask = self._parameter_int(operation, "z_mask")
            if remote and (len(quantum_inputs) != 2 or (x_mask, z_mask) not in {
                (0b11, 0),
                (0, 0b11),
            }):
                raise NotImplementedError(
                    "cross-region observable lowering currently supports only "
                    "an exact uniform two-body XX or ZZ product")
            attrs = {
                "instrument":
                    objective,
                "placements":
                    mlir_ir.ArrayAttr.get(placements, context=self.context),
                "site":
                    self._next_site(),
            }
            if remote:
                channel, spaces = self._remote_observable_channel(placements)
                with self.context:
                    channel_ref = mlir_ir.SymbolRefAttr.get(
                        [self.domain_symbol, channel.name],
                        context=self.context,
                    )
                    endpoint_refs = [
                        mlir_ir.SymbolRefAttr.get(
                            [self.domain_symbol, space],
                            context=self.context,
                        ) for space in spaces
                    ]
                attrs.update({
                    "channel":
                        channel_ref,
                    "channel_capability":
                        mlir_ir.Attribute.parse(
                            '#lvm.capability<"qlx.machine/observable_remote">',
                            context=self.context,
                        ),
                    "endpoints":
                        mlir_ir.ArrayAttr.get(
                            endpoint_refs,
                            context=self.context,
                        ),
                })
            self._copy_parameters(operation, attrs)
            new = self._insert(
                self.kernel_ip,
                "lvm.instrument",
                operands=[self._mapped(value) for value in operation.operands],
                results=result_types,
                attributes=attrs,
            )
            quantum_index = 0
            for old, result in zip(operation.results, new.results):
                self.value_map[old] = result
                if str(old.type) == "!qlx.logical_qubit":
                    self.placement_map[old] = placements[quantum_index]
                    quantum_index += 1
            return
        if name == "qlx.apply":
            quantum_inputs = self._quantum_inputs(operation)
            placements = self._placements_for(quantum_inputs)
            spaces = tuple(
                dict.fromkeys(self._space_name(item) for item in placements))
            if len(spaces) > 1:
                raise NotImplementedError(
                    "cross-region logical actions are not native; select an "
                    "explicit distributed protocol or move state first")
            result_types = []
            quantum_index = 0
            for result in operation.results:
                if str(result.type) == "!qlx.logical_qubit":
                    result_types.append(
                        self._placed_type(placements[quantum_index]))
                    quantum_index += 1
                else:
                    result_types.append(result.type)
            attrs = {
                "action":
                    operation.attributes["action"],
                "placements":
                    mlir_ir.ArrayAttr.get(placements, context=self.context),
                "site":
                    self._next_site(),
            }
            self._copy_parameters(operation, attrs)
            new = self._insert(
                self.kernel_ip,
                "lvm.apply",
                operands=[self._mapped(value) for value in operation.operands],
                results=result_types,
                attributes=attrs,
            )
            q_index = 0
            for old, result in zip(operation.results, new.results):
                self.value_map[old] = result
                if str(old.type) == "!qlx.logical_qubit":
                    self.placement_map[old] = placements[q_index]
                    q_index += 1
            return
        if name == "qlx.idle":
            quantum_inputs = list(operation.operands[:-1])
            placements = self._placements_for(quantum_inputs)
            new = self._insert(
                self.kernel_ip,
                "lvm.idle",
                operands=[
                    *(self._mapped(value) for value in quantum_inputs),
                    self._mapped(operation.operands[-1])
                ],
                results=[
                    self._placed_type(placement) for placement in placements
                ],
                attributes={
                    "placements":
                        mlir_ir.ArrayAttr.get(placements, context=self.context),
                },
            )
            for old, result, placement in zip(operation.results, new.results,
                                              placements):
                self.value_map[old] = result
                self.placement_map[old] = placement
            return
        if name == "qlx.discard":
            quantum_inputs = list(operation.operands)
            placements = self._placements_for(quantum_inputs)
            attrs = {
                "placements":
                    mlir_ir.ArrayAttr.get(placements, context=self.context)
            }
            if "reason" in operation.attributes:
                attrs["reason"] = operation.attributes["reason"]
            self._insert(
                self.kernel_ip,
                "lvm.discard",
                operands=[self._mapped(value) for value in quantum_inputs],
                attributes=attrs,
            )
            return
        if name == "qlx.resource_request":
            kind = str(operation.attributes["kind"]).strip('"')
            stream = self._stream_for(kind)
            new = self._insert(
                self.kernel_ip,
                "lvm.resource_request",
                results=[self._bound_event_type(kind, stream)],
                attributes={
                    "kind": operation.attributes["kind"],
                    "stream": stream
                },
            )
            self.value_map[operation.result] = new.result
            return
        if name == "event.test":
            new = self._insert(
                self.kernel_ip,
                "event.test",
                operands=[self._mapped(operation.operands[0])],
                results=[operation.result.type],
            )
            self.value_map[operation.result] = new.result
            return
        if name == "event.poll":
            new = self._insert(
                self.kernel_ip,
                "event.poll",
                operands=[self._mapped(operation.operands[0])],
                results=[operation.result.type],
            )
            self.value_map[operation.result] = new.result
            return
        if name == "event.is":
            new = self._insert(
                self.kernel_ip,
                "event.is",
                operands=[self._mapped(operation.operands[0])],
                results=[operation.result.type],
                attributes={"state": operation.attributes["state"]},
            )
            self.value_map[operation.result] = new.result
            return
        if name == "event.select_ready":
            new = self._insert(
                self.kernel_ip,
                "event.select_ready",
                operands=[self._mapped(value) for value in operation.operands],
                results=[operation.result.type],
                attributes={"policy": operation.attributes["policy"]},
            )
            self.value_map[operation.result] = new.result
            return
        if name == "event.try_take":
            event = operation.operands[0]
            carries = tuple(operation.operands[1:])
            result_types = [self._mapped(value).type for value in carries]
            new = self._insert(
                self.kernel_ip,
                "event.try_take",
                operands=[
                    self._mapped(event),
                    *(self._mapped(value) for value in carries)
                ],
                results=result_types,
                regions=3,
            )
            source_blocks = [region.blocks[0] for region in operation.regions]
            alternative_types = [
                self._placed_auxiliary_type(source_blocks[0].arguments[0].type),
                self._placed_auxiliary_type(source_blocks[1].arguments[0].type),
                source_blocks[2].arguments[0].type,
            ]
            for source_region, target_region, alternative_type in zip(
                    operation.regions, new.regions, alternative_types):
                self._convert_event_take_region(
                    source_region,
                    target_region,
                    alternative_type=alternative_type,
                    carries=carries,
                )
            for old, result, carry in zip(operation.results, new.results,
                                          carries):
                self.value_map[old] = result
                if str(carry.type) == "!qlx.logical_qubit":
                    self.placement_map[old] = self._infer_placement(carry)
            return
        if name == "event.cancel":
            attrs = {}
            if "reason" in operation.attributes:
                attrs["reason"] = operation.attributes["reason"]
            cancelled = self._insert(
                self.kernel_ip,
                "event.cancel",
                operands=[self._mapped(operation.operands[0])],
                results=[operation.result.type],
                attributes=attrs,
            )
            self.value_map[operation.result] = cancelled.result
            return
        if name == "event.await":
            kind = self._quoted_parameter(operation.result.type,
                                          '!qlx.logical_resource<"')
            stream = self._stream_for(kind)
            new = self._insert(
                self.kernel_ip,
                "event.await",
                operands=[self._mapped(operation.operands[0])],
                results=[self._bound_resource_type(kind, stream)],
            )
            self.value_map[operation.result] = new.result
            return
        if name == "event.fence":
            self._insert(
                self.kernel_ip,
                "event.fence",
                attributes={"effects": operation.attributes["effects"]},
            )
            return
        if name == "event.selection":
            self._insert(
                self.kernel_ip,
                "event.selection",
                operands=[self._mapped(operation.operands[0])],
                attributes={
                    "mode": operation.attributes["mode"],
                    "accept_when": operation.attributes["accept_when"],
                },
            )
            return
        if name == "qlx.consume_resource":
            quantum_inputs = list(operation.operands[1:])
            placements = self._placements_for(quantum_inputs)
            kind = self._quoted_parameter(operation.operands[0].type,
                                          '!qlx.logical_resource<"')
            if kind is None:
                raise ValueError(
                    "resource consume requires a typed logical resource kind")
            stream = self._stream_for(kind)
            new = self._insert(
                self.kernel_ip,
                "lvm.consume_resource",
                operands=[self._mapped(value) for value in operation.operands],
                results=[
                    self._placed_type(placement) for placement in placements
                ],
                attributes={
                    "action":
                        operation.attributes["action"],
                    "resource_kind":
                        mlir_ir.FlatSymbolRefAttr.get(kind,
                                                      context=self.context),
                    "resource_stream":
                        stream,
                    "placements":
                        mlir_ir.ArrayAttr.get(placements, context=self.context),
                    "site":
                        self._next_site(),
                },
            )
            for old, result, placement in zip(operation.results, new.results,
                                              placements):
                self.value_map[old] = result
                self.placement_map[old] = placement
            return
        if name == "qlx.frame_init":
            domain = str(operation.attributes["domain"]).strip('"')
            frame_type = mlir_ir.Type.parse(f'!lvm.logical_frame<"{domain}">',
                                            context=self.context)
            new = self._insert(
                self.kernel_ip,
                "lvm.frame_init",
                results=[frame_type],
                attributes={"domain": operation.attributes["domain"]},
            )
            self.value_map[operation.result] = new.result
            return
        if name in {"qlx.frame_update", "qlx.frame_transform"}:
            target_name = name.replace("qlx.", "lvm.")
            result_type = self._placed_auxiliary_type(operation.result.type)
            new = self._insert(
                self.kernel_ip,
                target_name,
                operands=[self._mapped(value) for value in operation.operands],
                results=[result_type],
                attributes=dict(operation.attributes),
            )
            self.value_map[operation.result] = new.result
            return
        if name == "cflow.if":
            result_types = [
                self._placed_result_type(result) for result in operation.results
            ]
            new = self._insert(
                self.kernel_ip,
                "cflow.if",
                operands=[self._mapped(operation.operands[0])],
                results=result_types,
                regions=2,
            )
            self._convert_region(operation.regions[0], new.regions[0])
            self._convert_region(operation.regions[1], new.regions[1])
            for old, result in zip(operation.results, new.results):
                self.value_map[old] = result
                if str(old.type) == "!qlx.logical_qubit":
                    self.placement_map[old] = self._infer_placement(old)
            return
        if name == "cflow.repeat":
            inits = tuple(operation.operands)
            result_types = [self._mapped(value).type for value in inits]
            new = self._insert(
                self.kernel_ip,
                "cflow.repeat",
                operands=[self._mapped(value) for value in inits],
                results=result_types,
                attributes={"count": operation.attributes["count"]},
                regions=1,
            )
            self._convert_region(operation.regions[0],
                                 new.regions[0],
                                 init_values=inits)
            for old, result, init in zip(operation.results, new.results, inits):
                self.value_map[old] = result
                if str(old.type) == "!qlx.logical_qubit":
                    self.placement_map[old] = self._infer_placement(init)
            return
        if name == "cflow.while":
            inits = tuple(operation.operands)
            result_types = [self._mapped(value).type for value in inits]
            attrs = {}
            if "max_iterations" in operation.attributes:
                attrs["max_iterations"] = operation.attributes["max_iterations"]
            new = self._insert(
                self.kernel_ip,
                "cflow.while",
                operands=[self._mapped(value) for value in inits],
                results=result_types,
                attributes=attrs,
                regions=2,
            )
            self._convert_region(operation.regions[0],
                                 new.regions[0],
                                 init_values=inits)
            self._convert_region(operation.regions[1],
                                 new.regions[1],
                                 init_values=inits)
            for old, result, init in zip(operation.results, new.results, inits):
                self.value_map[old] = result
                if str(old.type) == "!qlx.logical_qubit":
                    self.placement_map[old] = self._infer_placement(init)
            return
        raise NotImplementedError(f"qlx-to-lvm does not yet convert {name}")


def _place_build(
    program,
    *,
    device,
    placement=(),
    constraints=None,
    objective=None,
    experiment=None,
    _transient=False,
) -> Build:
    from .compile import compile

    if not isinstance(program, Build):
        program = compile(program, pipeline=pipelines.logical())
    if program.profile != "p0":
        raise ValueError(
            "cudaq.logical.place requires a P0 Build or portable definition")
    if constraints is not None:
        if placement:
            raise TypeError("specify placement= or constraints=, not both")
        placement = constraints
    if callable(placement):
        placement = placement(program.values)
    machine = _as_machine(device)
    module, witness, root_symbol = _P0ToP1(program, machine,
                                           tuple(placement or ()),
                                           objective).run()
    return Build(
        context=module.context,
        module=module,
        root=DefinitionHandle(root_symbol, "kernel", "p1"),
        profile="p1",
        pipeline=pipelines.placed(),
        evidence=program.evidence + (EvidenceRecord(
            kind="placement",
            producer="cudaq-logical-place-python@0.1",
            result="pass",
            obligations=("capacity", "placement-completeness", "p0-refinement"),
        ),),
        value_groups={
            name: len(group) for name, group in program.values._groups.items()
        },
        placement=witness,
        experiment=experiment or program.experiment,
        device=device,
        objective=objective,
        source_modules=program.source_modules,
        _transient=_transient,
    )


def place(
    program,
    *,
    device,
    placement=(),
    constraints=None,
    objective=None,
    experiment=None,
    _transient=False,
) -> Build:
    """Refine P0 through the inspectable problem/plan placement seam."""

    if _transient:
        # The public problem/plan seam commits portable P0 assembly into its
        # digest. The progressive compiler already owns the exact live P0
        # ModuleOp, so invoke the same deterministic first-fit materializer
        # directly. P1 still carries the canonical placement witness.
        return _place_build(
            program,
            device=device,
            placement=placement,
            constraints=constraints,
            objective=objective,
            experiment=experiment,
            _transient=True,
        )

    from ..architecture import placement as placement_api

    placement_problem = placement_api.problem(
        program,
        device=device,
        placement=placement,
        constraints=constraints,
        objective=objective,
        experiment=experiment,
    )
    placement_plan = placement_api.solve(placement_problem)
    return placement_api.apply(
        None,
        placement_problem,
        placement_plan,
    )
