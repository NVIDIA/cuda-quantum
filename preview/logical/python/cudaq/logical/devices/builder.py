# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Mutable authoring handles for immutable layered device models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, isfinite
import sys
from typing import Any, Iterable, Mapping

from cudaq.logical.devices.definition import (
    Device,
    FactoryModel,
    LogicalToQECBinding,
    PhysicalOperatingPoint,
    QECArchitecture,
    QECChannelPort,
    QECChannelRealization,
    QECChannelToPhysicalBinding,
    QECMachine,
    QECRegion,
    QECToPhysicalBinding,
    _compiler_key,
    _frozen_mapping,
    _member_name,
)
from cudaq.logical.architecture.logical import (
    CapabilityKey,
    Channel,
    LogicalMachine,
    RESOURCE_TRANSFER_CAPABILITY,
    Space,
    SpaceDeclaration,
    Stream,
    capability,
)
from cudaq.logical.devices.timing import TimingModel
from cudaq.logical.architecture.physical_definition import (
    NativeActionDecomposition,
    PatchKind,
    PatchTopology,
    PhysicalFootprint,
    PhysicalAction,
    PhysicalInstrument,
    PhysicalMachine,
    ResourceClass,
    ResourceGranularity,
    Topology,
)
from cudaq.logical.architecture.capabilities import (
    PhysicalCapability,
    PhysicalCapabilityBinding,
)
from cudaq.logical.protocols.definition import ProtocolDefinition
from cudaq.logical.codes import (
    Code,
    Encoding,
)


@dataclass(frozen=True, slots=True)
class CarrierSelection:
    resource: "PhysicalResourceBuilder"
    indices: tuple[int, ...]


class LogicalRegionBuilder:
    __slots__ = ("_builder", "space")

    def __init__(self, builder: "DeviceBuilder", space: Space) -> None:
        self._builder = builder
        self.space = space

    @property
    def name(self) -> str:
        return self.space.name or ""

    @property
    def capacity(self) -> int | None:
        return self.space.capacity

    def __getitem__(self, index: int):
        return self.space[index]

    def any_slot(self):
        return self.space.any_slot()

    @property
    def stream(self) -> Stream:
        """The typed output stream of this factory region.

        Ordinary regions deliberately have no implicit stream.  Looking one up
        therefore fails closed unless ``logical.add_factory`` derived exactly
        one backed stream for this handle.
        """

        matches = tuple(
            value for value in self._builder.logical._members.values()
            if isinstance(value, Stream) and value.region is not None and
            value.region.name == self.name)
        if len(matches) != 1:
            raise AttributeError(
                f"logical region @{self.name} is not a uniquely backed factory")
        return matches[0]

    @property
    def supply(self) -> Channel:
        """The P1 supply channel derived with this factory's output stream."""

        stream = self.stream
        matches = tuple(
            value for value in self._builder.logical._members.values()
            if isinstance(value, Channel) and value.source is self.space and
            value.destination is stream and
            RESOURCE_TRANSFER_CAPABILITY in value.capabilities)
        if len(matches) != 1:
            raise AttributeError(
                f"logical factory @{self.name} has no unique supply channel")
        return matches[0]

    def bind_encoding(
        self,
        encoding: Code | Encoding,
        *,
        block_capacity: int | None = None,
        packing: str = "dense",
        name: str | None = None,
    ) -> "QECRegionBuilder":
        return self._builder.qec.bind(
            self,
            encoding=encoding,
            block_capacity=block_capacity,
            packing=packing,
            name=name,
        )


class QECRegionBuilder:
    __slots__ = ("_builder", "_auxiliary_regions", "region", "logical")

    def __init__(
            self,
            builder: "DeviceBuilder",
            region: QECRegion,
            logical: LogicalRegionBuilder | None,
            auxiliary_regions: Iterable["QECRegionBuilder"] = (),
    ) -> None:
        self._builder = builder
        self.region = region
        self.logical = logical
        self._auxiliary_regions = tuple(auxiliary_regions)

    @property
    def name(self) -> str:
        return self.region.name

    @property
    def block_capacity(self) -> int:
        return self.region.block_capacity

    @property
    def auxiliary_regions(self) -> tuple["QECRegionBuilder", ...]:
        """Typed handles for scratch regions derived by this architecture.

        Architecture binding owns the names and number of these P2 regions.
        Returning their builder handles keeps later physical realization typed;
        callers never need to reconstruct generated names through ``getattr``.
        """

        return self._auxiliary_regions

    def realize_on(self, *resources, **kwargs) -> "QECRegionBuilder":
        self._builder.physical.bind(self, to=resources, **kwargs)
        return self


class QECChannelBuilder:
    """Builder-owned handle for one selected P2 channel realization."""

    __slots__ = ("_builder", "channel")

    def __init__(
        self,
        builder: "DeviceBuilder",
        channel: QECChannelRealization,
    ) -> None:
        self._builder = builder
        self.channel = channel

    @property
    def name(self) -> str:
        return self.channel.name

    @property
    def source(self) -> QECChannelPort:
        return self.channel.source

    @property
    def destination(self) -> QECChannelPort:
        return self.channel.destination

    @property
    def capabilities(self) -> tuple[CapabilityKey, ...]:
        return self.channel.capabilities

    def realize_on(self, *resources) -> "QECChannelBuilder":
        self._builder.physical.bind_channel(self, to=resources)
        return self


class PhysicalResourceBuilder:
    __slots__ = ("_builder", "name", "resource", "topology")

    def __init__(
        self,
        builder: "DeviceBuilder",
        name: str,
        resource: ResourceClass,
        topology: Topology | None,
    ) -> None:
        self._builder = builder
        self.name = name
        self.resource = resource
        self.topology = topology

    @property
    def count(self) -> int:
        return self.resource.count

    @property
    def kind(self) -> str:
        return self.resource.kind

    def __getitem__(self, key: int | tuple[int, ...]) -> CarrierSelection:
        indices = key if isinstance(key, tuple) else (key,)
        if not indices or len(set(indices)) != len(indices):
            raise ValueError("carrier selection must be nonempty and unique")
        if any(not isinstance(index, int) or isinstance(index, bool) or
               index < 0 or index >= self.count for index in indices):
            raise IndexError(
                f"carrier selection must use indices in [0, {self.count})")
        return CarrierSelection(self, tuple(indices))

    def claim(
        self,
        *,
        offset: int = 0,
        count: int | None = None,
        units: int | None = None,
    ):
        """Return one typed compact-model claim over this resource pool."""

        from cudaq.logical.devices.component_models import PhysicalResourceClaim

        return PhysicalResourceClaim(
            self.resource,
            offset=offset,
            count=count,
            units=units,
        )


class _LogicalNamespace:

    def __init__(self, builder: "DeviceBuilder",
                 machine: LogicalMachine | None) -> None:
        self._builder = builder
        self._members = {} if machine is None else dict(machine._members)
        self._regions = {
            space.name: LogicalRegionBuilder(builder, space)
            for space in (() if machine is None else machine.spaces)
        }

    def __getattr__(self, name: str):
        if name in self._regions:
            return self._regions[name]
        try:
            return self._members[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def _reserve(self, name: str) -> str:
        name = _member_name(name, what="logical member name")
        if name in self._members:
            raise ValueError(f"duplicate logical member name {name!r}")
        return name

    def _default(self, base: str) -> str:
        candidate = base
        ordinal = 2
        while candidate in self._members:
            candidate = f"{base}_{ordinal}"
            ordinal += 1
        return candidate

    def add_region(
            self,
            name: str,
            *,
            capacity: int | None = None,
            capabilities=(),
            tags=(),
    ) -> LogicalRegionBuilder:
        self._builder._assert_open()
        name = self._reserve(name)
        space = Space(
            name=name,
            capacity=capacity,
            capabilities=tuple(capabilities),
            tags=tuple(tags),
        )
        handle = LogicalRegionBuilder(self._builder, space)
        self._members[name] = space
        self._regions[name] = handle
        return handle

    def add_compute(
            self,
            *,
            capacity: int | None = None,
            capabilities=(),
            tags=(),
            name: str | None = None,
    ) -> LogicalRegionBuilder:
        values = tuple(capabilities)
        if capability.logical_compute not in values:
            values = (capability.logical_compute, *values)
        return self.add_region(
            name or self._default("compute"),
            capacity=capacity,
            capabilities=values,
            tags=tags,
        )

    def add_memory(
            self,
            *,
            capacity: int | None = None,
            capabilities=(),
            tags=(),
            name: str | None = None,
    ) -> LogicalRegionBuilder:
        values = tuple(capabilities)
        if capability.logical_memory not in values:
            values = (capability.logical_memory, *values)
        return self.add_region(
            name or self._default("memory"),
            capacity=capacity,
            capabilities=values,
            tags=tags,
        )

    def add_factory(
        self,
        *,
        produces,
        via,
        capacity: int | None = 1,
        buffer_size: int | None = 1,
        transfer=None,
        capabilities=(),
        tags=(),
        name: str | None = None,
        stream_name: str | None = None,
    ) -> LogicalRegionBuilder:
        """Add one resource-producing region and its backed output stream.

        ``capacity`` counts concurrent factory instances. ``buffer_size``
        counts produced resources that may wait in the stream. The returned
        handle is the ordinary logical region used by QEC/physical bindings
        and placement; the resource stream and supply channel are derived.
        """

        from cudaq.logical.protocols.definition import ProtocolDefinition
        from ..std import ResourceFlowRef, ResourceKind

        self._builder._assert_open()
        if not isinstance(produces, ResourceKind):
            raise TypeError("logical.add_factory produces= requires a "
                            "cudaq.logical.types.ResourceKind")
        if not isinstance(via, ProtocolDefinition):
            raise TypeError("logical.add_factory via= requires a "
                            "cudaq.logical.protocols.ProtocolDefinition")
        if via._factory_region is not None:
            raise ValueError(
                "logical.add_factory via= protocol is already attached to a "
                "factory")
        if via.signature.parameters:
            raise ValueError(
                "logical.add_factory via= producer must not require inputs")
        objective = via.implements
        if (not isinstance(objective, ResourceFlowRef) or
                objective.kind != "produce" or objective.resource != produces):
            raise ValueError("logical.add_factory via= must implement "
                             "cudaq.logical.logical.produce(produces)")
        if buffer_size is not None and (not isinstance(buffer_size, int) or
                                        isinstance(buffer_size, bool) or
                                        buffer_size < 0):
            raise TypeError(
                "logical.add_factory buffer_size must be a nonnegative int")
        if transfer is not None:
            if not isinstance(transfer, ProtocolDefinition):
                raise TypeError("logical.add_factory transfer= requires a "
                                "cudaq.logical.protocols.ProtocolDefinition")
            if produces not in transfer._resource_input_kinds():
                raise ValueError(
                    "logical.add_factory transfer= protocol must consume "
                    "cudaq.logical.types.resource[produces]")

        resource_name = _member_name(produces.name,
                                     what="factory resource name")
        region_name = _member_name(
            name or self._default(f"{resource_name}_factory"),
            what="logical factory name",
        )
        if region_name in self._members:
            raise ValueError(f"duplicate logical member name {region_name!r}")
        if stream_name is None:
            stream_base = f"{resource_name}_stream"
            stream_name = stream_base
            ordinal = 2
            while (stream_name in self._members or stream_name == region_name or
                   f"{stream_name}_supply" in self._members or
                   f"{stream_name}_supply" == region_name):
                stream_name = f"{stream_base}_{ordinal}"
                ordinal += 1
        stream_name = _member_name(stream_name,
                                   what="logical factory stream name")
        if stream_name == region_name or stream_name in self._members:
            raise ValueError(f"duplicate logical member name {stream_name!r}")
        supply_name = _member_name(f"{stream_name}_supply",
                                   what="logical factory supply name")
        if supply_name == region_name or supply_name in self._members:
            raise ValueError(f"duplicate logical member name {supply_name!r}")

        values = tuple(capabilities)
        if capability.logical_factory not in values:
            values = (capability.logical_factory, *values)
        factory = self.add_region(
            region_name,
            capacity=capacity,
            capabilities=values,
            tags=tags,
        )
        bound_via = via._bind_factory(factory.space)
        self.add_stream(
            produces,
            name=stream_name,
            buffer_size=buffer_size,
            region=factory,
            produced_by=bound_via,
            transfer=transfer,
        )
        return factory

    def add_channel(
        self,
        source: LogicalRegionBuilder | Space | Stream,
        destination: LogicalRegionBuilder | Space | Stream,
        *,
        name: str | None = None,
        capabilities=(),
        direction="forward",
        capacity: int | None = None,
        concurrency: int | None = None,
    ) -> Channel:
        self._builder._assert_open()
        name = self._reserve(name or self._default("channel"))

        def endpoint(value):
            if isinstance(value, LogicalRegionBuilder):
                if value._builder is not self._builder:
                    raise ValueError(
                        "channel endpoint belongs to another builder")
                return value.space
            if (isinstance(value, (Space, Stream)) and
                    value in self._members.values()):
                return value
            raise TypeError("channel endpoints must be logical regions")

        channel = Channel(
            endpoint(source),
            endpoint(destination),
            capabilities=capabilities,
            direction=direction,
            capacity=capacity,
            concurrency=concurrency,
            name=name,
        )
        self._members[name] = channel
        return channel

    def add_stream(
        self,
        produces,
        *,
        name: str,
        buffer_size: int | None = None,
        region: LogicalRegionBuilder | None = None,
        produced_by=None,
        transfer=None,
        external: bool = False,
    ) -> Stream:
        from cudaq.logical.protocols.definition import ProtocolDefinition

        self._builder._assert_open()
        name = self._reserve(name)
        if region is not None and region._builder is not self._builder:
            raise ValueError("stream region belongs to another builder")
        if external and region is not None:
            raise TypeError("a stream cannot be both backed and external")
        if (produced_by is not None and region is not None and
                capability.logical_factory not in region.space.capabilities):
            raise ValueError("a producer-backed stream region requires the "
                             "logical_factory capability")
        if region is not None and isinstance(produced_by, ProtocolDefinition):
            produced_by = produced_by._bind_factory(region.space)
        stream = Stream(
            produces=produces,
            buffer_size=buffer_size,
            region=(None if region is None else SpaceDeclaration(
                capacity=region.capacity,
                name=region.name,
            )),
            produced_by=produced_by,
            transfer=transfer,
            external=external,
            name=name,
        )
        self._members[name] = stream
        if region is not None:
            supply_name = self._reserve(f"{name}_supply")
            self._members[supply_name] = Channel(
                region.space,
                stream,
                capabilities=(RESOURCE_TRANSFER_CAPABILITY,),
                name=supply_name,
            )
        return stream


class _QECNamespace:

    def __init__(self, builder: "DeviceBuilder", machine: QECMachine | None):
        self._builder = builder
        self._regions = {
            region.name: region
            for region in (() if machine is None else machine.regions)
        }
        self._bindings: list[LogicalToQECBinding] = []
        self._handles: dict[str, QECRegionBuilder] = {
            name: QECRegionBuilder(builder, region, None)
            for name, region in self._regions.items()
        }
        self._ports: dict[str, QECChannelPort] = {
            port.name: port
            for port in (() if machine is None else machine.channel_ports)
        }
        self._channels: dict[str, QECChannelRealization] = {
            channel.name: channel
            for channel in (() if machine is None else machine.channels)
        }
        self._channel_handles: dict[str, QECChannelBuilder] = {
            name: QECChannelBuilder(builder, channel)
            for name, channel in self._channels.items()
        }

    def __getattr__(self, name: str):
        try:
            return self._handles[name]
        except KeyError as exc:
            try:
                return self._channel_handles[name]
            except KeyError:
                raise AttributeError(name) from exc

    def bind(
        self,
        logical: LogicalRegionBuilder,
        *,
        encoding: Code | Encoding | None = None,
        architecture: QECArchitecture | None = None,
        block_capacity: int | None = None,
        packing: str | None = None,
        name: str | None = None,
    ) -> QECRegionBuilder:
        self._builder._assert_open()
        if (not isinstance(logical, LogicalRegionBuilder) or
                logical._builder is not self._builder):
            raise TypeError(
                "qec.bind() requires a logical handle from this builder")
        if (encoding is None) == (architecture is None):
            raise TypeError(
                "qec.bind() requires exactly one of encoding= or architecture=")
        if architecture is not None:
            if not isinstance(architecture, QECArchitecture):
                raise TypeError(
                    "qec.bind() architecture= requires a QECArchitecture")
            if packing is not None:
                raise TypeError(
                    "qec.bind() packing= cannot override architecture packing")
            encoding = architecture.encoding
            packing = architecture.packing
        elif packing is None:
            packing = "dense"
        if isinstance(encoding, Code):
            encoding = encoding.default_encoding
        if not isinstance(encoding, Encoding):
            raise TypeError("qec.bind() encoding= requires Code or Encoding")
        if any(binding.logical_region is logical.space
               for binding in self._bindings):
            raise ValueError(
                f"logical region {logical.name!r} is already bound")
        if block_capacity is None:
            if logical.capacity is None:
                raise ValueError(
                    "block_capacity is required for an unbounded logical region"
                )
            block_capacity = ceil(logical.capacity / encoding.code.k)
        region_name = _member_name(name or logical.name, what="QEC region name")
        if region_name in self._regions:
            raise ValueError(f"duplicate QEC region name {region_name!r}")
        auxiliary_regions = tuple(
            replace(template, name=f"{region_name}_{template.name}")
            for template in (
                () if architecture is None else architecture.auxiliary_regions))
        collisions = sorted(
            region.name
            for region in auxiliary_regions
            if region.name in self._regions or region.name == region_name)
        if collisions:
            raise ValueError("duplicate architecture QEC region name(s): " +
                             ", ".join(collisions))
        region = QECRegion(
            region_name,
            encoding,
            block_capacity,
            packing,
        )
        binding = LogicalToQECBinding(
            logical.space,
            region,
            packing,
            architecture,
            auxiliary_regions,
        )
        auxiliary_handles = tuple(
            QECRegionBuilder(self._builder, auxiliary, None)
            for auxiliary in auxiliary_regions)
        handle = QECRegionBuilder(
            self._builder,
            region,
            logical,
            auxiliary_regions=auxiliary_handles,
        )
        self._regions[region_name] = region
        self._regions.update(
            (auxiliary.name, auxiliary) for auxiliary in auxiliary_regions)
        self._bindings.append(binding)
        self._handles[region_name] = handle
        self._handles.update((auxiliary.name, auxiliary_handle)
                             for auxiliary, auxiliary_handle in zip(
                                 auxiliary_regions, auxiliary_handles))
        return handle

    def bind_existing(
        self,
        logical: LogicalRegionBuilder,
        *,
        to: QECRegion,
    ) -> QECRegionBuilder:
        """Refine a logical region through a region from an imported machine."""

        self._builder._assert_open()
        if (not isinstance(logical, LogicalRegionBuilder) or
                logical._builder is not self._builder):
            raise TypeError(
                "qec.bind_existing() requires a logical handle from this builder"
            )
        if not isinstance(to, QECRegion) or to not in self._regions.values():
            raise TypeError(
                "qec.bind_existing() to= must come from the imported QECMachine"
            )
        binding = LogicalToQECBinding(logical.space, to, to.packing)
        handle = QECRegionBuilder(self._builder, to, logical)
        self._bindings.append(binding)
        self._handles[to.name] = handle
        return handle

    def bind_channel(
        self,
        logical: Channel,
        *,
        via: QECChannelRealization,
    ) -> QECChannelBuilder:
        """Bind one P1 channel to a provider-selected P2 realization."""

        self._builder._assert_open()
        if (not isinstance(logical, Channel) or
                logical not in self._builder.logical._members.values()):
            raise TypeError(
                "qec.bind_channel() requires a logical channel from this builder"
            )
        if not isinstance(via, QECChannelRealization):
            raise TypeError(
                "qec.bind_channel() via= requires a QECChannelRealization")
        if via.logical_channel is not logical:
            raise ValueError(
                "QEC channel realization is bound to a different logical channel"
            )
        if via.name in self._channels:
            raise ValueError(f"QEC channel {via.name!r} is already bound")
        if any(channel.logical_channel is logical
               for channel in self._channels.values()):
            raise ValueError(
                "logical channel already has a selected QEC realization")

        def endpoint_space(endpoint):
            if isinstance(endpoint, Space):
                return endpoint
            if isinstance(endpoint, Stream) and endpoint.region is not None:
                handle = self._builder.logical._regions.get(
                    endpoint.region.name)
                return None if handle is None else handle.space
            return None

        expected = (endpoint_space(logical.source),
                    endpoint_space(logical.destination))
        if any(value is None for value in expected):
            raise ValueError(
                "P2 channel refinement requires backed logical endpoints")
        qec_for = {
            id(binding.logical_region): binding.qec_region
            for binding in self._bindings
        }
        expected_qec = tuple(qec_for.get(id(value)) for value in expected)
        if None in expected_qec:
            raise ValueError(
                "bind both logical channel endpoints to QEC regions first")
        if (via.source.region, via.destination.region) != expected_qec:
            raise ValueError(
                "QEC channel ports do not refine the logical channel endpoints")
        canonical_ports = []
        for port in (via.source, via.destination):
            previous = self._ports.get(port.name)
            if previous is not None and previous != port:
                raise ValueError(
                    f"QEC channel-port name {port.name!r} has two definitions")
            if previous is None:
                self._ports[port.name] = port
                previous = port
            canonical_ports.append(previous)
        if any(canonical is not actual
               for canonical, actual in zip(canonical_ports, (
                   via.source, via.destination))):
            via = replace(
                via,
                source=canonical_ports[0],
                destination=canonical_ports[1],
            )
        handle = QECChannelBuilder(self._builder, via)
        self._channels[via.name] = via
        self._channel_handles[via.name] = handle
        return handle


class _PhysicalNamespace:

    def __init__(self, builder: "DeviceBuilder",
                 machine: PhysicalMachine | None) -> None:
        self._builder = builder
        self._topologies = {
            topology.name: topology
            for topology in (() if machine is None else machine.topologies)
        }
        inferred_topology = (machine.topologies[0] if machine is not None and
                             len(machine.topologies) == 1 else None)
        self._resources = {
            resource.name:
                PhysicalResourceBuilder(
                    builder,
                    resource.name,
                    resource,
                    inferred_topology,
                ) for resource in (
                    () if machine is None else machine.resource_classes)
        }
        self._bindings: list[QECToPhysicalBinding] = []
        self._channel_bindings: list[QECChannelToPhysicalBinding] = []
        self._spacetime_plans = []
        self._operating_point: PhysicalOperatingPoint | None = None

    def __getattr__(self, name: str):
        try:
            return self._resources[name]
        except KeyError as exc:
            try:
                return self._topologies[name]
            except KeyError:
                raise AttributeError(name) from exc

    def _default(self, kind: str) -> str:
        base = kind if kind.endswith("s") else f"{kind}s"
        candidate = base
        ordinal = 2
        while candidate in self._resources:
            candidate = f"{base}_{ordinal}"
            ordinal += 1
        return candidate

    def add_resources(
        self,
        kind: str,
        count: int,
        *,
        name: str | None = None,
        granularity: ResourceGranularity | str = ResourceGranularity.CARRIER,
        footprint: PhysicalFootprint | None = None,
        native_actions: Iterable[PhysicalAction | str] = (),
        native_action_decompositions: Iterable[NativeActionDecomposition] = (),
        native_instruments: Iterable[PhysicalInstrument] = (),
        topology: Topology | None = None,
        capabilities: Iterable[PhysicalCapability | str] = (),
        capability_bindings: Iterable[PhysicalCapabilityBinding] = (),
        erasure_indices: Iterable[int] | None = None,
    ) -> PhysicalResourceBuilder:
        self._builder._assert_open()
        name = _member_name(name or self._default(kind),
                            what="physical resource name")
        if name in self._resources:
            raise ValueError(f"duplicate physical resource name {name!r}")
        if topology is not None:
            topology = topology._bind_resource_count(count)
            if topology.name is None:
                topology = topology._named(f"{name}_topology")
            if topology.name in self._topologies:
                raise ValueError(
                    f"duplicate physical topology name {topology.name!r}")
            self._topologies[topology.name] = topology
        resource = ResourceClass(
            kind,
            count,
            granularity=granularity,
            footprint=footprint,
            native_actions=native_actions,
            native_action_decompositions=native_action_decompositions,
            native_instruments=native_instruments,
            capabilities=capabilities,
            capability_bindings=capability_bindings,
            erasure_indices=erasure_indices,
            name=name,
        )
        handle = PhysicalResourceBuilder(self._builder, name, resource,
                                         topology)
        self._resources[name] = handle
        return handle

    def add_qubits(self, count: int, **kwargs) -> PhysicalResourceBuilder:
        return self.add_resources("qubit", count, **kwargs)

    def add_atoms(self, count: int, **kwargs) -> PhysicalResourceBuilder:
        return self.add_resources("atom", count, **kwargs)

    def bind(
        self,
        qec: QECRegionBuilder,
        *,
        to: PhysicalResourceBuilder | Iterable[PhysicalResourceBuilder],
        topology: Topology | None = None,
        patches: Iterable[CarrierSelection] | None = None,
        categories: Mapping[int, PatchKind] | Iterable[PatchKind | None] |
        None = None,
        factory_model: FactoryModel | None = None,
    ) -> QECToPhysicalBinding:
        self._builder._assert_open()
        if not isinstance(
                qec, QECRegionBuilder) or qec._builder is not self._builder:
            raise TypeError(
                "physical.bind() requires a QEC handle from this builder")
        resources = (to,) if isinstance(to,
                                        PhysicalResourceBuilder) else tuple(to)
        if not resources or any(
                not isinstance(resource, PhysicalResourceBuilder) or
                resource._builder is not self._builder
                for resource in resources):
            raise TypeError(
                "physical.bind() to= requires physical handles from this builder"
            )
        if any(binding.qec_region is qec.region for binding in self._bindings):
            raise ValueError(f"QEC region {qec.name!r} is already realized")
        if factory_model is not None:
            if not isinstance(factory_model, FactoryModel):
                raise TypeError(
                    "physical.bind() factory_model= requires a FactoryModel")
            try:
                stream = qec.logical.stream
            except AttributeError as error:
                raise ValueError(
                    "physical factory model requires a uniquely backed "
                    "logical factory region") from error
            characterization = factory_model.characterization
            if characterization is not None:
                from cudaq.logical.compiler.protocol_identity import (
                    factory_protocol_semantics_sha256,)

                if characterization.resource_kind != stream.produces:
                    raise ValueError(
                        "compiled factory model produces a different resource "
                        "kind than the bound logical stream")
                if characterization.source_provider != stream.produced_by.name:
                    raise ValueError(
                        "compiled factory model was characterized from a "
                        "different source producer")
                target_digest = factory_protocol_semantics_sha256(
                    stream.produced_by)
                if characterization.source_provider_sha256 != target_digest:
                    raise ValueError(
                        "compiled factory model source producer semantics do "
                        "not match the bound logical stream")
                target_distance = (
                    qec.region.encoding.code.d.conservative_value)
                if target_distance is None:
                    raise ValueError(
                        "compiled factory model requires a selected factory "
                        "code with scalar distance evidence")
                if target_distance not in characterization.code_distances:
                    raise ValueError(
                        "compiled factory model does not contain the selected "
                        f"factory binding's distance-{target_distance} code")
                lanes = qec.logical.capacity
                if lanes is None or lanes <= 0:
                    raise ValueError(
                        "compiled factory model requires a positive logical "
                        "factory capacity")
                units = []
                for resource in resources:
                    if resource.resource.footprint is None:
                        units.append((resource.kind, resource.count))
                    else:
                        footprint = resource.resource.footprint
                        units.append((
                            footprint.unit_kind,
                            resource.count * footprint.units,
                        ))
                unit_kinds = {kind for kind, _ in units}
                if unit_kinds != {characterization.physical_unit_kind}:
                    raise ValueError(
                        "compiled factory model and selected physical binding "
                        "use different base-unit kinds")
                provisioned = sum(count for _, count in units)
                required = lanes * characterization.physical_units
                if provisioned < required:
                    raise ValueError(
                        "compiled factory model requires at least "
                        f"{required} {characterization.physical_unit_kind}s "
                        f"for {lanes} lanes, but the binding provides "
                        f"{provisioned}")
        inferred_topologies = {
            id(resource.topology): resource.topology
            for resource in resources
            if resource.topology is not None
        }
        if len(inferred_topologies) > 1:
            raise ValueError("one QEC binding requires one carrier topology")
        inferred_topology = next(iter(inferred_topologies.values()), None)
        if topology is not None:
            if (not isinstance(topology, Topology) or
                    topology not in self._topologies.values()):
                raise TypeError(
                    "physical.bind() topology= must come from this physical machine"
                )
            if inferred_topology is not None and topology is not inferred_topology:
                raise ValueError(
                    "physical.bind() topology= conflicts with the resource topology"
                )
        else:
            topology = inferred_topology
        patch_topology = None
        if patches is not None:
            if len(resources) != 1:
                raise ValueError("patch selections require one resource")
            selections = tuple(patches)
            if any(not isinstance(selection, CarrierSelection) or
                   selection.resource is not resources[0]
                   for selection in selections):
                raise TypeError(
                    "patches must be selections from the bound physical resource"
                )
            category_map = {}
            if isinstance(categories, Mapping):
                category_map = dict(categories)
            elif categories is not None:
                values = tuple(categories)
                if len(values) != len(selections):
                    raise ValueError("categories requires one value per patch")
                category_map = {
                    index: value
                    for index, value in enumerate(values)
                    if value is not None
                }
            patch_topology = PatchTopology(
                tuple(selection.indices for selection in selections),
                categories=category_map,
            )
        elif categories is not None:
            raise TypeError("categories= requires patches=")
        binding = QECToPhysicalBinding(
            qec.region,
            tuple(resource.resource for resource in resources),
            topology=topology,
            patch_topology=patch_topology,
            factory_model=factory_model,
        )
        self._bindings.append(binding)
        return binding

    def bind_channel(
        self,
        qec: QECChannelBuilder,
        *,
        to: PhysicalResourceBuilder | Iterable[PhysicalResourceBuilder],
        transport_claims=(),
        endpoint_occupancy=None,
        transport_model=None,
    ) -> QECChannelToPhysicalBinding:
        """Bind one P2 channel to P3 pools and optional detailed route facts."""

        self._builder._assert_open()
        if (not isinstance(qec, QECChannelBuilder) or
                qec._builder is not self._builder):
            raise TypeError(
                "physical.bind_channel() requires a QEC channel from this builder"
            )
        resources = (to,) if isinstance(to,
                                        PhysicalResourceBuilder) else tuple(to)
        if not resources or any(
                not isinstance(resource, PhysicalResourceBuilder) or
                resource._builder is not self._builder
                for resource in resources):
            raise TypeError(
                "physical.bind_channel() to= requires physical handles from this builder"
            )
        if any(binding.qec_channel is qec.channel
               for binding in self._channel_bindings):
            raise ValueError(f"QEC channel {qec.name!r} is already realized")
        binding = QECChannelToPhysicalBinding(
            qec.channel,
            tuple(resource.resource for resource in resources),
            transport_claims=transport_claims,
            endpoint_occupancy=endpoint_occupancy,
            transport_model=transport_model,
        )
        self._channel_bindings.append(binding)
        return binding

    def bind_protocol(self, model):
        """Attach one compact P3 plan to its exact typed P2 protocol."""

        from cudaq.logical.devices.component_models import SpacetimePlanModel

        self._builder._assert_open()
        if not isinstance(model, SpacetimePlanModel):
            raise TypeError(
                "physical.bind_protocol() requires a SpacetimePlanModel")
        if any(value.protocol.name == model.protocol.name
               for value in self._spacetime_plans):
            raise ValueError(
                f"protocol {model.protocol.name!r} already has a compact plan")
        available = {id(handle.resource) for handle in self._resources.values()}
        if any(
                id(claim.resource_class) not in available
                for phase in model.phases
                for claim in phase.resources):
            raise ValueError(
                "compact plan claims resources outside this physical machine")
        self._spacetime_plans.append(model)
        return model

    def set_operating_point(
        self,
        *,
        timing: TimingModel | Mapping[str, Any] | None = None,
        calibration: Mapping[str, Any] | None = None,
        costs: Mapping[str, Any] | None = None,
        target_compatibility: Iterable[str] = (),
        name: str = "default",
    ) -> PhysicalOperatingPoint:
        self._builder._assert_open()
        if self._operating_point is not None:
            raise ValueError("DeviceBuilder already has an operating point")
        self._operating_point = PhysicalOperatingPoint(
            timing=timing,
            calibration=calibration,
            costs=costs,
            target_compatibility=tuple(target_compatibility),
            name=name,
        )
        return self._operating_point


class DeviceBuilder:
    """Canonical mutable authoring facade for an immutable layered Device."""

    __slots__ = (
        "name",
        "logical",
        "qec",
        "physical",
        "metadata",
        "source_module",
        "_base_logical",
        "_base_qec",
        "_base_physical",
        "_compilers",
        "_built",
    )

    def __init__(
        self,
        name: str,
        *,
        logical: LogicalMachine | None = None,
        qec: QECMachine | None = None,
        physical: PhysicalMachine | None = None,
        compilers: Iterable[Any] = (),
        metadata: Mapping[str, Any] | None = None,
        source_module: str | None = None,
    ) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError("DeviceBuilder name must be a nonempty string")
        self.name = name
        self.metadata = _frozen_mapping(metadata)
        if source_module is None:
            import sys

            source_module = sys._getframe(1).f_globals.get("__name__")
        self.source_module = source_module
        self._base_logical = logical
        self._base_qec = qec
        self._base_physical = physical
        self._built = None
        self._compilers = []
        for compiler in compilers:
            self.add_compiler(compiler)
        self.logical = _LogicalNamespace(self, logical)
        self.qec = _QECNamespace(self, qec)
        self.physical = _PhysicalNamespace(self, physical)

    def _assert_open(self) -> None:
        if self._built is not None:
            raise RuntimeError(
                "DeviceBuilder is frozen after build(); create a new builder")

    def add_compiler(self, compiler):
        """Attach one versioned compiler capability to the immutable device."""

        self._assert_open()
        key = _compiler_key(compiler)
        if any(_compiler_key(value) == key for value in self._compilers):
            raise ValueError(f"duplicate device compiler key {key!r}")
        self._compilers.append(compiler)
        return compiler

    def build(self) -> Device:
        if self._built is not None:
            return self._built
        if not self.logical._regions:
            raise ValueError(
                "DeviceBuilder requires at least one logical region")
        logical = LogicalMachine(f"{self.name}LogicalMachine",
                                 self.logical._members)
        # Rebind logical objects after LogicalMachine normalizes its member map.
        logical_by_name = {space.name: space for space in logical.spaces}
        logical_to_qec = tuple(
            LogicalToQECBinding(
                logical_by_name[binding.logical_region.name],
                binding.qec_region,
                binding.packing,
                binding.architecture,
                binding.auxiliary_regions,
            ) for binding in self.qec._bindings)
        qec = None
        if self.qec._regions:
            imported_qec = self._base_qec is not None and {
                id(region) for region in self._base_qec.regions
            } == {
                id(region) for region in self.qec._regions.values()
            } and tuple(self.qec._ports.values()) == (
                self._base_qec.channel_ports) and tuple(
                    self.qec._channels.values()) == (self._base_qec.channels)
            qec = QECMachine(
                f"{self.name}QECMachine",
                self.qec._regions.values(),
                channel_ports=self.qec._ports.values(),
                channels=self.qec._channels.values(),
            ) if not imported_qec else self._base_qec
        resources = {
            name: handle.resource
            for name, handle in self.physical._resources.items()
        }
        topologies = dict(self.physical._topologies)
        physical = None
        if resources:
            imported_physical = self._base_physical is not None and {
                id(resource)
                for resource in self._base_physical.resource_classes
            } == {id(resource) for resource in resources.values()}
            physical = (self._base_physical
                        if imported_physical else PhysicalMachine(
                            f"{self.name}PhysicalMachine",
                            resource_classes=resources.values(),
                            topologies=topologies,
                        ))
        operating_point = self.physical._operating_point
        for binding in self.physical._bindings:
            model = binding.factory_model
            characterization = (None
                                if model is None else model.characterization)
            if characterization is None:
                continue
            if operating_point is None:
                raise ValueError(
                    "compiled factory model requires a selected operating "
                    "point")
            if (characterization.timing_source
                    != operating_point.timing_source):
                raise ValueError(
                    "compiled factory model timing source differs from the "
                    "selected operating point")
            for name, expected in characterization.timing_profile:
                source_name = "surface_cycle_ns" if name == "cycle_ns" else name
                try:
                    actual = float(operating_point.timing[source_name])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        "compiled factory model operating point lacks "
                        f"characterized timing {source_name!r}") from error
                if not isfinite(actual) or actual != expected:
                    raise ValueError(
                        "compiled factory model timing "
                        f"{source_name!r}={actual!r} differs from the compiled "
                        f"characterization value {expected!r}")
        self._built = Device(
            self.name,
            logical=logical,
            qec=qec,
            physical=physical,
            logical_to_qec=logical_to_qec,
            qec_to_physical=self.physical._bindings,
            qec_channels_to_physical=self.physical._channel_bindings,
            spacetime_plans=self.physical._spacetime_plans,
            compilers=self._compilers,
            operating_point=operating_point,
            metadata=self.metadata,
            source_module=self.source_module,
        )
        return self._built


__all__ = [
    "CarrierSelection",
    "LogicalRegionBuilder",
    "QECRegionBuilder",
    "QECChannelBuilder",
    "PhysicalResourceBuilder",
    "DeviceBuilder",
]
