# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Mutable authoring handles for immutable layered device models."""

from __future__ import annotations

from dataclasses import replace
from math import ceil
import sys
from typing import Any, Iterable, Mapping

from ..devices.definition import (
    Device,
    LogicalToQECBinding,
    QECArchitecture,
    QECMachine,
    QECRegion,
    _frozen_mapping,
    _member_name,
)
from ..architecture.logical import (
    CapabilityKey,
    LogicalMachine,
    Space,
    SpaceDeclaration,
    Stream,
    capability,
)
from ..protocols.definition import ProtocolDefinition
from ..codes import (
    Code,
    Encoding,
)


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
        Returning their builder handles keeps later P2 authoring typed;
        callers never need to reconstruct generated names through ``getattr``.
        """

        return self._auxiliary_regions


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
        handle is the ordinary logical region used by QEC bindings
        and placement; the resource stream and supply channel are derived.
        """

        from ..protocols.definition import ProtocolDefinition
        from ..std import ResourceFlowRef, ResourceKind

        self._builder._assert_open()
        if not isinstance(produces, ResourceKind):
            raise TypeError(
                "logical.add_factory produces= requires a cudaq.logical.types.ResourceKind"
            )
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
                             "cudaq.logical.std.produce(produces)")
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
        from ..protocols.definition import ProtocolDefinition

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

    def __getattr__(self, name: str):
        try:
            return self._handles[name]
        except KeyError as exc:
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


class DeviceBuilder:
    """Canonical mutable authoring facade for an immutable P1/P2 Device."""

    __slots__ = (
        "name",
        "logical",
        "qec",
        "metadata",
        "source_module",
        "_base_logical",
        "_base_qec",
        "_built",
    )

    def __init__(
        self,
        name: str,
        *,
        logical: LogicalMachine | None = None,
        qec: QECMachine | None = None,
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
        self._built = None
        self.logical = _LogicalNamespace(self, logical)
        self.qec = _QECNamespace(self, qec)

    def _assert_open(self) -> None:
        if self._built is not None:
            raise RuntimeError(
                "DeviceBuilder is frozen after build(); create a new builder")

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
            } == {id(region) for region in self.qec._regions.values()}
            qec = QECMachine(
                f"{self.name}QECMachine",
                self.qec._regions.values(),
            ) if not imported_qec else self._base_qec
        self._built = Device(
            self.name,
            logical=logical,
            qec=qec,
            logical_to_qec=logical_to_qec,
            metadata=self.metadata,
            source_module=self.source_module,
        )
        return self._built


# These builder classes historically lived in cudaq.logical.model.device.  Preserve that
# durable identity for pickles, provenance, annotation resolution, wildcard
# imports, and compatibility diagnostics while direct imports migrate to this
# implementation owner.
_COMPATIBILITY_EXPORTS = (
    "LogicalRegionBuilder",
    "QECRegionBuilder",
    "DeviceBuilder",
)
for _compatibility_name in _COMPATIBILITY_EXPORTS:
    _compatibility_class = globals()[_compatibility_name]
    _compatibility_class.__module__ = "cudaq.logical.model.device"
del _compatibility_class, _compatibility_name

__all__ = list(_COMPATIBILITY_EXPORTS)
