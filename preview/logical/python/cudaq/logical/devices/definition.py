# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Layered device machines and the canonical whole-stack authoring builder."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .._core.immutable import (
    ImmutableValue,
    freeze_mapping,
)
from ..architecture.logical import (
    LogicalMachine,
    Space,
    Stream,
    capability,
)
from ..protocols.definition import ProtocolDefinition
from ..codes import (
    Code,
    CodeProfile,
    Encoding,
    EncodingHierarchy,
    EncodingProjection,
)


def _frozen_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return freeze_mapping(value)


def _member_name(value: str, *, what: str) -> str:
    if not isinstance(value, str) or not value or not value.isidentifier():
        raise ValueError(f"{what} must be a nonempty Python identifier")
    return value


@dataclass(frozen=True, slots=True)
class QECRegion:
    """One P2 encoded block pool.

    ``block_capacity`` counts encoded blocks. It is intentionally independent
    of the number of P1 logical owners that those blocks can host.
    """

    name: str
    encoding: Encoding
    block_capacity: int
    packing: str = "dense"
    metadata: Mapping[str, Any] | None = None
    role: str | None = None

    def __post_init__(self) -> None:
        _member_name(self.name, what="QEC region name")
        if not isinstance(self.encoding, Encoding):
            raise TypeError(
                "QECRegion.encoding must be a cudaq.logical.Encoding")
        if (not isinstance(self.block_capacity, int) or
                isinstance(self.block_capacity, bool) or
                self.block_capacity <= 0):
            raise TypeError("QECRegion.block_capacity must be a positive int")
        if not isinstance(self.packing, str) or not self.packing:
            raise ValueError("QECRegion.packing must be nonempty")
        if self.role not in (None, "compute", "memory", "factory", "scratch"):
            raise ValueError(
                "QECRegion.role must be compute, memory, factory, scratch, or None"
            )
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class QECArchitecture:
    """An explicitly bound P2 realization bundle."""

    name: str
    encoding: Encoding
    link_roots: tuple[Any, ...] = ()
    packing: str = "dense"
    auxiliary_regions: tuple[QECRegion, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _member_name(self.name, what="QEC architecture name")
        if not isinstance(self.encoding, Encoding):
            raise TypeError(
                "QECArchitecture.encoding must be a cudaq.logical.Encoding")
        if not isinstance(self.packing, str) or not self.packing:
            raise ValueError("QECArchitecture.packing must be nonempty")

        from ..gadgets import GadgetDefinition
        from ..qec.lowering import QECLowering

        definition_types = (
            Code,
            CodeProfile,
            Encoding,
            EncodingHierarchy,
            EncodingProjection,
            GadgetDefinition,
            ProtocolDefinition,
            QECLowering,
        )
        supplied = tuple(self.link_roots)
        if any(not isinstance(item, definition_types) for item in supplied):
            invalid = next(item for item in supplied
                           if not isinstance(item, definition_types))
            raise TypeError(
                "QECArchitecture.link_roots must contain CUDA-Q Logical QEC definitions; "
                f"got {type(invalid).__name__}")
        unique = []
        seen = set()
        for item in supplied:
            if id(item) in seen:
                continue
            seen.add(id(item))
            unique.append(item)

        auxiliary = tuple(self.auxiliary_regions)
        if any(not isinstance(region, QECRegion) for region in auxiliary):
            raise TypeError(
                "QECArchitecture.auxiliary_regions must contain QECRegion values"
            )
        names = tuple(region.name for region in auxiliary)
        if len(set(names)) != len(names):
            raise ValueError(
                "QECArchitecture auxiliary region template names must be unique"
            )
        if any(region.role != "scratch" for region in auxiliary):
            raise ValueError(
                "QECArchitecture auxiliary regions must use role='scratch'")

        object.__setattr__(self, "link_roots", tuple(unique))
        object.__setattr__(self, "auxiliary_regions", auxiliary)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))

    @property
    def code(self) -> Code:
        return self.encoding.code


class QECMachine(ImmutableValue):
    """Immutable P2 virtual machine of encoded block pools."""

    __slots__ = (
        "name",
        "regions",
        "metadata",
        "_members",
    )

    def __init__(
        self,
        name: str,
        regions: Mapping[str, QECRegion] | Iterable[QECRegion],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(regions, Mapping):
            normalized = tuple(
                replace(region, name=member_name)
                for member_name, region in regions.items())
        else:
            normalized = tuple(regions)
        if not normalized:
            raise ValueError("QECMachine requires at least one QEC region")
        if any(not isinstance(region, QECRegion) for region in normalized):
            raise TypeError("QECMachine regions must be QECRegion values")
        names = tuple(region.name for region in normalized)
        if len(set(names)) != len(names):
            raise ValueError("QECMachine region names must be unique")
        self.name = str(name)
        self.regions = normalized
        self.metadata = _frozen_mapping(metadata)
        self._members = MappingProxyType(
            {value.name: value for value in normalized})
        self._seal()

    def __getattr__(self, name: str) -> QECRegion:
        try:
            return self._members[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)


@dataclass(frozen=True, slots=True)
class LogicalToQECBinding:
    logical_region: Space
    qec_region: QECRegion
    packing: str = "dense"
    architecture: QECArchitecture | None = None
    auxiliary_regions: tuple[QECRegion, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.logical_region, Space):
            raise TypeError("logical-to-QEC binding requires a logical Space")
        if not isinstance(self.qec_region, QECRegion):
            raise TypeError("logical-to-QEC binding requires a QECRegion")
        if not isinstance(self.packing, str) or not self.packing:
            raise ValueError("logical-to-QEC packing must be nonempty")
        if self.packing != self.qec_region.packing:
            raise ValueError(
                "logical-to-QEC packing must match the bound QEC region packing"
            )
        if self.architecture is not None:
            if not isinstance(self.architecture, QECArchitecture):
                raise TypeError(
                    "logical-to-QEC architecture must be a QECArchitecture")
            if self.qec_region.encoding is not self.architecture.encoding:
                raise ValueError(
                    "logical-to-QEC architecture encoding must be the bound "
                    "QEC region encoding")
            if self.packing != self.architecture.packing:
                raise ValueError(
                    "logical-to-QEC packing must match the architecture packing"
                )
        auxiliary = tuple(self.auxiliary_regions)
        if any(not isinstance(region, QECRegion) for region in auxiliary):
            raise TypeError(
                "logical-to-QEC auxiliary regions must be QECRegion values")
        if self.architecture is None and auxiliary:
            raise ValueError(
                "logical-to-QEC auxiliary regions require an architecture binding"
            )
        if self.architecture is not None:
            expected = tuple(
                replace(template,
                        name=f"{self.qec_region.name}_{template.name}")
                for template in self.architecture.auxiliary_regions)
            if auxiliary != expected:
                raise ValueError(
                    "logical-to-QEC auxiliary regions must exactly expand the "
                    "architecture scratch templates")
        object.__setattr__(self, "auxiliary_regions", auxiliary)
        capacity = self.logical_region.capacity
        if capacity is not None:
            available = (self.qec_region.block_capacity *
                         self.qec_region.encoding.code.k)
            if capacity > available:
                raise ValueError(
                    f"logical region @{self.logical_region.name} requires "
                    f"{capacity} owners but QEC region @{self.qec_region.name} "
                    f"provides only {available} logical ports")


class Device(ImmutableValue):
    """Immutable manifest for one P1/P2 device stack."""

    __slots__ = (
        "name",
        "logical",
        "qec",
        "logical_to_qec",
        "metadata",
        "source_module",
    )

    def __init__(
        self,
        name: str,
        *,
        logical: LogicalMachine,
        qec: QECMachine | None = None,
        logical_to_qec: Iterable[LogicalToQECBinding] = (),
        metadata: Mapping[str, Any] | None = None,
        source_module: str | None = None,
    ) -> None:
        if not isinstance(logical, LogicalMachine):
            raise TypeError("Device.logical must be a LogicalMachine")
        if qec is not None and not isinstance(qec, QECMachine):
            raise TypeError("Device.qec must be a QECMachine or None")
        logical_to_qec = tuple(logical_to_qec)
        self._validate_refinements(logical, qec, logical_to_qec)
        self.name = str(name)
        self.logical = logical
        self.qec = qec
        self.logical_to_qec = logical_to_qec
        self.metadata = _frozen_mapping(metadata)
        self.source_module = source_module
        self._seal()

    @staticmethod
    def _validate_refinements(logical, qec, logical_to_qec) -> None:
        if qec is None:
            if logical_to_qec:
                raise ValueError(
                    "Device has refinements beyond its final layer")
            return
        if any(not isinstance(binding, LogicalToQECBinding)
               for binding in logical_to_qec):
            raise TypeError(
                "Device.logical_to_qec must contain LogicalToQECBinding values")
        logical_spaces = {id(space) for space in logical.spaces}
        qec_regions = {id(region) for region in qec.regions}
        if (len(logical_to_qec) != len(logical_spaces) or
            {id(binding.logical_region) for binding in logical_to_qec
            } != logical_spaces):
            raise ValueError(
                "LogicalToQEC bindings must cover every logical space exactly")
        if any(
                id(binding.qec_region) not in qec_regions
                for binding in logical_to_qec):
            raise ValueError(
                "LogicalToQEC binding references another QEC machine")
        if any(
                id(region) not in qec_regions
                for binding in logical_to_qec
                for region in binding.auxiliary_regions):
            raise ValueError(
                "LogicalToQEC auxiliary region references another QEC machine")
        primary_regions = {id(binding.qec_region) for binding in logical_to_qec}
        auxiliary_owners: dict[int, int] = {}
        for binding in logical_to_qec:
            for region in binding.auxiliary_regions:
                identity = id(region)
                auxiliary_owners[identity] = auxiliary_owners.get(identity,
                                                                  0) + 1
                if identity in primary_regions:
                    raise ValueError(
                        "a QEC region cannot be both a logical refinement and "
                        "architecture scratch")
                if region.role != "scratch":
                    raise ValueError(
                        "architecture auxiliary QEC regions must use role='scratch'"
                    )
        for region in qec.regions:
            identity = id(region)
            if identity in primary_regions:
                if region.role == "scratch":
                    raise ValueError(
                        "a logical refinement cannot target a scratch QEC region"
                    )
                continue
            if region.role != "scratch" or auxiliary_owners.get(identity) != 1:
                raise ValueError(
                    f"unrefined QEC region @{region.name} must be owned exactly "
                    "once as architecture scratch")

    @property
    def refinements(self):
        return self.logical_to_qec

    @property
    def layers(self):
        from ..stages import P1, P2

        layers = [P1]
        if self.qec is not None:
            layers.append(P2)
        return tuple(layers)

    @property
    def default_qec_region(self) -> QECRegion:
        """The first compute region's typed P2 refinement.

        This deterministic default exists for objective-free top-level entry
        gadgets. Reusable gadgets and heterogeneous workflows continue to use
        explicit regions.
        """

        compute = next(
            (space for space in self.logical.spaces
             if capability.logical_compute in space.capabilities),
            None,
        )
        if compute is None:
            raise ValueError(
                f"device @{self.name} has no logical compute region")
        binding = next(
            (binding for binding in self.logical_to_qec
             if binding.logical_region is compute or
             binding.logical_region.name == compute.name),
            None,
        )
        if binding is None:
            raise ValueError(
                f"device @{self.name} compute region @{compute.name} has no "
                "QEC refinement")
        return binding.qec_region

    def _has_same_static_stack(self, other) -> bool:
        """Whether ``other`` names the same immutable P1/P2 stack."""
        return (isinstance(other, Device) and self.name == other.name and
                self.logical is other.logical and self.qec is other.qec and
                self.logical_to_qec == other.logical_to_qec)

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)

    @property
    def machine_graph(self):
        from ..compiler.topology_view import MachineGraphView

        return MachineGraphView.from_device(self)


_BUILDER_EXPORTS = frozenset({
    "LogicalRegionBuilder",
    "QECRegionBuilder",
    "DeviceBuilder",
})

__all__ = [
    "QECRegion",
    "QECArchitecture",
    "QECMachine",
    "LogicalToQECBinding",
    "Device",
    *sorted(_BUILDER_EXPORTS),
]


def __getattr__(name: str):
    """Resolve historical builder imports from their mutable owner."""

    if name not in _BUILDER_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module(".builder", __package__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
