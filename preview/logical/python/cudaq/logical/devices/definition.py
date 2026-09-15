# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Layered device machines and the canonical whole-stack authoring builder."""

from __future__ import annotations

from collections.abc import Set as AbstractSet
from dataclasses import InitVar, dataclass, field, replace
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from cudaq.logical._core.immutable import (
    ImmutableValue,
    freeze_mapping,
)
from cudaq.logical.analysis.evidence import Provenance
from cudaq.logical.architecture.logical import (
    Channel,
    CapabilityKey,
    LogicalMachine,
    RESOURCE_TRANSFER_CAPABILITY,
    Space,
    Stream,
    capability,
)
from cudaq.logical.devices.timing import TimingModel
from cudaq.logical.architecture.physical_definition import (
    PatchTopology,
    PhysicalMachine,
    ResourceClass,
    Topology,
)
from cudaq.logical.protocols.definition import ProtocolDefinition
from cudaq.logical.std import ResourceKind
from cudaq.logical.codes import (
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


def _compiler_key(value) -> str:
    """Validate one device-contributed compiler without importing its plugin.

    One immutable capability may implement the lattice-surgery P2 materializer,
    the physical projector, or both.  Keeping both on the existing device
    compiler tuple avoids a second registry while preserving distinct stage
    contracts and pipeline provenance.
    """

    key = getattr(value, "key", None)
    if not isinstance(key, str) or not key:
        raise TypeError(
            "device compilers require a nonempty versioned string key")
    materializer_methods = (
        "accepts",
        "architecture_digest",
        "accepts_pipeline",
        "solve",
        "materialize_p2",
        "verify_p2_materialization",
    )
    projector_methods = (
        "accepts_projection",
        "projection_architecture_digest",
        "accepts_projection_pipeline",
        "project_p3",
        "verify_projection",
    )
    materializer = all(
        callable(getattr(value, method, None))
        for method in materializer_methods)
    projector = all(
        callable(getattr(value, method, None)) for method in projector_methods)
    if not materializer and not projector:
        raise TypeError(
            f"device compiler {key!r} must implement the complete "
            "lattice-surgery materializer or physical-projector contract")
    if materializer and getattr(value, "pipeline", None) is None:
        raise TypeError(
            f"device compiler {key!r} must contribute a default P2 pipeline")
    if projector and getattr(value, "projection_pipeline", None) is None:
        raise TypeError(
            f"device compiler {key!r} must contribute a default P3 pipeline")
    return key


_FACTORY_CHARACTERIZATION_AUTH = object()


def _compiled_factory_evidence(characterization) -> Provenance:
    return Provenance(
        "computation",
        "qlx.factory-characterization/v1:"
        f"{characterization.build_sha256}:"
        f"{characterization.schedule_sha256}",
    )


@dataclass(frozen=True, slots=True)
class FactoryCharacterization:
    """Compiler evidence behind one compact physical factory model.

    The characterization identifies the exact verified P3 build and schedule,
    the typed output resource, the output event used for startup timing, and
    the physical resources exercised by the detailed graph or provisioned by
    an authenticated periodic plan. It is separate from :class:`FactoryModel`
    so a literature-backed early-study model remains lightweight and visibly
    distinct from a compiled result.
    """

    resource_kind: ResourceKind
    source_provider: str
    source_provider_sha256: str
    startup_cycles: float
    output_interval_cycles: float
    build_sha256: str
    schedule_sha256: str
    operating_point: str
    output_events: tuple[str, ...]
    physical_units: int
    physical_unit_kind: str
    code_distances: tuple[int, ...]
    timing_profile: tuple[tuple[str, float], ...]
    timing_source: str | None
    selection_events: tuple[str, ...] = ()
    _auth: InitVar[object] = None

    def __post_init__(self, _auth) -> None:
        if _auth is not _FACTORY_CHARACTERIZATION_AUTH:
            raise TypeError(
                "FactoryCharacterization is an opaque compiler artifact; "
                "obtain it from cudaq.logical.compiler.factory_model()")
        if not isinstance(self.resource_kind, ResourceKind):
            raise TypeError(
                "FactoryCharacterization.resource_kind must be a ResourceKind")
        if not isinstance(self.source_provider,
                          str) or not self.source_provider:
            raise ValueError(
                "FactoryCharacterization.source_provider must be nonempty")
        digest = self.source_provider_sha256
        if (not isinstance(digest, str) or not digest.startswith("sha256:") or
                len(digest) != 71 or
                any(character not in "0123456789abcdef"
                    for character in digest.removeprefix("sha256:"))):
            raise ValueError(
                "FactoryCharacterization.source_provider_sha256 must be a "
                "canonical lowercase sha256 commitment")
        for name in ("startup_cycles", "output_interval_cycles"):
            raw = getattr(self, name)
            if (isinstance(raw, bool) or not isinstance(raw, (int, float)) or
                    not isfinite(float(raw)) or float(raw) <= 0.0):
                raise TypeError(
                    f"FactoryCharacterization.{name} must be a finite "
                    "positive number")
            object.__setattr__(self, name, float(raw))
        if self.output_interval_cycles > self.startup_cycles:
            raise ValueError(
                "FactoryCharacterization output interval cannot exceed "
                "startup time")
        for name in ("build_sha256", "schedule_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64 or
                    any(character not in "0123456789abcdef"
                        for character in value)):
                raise ValueError(
                    f"FactoryCharacterization.{name} must be a lowercase "
                    "SHA-256 digest")
        if not isinstance(self.operating_point,
                          str) or not self.operating_point:
            raise ValueError(
                "FactoryCharacterization.operating_point must be nonempty")
        for name in ("output_events", "selection_events"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, str) or not value for value in values):
                raise TypeError(
                    f"FactoryCharacterization.{name} must contain nonempty "
                    "event identities")
            if len(values) != len(set(values)):
                raise ValueError(
                    f"FactoryCharacterization.{name} must be unique")
            object.__setattr__(self, name, values)
        if not self.output_events:
            raise ValueError(
                "FactoryCharacterization requires at least one output event")
        if (isinstance(self.physical_units, bool) or
                not isinstance(self.physical_units, int) or
                self.physical_units <= 0):
            raise TypeError(
                "FactoryCharacterization.physical_units must be a positive int")
        if (not isinstance(self.physical_unit_kind, str) or
                not self.physical_unit_kind):
            raise ValueError(
                "FactoryCharacterization.physical_unit_kind must be nonempty")
        code_distances = tuple(self.code_distances)
        if (not code_distances or any(
                isinstance(distance, bool) or not isinstance(distance, int) or
                distance <= 0 for distance in code_distances)):
            raise TypeError(
                "FactoryCharacterization.code_distances must contain "
                "positive integers")
        if code_distances != tuple(sorted(set(code_distances))):
            raise ValueError(
                "FactoryCharacterization.code_distances must be sorted and "
                "unique")
        object.__setattr__(self, "code_distances", code_distances)
        timing_profile = tuple(
            (name, float(value)) for name, value in self.timing_profile)
        if timing_profile != tuple(sorted(timing_profile)):
            raise ValueError(
                "FactoryCharacterization.timing_profile must be sorted")
        if len({name for name, _ in timing_profile}) != len(timing_profile):
            raise ValueError(
                "FactoryCharacterization.timing_profile keys must be unique")
        if any(not isinstance(name, str) or not name or not isfinite(value) or
               value < 0.0 for name, value in timing_profile):
            raise ValueError(
                "FactoryCharacterization.timing_profile must contain finite "
                "nonnegative named values")
        object.__setattr__(self, "timing_profile", timing_profile)
        if (self.timing_source is not None and
            (not isinstance(self.timing_source, str) or
             not self.timing_source)):
            raise ValueError(
                "FactoryCharacterization.timing_source must be a nonempty "
                "string or None")


def _factory_characterization_from_verified_schedule(**values):
    """Internal constructor for compiler-authenticated characterization."""

    return FactoryCharacterization(
        **values,
        _auth=_FACTORY_CHARACTERIZATION_AUTH,
    )


@dataclass(frozen=True, slots=True)
class FactoryModel:
    """Deterministic compact P3 timing model for one physical factory lane.

    ``startup_cycles`` is the time from an empty factory to its first
    available payload. ``output_interval_cycles`` is the proven spacing between
    subsequent payloads from one lane. It may be shorter than startup when a
    registered P3 recurrence independently verifies the pipeline initiation
    interval; otherwise one compiled output conservatively uses startup and
    does not claim pipelining. Both are expressed in the selected operating
    point's surface-code cycle so that the immutable device retains
    dimensionless architecture facts and P3 derives concrete nanoseconds.
    ``evidence`` states where a declared model came from. A future model
    characterized from a compiled factory retains computation provenance here.

    A ``guaranteed`` model has no selected rejection event.  A
    ``single_shot`` model is explicitly conditional on the retained P3
    selections accepting; it is useful for the same single-shot resource
    studies as a detailed factory attempt, but does not claim retry latency.
    ``characterization`` is present only when the timing was computed from a
    verified P3 schedule.
    """

    startup_cycles: float
    output_interval_cycles: float
    evidence: Provenance
    policy: str = "guaranteed"
    characterization: FactoryCharacterization | None = None

    def __post_init__(self) -> None:
        for name, raw in (
            ("startup_cycles", self.startup_cycles),
            ("output_interval_cycles", self.output_interval_cycles),
        ):
            if (isinstance(raw, bool) or not isinstance(raw, (int, float)) or
                    not isfinite(float(raw)) or float(raw) <= 0.0):
                raise TypeError(
                    f"FactoryModel.{name} must be a finite positive number")
            object.__setattr__(self, name, float(raw))
        if self.output_interval_cycles > self.startup_cycles:
            raise ValueError(
                "FactoryModel output interval cannot exceed startup time")
        if not isinstance(self.evidence, Provenance):
            raise TypeError("FactoryModel.evidence must be "
                            "cudaq.logical.analysis.Provenance")
        if self.policy not in {"guaranteed", "single_shot"}:
            raise ValueError(
                "FactoryModel policy must be guaranteed or single_shot")
        if (self.characterization is not None and
                not isinstance(self.characterization, FactoryCharacterization)):
            raise TypeError("FactoryModel.characterization must be a "
                            "FactoryCharacterization or None")
        if self.characterization is not None:
            if (self.startup_cycles != self.characterization.startup_cycles or
                    self.output_interval_cycles
                    != self.characterization.output_interval_cycles):
                raise ValueError(
                    "a characterized FactoryModel must retain the exact "
                    "compiler-derived startup and output interval")
            if self.evidence != _compiled_factory_evidence(
                    self.characterization):
                raise ValueError(
                    "a characterized FactoryModel must retain its exact "
                    "compiler-derived evidence")
            selected = bool(self.characterization.selection_events)
            if selected and self.policy != "single_shot":
                raise ValueError(
                    "a factory characterization with selection events must "
                    "use policy='single_shot'")
            if not selected and self.policy == "single_shot":
                raise ValueError(
                    "policy='single_shot' requires retained selection events")


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

        from cudaq.logical.gadgets import GadgetDefinition, GadgetProfile
        from cudaq.logical.qec.lowering import QECLowering

        definition_types = (
            Code,
            CodeProfile,
            Encoding,
            EncodingHierarchy,
            EncodingProjection,
            GadgetDefinition,
            GadgetProfile,
            ProtocolDefinition,
            QECLowering,
        )
        supplied = tuple(self.link_roots)
        if any(not isinstance(item, definition_types) for item in supplied):
            invalid = next(item for item in supplied
                           if not isinstance(item, definition_types))
            raise TypeError(
                "QECArchitecture.link_roots must contain QLX QEC definitions; "
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
    """Immutable P2 virtual machine of encoded block pools and channel ports."""

    __slots__ = (
        "name",
        "regions",
        "channel_ports",
        "channels",
        "metadata",
        "_members",
    )

    def __init__(
        self,
        name: str,
        regions: Mapping[str, QECRegion] | Iterable[QECRegion],
        *,
        channel_ports: Iterable["QECChannelPort"] = (),
        channels: Iterable["QECChannelRealization"] = (),
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
        channel_ports = tuple(channel_ports)
        channels = tuple(channels)
        if any(not isinstance(value, QECChannelPort)
               for value in channel_ports):
            raise TypeError(
                "QECMachine channel_ports must be QECChannelPort values")
        if any(not isinstance(value, QECChannelRealization)
               for value in channels):
            raise TypeError(
                "QECMachine channels must be QECChannelRealization values")
        all_names = (*names, *(value.name for value in channel_ports),
                     *(value.name for value in channels))
        if len(set(all_names)) != len(all_names):
            raise ValueError("QECMachine member names must be unique")
        region_ids = {id(region) for region in normalized}
        if any(id(port.region) not in region_ids for port in channel_ports):
            raise ValueError("QEC channel port references another QEC machine")
        port_ids = {id(port) for port in channel_ports}
        if any(
                id(channel.source) not in port_ids or
                id(channel.destination) not in port_ids
                for channel in channels):
            raise ValueError(
                "QEC channel references a port outside its machine")
        self.name = str(name)
        self.regions = normalized
        self.channel_ports = channel_ports
        self.channels = channels
        self.metadata = _frozen_mapping(metadata)
        self._members = MappingProxyType({
            value.name: value
            for value in (*normalized, *channel_ports, *channels)
        })
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


@dataclass(frozen=True, slots=True)
class QECChannelPort:
    """Persistent P2 channel boundary owned by one encoded region.

    This is a device-level port that may participate in many protocol epochs;
    it is distinct from :class:`cudaq.logical.Port`, which describes one
    gadget call ABI.
    Provider-specific geometry remains in immutable ``metadata``.
    """

    name: str
    region: QECRegion
    slot: int
    capabilities: tuple[CapabilityKey, ...] = ()
    concurrency: int = 1
    provider: str = "qlx"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _member_name(self.name, what="QEC channel-port name")
        if not isinstance(self.region, QECRegion):
            raise TypeError(
                "QECChannelPort.region must be a cudaq.logical.QECRegion")
        if (not isinstance(self.slot, int) or isinstance(self.slot, bool) or
                self.slot < 0 or self.slot >= self.region.block_capacity):
            raise ValueError(
                "QECChannelPort.slot must select an encoded block in its region"
            )
        if (not isinstance(self.concurrency, int) or
                isinstance(self.concurrency, bool) or self.concurrency <= 0):
            raise TypeError("QECChannelPort.concurrency must be a positive int")
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("QECChannelPort.provider must be nonempty")
        capabilities = tuple(self.capabilities)
        if not capabilities or any(
                not isinstance(value, CapabilityKey) for value in capabilities):
            raise TypeError(
                "QECChannelPort.capabilities must contain typed qlx capabilities"
            )
        if len(set(capabilities)) != len(capabilities):
            raise ValueError("QECChannelPort.capabilities must be unique")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class QECChannelRealization:
    """Selected P2 realization of one P1 capability-bearing channel."""

    name: str
    logical_channel: Channel
    source: QECChannelPort
    destination: QECChannelPort
    capabilities: tuple[CapabilityKey, ...] = ()
    concurrency: int = 1
    provider: str = "qlx"
    metadata: Mapping[str, Any] | None = None
    protocol: ProtocolDefinition | None = None

    def __post_init__(self) -> None:
        _member_name(self.name, what="QEC channel-realization name")
        if not isinstance(self.logical_channel, Channel):
            raise TypeError(
                "QECChannelRealization.logical_channel must be a cudaq.logical.Channel"
            )
        if not isinstance(self.source, QECChannelPort) or not isinstance(
                self.destination, QECChannelPort):
            raise TypeError(
                "QECChannelRealization endpoints must be QECChannelPort values")
        capabilities = tuple(self.capabilities or
                             self.logical_channel.capabilities)
        if not capabilities:
            raise ValueError(
                "QEC channel realizations require at least one P1 capability")
        if any(not isinstance(value, CapabilityKey) for value in capabilities):
            raise TypeError(
                "QECChannelRealization.capabilities must be typed qlx capabilities"
            )
        if len(set(capabilities)) != len(capabilities):
            raise ValueError(
                "QECChannelRealization.capabilities must be unique")
        if any(value not in self.logical_channel.capabilities
               for value in capabilities):
            raise ValueError(
                "QEC channel realization cannot add a capability absent from P1"
            )
        for port in (self.source, self.destination):
            if any(value not in self.logical_channel.capabilities
                   for value in port.capabilities):
                raise ValueError(
                    "QEC channel-port capabilities must be authorized by the "
                    "P1 logical channel")
        if any(value not in self.source.capabilities
               for value in capabilities) or any(
                   value not in self.destination.capabilities
                   for value in capabilities):
            raise ValueError(
                "QEC channel-realization capabilities must be offered by both ports"
            )
        if (not isinstance(self.concurrency, int) or
                isinstance(self.concurrency, bool) or self.concurrency <= 0):
            raise TypeError(
                "QECChannelRealization.concurrency must be a positive int")
        limits = (
            self.logical_channel.concurrency,
            self.source.concurrency,
            self.destination.concurrency,
        )
        if any(limit is not None and self.concurrency > limit
               for limit in limits):
            raise ValueError(
                "QEC channel concurrency exceeds its P1 channel or channel port"
            )
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("QECChannelRealization.provider must be nonempty")
        if self.protocol is not None and not isinstance(self.protocol,
                                                        ProtocolDefinition):
            raise TypeError(
                "QECChannelRealization.protocol must be a cudaq.logical.ProtocolDefinition"
            )
        if RESOURCE_TRANSFER_CAPABILITY in capabilities:
            if self.protocol is None:
                raise ValueError(
                    "resource-transfer channel realizations require a typed "
                    "delivery protocol")
            kinds = tuple(endpoint.produces for endpoint in (
                self.logical_channel.source,
                self.logical_channel.destination,
            ) if isinstance(endpoint, Stream))
            if not kinds or len(set(kinds)) != 1:
                raise ValueError(
                    "resource-transfer channel endpoints must determine one "
                    "resource kind")
            expected = kinds[0]
            if (len(self.protocol.signature.parameters) != 1 or
                    len(self.protocol._boundary_output_annotations()) != 1 or
                    self.protocol._resource_input_kinds() != (expected,) or
                    self.protocol._resource_output_kinds() != (expected,)):
                raise TypeError(
                    "resource-transfer delivery protocol must consume and "
                    "return exactly one resource of the channel kind")
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "metadata", _frozen_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class QECToPhysicalBinding:
    qec_region: QECRegion
    resources: tuple[ResourceClass, ...]
    topology: Topology | None = None
    patch_topology: PatchTopology | None = None
    factory_model: FactoryModel | None = None
    name: str | None = None

    def __init__(
        self,
        qec_region: QECRegion,
        resources: Iterable[ResourceClass],
        *,
        topology: Topology | None = None,
        patch_topology: PatchTopology | None = None,
        factory_model: FactoryModel | None = None,
        name: str | None = None,
    ) -> None:
        resources = tuple(resources)
        if not isinstance(qec_region, QECRegion):
            raise TypeError("QEC-to-physical binding requires a QECRegion")
        if not resources or any(not isinstance(resource, ResourceClass)
                                for resource in resources):
            raise TypeError(
                "QEC-to-physical binding requires physical ResourceClass values"
            )
        if topology is not None:
            if not isinstance(topology, Topology):
                raise TypeError(
                    "QEC-to-physical topology must be a cudaq.logical.Topology")
            if topology.num_nodes is not None:
                counts = {resource.count for resource in resources}
                if len(counts) != 1:
                    raise ValueError(
                        "one fixed-size carrier topology cannot refine resource "
                        "pools with different counts")
                topology = topology._bind_resource_count(next(iter(counts)))
        if patch_topology is not None:
            if not isinstance(patch_topology, PatchTopology):
                raise TypeError(
                    "QEC-to-physical patch_topology must be a cudaq.logical.PatchTopology"
                )
            if topology is None:
                raise ValueError("patch topology requires a carrier topology")
            if len(resources) != 1:
                raise ValueError(
                    "patch topology currently requires one physical resource class"
                )
            patch_topology = patch_topology._bind(
                topology=topology,
                capacity=qec_region.block_capacity,
                resource_count=resources[0].count,
            )
        if factory_model is not None and not isinstance(factory_model,
                                                        FactoryModel):
            raise TypeError(
                "QEC-to-physical factory_model must be a FactoryModel")
        object.__setattr__(self, "qec_region", qec_region)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "topology", topology)
        object.__setattr__(self, "patch_topology", patch_topology)
        object.__setattr__(self, "factory_model", factory_model)
        object.__setattr__(self, "name", name or qec_region.name)


@dataclass(frozen=True, slots=True)
class QECChannelToPhysicalBinding:
    """P3 carrier pools available to one selected P2 channel realization.

    ``transport_claims`` and ``endpoint_occupancy`` are the optional detailed
    P3 route facts used to schedule a representative transport before a
    compact model exists.  They refine, rather than replace, ``resources``:
    every claim must select a slice from one of those bound pools.
    """

    qec_channel: QECChannelRealization
    resources: tuple[ResourceClass, ...]
    transport_claims: tuple[Any, ...]
    endpoint_occupancy: tuple[int, int] | None
    transport_model: Any | None

    def __init__(
        self,
        qec_channel: QECChannelRealization,
        resources: Iterable[ResourceClass],
        *,
        transport_claims=(),
        endpoint_occupancy=None,
        transport_model=None,
    ) -> None:
        resources = tuple(resources)
        if not isinstance(qec_channel, QECChannelRealization):
            raise TypeError(
                "QEC-channel-to-physical binding requires a QEC realization")
        if not resources or any(not isinstance(resource, ResourceClass)
                                for resource in resources):
            raise TypeError(
                "QEC-channel-to-physical binding requires physical resources")
        if len({id(resource) for resource in resources}) != len(resources):
            raise ValueError("QEC-channel-to-physical resources must be unique")
        from .component_models import (PhysicalResourceClaim, TransportModel,
                                       _reject_overlapping_claims)
        transport_claims = tuple(transport_claims)
        if any(not isinstance(claim, PhysicalResourceClaim)
               for claim in transport_claims):
            raise TypeError(
                "QEC-channel-to-physical transport_claims must contain "
                "PhysicalResourceClaim values")
        permitted = {id(resource) for resource in resources}
        if any(
                id(claim.resource_class) not in permitted
                for claim in transport_claims):
            raise ValueError(
                "detailed transport claims resources outside its selected "
                "P3 channel binding")
        _reject_overlapping_claims(
            transport_claims,
            what="QECChannelToPhysicalBinding",
        )
        if endpoint_occupancy is not None:
            endpoint_occupancy = tuple(endpoint_occupancy)
            if len(endpoint_occupancy) != 2 or any(
                    isinstance(value, bool) or not isinstance(value, int) or
                    value <= 0 for value in endpoint_occupancy):
                raise TypeError(
                    "QEC-channel-to-physical endpoint_occupancy requires two "
                    "positive ints")
        if bool(transport_claims) != (endpoint_occupancy is not None):
            raise ValueError(
                "detailed transport claims and endpoint occupancy must appear "
                "together")
        if endpoint_occupancy is not None:
            source_units, destination_units = endpoint_occupancy
            if (source_units > qec_channel.source.concurrency or
                    destination_units > qec_channel.destination.concurrency or
                    max(source_units,
                        destination_units) > qec_channel.concurrency):
                raise ValueError(
                    "transport endpoint occupancy exceeds P1/P2 channel limits")
        if transport_model is not None:
            if not isinstance(transport_model, TransportModel):
                raise TypeError(
                    "QEC-channel-to-physical transport_model must be a "
                    "TransportModel")
            if any(
                    id(claim.resource_class) not in permitted
                    for claim in transport_model.resources):
                raise ValueError(
                    "transport model claims resources outside its selected "
                    "P3 channel binding")
            source_units, destination_units = transport_model.endpoint_occupancy
            if (source_units > qec_channel.source.concurrency or
                    destination_units > qec_channel.destination.concurrency or
                    max(source_units,
                        destination_units) > qec_channel.concurrency):
                raise ValueError(
                    "transport endpoint occupancy exceeds P1/P2 channel limits")
            if transport_claims and (
                    transport_model.resources != transport_claims or
                    transport_model.endpoint_occupancy != endpoint_occupancy):
                raise ValueError(
                    "transport model must retain the exact detailed P3 route "
                    "claims and endpoint occupancy")
        object.__setattr__(self, "qec_channel", qec_channel)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "transport_claims", transport_claims)
        object.__setattr__(self, "endpoint_occupancy", endpoint_occupancy)
        object.__setattr__(self, "transport_model", transport_model)


@dataclass(frozen=True, slots=True)
class PhysicalOperatingPoint:
    """Replaceable timing, calibration, and cost assumptions."""

    timing: TimingModel | Mapping[str, Any] | None = None
    calibration: Mapping[str, Any] | None = None
    costs: Mapping[str, Any] | None = None
    target_compatibility: tuple[str, ...] = ()
    name: str = "default"
    timing_source: str | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        timing_source = (self.timing.source if isinstance(
            self.timing, TimingModel) else None)
        object.__setattr__(self, "timing", _frozen_mapping(self.timing))
        object.__setattr__(self, "timing_source", timing_source)
        object.__setattr__(self, "calibration",
                           _frozen_mapping(self.calibration))
        object.__setattr__(self, "costs", _frozen_mapping(self.costs))
        object.__setattr__(
            self,
            "target_compatibility",
            tuple(map(str, self.target_compatibility)),
        )


class Device(ImmutableValue):
    """Immutable manifest for one contiguous layered device stack."""

    __slots__ = (
        "name",
        "logical",
        "qec",
        "physical",
        "logical_to_qec",
        "qec_to_physical",
        "qec_channels_to_physical",
        "spacetime_plans",
        "_compilers",
        "operating_point",
        "metadata",
        "source_module",
    )

    def __init__(
        self,
        name: str,
        *,
        logical: LogicalMachine,
        qec: QECMachine | None = None,
        physical: PhysicalMachine | None = None,
        logical_to_qec: Iterable[LogicalToQECBinding] = (),
        qec_to_physical: Iterable[QECToPhysicalBinding] = (),
        qec_channels_to_physical: Iterable[QECChannelToPhysicalBinding] = (),
        spacetime_plans: Iterable[Any] = (),
        compilers: Iterable[Any] = (),
        operating_point: PhysicalOperatingPoint | None = None,
        metadata: Mapping[str, Any] | None = None,
        source_module: str | None = None,
    ) -> None:
        if not isinstance(logical, LogicalMachine):
            raise TypeError("Device.logical must be a LogicalMachine")
        if qec is not None and not isinstance(qec, QECMachine):
            raise TypeError("Device.qec must be a QECMachine or None")
        if physical is not None and not isinstance(physical, PhysicalMachine):
            raise TypeError("Device.physical must be a PhysicalMachine or None")
        if physical is not None and qec is None:
            raise ValueError(
                "Device cannot skip QEC between logical and physical")
        if operating_point is not None and physical is None:
            raise ValueError(
                "PhysicalOperatingPoint requires a PhysicalMachine")

        logical_to_qec = tuple(logical_to_qec)
        qec_to_physical = self._canonicalize_qec_to_physical(
            physical, tuple(qec_to_physical))
        qec_channels_to_physical = tuple(qec_channels_to_physical)
        spacetime_plans = tuple(spacetime_plans)
        compilers = tuple(compilers)
        compiler_keys = tuple(_compiler_key(value) for value in compilers)
        if len(set(compiler_keys)) != len(compiler_keys):
            raise ValueError("Device compiler keys must be unique")
        self._validate_refinements(
            logical,
            qec,
            physical,
            logical_to_qec,
            qec_to_physical,
            qec_channels_to_physical,
        )
        from .component_models import SpacetimePlanModel
        if any(not isinstance(model, SpacetimePlanModel)
               for model in spacetime_plans):
            raise TypeError(
                "Device.spacetime_plans must contain SpacetimePlanModel values")
        if len({model.protocol.name for model in spacetime_plans
               }) != len(spacetime_plans):
            raise ValueError(
                "Device may bind only one spacetime plan per source protocol")
        if physical is None and spacetime_plans:
            raise ValueError("spacetime plans require a physical machine")
        physical_resources = ({
            id(resource) for resource in physical.resource_classes
        } if physical is not None else set())
        if any(
                id(claim.resource_class) not in physical_resources
                for model in spacetime_plans
                for phase in model.phases
                for claim in phase.resources):
            raise ValueError(
                "spacetime plan claims a resource outside its physical machine")
        if spacetime_plans or any(binding.transport_model is not None
                                  for binding in qec_channels_to_physical):
            if operating_point is None:
                raise ValueError(
                    "compact physical component models require an operating point"
                )
            from cudaq.logical.compiler.component_identity import (
                channel_binding_sha256,
                channel_identity_sha256,
                logical_channel_sha256,
                physical_architecture_sha256,
                protocol_contract,
                timing_profile,
                transport_model_sha256,
            )
            architecture_sha256 = physical_architecture_sha256(physical)
            selected_timing = timing_profile(operating_point)
            selected_distances = {
                region.encoding.code.d.conservative_value
                for region in (() if qec is None else qec.regions)
                if region.encoding.code.d.conservative_value is not None
            }
            for model in spacetime_plans:
                characterization = model.characterization
                if characterization is None:
                    if model.code_distances and not set(
                            model.code_distances).issubset(selected_distances):
                        raise ValueError(
                            "asserted spacetime model code distance is not "
                            "selected by this device")
                    continue
                contract = protocol_contract(model.protocol)
                if (characterization.source_protocol_sha256
                        != contract["source_sha256"] or
                        characterization.objective_sha256
                        != contract["objective_sha256"] or
                        characterization.boundary_sha256
                        != contract["boundary_sha256"]):
                    raise ValueError(
                        "compiled spacetime plan source protocol, objective, "
                        "or boundary ABI differs from the selected protocol")
                if (characterization.architecture_sha256
                        != architecture_sha256):
                    raise ValueError(
                        "compiled spacetime plan targets another physical "
                        "architecture")
                if (characterization.operating_point != operating_point.name or
                        characterization.timing_source
                        != operating_point.timing_source or
                        characterization.timing_profile != selected_timing):
                    raise ValueError(
                        "compiled spacetime plan operating point or timing "
                        "source differs from the selected device")
            for binding in qec_channels_to_physical:
                model = binding.transport_model
                if model is None or model.characterization is None:
                    continue
                characterization = model.characterization
                protocol = binding.qec_channel.protocol
                if protocol is None:
                    raise ValueError(
                        "characterized transport requires one selected typed "
                        "delivery protocol")
                if (characterization.channel_sha256 != logical_channel_sha256(
                        binding.qec_channel) or
                        characterization.realization_sha256
                        != channel_identity_sha256(binding.qec_channel) or
                        characterization.protocol_sha256
                        != protocol_contract(protocol)["source_sha256"] or
                        characterization.binding_sha256
                        != channel_binding_sha256(
                            QECChannelToPhysicalBinding(
                                binding.qec_channel,
                                binding.resources,
                                transport_claims=(binding.transport_claims),
                                endpoint_occupancy=(
                                    binding.endpoint_occupancy)))):
                    raise ValueError(
                        "compiled transport source channel, protocol, or "
                        "physical binding differs from the selected device")
                if (characterization.architecture_sha256 != architecture_sha256
                        or characterization.operating_point
                        != operating_point.name or
                        characterization.timing_source
                        != operating_point.timing_source or
                        characterization.timing_profile != selected_timing or
                        characterization.model_sha256
                        != transport_model_sha256(model)):
                    raise ValueError(
                        "compiled transport physical model or operating point "
                        "differs from the selected device")
        self.name = str(name)
        self.logical = logical
        self.qec = qec
        self.physical = physical
        self.logical_to_qec = logical_to_qec
        self.qec_to_physical = qec_to_physical
        self.qec_channels_to_physical = qec_channels_to_physical
        self.spacetime_plans = spacetime_plans
        self._compilers = compilers
        self.operating_point = operating_point
        self.metadata = _frozen_mapping(metadata)
        self.source_module = source_module
        self._seal()

    @staticmethod
    def _canonicalize_qec_to_physical(
        physical,
        bindings,
    ) -> tuple[QECToPhysicalBinding, ...]:
        """Rebind exact refinements to the owning machine's topology value."""

        if physical is None:
            return bindings
        topologies = {
            topology.name: topology for topology in physical.topologies
        }
        normalized = []
        for binding in bindings:
            if (not isinstance(binding, QECToPhysicalBinding) or
                    binding.topology is None):
                normalized.append(binding)
                continue
            topology = topologies.get(binding.topology.name)
            if topology is None or topology != binding.topology:
                normalized.append(binding)
                continue
            if binding.topology is topology:
                normalized.append(binding)
                continue
            normalized.append(
                QECToPhysicalBinding(
                    binding.qec_region,
                    binding.resources,
                    topology=topology,
                    patch_topology=binding.patch_topology,
                    factory_model=binding.factory_model,
                    name=binding.name,
                ))
        return tuple(normalized)

    @staticmethod
    def _validate_refinements(
        logical,
        qec,
        physical,
        logical_to_qec,
        qec_to_physical,
        qec_channels_to_physical,
    ) -> None:
        if qec is None:
            if logical_to_qec or qec_to_physical or qec_channels_to_physical:
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
        logical_channels = {id(channel) for channel in logical.channels}
        if any(
                id(channel.logical_channel) not in logical_channels
                for channel in qec.channels):
            raise ValueError("P2 channel references another logical machine")
        selected_logical_channels = tuple(
            id(channel.logical_channel) for channel in qec.channels)
        if len(set(selected_logical_channels)) != len(
                selected_logical_channels):
            raise ValueError(
                "one P1 channel cannot have multiple selected P2 realizations")
        qec_for_logical_name = {
            binding.logical_region.name: binding.qec_region
            for binding in logical_to_qec
        }

        def endpoint_region(endpoint):
            if isinstance(endpoint, Space):
                return qec_for_logical_name.get(endpoint.name)
            if isinstance(endpoint, Stream) and endpoint.region is not None:
                return qec_for_logical_name.get(endpoint.region.name)
            return None

        for channel in qec.channels:
            expected = (
                endpoint_region(channel.logical_channel.source),
                endpoint_region(channel.logical_channel.destination),
            )
            if None in expected or (
                    channel.source.region,
                    channel.destination.region,
            ) != expected:
                raise ValueError(
                    "P2 channel ports must refine their P1 channel endpoints")
        if physical is None:
            if qec_to_physical or qec_channels_to_physical:
                raise ValueError(
                    "Device has refinements beyond its final layer")
            return
        if any(not isinstance(binding, QECToPhysicalBinding)
               for binding in qec_to_physical):
            raise TypeError(
                "Device.qec_to_physical must contain QECToPhysicalBinding values"
            )
        if (len(qec_to_physical) != len(qec_regions) or
            {id(binding.qec_region) for binding in qec_to_physical
            } != qec_regions):
            raise ValueError(
                "QECToPhysical bindings must cover every QEC region exactly")
        physical_resources = {
            id(resource) for resource in physical.resource_classes
        }
        if any(
                id(resource) not in physical_resources
                for binding in qec_to_physical
                for resource in binding.resources):
            raise ValueError(
                "QECToPhysical binding references another PhysicalMachine")
        qec_channels = {id(channel) for channel in qec.channels}
        if any(not isinstance(binding, QECChannelToPhysicalBinding)
               for binding in qec_channels_to_physical):
            raise TypeError("Device.qec_channels_to_physical must contain "
                            "QECChannelToPhysicalBinding values")
        if (len(qec_channels_to_physical) != len(qec_channels) or
            {id(binding.qec_channel) for binding in qec_channels_to_physical
            } != qec_channels):
            raise ValueError(
                "QEC channel physical bindings must cover every P2 channel "
                "exactly")
        if any(
                id(resource) not in physical_resources
                for binding in qec_channels_to_physical
                for resource in binding.resources):
            raise ValueError(
                "QEC channel physical binding references another machine")
        physical_topologies = {
            topology.name: topology for topology in physical.topologies
        }
        for binding in qec_to_physical:
            if binding.topology is None:
                continue
            topology = physical_topologies.get(binding.topology.name)
            if topology is None:
                raise ValueError(
                    "QECToPhysical binding references another PhysicalMachine")
            if topology is not binding.topology:
                raise ValueError(
                    "QECToPhysical binding topology conflicts with its "
                    "PhysicalMachine declaration")
        claimed_carriers: dict[tuple[int, int], tuple[str, int]] = {}
        for binding in qec_to_physical:
            if binding.patch_topology is None:
                continue
            resource = binding.resources[0]
            for slot, group in enumerate(binding.patch_topology.carrier_groups):
                for carrier in group:
                    key = (id(resource), carrier)
                    previous = claimed_carriers.get(key)
                    if previous is not None:
                        previous_region, previous_slot = previous
                        raise ValueError(
                            f"carrier {carrier} in resource class "
                            f"@{resource.name} is claimed by both "
                            f"{previous_region}[{previous_slot}] and "
                            f"{binding.qec_region.name}[{slot}]")
                    claimed_carriers[key] = (
                        binding.qec_region.name,
                        slot,
                    )

    @property
    def refinements(self):
        return (
            *self.logical_to_qec,
            *self.qec_to_physical,
            *self.qec_channels_to_physical,
        )

    @property
    def layers(self):
        from cudaq.logical.stages import (
            P1,
            P2,
            P3,
        )

        layers = [P1]
        if self.qec is not None:
            layers.append(P2)
        if self.physical is not None:
            layers.append(P3)
        return tuple(layers)

    @property
    def compilers(self) -> tuple[Any, ...]:
        """Versioned compiler capabilities frozen into this device."""

        return self._compilers

    def with_operating_point(
            self, operating_point: PhysicalOperatingPoint) -> "Device":
        if not isinstance(operating_point, PhysicalOperatingPoint):
            raise TypeError(
                "Device.with_operating_point requires PhysicalOperatingPoint")
        return Device(
            self.name,
            logical=self.logical,
            qec=self.qec,
            physical=self.physical,
            logical_to_qec=self.logical_to_qec,
            qec_to_physical=self.qec_to_physical,
            qec_channels_to_physical=self.qec_channels_to_physical,
            spacetime_plans=self.spacetime_plans,
            compilers=self.compilers,
            operating_point=operating_point,
            metadata=self.metadata,
            source_module=self.source_module,
        )

    def with_spacetime_plan(self, model) -> "Device":
        """Return this exact stack with one compact synchronous P3 plan."""

        from .component_models import SpacetimePlanModel

        if not isinstance(model, SpacetimePlanModel):
            raise TypeError(
                "Device.with_spacetime_plan requires a SpacetimePlanModel")
        if self.physical is None or self.operating_point is None:
            raise ValueError(
                "compact spacetime plans require a physical machine and "
                "operating point")
        return Device(
            self.name,
            logical=self.logical,
            qec=self.qec,
            physical=self.physical,
            logical_to_qec=self.logical_to_qec,
            qec_to_physical=self.qec_to_physical,
            qec_channels_to_physical=self.qec_channels_to_physical,
            spacetime_plans=(*self.spacetime_plans, model),
            compilers=self.compilers,
            operating_point=self.operating_point,
            metadata=self.metadata,
            source_module=self.source_module,
        )

    def with_transport_model(self, channel, model) -> "Device":
        """Return this exact stack with one selected channel model replaced."""

        from .component_models import TransportModel

        if not isinstance(channel, QECChannelRealization):
            raise TypeError(
                "Device.with_transport_model requires a QECChannelRealization")
        if not isinstance(model, TransportModel):
            raise TypeError(
                "Device.with_transport_model requires a TransportModel")
        matches = tuple(binding for binding in self.qec_channels_to_physical
                        if binding.qec_channel is channel)
        if len(matches) != 1:
            raise ValueError(
                "transport model channel must have one exact P3 binding")
        replaced = tuple(
            QECChannelToPhysicalBinding(
                binding.qec_channel,
                binding.resources,
                transport_claims=binding.transport_claims,
                endpoint_occupancy=binding.endpoint_occupancy,
                transport_model=model,
            ) if binding is matches[0] else binding
            for binding in self.qec_channels_to_physical)
        return Device(
            self.name,
            logical=self.logical,
            qec=self.qec,
            physical=self.physical,
            logical_to_qec=self.logical_to_qec,
            qec_to_physical=self.qec_to_physical,
            qec_channels_to_physical=replaced,
            spacetime_plans=self.spacetime_plans,
            compilers=self.compilers,
            operating_point=self.operating_point,
            metadata=self.metadata,
            source_module=self.source_module,
        )

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

    def with_noise(self, noise: NoiseModel) -> "Device":
        """Derive a device variant with one replacement typed noise model."""

        if not isinstance(noise, NoiseModel):
            raise TypeError(
                "Device.with_noise requires a cudaq.logical.NoiseModel")
        point = self.operating_point
        if point is None:
            point = PhysicalOperatingPoint(noise=noise)
        else:
            point = replace(point, noise=noise)
        return self.with_operating_point(point)

    def with_added_noise(self, *mechanisms: NoiseMechanism) -> "Device":
        """Derive a device variant by appending physical noise mechanisms."""

        if not mechanisms:
            raise ValueError("Device.with_added_noise requires a mechanism")
        if any(not isinstance(item, NoiseMechanism) for item in mechanisms):
            raise TypeError(
                "Device.with_added_noise expects NoiseMechanism values")
        current = (None if self.operating_point is None else
                   self.operating_point.noise)
        if isinstance(current, NoiseModel):
            noise = current.with_added_noise(*mechanisms)
        elif not current:
            noise = NoiseModel(
                mechanisms,
                name=f"{self.name}_noise",
            )
        else:
            raise TypeError(
                "Device.with_added_noise requires a typed NoiseModel "
                "operating point")
        return self.with_noise(noise)

    def _has_same_static_stack(self, other) -> bool:
        """Whether ``other`` differs only in its replaceable operating point."""

        return (isinstance(other, Device) and self.name == other.name and
                self.logical is other.logical and self.qec is other.qec and
                self.physical is other.physical and
                self.logical_to_qec == other.logical_to_qec and
                self.qec_to_physical == other.qec_to_physical and
                self.qec_channels_to_physical == other.qec_channels_to_physical
                and self.spacetime_plans == other.spacetime_plans and
                self.compilers == other.compilers)

    def incomplete_streams(self) -> tuple[Stream, ...]:
        return tuple(stream for stream in self.logical.streams
                     if not stream.is_physically_complete)

    def require_physical_completeness(self, *, context: str) -> None:
        from ..errors import IncompletePhysicalModel

        incomplete = self.incomplete_streams()
        if incomplete:
            names = ", ".join(
                sorted(stream.name or "?" for stream in incomplete))
            raise IncompletePhysicalModel(
                f"{context} requires a complete physical model, but device "
                f"@{self.name} has resource streams with no declared supply: "
                f"{names}")

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)

    @property
    def machine_graph(self):
        from ..compiler.topology_view import MachineGraphView

        return MachineGraphView.from_device(self)

    @property
    def patch_topology_graph(self):
        from ..compiler.topology_view import PatchTopologyView

        return PatchTopologyView.from_device(self)

    @property
    def carrier_graph(self):
        from ..compiler.topology_view import CarrierGraphView

        return CarrierGraphView.from_device(self)

    @property
    def stack_graph(self):
        from ..compiler.topology_view import DeviceStackGraphView

        return DeviceStackGraphView.from_device(self)


_BUILDER_EXPORTS = frozenset({
    "CarrierSelection",
    "LogicalRegionBuilder",
    "QECRegionBuilder",
    "QECChannelBuilder",
    "PhysicalResourceBuilder",
    "DeviceBuilder",
})

__all__ = [
    "QECRegion",
    "QECArchitecture",
    "QECMachine",
    "LogicalToQECBinding",
    "QECChannelPort",
    "QECChannelRealization",
    "QECToPhysicalBinding",
    "QECChannelToPhysicalBinding",
    "PhysicalOperatingPoint",
    "Device",
    *sorted(_BUILDER_EXPORTS),
]


def __getattr__(name: str):
    """Resolve historical builder imports from their mutable owner."""

    if name not in _BUILDER_EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module = import_module("cudaq.logical.devices.builder")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
