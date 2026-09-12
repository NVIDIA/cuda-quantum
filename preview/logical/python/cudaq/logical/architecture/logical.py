# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from cudaq.logical._core.immutable import ImmutableValue

_MACHINE_CAPABILITY_NAMESPACE = "qlx.machine"
_RESERVED_QLX_CAPABILITY_NAMESPACES = frozenset({
    _MACHINE_CAPABILITY_NAMESPACE,
    "qlx.launch",
})


class Direction(str, Enum):
    """Typed transfer direction of one machine channel.

    Channels accept either the enum member or its lowercase string value;
    both normalize to the enum, which remains a ``str`` for IR emission.
    """

    FORWARD = "forward"
    REVERSE = "reverse"
    BIDIRECTIONAL = "bidirectional"

    # Keep ``str(Direction.FORWARD) == "forward"`` on every Python version so
    # string-based emission and comparison stay stable.
    __str__ = str.__str__


@dataclass(frozen=True, slots=True)
class CapabilityKey:
    key: str

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key or "/" not in self.key:
            raise TypeError(
                "capability key must be a nonempty qualified string")
        namespace, name = self.key.split("/", 1)
        if not namespace or not name:
            raise TypeError(
                "capability key must be a nonempty qualified string")
        if (namespace.startswith("qlx.") and
                namespace not in _RESERVED_QLX_CAPABILITY_NAMESPACES):
            raise ValueError("unsupported reserved QLX capability namespace")


class _OpenVocabulary:

    def __init__(self, prefix: str, value_type) -> None:
        self._prefix = prefix
        self._value_type = value_type

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return self._value_type(f"{self._prefix}/{name}")

    def __call__(self, key: str):
        if "/" not in key:
            key = f"{self._prefix}/{key}"
        return self._value_type(key)


capability = _OpenVocabulary(_MACHINE_CAPABILITY_NAMESPACE, CapabilityKey)


def _machine_capabilities(values, *, what: str) -> tuple[CapabilityKey, ...]:
    normalized = tuple(values)
    if any(not isinstance(value, CapabilityKey) for value in normalized):
        raise TypeError(
            f"{what} must contain cudaq.logical.CapabilityKey values")
    if any(
            value.key.startswith("qlx.") and
            not value.key.startswith(f"{_MACHINE_CAPABILITY_NAMESPACE}/")
            for value in normalized):
        raise ValueError(
            f"{what} must use the QLX machine-capability namespace")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{what} must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class SpaceSlot:
    space: "Space"
    index: int | None

    @property
    def semantic_ref(self) -> tuple[object, ...]:
        """Stable Pauli-product ordering within one logical machine."""

        return (
            "space_slot",
            self.space.name or "",
            -1 if self.index is None else self.index,
            id(self.space),
        )


@dataclass(frozen=True, slots=True)
class Space:
    capabilities: tuple[CapabilityKey, ...] = ()
    capacity: int | None = None
    tags: tuple[str, ...] = ()
    name: str | None = None

    def __init__(
            self,
            *,
            capabilities: Iterable[CapabilityKey] = (),
            capacity: int | None = None,
            tags: Iterable[str] = (),
            name: str | None = None,
    ) -> None:
        if capacity is not None and (not isinstance(capacity, int) or
                                     isinstance(capacity, bool) or
                                     capacity < 0):
            raise TypeError("Space capacity must be a nonnegative Python int")
        object.__setattr__(
            self,
            "capabilities",
            _machine_capabilities(capabilities, what="Space capabilities"),
        )
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "tags", tuple(tags))
        object.__setattr__(self, "name", name)

    def __getitem__(self, index: int) -> SpaceSlot:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise TypeError("space slot index must be a nonnegative int")
        if self.capacity is not None and index >= self.capacity:
            raise IndexError(
                f"slot {index} exceeds space capacity {self.capacity}")
        return SpaceSlot(self, index)

    def any_slot(self) -> SpaceSlot:
        return SpaceSlot(self, None)


@dataclass(frozen=True, slots=True)
class SpaceDeclaration:
    """Concise source declaration projected into one logical machine space."""

    capabilities: tuple[CapabilityKey, ...] = ()
    capacity: int | None = None
    tags: tuple[str, ...] = ()
    name: str | None = None

    def __init__(
            self,
            *,
            capabilities: Iterable[CapabilityKey] = (),
            capacity: int | None = None,
            tags: Iterable[str] = (),
            name: str | None = None,
    ) -> None:
        if capacity is not None and (not isinstance(capacity, int) or
                                     isinstance(capacity, bool) or
                                     capacity < 0):
            raise TypeError("space capacity must be a nonnegative Python int")
        object.__setattr__(
            self,
            "capabilities",
            _machine_capabilities(
                capabilities,
                what="SpaceDeclaration capabilities",
            ),
        )
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "tags", tuple(tags))
        object.__setattr__(self, "name", name)

    def logical(self, name: str | None = None) -> Space:
        return Space(
            capabilities=self.capabilities,
            capacity=self.capacity,
            tags=self.tags,
            name=self.name if name is None else name,
        )


def region(
        *,
        capabilities: Iterable[CapabilityKey] = (),
        capacity: int | None = None,
        tags: Iterable[str] = (),
        name: str | None = None,
) -> SpaceDeclaration:
    """Declare a logical region.

    QEC and physical realizations belong to the corresponding
    :class:`DeviceBuilder` namespaces. The normalized machine and LVM IR
    continue to use the internal :class:`Space` model.
    """

    return SpaceDeclaration(
        capabilities=capabilities,
        capacity=capacity,
        tags=tags,
        name=name,
    )


@dataclass(frozen=True, slots=True)
class Stream:
    """A typed resource-flow endpoint.

    ``produces`` is the resource kind that flows, ``buffer_size`` the number of
    in-flight/buffered resources the endpoint can hold — the output queue
    depth, not a qubit count and distinct from a region's ``capacity`` (which
    counts logical slots / concurrent factory instances). A stream may name
    the *factory* that supplies it: ``region`` is the producer's working
    region (a device-only QEC binding when it carries ``code``/``encoding``),
    ``produced_by`` the production protocol, and ``transfer`` the
    consumer-side transfer/injection protocol. A stream that carries a backing
    ``region`` has a physical home once that region is refined through a
    device's QEC and physical namespaces. A stream with neither a backing
    region nor
    ``external=True`` is physically incomplete: its supply has no home, and
    stages that require physical truth (P3 lowering, schedule-aware and
    digital-twin estimation) fail closed on it.
    """

    produces: Any
    buffer_size: int | None = None
    region: "SpaceDeclaration | None" = None
    produced_by: Any = None
    transfer: Any = None
    external: bool = False
    name: str | None = None

    def __post_init__(self) -> None:
        from ..std import ResourceFlowRef, ResourceKind
        from cudaq.logical.protocols.definition import ProtocolDefinition

        if not isinstance(self.produces, ResourceKind):
            raise TypeError(
                "stream produces must be a cudaq.logical.types.ResourceKind")
        if not isinstance(self.external, bool):
            raise TypeError("stream external must be a bool")
        if self.buffer_size is not None and (
                not isinstance(self.buffer_size, int) or
                isinstance(self.buffer_size, bool) or self.buffer_size < 0):
            raise TypeError(
                "stream buffer_size must be a nonnegative int or None")
        if self.region is not None and not isinstance(self.region,
                                                      SpaceDeclaration):
            raise TypeError(
                "stream region must be a cudaq.logical.SpaceDeclaration")
        if self.external and self.region is not None:
            raise TypeError("a stream cannot be both backed and external")
        if self.produced_by is not None:
            if not isinstance(self.produced_by, ProtocolDefinition):
                raise TypeError("stream produced_by must be a "
                                "cudaq.logical.protocols.ProtocolDefinition")
            objective = self.produced_by.implements
            if (not isinstance(objective, ResourceFlowRef) or
                    objective.kind != "produce" or
                    objective.resource != self.produces):
                raise ValueError("stream produced_by must implement "
                                 "cudaq.logical.logical.produce(produces)")
            if self.produced_by.signature.parameters:
                raise ValueError("stream producer must not require inputs")
        if self.transfer is not None:
            if not isinstance(self.transfer, ProtocolDefinition):
                raise TypeError("stream transfer must be a "
                                "cudaq.logical.protocols.ProtocolDefinition")
            if self.produces not in self.transfer._resource_input_kinds():
                raise ValueError("stream transfer must consume "
                                 "cudaq.logical.types.resource[produces]")

    @property
    def is_backed(self) -> bool:
        """The stream names a backing factory region with a physical home."""

        return self.region is not None

    @property
    def is_physically_complete(self) -> bool:
        """The supply boundary is declared: backed, or explicitly external."""

        return self.is_backed or self.external


def stream(
    produces: Any,
    *,
    buffer_size: int | None = None,
    region: "SpaceDeclaration | None" = None,
    produced_by: Any = None,
    transfer: Any = None,
    external: bool = False,
    name: str | None = None,
) -> Stream:
    """Declare a logical resource stream, optionally naming its factory.

    ``buffer_size`` is the output queue depth (in-flight resources), distinct
    from the *region* ``capacity`` that counts concurrent factory instances.
    Give ``region=cudaq.logical.region(...)`` to name the producer region. In a device,
    refine that region through ``DeviceBuilder.qec`` and
    ``DeviceBuilder.physical`` so the stream has a physical home. ``external``
    declares that the supply is deliberately out of frame. A stream that is
    neither backed nor external is physically incomplete and fails closed
    where physical truth is required.
    """

    if region is not None and not isinstance(region, SpaceDeclaration):
        raise TypeError(
            "cudaq.logical.stream region= requires a cudaq.logical.region")
    if external and region is not None:
        raise TypeError(
            "a stream is either backed by a factory region or external, "
            "not both")
    return Stream(
        produces=produces,
        buffer_size=buffer_size,
        region=region,
        produced_by=produced_by,
        transfer=transfer,
        external=external,
        name=name,
    )


@dataclass(frozen=True, slots=True)
class Channel:
    source: Space | SpaceDeclaration | Stream
    destination: Space | SpaceDeclaration | Stream
    capabilities: tuple[CapabilityKey, ...] = ()
    direction: str = "forward"
    concurrency: int | None = None
    name: str | None = None

    def __init__(
        self,
        source: Space | SpaceDeclaration | Stream,
        destination: Space | SpaceDeclaration | Stream,
        *,
        capabilities: Iterable[CapabilityKey] = (),
        direction: str = "forward",
        capacity: int | None = None,
        concurrency: int | None = None,
        name: str | None = None,
    ) -> None:
        if capacity is not None and concurrency is not None:
            raise TypeError(
                "Channel accepts concurrency=; capacity= is its compatibility "
                "alias and the two cannot be supplied together")
        concurrency = capacity if concurrency is None else concurrency
        if concurrency is not None and (not isinstance(concurrency, int) or
                                        isinstance(concurrency, bool) or
                                        concurrency <= 0):
            raise TypeError("Channel concurrency must be a positive Python int")
        normalized = str(direction).lower()
        try:
            normalized = Direction(normalized)
        except ValueError:
            pass  # Open vocabulary: unknown directions stay plain strings.
        capabilities = _machine_capabilities(
            capabilities,
            what="Channel capabilities",
        )
        derived = RESOURCE_TRANSFER_CAPABILITY if isinstance(
            source, Stream) or isinstance(destination, Stream) else None
        if derived is not None and any(
                value.key.rsplit("/", 1)[-1] == "resource_transfer" and
                value != derived for value in capabilities):
            raise ValueError("stream-derived resource transfer must use "
                             "qlx.capability.resource_transfer")
        if derived is not None and derived not in capabilities:
            capabilities = (*capabilities, derived)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "destination", destination)
        object.__setattr__(self, "capabilities", capabilities)
        object.__setattr__(self, "direction", normalized)
        object.__setattr__(self, "concurrency", concurrency)
        object.__setattr__(self, "name", name)

    @property
    def capacity(self) -> int | None:
        """Compatibility alias for the P1 maximum-in-flight concurrency."""

        return self.concurrency


def channel(
    source: Space | SpaceDeclaration | Stream,
    destination: Space | SpaceDeclaration | Stream,
    *,
    capabilities: Iterable[CapabilityKey] = (),
    direction: str = "forward",
    capacity: int | None = None,
    concurrency: int | None = None,
    name: str | None = None,
) -> Channel:
    """Declare a logical channel in a logical machine class."""

    return Channel(
        source,
        destination,
        capabilities=capabilities,
        direction=direction,
        capacity=capacity,
        concurrency=concurrency,
        name=name,
    )


class LogicalMachine(ImmutableValue):
    """Immutable P1 logical virtual platform; never physical topology."""

    __slots__ = ("name", "spaces", "streams", "channels", "_members")

    def __init__(self, name: str, members: Mapping[str, Any]) -> None:
        self.name = name
        self._members = MappingProxyType(dict(members))
        self.spaces = tuple(
            value for value in members.values() if isinstance(value, Space))
        self.streams = tuple(
            value for value in members.values() if isinstance(value, Stream))
        self.channels = tuple(
            value for value in members.values() if isinstance(value, Channel))
        self._seal()

    def __getattr__(self, name: str):
        try:
            return self._members[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __repr__(self) -> str:
        return f"LogicalMachine({self.name!r})"


RESOURCE_TRANSFER_CAPABILITY = capability("resource_transfer")


def synthesize_stream_supply(raw, renamed, members):
    """Expand every backed stream into a factory region and a supply channel.

    For a stream whose ``region`` names a factory, this materializes that
    region as a real logical :class:`Space` (``<stream>_factory``) and a
    :class:`Channel` (``<stream>_supply``) carrying the resource-transfer
    capability from the factory to the stream. Device authors refine the
    synthesized logical factory region explicitly through the QEC and physical
    namespaces.

    Returns ``{id(original_stream): factory_space}`` so a physical-resource
    declaration bound with ``for_=stream`` resolves to the factory region.
    """

    factory_spaces: dict[int, Space] = {}
    for key, value in raw.items():
        if not isinstance(value, Stream) or value.region is None:
            continue
        region_decl = value.region
        factory_name = region_decl.name or f"{key}_factory"
        if factory_name in members:
            raise ValueError(
                f"stream {key!r} factory region collides with an existing "
                f"member {factory_name!r}")
        factory_space = region_decl.logical(factory_name)
        if capability.logical_factory not in factory_space.capabilities:
            factory_space = replace(
                factory_space,
                capabilities=(
                    capability.logical_factory,
                    *factory_space.capabilities,
                ),
            )
        members[factory_name] = factory_space
        factory_spaces[id(value)] = factory_space
        channel_name = f"{key}_supply"
        if channel_name in members:
            raise ValueError(
                f"stream {key!r} supply channel collides with an existing "
                f"member {channel_name!r}")
        members[channel_name] = Channel(
            factory_space,
            renamed.get(id(value), value),
            name=channel_name,
        )
    return factory_spaces


def machine(cls=None, *, name: str | None = None):

    def decorate(machine_class) -> LogicalMachine:
        raw = {
            key: value
            for key, value in vars(machine_class).items()
            if isinstance(value, (Space, SpaceDeclaration, Stream, Channel))
        }
        renamed: dict[int, Any] = {}
        members: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, SpaceDeclaration):
                named = value.logical(key)
                renamed[id(value)] = named
                members[key] = named
            elif isinstance(value, Space):
                named = replace(value, name=key)
                renamed[id(value)] = named
                members[key] = named
            elif isinstance(value, Stream):
                named = replace(value, name=key)
                renamed[id(value)] = named
                members[key] = named
        synthesize_stream_supply(raw, renamed, members)
        for key, value in raw.items():
            if isinstance(value, Channel):
                members[key] = replace(
                    value,
                    source=renamed.get(id(value.source), value.source),
                    destination=renamed.get(id(value.destination),
                                            value.destination),
                    name=key,
                )
        return LogicalMachine(name or machine_class.__name__, members)

    return decorate(cls) if cls is not None else decorate


@dataclass(frozen=True, slots=True)
class LogicalValueRef:
    program: str
    group: str
    path: tuple[int, ...] = ()
    allocation: int | None = None


class LogicalValueGroup:
    __slots__ = ("program", "name", "count", "allocation")

    def __init__(self, program: str, name: str, count: int,
                 allocation: int) -> None:
        self.program = program
        self.name = name
        self.count = count
        self.allocation = allocation

    def __getitem__(self, index: int) -> LogicalValueRef:
        if index < 0 or index >= self.count:
            raise IndexError(index)
        return LogicalValueRef(self.program, self.name, (index,),
                               self.allocation)

    def __iter__(self):
        return iter(tuple(self[i] for i in range(self.count)))

    def __len__(self) -> int:
        return self.count


class ProgramValueSchema:
    __slots__ = ("_groups", "_allocations")

    def __init__(self, program: str, groups) -> None:
        if isinstance(groups, Mapping):
            entries = tuple(
                (allocation, name, count)
                for allocation, (name, count) in enumerate(groups.items()))
        else:
            entries = tuple((
                int(item["allocation"]),
                str(item["name"]),
                int(item["count"]),
            ) for item in groups)
        values = {
            name: LogicalValueGroup(program, name, count, allocation)
            for allocation, name, count in entries
        }
        self._groups = MappingProxyType(values)
        self._allocations = tuple(value for value in sorted(
            values.values(), key=lambda value: value.allocation))

    def __getattr__(self, name: str):
        try:
            return self._groups[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __getitem__(self, allocation: int) -> LogicalValueGroup:
        if not isinstance(allocation, int) or isinstance(allocation, bool):
            raise TypeError("logical allocation index must be an int")
        return self._allocations[allocation]

    def __iter__(self):
        return iter(self._allocations)

    def __len__(self) -> int:
        return len(self._allocations)
