# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Composable library recipes for common QLX device stacks.

The helpers in this module are immutable inputs to the ordinary
:class:`cudaq.logical.DeviceBuilder`. They preserve its staged semantics: a
recipe may
stop at P1, bind an encoding at P2, or realize that region on physical carriers
at P3.  Use :func:`compose` when a device needs more structure than one of the
four common convenience constructors.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto
from typing import Any, Iterable, Mapping

from cudaq.logical.architecture.capabilities import (
    HeraldedErasure,
    PhysicalCapability,
    PhysicalCapabilityBinding,
)
from cudaq.logical.architecture.logical import (
    CapabilityKey,
    Direction,
    capability,
)
from cudaq.logical.architecture.physical_definition import (
    NativeActionDecomposition,
    PatchKind,
    PhysicalFootprint,
    PhysicalAction,
    PhysicalMachine,
    PhysicalInstrument,
    ResourceGranularity,
    Topology,
)
from cudaq.logical.devices.definition import Device
from cudaq.logical.devices.builder import DeviceBuilder
from cudaq.logical.codes import (
    Code,
    Encoding,
)

EncodingChoice = Code | Encoding
_not_provided = object()


class _CarrierSetKind(Enum):
    ALL = auto()
    NONE = auto()
    ONLY = auto()


@dataclass(frozen=True, slots=True)
class _CarrierSet:
    """Internal value behind ``cudaq.logical.devices.carriers``."""

    kind: _CarrierSetKind
    indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        indices = tuple(self.indices)
        if self.kind is not _CarrierSetKind.ONLY and indices:
            raise ValueError(
                f"carrier set {self.kind.name.lower()!r} cannot contain "
                "explicit indices")
        if self.kind is _CarrierSetKind.ONLY and not indices:
            raise ValueError("an explicit carrier subset cannot be empty; use "
                             "cudaq.logical.devices.carriers.none")
        if any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in indices):
            raise TypeError("carrier indices must be Python ints")
        if any(index < 0 for index in indices):
            raise ValueError("carrier indices must be nonnegative")
        if len(set(indices)) != len(indices):
            raise ValueError("carrier indices must not contain duplicates")
        object.__setattr__(self, "indices", tuple(sorted(indices)))


class _CarrierSets:
    """Namespace for explicit capability-carrier selections."""

    __slots__ = ()

    all = _CarrierSet(_CarrierSetKind.ALL)
    none = _CarrierSet(_CarrierSetKind.NONE)

    @staticmethod
    def only(*indices: int) -> _CarrierSet:
        """Select exactly the listed local carrier indices."""

        return _CarrierSet(_CarrierSetKind.ONLY, tuple(indices))


carriers = _CarrierSets()


def _member_name(value: str, *, what: str) -> str:
    if not isinstance(value, str) or not value or not value.isidentifier():
        raise ValueError(f"{what} must be a nonempty Python identifier")
    return value


def _nonnegative(value: int | None, *, what: str) -> int | None:
    if value is not None and (not isinstance(value, int) or
                              isinstance(value, bool) or value < 0):
        raise TypeError(f"{what} must be a nonnegative Python int or None")
    return value


def _positive(value: int, *, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise TypeError(f"{what} must be a positive Python int")
    return value


def _encoding_choice(
    value: EncodingChoice | None,
    *,
    what: str,
) -> Code | Encoding | None:
    if value is None:
        return None
    if not isinstance(value, (Code, Encoding)):
        raise TypeError(
            f"{what} requires a cudaq.logical.Code or cudaq.logical.Encoding")
    return value


def _items(value, expected_type, *, what: str):
    if value is None:
        return ()
    if isinstance(value, expected_type):
        return (value,)
    try:
        result = tuple(value)
    except TypeError as exc:
        raise TypeError(
            f"{what} requires a {expected_type.__name__} or an iterable of them"
        ) from exc
    if any(not isinstance(item, expected_type) for item in result):
        raise TypeError(
            f"every {what} entry must be a {expected_type.__name__}")
    return result


def _categories(value, count: int, *, what: str):
    if value is None:
        return ()
    if isinstance(value, Mapping):
        unknown = set(value) - set(range(count))
        if unknown:
            raise ValueError(
                f"{what} categories reference unknown slots: {sorted(unknown)!r}"
            )
        result = tuple(value.get(slot) for slot in range(count))
    else:
        result = tuple(value)
        if len(result) != count:
            raise ValueError(
                f"{what} categories require one entry per patch selection")
    if any(item is not None and not isinstance(item, PatchKind)
           for item in result):
        raise TypeError(
            f"{what} categories must be cudaq.logical.PatchKind values")
    return result


@dataclass(frozen=True, slots=True)
class PhysicalResource:
    """An unbound physical resource declaration.

    Counts on a normal region are totals.  Counts used as a factory bank's
    ``workspace`` are per factory instance and are multiplied by ``instances``
    when the device is built. ``erasure_indices`` are local to this declaration;
    factory replication repeats the same equipped positions in every instance.
    """

    kind: str
    count: int
    granularity: ResourceGranularity = ResourceGranularity.CARRIER
    footprint: PhysicalFootprint | None = None
    native_actions: tuple[PhysicalAction | str, ...] = ()
    native_action_decompositions: tuple[NativeActionDecomposition, ...] = ()
    native_instruments: tuple[PhysicalInstrument, ...] = ()
    topology: Topology | None = None
    capabilities: tuple[PhysicalCapability | str, ...] = ()
    capability_bindings: tuple[PhysicalCapabilityBinding, ...] = ()
    erasure_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("physical resource kind must be nonempty")
        _nonnegative(self.count, what="physical resource count")
        try:
            granularity = ResourceGranularity(self.granularity)
        except ValueError as error:
            raise ValueError(
                "physical resource granularity must be 'carrier' or 'patch'"
            ) from error
        if granularity is ResourceGranularity.PATCH:
            if not isinstance(self.footprint, PhysicalFootprint):
                raise TypeError(
                    "patch-granularity resources require a PhysicalFootprint")
        elif self.footprint is not None:
            raise ValueError(
                "carrier-granularity resources must not declare a patch footprint"
            )
        object.__setattr__(self, "granularity", granularity)
        object.__setattr__(self, "native_actions", tuple(self.native_actions))
        object.__setattr__(
            self,
            "native_action_decompositions",
            tuple(self.native_action_decompositions),
        )
        object.__setattr__(self, "native_instruments",
                           tuple(self.native_instruments))
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        bindings = tuple(self.capability_bindings)
        if any(not isinstance(binding, PhysicalCapabilityBinding)
               for binding in bindings):
            raise TypeError(
                "capability_bindings entries must be PhysicalCapabilityBinding values"
            )
        for binding in bindings:
            if any(index < 0 or index >= self.count
                   for index in binding.indices):
                raise ValueError(
                    "physical capability indices must lie within the resource")
        keys = tuple(binding.capability.key for binding in bindings)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "physical resource has duplicate capability bindings")
        object.__setattr__(self, "capability_bindings", bindings)
        if self.erasure_indices is not None:
            indices = tuple(self.erasure_indices)
            if any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in indices):
                raise TypeError("erasure_indices must contain Python ints")
            if len(set(indices)) != len(indices):
                raise ValueError("erasure_indices must not contain duplicates")
            if any(index < 0 or index >= self.count for index in indices):
                raise ValueError(
                    "erasure_indices entries must lie within the resource")
            object.__setattr__(self, "erasure_indices", tuple(sorted(indices)))
        if self.erasure_indices is not None and any(
                isinstance(binding.capability, HeraldedErasure)
                for binding in bindings):
            raise ValueError(
                "use either typed HeraldedErasure capability bindings or "
                "legacy erasure_indices, not both")

    def __getitem__(self, key: int | tuple[int, ...]) -> "CarrierSelection":
        indices = key if isinstance(key, tuple) else (key,)
        return CarrierSelection(self, tuple(indices))

    def with_capability(
        self,
        capability: PhysicalCapability,
        *,
        carriers: _CarrierSet | object = _not_provided,
        indices: Iterable[int] | None | object = _not_provided,
    ) -> "PhysicalResource":
        """Return this resource with a typed capability installed on carriers.

        New code must make the architecture point explicit with
        ``carriers=cudaq.logical.devices.carriers.all``, ``.none``, or
        ``.only(0, 1, ...)``. ``indices=`` remains a compatibility spelling;
        its historical ``None``/empty distinction is intentionally not used by
        the canonical API.
        """

        if not isinstance(capability, PhysicalCapability):
            raise TypeError("with_capability() requires PhysicalCapability")
        if carriers is _not_provided and indices is _not_provided:
            raise TypeError(
                "with_capability() requires an explicit carriers= selection; "
                "use cudaq.logical.devices.carriers.all, .none, or .only(...)")
        if carriers is not _not_provided and indices is not _not_provided:
            raise TypeError(
                "with_capability() accepts either carriers= or legacy "
                "indices=, not both")
        if carriers is not _not_provided:
            if not isinstance(carriers, _CarrierSet):
                raise TypeError(
                    "carriers= must be cudaq.logical.devices.carriers.all, "
                    ".none, or "
                    ".only(...)")
            if carriers.kind is _CarrierSetKind.ALL:
                selected = tuple(range(self.count))
            elif carriers.kind is _CarrierSetKind.NONE:
                selected = ()
            else:
                selected = carriers.indices
        else:
            selected = (tuple(range(self.count))
                        if indices is None else tuple(indices))
        binding = PhysicalCapabilityBinding(capability, selected)
        if any(index < 0 or index >= self.count for index in binding.indices):
            raise ValueError(
                "physical capability indices must lie within the resource")
        if any(existing.capability.key == capability.key
               for existing in self.capability_bindings):
            raise ValueError(
                f"physical resource already binds capability {capability.key!r}"
            )
        if (isinstance(capability, HeraldedErasure) and
                self.erasure_indices is not None):
            raise ValueError(
                "use either typed HeraldedErasure capability bindings or "
                "legacy erasure_indices, not both")
        return replace(
            self,
            capability_bindings=(*self.capability_bindings, binding),
        )

    def _add(self, builder: DeviceBuilder, *, name: str, multiplier: int = 1):
        erasure_indices = self.erasure_indices
        capability_bindings = self.capability_bindings
        if erasure_indices is not None and multiplier != 1:
            erasure_indices = tuple(instance * self.count + index
                                    for instance in range(multiplier)
                                    for index in erasure_indices)
        if capability_bindings and multiplier != 1:
            capability_bindings = tuple(
                PhysicalCapabilityBinding(
                    binding.capability,
                    tuple(instance * self.count + index
                          for instance in range(multiplier)
                          for index in binding.indices),
                )
                for binding in capability_bindings)
        return builder.physical.add_resources(
            self.kind,
            self.count * multiplier,
            name=name,
            granularity=self.granularity,
            footprint=self.footprint,
            native_actions=self.native_actions,
            native_action_decompositions=self.native_action_decompositions,
            native_instruments=self.native_instruments,
            topology=self.topology,
            capabilities=self.capabilities,
            capability_bindings=capability_bindings,
            erasure_indices=erasure_indices,
        )


@dataclass(frozen=True, slots=True)
class CarrierSelection:
    """Typed carrier group selected from one physical-resource recipe."""

    resource: PhysicalResource
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.resource, PhysicalResource):
            raise TypeError("carrier selection requires a "
                            "cudaq.logical.devices.PhysicalResource")
        indices = tuple(self.indices)
        if not indices:
            raise ValueError("carrier selection must not be empty")
        if any(not isinstance(index, int) or isinstance(index, bool) or
               index < 0 or index >= self.resource.count for index in indices):
            raise IndexError("carrier selection must use indices in "
                             f"[0, {self.resource.count})")
        if len(set(indices)) != len(indices):
            raise ValueError("carrier selection contains a duplicate index")
        object.__setattr__(self, "indices", indices)


def resources(
    kind: str,
    count: int,
    *,
    granularity: ResourceGranularity | str = ResourceGranularity.CARRIER,
    footprint: PhysicalFootprint | None = None,
    native_actions: Iterable[PhysicalAction | str] = (),
    native_action_decompositions: Iterable[NativeActionDecomposition] = (),
    native_instruments: Iterable[PhysicalInstrument] = (),
    topology: Topology | None = None,
    capabilities: Iterable[PhysicalCapability | str] = (),
    capability_bindings: Iterable[PhysicalCapabilityBinding] = (),
    erasure_indices: Iterable[int] | None = None,
) -> PhysicalResource:
    """Describe physical resources before binding them to a recipe region."""

    return PhysicalResource(
        kind=kind,
        count=count,
        granularity=granularity,
        footprint=footprint,
        native_actions=tuple(native_actions),
        native_action_decompositions=tuple(native_action_decompositions),
        native_instruments=tuple(native_instruments),
        topology=topology,
        capabilities=tuple(capabilities),
        capability_bindings=tuple(capability_bindings),
        erasure_indices=(None if erasure_indices is None else
                         tuple(erasure_indices)),
    )


def qubits(
    count: int,
    *,
    native_actions: Iterable[PhysicalAction | str] = (),
    native_action_decompositions: Iterable[NativeActionDecomposition] = (),
    native_instruments: Iterable[PhysicalInstrument] = (),
    topology: Topology | None = None,
    capabilities: Iterable[PhysicalCapability | str] = (),
    capability_bindings: Iterable[PhysicalCapabilityBinding] = (),
    erasure_indices: Iterable[int] | None = None,
) -> PhysicalResource:
    """Specialize :func:`resources` for physical qubits.

    Install typed carrier properties with :meth:`PhysicalResource.with_capability`.
    ``erasure_indices`` remains as a compatibility input for located-erasure
    hardware descriptions.
    """

    return resources(
        "qubit",
        count,
        native_actions=native_actions,
        native_action_decompositions=native_action_decompositions,
        native_instruments=native_instruments,
        topology=topology,
        capabilities=capabilities,
        capability_bindings=capability_bindings,
        erasure_indices=erasure_indices,
    )


@dataclass(frozen=True, slots=True)
class Region:
    """One reusable P1 region with optional P2 and P3 refinements."""

    name: str
    encoding: EncodingChoice | None = None
    capacity: int | None = 1
    capabilities: tuple[CapabilityKey, ...] = ()
    tags: tuple[str, ...] = ()
    physical: tuple[PhysicalResource, ...] = ()
    patches: tuple[CarrierSelection, ...] = ()
    categories: tuple[PatchKind | None, ...] = ()

    def __post_init__(self) -> None:
        _member_name(self.name, what="region name")
        _encoding_choice(
            self.encoding,
            what=f"region {self.name!r} encoding=",
        )
        _nonnegative(self.capacity, what="region capacity")
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(
            self,
            "physical",
            _items(self.physical, PhysicalResource, what="physical"),
        )
        patches = _items(self.patches, CarrierSelection, what="patches")
        if patches:
            owners = {id(selection.resource) for selection in patches}
            if len(owners) != 1:
                raise ValueError(
                    f"region {self.name!r} patches must select one resource")
            if not any(patches[0].resource is item for item in self.physical):
                raise ValueError(
                    f"region {self.name!r} patch selections must come from "
                    "its physical resources")
        object.__setattr__(self, "patches", patches)
        object.__setattr__(
            self,
            "categories",
            _categories(
                self.categories or None,
                len(patches),
                what=f"region {self.name!r}",
            ),
        )
        if self.categories and not patches:
            raise TypeError(f"region {self.name!r} categories require patches")


def region(
    name: str,
    *,
    encoding: EncodingChoice | None = None,
    capacity: int | None = 1,
    capabilities: Iterable[CapabilityKey] = (),
    tags: Iterable[str] = (),
    physical: PhysicalResource | Iterable[PhysicalResource] | None = None,
    patches: CarrierSelection | Iterable[CarrierSelection] | None = None,
    categories: Mapping[int, PatchKind] | Iterable[PatchKind | None] |
    None = None,
) -> Region:
    """Build a logical region recipe that may stop after any device layer."""

    patch_values = _items(patches, CarrierSelection, what="patches")
    return Region(
        name=name,
        encoding=encoding,
        capacity=capacity,
        capabilities=tuple(capabilities),
        tags=tuple(tags),
        physical=_items(physical, PhysicalResource, what="physical"),
        patches=patch_values,
        categories=_categories(
            categories,
            len(patch_values),
            what=f"region {name!r}",
        ),
    )


def compute_region(
    *,
    encoding: EncodingChoice | None = None,
    capacity: int | None = 1,
    capabilities: Iterable[CapabilityKey] = (capability.logical_compute,),
    tags: Iterable[str] = (),
    physical: PhysicalResource | Iterable[PhysicalResource] | None = None,
    patches: CarrierSelection | Iterable[CarrierSelection] | None = None,
    categories: Mapping[int, PatchKind] | Iterable[PatchKind | None] |
    None = None,
    name: str = "compute",
) -> Region:
    """Build a compute region with the standard compute capability."""

    return region(
        name,
        encoding=encoding,
        capacity=capacity,
        capabilities=capabilities,
        tags=tags,
        physical=physical,
        patches=patches,
        categories=categories,
    )


def memory_region(
    *,
    encoding: EncodingChoice | None = None,
    capacity: int | None = 1,
    capabilities: Iterable[CapabilityKey] = (capability.logical_memory,),
    tags: Iterable[str] = (),
    physical: PhysicalResource | Iterable[PhysicalResource] | None = None,
    patches: CarrierSelection | Iterable[CarrierSelection] | None = None,
    categories: Mapping[int, PatchKind] | Iterable[PatchKind | None] |
    None = None,
    name: str = "memory",
) -> Region:
    """Build a memory region with the standard memory capability."""

    return region(
        name,
        encoding=encoding,
        capacity=capacity,
        capabilities=capabilities,
        tags=tags,
        physical=physical,
        patches=patches,
        categories=categories,
    )


@dataclass(frozen=True, slots=True)
class FactoryBank:
    """A resource stream backed by ``instances`` concurrent factories."""

    produces: Any
    name: str = "magic"
    instances: int = 1
    buffer_size: int | None = 1
    encoding: EncodingChoice | None = None
    capabilities: tuple[CapabilityKey, ...] = (capability.logical_factory,)
    tags: tuple[str, ...] = ()
    produced_by: Any = None
    transfer: Any = None
    workspace: tuple[PhysicalResource, ...] = ()
    region_name: str | None = None

    def __post_init__(self) -> None:
        _member_name(self.name, what="factory bank name")
        _positive(self.instances, what="factory instances")
        _nonnegative(self.buffer_size, what="factory buffer_size")
        _encoding_choice(
            self.encoding,
            what=f"factory bank {self.name!r} encoding=",
        )
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(
            self,
            "workspace",
            _items(self.workspace, PhysicalResource, what="workspace"),
        )
        if self.region_name is not None and (
                not isinstance(self.region_name, str) or not self.region_name):
            raise ValueError("factory region_name must be nonempty or None")

    @property
    def backing_region_name(self) -> str:
        return self.region_name or f"{self.name}_factory"


def factory_bank(
    produces: Any,
    *,
    instances: int = 1,
    buffer_size: int | None = 1,
    encoding: EncodingChoice | None = None,
    capabilities: Iterable[CapabilityKey] = (capability.logical_factory,),
    tags: Iterable[str] = (),
    produced_by: Any = None,
    transfer: Any = None,
    workspace: PhysicalResource | Iterable[PhysicalResource] | None = None,
    name: str = "magic",
    region_name: str | None = None,
) -> FactoryBank:
    """Describe one resource kind supplied by ``instances`` factories."""

    return FactoryBank(
        produces=produces,
        name=name,
        instances=instances,
        buffer_size=buffer_size,
        encoding=encoding,
        capabilities=tuple(capabilities),
        tags=tuple(tags),
        produced_by=produced_by,
        transfer=transfer,
        workspace=_items(workspace, PhysicalResource, what="workspace"),
        region_name=region_name,
    )


def factory(*args, **kwargs) -> FactoryBank:
    """Short spelling of :func:`factory_bank`."""

    return factory_bank(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class Link:
    """A typed P1 capability-bearing channel between two recipe regions."""

    source: Region
    destination: Region
    capabilities: tuple[CapabilityKey, ...] = ()
    direction: str = Direction.FORWARD
    capacity: int | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, Region) or not isinstance(
                self.destination, Region):
            raise TypeError(
                "link endpoints must be cudaq.logical.devices.Region values")
        object.__setattr__(self, "capabilities", tuple(self.capabilities))
        _nonnegative(self.capacity, what="link capacity")
        if self.name is not None:
            _member_name(self.name, what="link name")


def link(
    source: Region,
    destination: Region,
    *,
    capabilities: Iterable[CapabilityKey] = (),
    direction: str = Direction.FORWARD,
    capacity: int | None = None,
    name: str | None = None,
) -> Link:
    """Build a capability-bearing channel from typed recipe endpoints."""

    return Link(
        source,
        destination,
        tuple(capabilities),
        direction,
        capacity,
        name,
    )


@dataclass(frozen=True, slots=True)
class DeviceRecipe:
    """A reusable immutable input to the ordinary :class:`cudaq.logical.Device`."""

    regions: tuple[Region, ...]
    factories: tuple[FactoryBank, ...] = ()
    links: tuple[Link, ...] = ()
    name: str | None = None
    architecture: PhysicalMachine | None = None
    timing: Mapping[str, Any] | None = None
    calibration: Mapping[str, Any] | None = None
    costs: Mapping[str, Any] | None = None
    target_compatibility: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.name is not None and (not isinstance(self.name, str) or
                                      not self.name):
            raise ValueError("device name must be nonempty or None")
        object.__setattr__(self, "regions",
                           _items(self.regions, Region, what="regions"))
        object.__setattr__(
            self,
            "factories",
            _items(self.factories, FactoryBank, what="factories"),
        )
        object.__setattr__(self, "links", _items(self.links, Link,
                                                 what="links"))
        if self.architecture is not None and not isinstance(
                self.architecture, PhysicalMachine):
            raise TypeError(
                "device recipe architecture must be PhysicalMachine or None")
        object.__setattr__(self, "target_compatibility",
                           tuple(self.target_compatibility))
        if not self.regions:
            raise ValueError("a device recipe requires at least one region")
        region_ids = {id(region) for region in self.regions}
        for connection in self.links:
            if (id(connection.source) not in region_ids or
                    id(connection.destination) not in region_ids):
                raise ValueError(
                    "link endpoints must be region objects included in this "
                    "device recipe")
        has_inline_physical = any(region.physical for region in self.regions)
        has_inline_physical |= any(
            factory.workspace for factory in self.factories)
        if self.architecture is not None and has_inline_physical:
            raise ValueError(
                "use either recipe physical resources or architecture=, not both"
            )

    def build(self, *, name: str | None = None) -> Device:
        """Build and normalize this thin recipe through ``DeviceBuilder``."""

        device_name = (self.name if name is None else
                       name) or "device_" + "_".join(region.name
                                                     for region in self.regions)
        builder = DeviceBuilder(
            device_name,
            physical=self.architecture,
            metadata=self.metadata,
            source_module=__name__,
        )

        def bind_encoding(handle, encoding, *, what):
            target = _encoding_choice(encoding, what=what)
            if target is not None:
                return builder.qec.bind(
                    handle,
                    encoding=target,
                )
            return None

        region_handles = {}
        qec_handles = {}
        region_encodings = tuple(target for recipe in self.regions
                                 if (target := _encoding_choice(
                                     recipe.encoding,
                                     what=f"region {recipe.name!r} encoding=",
                                 )) is not None)
        inherited_factory_encoding = (
            region_encodings[0] if region_encodings and
            all(value is region_encodings[0] for value in region_encodings) else
            None)
        for recipe in self.regions:
            handle = builder.logical.add_region(
                recipe.name,
                capacity=recipe.capacity,
                capabilities=recipe.capabilities,
                tags=recipe.tags,
            )
            qec_handles[id(recipe)] = bind_encoding(
                handle,
                recipe.encoding,
                what=f"region {recipe.name!r} encoding=",
            )
            region_handles[id(recipe)] = handle

        factory_handles = {}
        for recipe in self.factories:
            handle = builder.logical.add_region(
                recipe.backing_region_name,
                capacity=recipe.instances,
                capabilities=recipe.capabilities,
                tags=recipe.tags,
            )
            qec_handles[id(recipe)] = bind_encoding(
                handle,
                recipe.encoding or inherited_factory_encoding,
                what=f"factory bank {recipe.name!r} encoding=",
            )
            builder.logical.add_stream(
                recipe.produces,
                name=recipe.name,
                buffer_size=recipe.buffer_size,
                region=handle,
                produced_by=recipe.produced_by,
                transfer=recipe.transfer,
            )
            factory_handles[id(recipe)] = handle

        for recipe in self.links:
            source = region_handles[id(recipe.source)]
            destination = region_handles[id(recipe.destination)]
            builder.logical.add_channel(
                source,
                destination,
                name=(recipe.name or
                      f"{recipe.source.name}_to_{recipe.destination.name}"),
                capabilities=recipe.capabilities,
                direction=recipe.direction,
                capacity=recipe.capacity,
            )

        def add_resources(
                owner_name,
                qec_owner,
                specifications,
                patches=(),
                categories=(),
                multiplier=1,
        ):
            handles = []
            handles_by_recipe = {}
            for index, specification in enumerate(specifications):
                suffix = "".join(character if character.isalnum() else "_"
                                 for character in specification.kind)
                resource_name = f"{owner_name}_{suffix}_resources"
                if index:
                    resource_name += f"_{index}"
                handle = specification._add(
                    builder,
                    name=resource_name,
                    multiplier=multiplier,
                )
                handles.append(handle)
                handles_by_recipe[id(specification)] = handle
            if not handles:
                return
            if qec_owner is None:
                raise ValueError(
                    f"physical resources for {owner_name!r} require encoding=")
            if patches:
                if multiplier != 1:
                    raise ValueError(
                        "factory patch selections require an explicit "
                        "architecture when instances is greater than one")
                patch_handle = handles_by_recipe[id(patches[0].resource)]
                builder.physical.bind(
                    qec_owner,
                    to=tuple(handles),
                    patches=tuple(patch_handle[selection.indices]
                                  for selection in patches),
                    categories=categories,
                )
            else:
                builder.physical.bind(qec_owner, to=tuple(handles))

        for recipe in self.regions:
            add_resources(
                recipe.name,
                qec_handles[id(recipe)],
                recipe.physical,
                recipe.patches,
                recipe.categories,
            )
        for recipe in self.factories:
            add_resources(
                recipe.name,
                qec_handles[id(recipe)],
                recipe.workspace,
                multiplier=recipe.instances,
            )

        all_resources = tuple(builder.physical._resources.values())
        if all_resources:
            for qec_handle in qec_handles.values():
                if qec_handle is not None and not any(
                        binding.qec_region is qec_handle.region
                        for binding in builder.physical._bindings):
                    builder.physical.bind(qec_handle, to=all_resources)
        if any(value is not None for value in (
                self.timing,
                self.calibration,
                self.costs,
        )) or self.target_compatibility:
            builder.physical.set_operating_point(
                timing=self.timing,
                calibration=self.calibration,
                costs=self.costs,
                target_compatibility=self.target_compatibility,
            )
        return builder.build()


def compose(
    *,
    regions: Region | Iterable[Region],
    factories: FactoryBank | Iterable[FactoryBank] | None = None,
    links: Link | Iterable[Link] | None = None,
    name: str | None = None,
    architecture: PhysicalMachine | None = None,
    timing: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    costs: Mapping[str, Any] | None = None,
    target_compatibility: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Device:
    """Compose arbitrary regions, factory banks, and links into a device."""

    return DeviceRecipe(
        regions=_items(regions, Region, what="regions"),
        factories=_items(factories, FactoryBank, what="factories"),
        links=_items(links, Link, what="links"),
        name=name,
        architecture=architecture,
        timing=timing,
        calibration=calibration,
        costs=costs,
        target_compatibility=tuple(target_compatibility),
        metadata=metadata,
    ).build()


def _facets(
    *,
    architecture,
    timing,
    calibration,
    costs,
    target_compatibility,
    metadata,
):
    return {
        "architecture": architecture,
        "timing": timing,
        "calibration": calibration,
        "costs": costs,
        "target_compatibility": target_compatibility,
        "metadata": metadata,
    }


def compute_only(
    *,
    encoding: EncodingChoice | None = None,
    capacity: int | None = 1,
    capabilities: Iterable[CapabilityKey] = (capability.logical_compute,),
    tags: Iterable[str] = (),
    physical: PhysicalResource | Iterable[PhysicalResource] | None = None,
    patches: CarrierSelection | Iterable[CarrierSelection] | None = None,
    categories: Mapping[int, PatchKind] | Iterable[PatchKind | None] |
    None = None,
    name: str | None = None,
    architecture: PhysicalMachine | None = None,
    timing: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    costs: Mapping[str, Any] | None = None,
    target_compatibility: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Device:
    """Instantiate a device containing one configurable compute region."""

    return compose(
        regions=compute_region(
            encoding=encoding,
            capacity=capacity,
            capabilities=capabilities,
            tags=tags,
            physical=physical,
            patches=patches,
            categories=categories,
        ),
        name=name,
        **_facets(
            architecture=architecture,
            timing=timing,
            calibration=calibration,
            costs=costs,
            target_compatibility=target_compatibility,
            metadata=metadata,
        ),
    )


def compute_factory(
    *,
    factory: FactoryBank | Iterable[FactoryBank],
    encoding: EncodingChoice | None = None,
    capacity: int | None = 1,
    capabilities: Iterable[CapabilityKey] = (capability.logical_compute,),
    tags: Iterable[str] = (),
    physical: PhysicalResource | Iterable[PhysicalResource] | None = None,
    patches: CarrierSelection | Iterable[CarrierSelection] | None = None,
    categories: Mapping[int, PatchKind] | Iterable[PatchKind | None] |
    None = None,
    name: str | None = None,
    architecture: PhysicalMachine | None = None,
    timing: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    costs: Mapping[str, Any] | None = None,
    target_compatibility: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Device:
    """Instantiate a compute region plus one or more factory banks."""

    return compose(
        regions=compute_region(
            encoding=encoding,
            capacity=capacity,
            capabilities=capabilities,
            tags=tags,
            physical=physical,
            patches=patches,
            categories=categories,
        ),
        factories=factory,
        name=name,
        **_facets(
            architecture=architecture,
            timing=timing,
            calibration=calibration,
            costs=costs,
            target_compatibility=target_compatibility,
            metadata=metadata,
        ),
    )


def compute_memory(
    *,
    encoding: EncodingChoice | None = None,
    capacity: int | None = 1,
    memory_encoding: EncodingChoice | None = None,
    memory_capacity: int | None = 1,
    capabilities: Iterable[CapabilityKey] = (capability.logical_compute,),
    memory_capabilities: Iterable[CapabilityKey] = (capability.logical_memory,),
    tags: Iterable[str] = (),
    memory_tags: Iterable[str] = (),
    physical: PhysicalResource | Iterable[PhysicalResource] | None = None,
    patches: CarrierSelection | Iterable[CarrierSelection] | None = None,
    categories: Mapping[int, PatchKind] | Iterable[PatchKind | None] |
    None = None,
    memory_physical: PhysicalResource | Iterable[PhysicalResource] |
    None = None,
    memory_patches: CarrierSelection | Iterable[CarrierSelection] | None = None,
    memory_categories: Mapping[int, PatchKind] | Iterable[PatchKind | None] |
    None = None,
    memory_channel_capabilities: Iterable[CapabilityKey] = (
        capability.state_transport,),
    link_capacity: int | None = None,
    name: str | None = None,
    architecture: PhysicalMachine | None = None,
    timing: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    costs: Mapping[str, Any] | None = None,
    target_compatibility: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Device:
    """Instantiate linked compute and memory regions."""

    memory_encoding = (encoding if memory_encoding is None else memory_encoding)
    compute = compute_region(
        encoding=encoding,
        capacity=capacity,
        capabilities=capabilities,
        tags=tags,
        physical=physical,
        patches=patches,
        categories=categories,
    )
    memory = memory_region(
        encoding=memory_encoding,
        capacity=memory_capacity,
        capabilities=memory_capabilities,
        tags=memory_tags,
        physical=memory_physical,
        patches=memory_patches,
        categories=memory_categories,
    )
    return compose(
        regions=(compute, memory),
        links=link(
            compute,
            memory,
            capabilities=memory_channel_capabilities,
            direction=Direction.BIDIRECTIONAL,
            capacity=link_capacity,
            name="compute_memory",
        ),
        name=name,
        **_facets(
            architecture=architecture,
            timing=timing,
            calibration=calibration,
            costs=costs,
            target_compatibility=target_compatibility,
            metadata=metadata,
        ),
    )


def compute_memory_factory(
    *,
    factory: FactoryBank | Iterable[FactoryBank],
    encoding: EncodingChoice | None = None,
    capacity: int | None = 1,
    memory_encoding: EncodingChoice | None = None,
    memory_capacity: int | None = 1,
    capabilities: Iterable[CapabilityKey] = (capability.logical_compute,),
    memory_capabilities: Iterable[CapabilityKey] = (capability.logical_memory,),
    tags: Iterable[str] = (),
    memory_tags: Iterable[str] = (),
    physical: PhysicalResource | Iterable[PhysicalResource] | None = None,
    patches: CarrierSelection | Iterable[CarrierSelection] | None = None,
    categories: Mapping[int, PatchKind] | Iterable[PatchKind | None] |
    None = None,
    memory_physical: PhysicalResource | Iterable[PhysicalResource] |
    None = None,
    memory_patches: CarrierSelection | Iterable[CarrierSelection] | None = None,
    memory_categories: Mapping[int, PatchKind] | Iterable[PatchKind | None] |
    None = None,
    memory_channel_capabilities: Iterable[CapabilityKey] = (
        capability.state_transport,),
    link_capacity: int | None = None,
    name: str | None = None,
    architecture: PhysicalMachine | None = None,
    timing: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    costs: Mapping[str, Any] | None = None,
    target_compatibility: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> Device:
    """Instantiate linked compute/memory regions and factory banks."""

    memory_encoding = (encoding if memory_encoding is None else memory_encoding)
    compute = compute_region(
        encoding=encoding,
        capacity=capacity,
        capabilities=capabilities,
        tags=tags,
        physical=physical,
        patches=patches,
        categories=categories,
    )
    memory = memory_region(
        encoding=memory_encoding,
        capacity=memory_capacity,
        capabilities=memory_capabilities,
        tags=memory_tags,
        physical=memory_physical,
        patches=memory_patches,
        categories=memory_categories,
    )
    return compose(
        regions=(compute, memory),
        factories=factory,
        links=link(
            compute,
            memory,
            capabilities=memory_channel_capabilities,
            direction=Direction.BIDIRECTIONAL,
            capacity=link_capacity,
            name="compute_memory",
        ),
        name=name,
        **_facets(
            architecture=architecture,
            timing=timing,
            calibration=calibration,
            costs=costs,
            target_compatibility=target_compatibility,
            metadata=metadata,
        ),
    )


__all__ = [
    "CarrierSelection",
    "DeviceRecipe",
    "FactoryBank",
    "Link",
    "PhysicalResource",
    "Region",
    "compose",
    "compute_factory",
    "compute_memory",
    "compute_memory_factory",
    "compute_only",
    "compute_region",
    "carriers",
    "factory",
    "factory_bank",
    "link",
    "memory_region",
    "qubits",
    "region",
    "resources",
]
