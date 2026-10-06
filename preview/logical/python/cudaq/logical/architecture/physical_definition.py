# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from inspect import Signature, signature
from math import isfinite
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

from cudaq.logical._core.immutable import (
    ImmutableValue,
    freeze_mapping,
)
from cudaq.logical.architecture.capabilities import (
    HeraldedErasure,
    PhysicalCapability,
    PhysicalCapabilityBinding,
)


class physical_qubit:
    """Annotation/marker for one physical two-level carrier."""


class Basis:
    X = "x"
    Y = "y"
    Z = "z"


class ResourceGranularity(str, Enum):
    """Allocation unit represented by one physical resource-class member."""

    CARRIER = "carrier"
    PATCH = "patch"


def _quantum_process_parameter(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("process parameters must be finite real numbers")
    result = float(value)
    if not isfinite(result):
        raise ValueError("process parameters must be finite")
    return result


@dataclass(frozen=True, slots=True)
class QuantumProcess:
    """A stable QLX-defined implementation of one physical operation.

    Physical actions and instruments retain this target-neutral descriptor in
    their P3 definition so materialization can emit the process contract
    without introducing a simulator-provider API.
    """

    name: str
    parameters: Mapping[str, float]
    kind = "builtin"

    def __init__(self,
                 name: str,
                 parameters: Mapping[str, Any] | None = None) -> None:
        name = str(name)
        if not name:
            raise ValueError("quantum process name must be nonempty")
        normalized = {
            str(parameter): _quantum_process_parameter(value)
            for parameter, value in (parameters or {}).items()
        }
        if any(not parameter for parameter in normalized):
            raise ValueError("process parameter names must be nonempty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "parameters", MappingProxyType(normalized))

    def to_dict(self) -> dict[str, Any]:
        """Return the stable P3 process attribute representation."""

        return {
            "kind": self.kind,
            "name": self.name,
            "parameters": dict(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class PhysicalFootprint:
    """Authenticated base-unit footprint of one schedulable resource.

    Patch-granularity resources use this value to retain their physical cost
    without expanding one encoded patch into thousands of scalar carrier SSA
    values. ``evidence`` names the counting convention or calibration source.
    """

    unit_kind: str
    units: int
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.unit_kind, str) or not self.unit_kind:
            raise ValueError("physical footprint unit_kind must be nonempty")
        if (not isinstance(self.units, int) or isinstance(self.units, bool) or
                self.units <= 0):
            raise TypeError("physical footprint units must be a positive int")
        if not isinstance(self.evidence, str) or not self.evidence:
            raise ValueError("physical footprint evidence must be nonempty")


@dataclass(frozen=True, slots=True)
class PhysicalAction:
    """One reusable physical instruction with target-neutral semantics.

    The action is the semantic object named by ``phys.apply``.  Architectures
    advertise action values, not magic strings. ``process`` is the physical
    implementation contract; optional controller bindings are kept separate.
    """

    name: str
    arity: int
    process: QuantumProcess
    controller_bindings: Mapping[str, str]
    parameters: tuple[str, ...]
    broadcast: bool
    metadata: Mapping[str, Any]

    def __init__(
        self,
        name: str,
        *,
        arity: int,
        process: QuantumProcess,
        controller_bindings: Mapping[str, str] | None = None,
        parameters: Iterable[str] = (),
        broadcast: bool = False,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        name = str(name)
        if not name:
            raise ValueError("physical action name must be nonempty")
        if not isinstance(arity, int) or isinstance(arity, bool) or arity <= 0:
            raise TypeError("physical action arity must be a positive int")
        if not isinstance(broadcast, bool):
            raise TypeError("physical action broadcast must be bool")
        if broadcast and arity != 1:
            raise ValueError("broadcast physical actions must have arity one")
        if not isinstance(process, QuantumProcess):
            raise TypeError("physical action process must be a QuantumProcess")
        bindings = {
            str(target): str(semantic)
            for target, semantic in (controller_bindings or {}).items()
        }
        if any(not key or not value for key, value in bindings.items()):
            raise ValueError(
                "physical action controller bindings must be nonempty "
                "strings")
        parameter_names = tuple(map(str, parameters))
        if any(not value for value in parameter_names):
            raise ValueError("physical action parameter names must be nonempty")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError("physical action has duplicate parameter names")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "arity", arity)
        object.__setattr__(self, "process", process)
        object.__setattr__(self, "controller_bindings",
                           MappingProxyType(bindings))
        object.__setattr__(self, "parameters", parameter_names)
        object.__setattr__(self, "broadcast", broadcast)
        object.__setattr__(self, "metadata", freeze_mapping(metadata))

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)


@dataclass(frozen=True, slots=True)
class NativeActionStep:
    """One native instruction in a device-owned action decomposition.

    ``operands`` indexes the source action's ordered carrier tuple.  Keeping
    the permutation explicit is essential for directional decompositions such
    as CX-through-CZ and for SWAP's alternating control/target roles.
    """

    action: PhysicalAction
    operands: tuple[int, ...]

    def __init__(self, action: PhysicalAction, *operands: int) -> None:
        if not isinstance(action, PhysicalAction):
            raise TypeError("native action steps require a PhysicalAction")
        indices = tuple(operands)
        if len(indices) != action.arity:
            raise ValueError(
                f"native action @{action.name} requires {action.arity} "
                f"operands, got {len(indices)}")
        if any(not isinstance(index, int) or isinstance(index, bool) or
               index < 0 for index in indices):
            raise TypeError(
                "native action step operands must be nonnegative ints")
        if len(set(indices)) != len(indices):
            raise ValueError("native action step operands must be distinct")
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "operands", indices)


@dataclass(frozen=True, slots=True)
class NativeActionDecomposition:
    """A fail-closed semantic-action legalization supplied by a device.

    The source action is never inferred from a target spelling.  Every step is
    a typed native action, and ``ResourceClass`` verifies that the recipe only
    references instructions the resource class actually advertises.
    """

    source: PhysicalAction
    steps: tuple[NativeActionStep, ...]

    def __init__(
        self,
        source: PhysicalAction,
        steps: Iterable[NativeActionStep],
    ) -> None:
        if not isinstance(source, PhysicalAction):
            raise TypeError("native action decomposition source must be typed")
        normalized = tuple(steps)
        if not normalized:
            raise ValueError(
                "native action decomposition requires at least one step")
        if any(not isinstance(step, NativeActionStep) for step in normalized):
            raise TypeError(
                "native action decomposition steps must be NativeActionStep values"
            )
        for step in normalized:
            if any(index >= source.arity for index in step.operands):
                raise ValueError(
                    f"native action @{step.action.name} operand exceeds source "
                    f"@{source.name} arity {source.arity}")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "steps", normalized)


@dataclass(frozen=True, slots=True)
class PhysicalInstrument:
    """A physical operation that produces a typed classical record.

    ``arity=None`` denotes a nonempty variadic instrument such as a native
    Pauli-product measurement. Instruments are advertised separately from
    state-only actions so measurement capability is never encoded as a magic
    action string.
    """

    name: str
    operation: str
    arity: int | None
    record_schema: str
    preserves_inputs: bool
    process: QuantumProcess
    controller_bindings: Mapping[str, str]
    parameters: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __init__(
        self,
        name: str,
        *,
        operation: str,
        arity: int | None,
        record_schema: str,
        preserves_inputs: bool,
        process: QuantumProcess,
        controller_bindings: Mapping[str, str] | None = None,
        parameters: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        name = str(name)
        operation = str(operation)
        record_schema = str(record_schema)
        if not name or not operation or not record_schema:
            raise ValueError(
                "physical instrument name, operation, and record schema "
                "must be nonempty")
        if arity is not None and (not isinstance(arity, int) or
                                  isinstance(arity, bool) or arity <= 0):
            raise TypeError(
                "physical instrument arity must be positive or None")
        if not isinstance(preserves_inputs, bool):
            raise TypeError("physical instrument preserves_inputs must be bool")
        if not isinstance(process, QuantumProcess):
            raise TypeError(
                "physical instrument process must be a QuantumProcess")
        bindings = {
            str(target): str(semantic)
            for target, semantic in (controller_bindings or {}).items()
        }
        if any(not key or not value for key, value in bindings.items()):
            raise ValueError(
                "physical instrument controller bindings must be nonempty")
        parameter_names = tuple(map(str, parameters))
        if any(not value for value in parameter_names):
            raise ValueError(
                "physical instrument parameter names must be nonempty")
        if len(set(parameter_names)) != len(parameter_names):
            raise ValueError(
                "physical instrument has duplicate parameter names")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "operation", operation)
        object.__setattr__(self, "arity", arity)
        object.__setattr__(self, "record_schema", record_schema)
        object.__setattr__(self, "preserves_inputs", preserves_inputs)
        object.__setattr__(self, "process", process)
        object.__setattr__(self, "controller_bindings",
                           MappingProxyType(bindings))
        object.__setattr__(self, "parameters", parameter_names)
        object.__setattr__(self, "metadata", freeze_mapping(metadata))

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)


class PhysicalDefinition:
    __slots__ = ("provider", "architecture", "name", "signature", "profile",
                 "__dict__")

    def __init__(
        self,
        provider: Callable[..., Any],
        architecture: "PhysicalMachine",
        *,
        name: str | None = None,
    ) -> None:
        if not isinstance(architecture, PhysicalMachine):
            raise TypeError(
                "@cudaq.logical.physical requires a PhysicalMachine")
        self.provider = provider
        self.architecture = architecture
        self.name = name or provider.__name__
        self.signature: Signature = signature(provider)
        if self.signature.parameters:
            raise TypeError(
                "this @cudaq.logical.physical slice uses acquired resources, not arguments"
            )
        self.profile = "p3"
        self.__name__ = provider.__name__
        self.__qualname__ = provider.__qualname__
        self.__doc__ = provider.__doc__
        self.__module__ = provider.__module__

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)


def physical(architecture, *, name: str | None = None):

    def decorate(provider):
        return PhysicalDefinition(provider, architecture, name=name)

    return decorate


@dataclass(frozen=True, slots=True)
class ResourceClass:
    kind: str
    count: int
    granularity: ResourceGranularity = ResourceGranularity.CARRIER
    footprint: PhysicalFootprint | None = None
    native_actions: tuple[PhysicalAction | str, ...] = ()
    native_action_decompositions: tuple[NativeActionDecomposition, ...] = ()
    native_instruments: tuple[PhysicalInstrument, ...] = ()
    capabilities: tuple[PhysicalCapability | str, ...] = ()
    capability_bindings: tuple[PhysicalCapabilityBinding, ...] = ()
    erasure_indices: tuple[int, ...] | None = None
    name: str | None = None

    def __init__(
        self,
        kind: str,
        count: int,
        *,
        granularity: ResourceGranularity | str = ResourceGranularity.CARRIER,
        footprint: PhysicalFootprint | None = None,
        native_actions: Iterable[PhysicalAction | str] = (),
        native_action_decompositions: Iterable[NativeActionDecomposition] = (),
        native_instruments: Iterable[PhysicalInstrument] = (),
        capabilities: Iterable[PhysicalCapability | str] = (),
        capability_bindings: Iterable[PhysicalCapabilityBinding] = (),
        erasure_indices: Iterable[int] | None = None,
        name: str | None = None,
    ) -> None:
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise TypeError("physical resource count must be a nonnegative int")
        kind = str(kind)
        if not kind:
            raise ValueError("physical resource kind must be nonempty")
        try:
            granularity = ResourceGranularity(granularity)
        except ValueError as error:
            raise ValueError(
                "physical resource granularity must be 'carrier' or 'patch'"
            ) from error
        if granularity is ResourceGranularity.PATCH:
            if not isinstance(footprint, PhysicalFootprint):
                raise TypeError(
                    "patch-granularity resources require a PhysicalFootprint")
        elif footprint is not None:
            raise ValueError(
                "carrier-granularity resources must not declare a patch footprint"
            )
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "granularity", granularity)
        object.__setattr__(self, "footprint", footprint)
        actions = tuple(native_actions)
        if any(not isinstance(action, (PhysicalAction, str))
               for action in actions):
            raise TypeError(
                "native_actions entries must be PhysicalAction values or "
                "compatibility strings")
        if any(isinstance(action, str) and not action for action in actions):
            raise ValueError("native action names must be nonempty")
        names = tuple(
            action.name if isinstance(action, PhysicalAction) else action
            for action in actions)
        if len(set(names)) != len(names):
            raise ValueError("resource class has duplicate native actions")
        decompositions = tuple(native_action_decompositions)
        if any(not isinstance(item, NativeActionDecomposition)
               for item in decompositions):
            raise TypeError("native_action_decompositions entries must be "
                            "NativeActionDecomposition values")
        sources = tuple(item.source.name for item in decompositions)
        if len(set(sources)) != len(sources):
            raise ValueError(
                "resource class has duplicate native action decompositions")
        advertised = set(names)
        for decomposition in decompositions:
            if decomposition.source.name in advertised:
                raise ValueError(
                    f"native action @{decomposition.source.name} must not also "
                    "have a decomposition")
            missing = {
                step.action.name
                for step in decomposition.steps
                if step.action.name not in advertised
            }
            if missing:
                raise ValueError(
                    f"native action decomposition @{decomposition.source.name} "
                    f"uses unadvertised actions {sorted(missing)!r}")
        instruments = tuple(native_instruments)
        if any(not isinstance(instrument, PhysicalInstrument)
               for instrument in instruments):
            raise TypeError(
                "native_instruments entries must be PhysicalInstrument values")
        instrument_names = tuple(instrument.name for instrument in instruments)
        if len(set(instrument_names)) != len(instrument_names):
            raise ValueError("resource class has duplicate native instruments")
        object.__setattr__(self, "native_actions", actions)
        object.__setattr__(self, "native_action_decompositions", decompositions)
        object.__setattr__(self, "native_instruments", instruments)
        capabilities = tuple(capabilities)
        if any(not isinstance(value, (PhysicalCapability, str))
               for value in capabilities):
            raise TypeError(
                "capabilities entries must be PhysicalCapability values or "
                "compatibility strings")
        if any(isinstance(value, str) and not value for value in capabilities):
            raise ValueError("physical capability keys must be nonempty")
        capability_keys = tuple(
            value.key if isinstance(value, PhysicalCapability) else value
            for value in capabilities)
        if len(set(capability_keys)) != len(capability_keys):
            raise ValueError(
                "resource class has duplicate physical capabilities")
        object.__setattr__(self, "capabilities", capabilities)
        bindings = tuple(capability_bindings)
        if any(not isinstance(binding, PhysicalCapabilityBinding)
               for binding in bindings):
            raise TypeError(
                "capability_bindings entries must be PhysicalCapabilityBinding values"
            )
        for binding in bindings:
            if any(index < 0 or index >= count for index in binding.indices):
                raise ValueError(
                    "physical capability indices must lie within the resource class"
                )
        seen_capabilities = set()
        for binding in bindings:
            key = binding.capability.key
            if key in seen_capabilities or key in capability_keys:
                raise ValueError(
                    f"resource class has duplicate physical capability {key!r}")
            seen_capabilities.add(key)
        object.__setattr__(self, "capability_bindings", bindings)
        if erasure_indices is not None:
            erasure_indices = tuple(erasure_indices)
            if any(
                    isinstance(index, bool) or not isinstance(index, int)
                    for index in erasure_indices):
                raise TypeError("erasure_indices must contain Python ints")
            if len(set(erasure_indices)) != len(erasure_indices):
                raise ValueError("erasure_indices must not contain duplicates")
            if any(index < 0 or index >= count for index in erasure_indices):
                raise ValueError(
                    "erasure_indices entries must lie within the resource class"
                )
            erasure_indices = tuple(sorted(erasure_indices))
        if erasure_indices is not None and any(
                isinstance(binding.capability, HeraldedErasure)
                for binding in bindings):
            raise ValueError(
                "use either typed HeraldedErasure capability bindings or "
                "legacy erasure_indices, not both")
        object.__setattr__(self, "erasure_indices", erasure_indices)
        object.__setattr__(self, "name", name)


@dataclass(frozen=True, slots=True)
class _TopologyEdge:
    """Private normalized form of one undirected adjacency pair."""

    source: int
    target: int


@dataclass(frozen=True, slots=True)
class Topology:
    """Physical carrier adjacency for one homogeneous resource pool.

    Integer-pair edges are undirected and support every native arity-two
    action or instrument advertised by the resource class. ``num_nodes`` is
    optional: a bound resource pool supplies it, otherwise it is inferred as
    one plus the largest endpoint. Unknown legacy kinds remain non-strict
    parameterized descriptions until a device-specific lowering expands them.
    When supplied, ``coordinates`` is an immutable complete embedding from
    carrier index to a unique two-dimensional integer point.
    """

    kind: str
    parameters: Mapping[str, Any]
    num_nodes: int | None
    edges: tuple[_TopologyEdge, ...]
    coordinates: Mapping[int, tuple[int, int]] | None
    strict: bool
    _name: str | None
    _adjacency: Mapping[int, tuple[int, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _edge_lookup: Mapping[tuple[int, int], _TopologyEdge] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        kind: str | None = None,
        *,
        parameters: Mapping[str, Any] | None = None,
        num_nodes: int | None = None,
        edges: Iterable[tuple[int, int]] | None = None,
        coordinates: Mapping[int, tuple[int, int]] | None = None,
        **parameter_values: Any,
    ) -> None:
        if parameters is not None and parameter_values:
            raise TypeError(
                "Topology accepts either parameters= or keyword parameter values"
            )
        kind = "explicit" if kind is None else str(kind)
        if not kind:
            raise ValueError("topology kind must be nonempty")
        if num_nodes is not None and (not isinstance(num_nodes, int) or
                                      isinstance(num_nodes, bool) or
                                      num_nodes < 0):
            raise TypeError("topology num_nodes must be a nonnegative int")
        normalized_coordinates = None
        if coordinates is not None:
            if not isinstance(coordinates, Mapping):
                raise TypeError("topology coordinates must be a mapping")
            normalized = {}
            occupied = set()
            for index, coordinate in coordinates.items():
                if (not isinstance(index, int) or isinstance(index, bool) or
                        index < 0):
                    raise TypeError(
                        "topology coordinate indices must be nonnegative ints")
                if (not isinstance(coordinate, tuple) or len(coordinate) != 2 or
                        any(not isinstance(component, int) or
                            isinstance(component, bool)
                            for component in coordinate)):
                    raise TypeError(
                        "topology coordinates must be exact 2D integer tuples")
                if coordinate in occupied:
                    raise ValueError("topology coordinates must be unique")
                normalized[index] = coordinate
                occupied.add(coordinate)
            normalized_coordinates = dict(sorted(normalized.items()))
        values = dict(parameters or parameter_values)
        expanded_edges: tuple[_TopologyEdge, ...] = ()
        is_strict = edges is not None
        if edges is not None:
            normalized = []
            for edge in edges:
                if (not isinstance(edge, (tuple, list)) or len(edge) != 2 or
                        any(not isinstance(endpoint, int) or
                            isinstance(endpoint, bool) or endpoint < 0
                            for endpoint in edge)):
                    raise TypeError(
                        "topology edges must be pairs of nonnegative ints")
                source, target = edge
                if source == target:
                    raise ValueError("topology edges cannot be self-loops")
                normalized.append(
                    _TopologyEdge(min(source, target), max(source, target)))
            expanded_edges = tuple(normalized)
        elif kind == "line" and "length" in values:
            length = int(values["length"])
            if length < 0:
                raise ValueError("line topology length must be nonnegative")
            if num_nodes is not None and num_nodes != length:
                raise ValueError(
                    "line topology num_nodes must equal its length")
            num_nodes = length
            expanded_edges = tuple(
                _TopologyEdge(index, index + 1)
                for index in range(max(0, length - 1)))
            is_strict = True
        elif kind == "grid" and {"rows", "columns"} <= values.keys():
            rows, columns = int(values["rows"]), int(values["columns"])
            if rows < 0 or columns < 0:
                raise ValueError("grid topology dimensions must be nonnegative")
            if num_nodes is not None and num_nodes != rows * columns:
                raise ValueError(
                    "grid topology num_nodes must equal rows * columns")
            num_nodes = rows * columns
            pairs = []
            for row in range(rows):
                for column in range(columns):
                    node = row * columns + column
                    if column + 1 < columns:
                        pairs.append((node, node + 1))
                    if row + 1 < rows:
                        pairs.append((node, node + columns))
            expanded_edges = tuple(
                _TopologyEdge(left, right) for left, right in pairs)
            is_strict = True

        seen = set()
        for edge in expanded_edges:
            key = (edge.source, edge.target)
            if key in seen:
                raise ValueError("topology contains a duplicate edge")
            seen.add(key)
        inferred = max(
            (endpoint for edge in expanded_edges
             for endpoint in (edge.source, edge.target)),
            default=-1,
        ) + 1
        if normalized_coordinates is not None:
            coordinate_count = max(normalized_coordinates, default=-1) + 1
            required_count = (num_nodes if num_nodes is not None else max(
                inferred, coordinate_count))
            expected = set(range(required_count))
            if set(normalized_coordinates) != expected:
                raise ValueError(
                    "topology coordinates must cover every carrier index")
            if num_nodes is None:
                num_nodes = required_count
        if num_nodes is not None and inferred > num_nodes:
            raise ValueError("topology edge endpoint exceeds num_nodes")
        if kind == "explicit" and num_nodes is not None:
            is_strict = True
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "parameters",
            freeze_mapping(values),
        )
        object.__setattr__(self, "num_nodes", num_nodes)
        object.__setattr__(self, "edges", expanded_edges)
        object.__setattr__(
            self,
            "coordinates",
            (None if normalized_coordinates is None else
             MappingProxyType(normalized_coordinates)),
        )
        object.__setattr__(self, "strict", is_strict)
        object.__setattr__(self, "_name", None)
        adjacency = {index: [] for index in range(num_nodes or inferred)}
        edge_lookup = {}
        for edge in expanded_edges:
            adjacency.setdefault(edge.source, []).append(edge.target)
            adjacency.setdefault(edge.target, []).append(edge.source)
            edge_lookup[(edge.source, edge.target)] = edge
        object.__setattr__(
            self,
            "_adjacency",
            MappingProxyType({
                node: tuple(sorted(neighbors))
                for node, neighbors in adjacency.items()
            }),
        )
        object.__setattr__(
            self,
            "_edge_lookup",
            MappingProxyType(edge_lookup),
        )

    @property
    def name(self) -> str | None:
        """Compiler-assigned symbol after architecture normalization."""

        return self._name

    @property
    def nodes(self) -> tuple[int, ...]:
        count = self.num_nodes
        if count is None:
            count = max(
                (endpoint for edge in self.edges
                 for endpoint in (edge.source, edge.target)),
                default=-1,
            ) + 1
        return tuple(range(count))

    def _clone(self, *, name=None, num_nodes=None, strict=None) -> "Topology":
        clone = object.__new__(Topology)
        clone_count = self.num_nodes if num_nodes is None else num_nodes
        object.__setattr__(clone, "kind", self.kind)
        object.__setattr__(clone, "parameters", self.parameters)
        object.__setattr__(clone, "num_nodes", clone_count)
        object.__setattr__(clone, "edges", self.edges)
        object.__setattr__(clone, "coordinates", self.coordinates)
        object.__setattr__(clone, "strict",
                           self.strict if strict is None else strict)
        object.__setattr__(clone, "_name", self._name if name is None else name)
        if clone_count is None or all(
                index in self._adjacency for index in range(clone_count)):
            adjacency = self._adjacency
        else:
            adjacency = MappingProxyType({
                index: self._adjacency.get(index, ())
                for index in range(clone_count)
            })
        object.__setattr__(clone, "_adjacency", adjacency)
        object.__setattr__(clone, "_edge_lookup", self._edge_lookup)
        return clone

    def _named(self, name: str) -> "Topology":
        return self._clone(name=name)

    def _bind_resource_count(self, count: int) -> "Topology":
        inferred = len(self.nodes)
        if inferred > count:
            raise ValueError(
                f"topology requires {inferred} nodes, but the bound resource "
                f"pool contains {count}")
        if self.num_nodes is not None and self.num_nodes != count:
            raise ValueError(
                f"topology num_nodes={self.num_nodes} does not match the "
                f"bound resource count {count}")
        strict = self.strict or self.kind == "explicit"
        if self.num_nodes == count and self.strict == strict:
            return self
        return self._clone(
            num_nodes=count,
            strict=strict,
        )

    @classmethod
    def line(cls, length: int) -> "Topology":
        return cls("line", length=length)

    @classmethod
    def grid(cls, rows: int, columns: int) -> "Topology":
        return cls("grid", rows=rows, columns=columns)

    def edge(self, source: int, target: int) -> _TopologyEdge | None:
        return self._edge_lookup.get((min(source, target), max(source, target)))

    def shortest_path(
        self,
        source: int,
        target: int,
        *,
        allowed: Iterable[int] | None = None,
    ) -> tuple[int, ...] | None:
        """Return the deterministic shortest adjacency path."""

        allowed_nodes = None if allowed is None else set(allowed)
        if allowed_nodes is None:
            if source not in self._adjacency or target not in self._adjacency:
                return None
        elif source not in allowed_nodes or target not in allowed_nodes:
            return None
        queue = deque((source,))
        predecessor = {source: None}
        while queue:
            node = queue.popleft()
            if node == target:
                path = []
                cursor = target
                while cursor is not None:
                    path.append(cursor)
                    cursor = predecessor[cursor]
                return tuple(reversed(path))
            for neighbor in self._adjacency.get(node, ()):
                if allowed_nodes is not None and neighbor not in allowed_nodes:
                    continue
                if neighbor in predecessor:
                    continue
                predecessor[neighbor] = node
                queue.append(neighbor)
        return None


@dataclass(frozen=True, slots=True)
class PatchKind:
    """Open, typed category for an implicit patch slot."""

    name: str

    def __post_init__(self) -> None:
        name = str(self.name)
        if not name:
            raise ValueError("patch kind name must be nonempty")
        object.__setattr__(self, "name", name)


PatchKind.DATA = PatchKind("data")
PatchKind.ANCILLA = PatchKind("ancilla")
PatchKind.ROUTING = PatchKind("routing")


@dataclass(frozen=True, slots=True)
class PatchTopology:
    """Physical carrier groups for the implicit slots of one device region.

    Region capacity defines slots ``0..capacity-1``.  This value only binds
    those slots to physical carriers and optionally categorizes them.  Its
    coarse ``edges`` are derived when the containing architecture supplies the
    authoritative carrier topology; users never author a second connectivity
    graph here.
    """

    carrier_groups: tuple[tuple[int, ...], ...]
    categories: tuple[PatchKind | None, ...]
    edges: tuple[tuple[int, int], ...]

    def __init__(
        self,
        carrier_groups: Mapping[int, Iterable[int]] | Iterable[Iterable[int]],
        *,
        categories: Mapping[int, PatchKind] | None = None,
    ) -> None:
        if isinstance(carrier_groups, Mapping):
            keys = tuple(sorted(carrier_groups))
            if keys != tuple(range(len(keys))):
                raise ValueError(
                    "patch carrier groups must name contiguous implicit slots "
                    "starting at zero")
            groups = tuple(tuple(carrier_groups[index]) for index in keys)
        else:
            groups = tuple(tuple(group) for group in carrier_groups)
        if not groups:
            raise ValueError(
                "patch topology requires at least one carrier group")
        normalized = []
        claimed: dict[int, int] = {}
        for slot, group in enumerate(groups):
            if not group:
                raise ValueError(
                    f"patch slot {slot} requires at least one carrier")
            if any(not isinstance(index, int) or isinstance(index, bool) or
                   index < 0 for index in group):
                raise TypeError(
                    "patch carrier indices must be nonnegative ints")
            if len(set(group)) != len(group):
                raise ValueError(
                    f"patch slot {slot} contains a duplicate carrier")
            for index in group:
                if index in claimed:
                    raise ValueError(
                        f"carrier {index} belongs to both patch slots "
                        f"{claimed[index]} and {slot}")
                claimed[index] = slot
            normalized.append(tuple(group))
        category_values: list[PatchKind | None] = [None] * len(normalized)
        for slot, category in (categories or {}).items():
            if not isinstance(slot, int) or isinstance(slot, bool):
                raise TypeError(
                    "patch category keys must be integer slot indices")
            if slot < 0 or slot >= len(normalized):
                raise ValueError(
                    f"patch category references unknown slot {slot}")
            if not isinstance(category, PatchKind):
                raise TypeError(
                    "patch categories must be cudaq.logical.PatchKind values")
            category_values[slot] = category
        object.__setattr__(self, "carrier_groups", tuple(normalized))
        object.__setattr__(self, "categories", tuple(category_values))
        object.__setattr__(self, "edges", ())

    @property
    def capacity(self) -> int:
        return len(self.carrier_groups)

    def is_adjacent(self, left: int, right: int) -> bool:
        return (min(left, right), max(left, right)) in self.edges

    def _bind(
        self,
        *,
        topology: Topology,
        capacity: int | None,
        resource_count: int,
    ) -> "PatchTopology":
        if capacity is None:
            raise ValueError(
                "patch topology requires a finite region capacity so its "
                "implicit slots are well-defined")
        if capacity != self.capacity:
            raise ValueError(
                f"patch topology defines {self.capacity} slots, but the bound "
                f"region has capacity {capacity}")
        if not topology.strict:
            raise ValueError(
                "patch topology requires a strict explicit carrier topology")
        topology_nodes = set(topology.nodes)
        for slot, group in enumerate(self.carrier_groups):
            unknown = set(group) - topology_nodes
            if unknown:
                raise ValueError(
                    f"patch slot {slot} references carriers absent from the "
                    f"bound topology: {sorted(unknown)!r}")
            if max(group) >= resource_count:
                raise ValueError(
                    f"patch slot {slot} carrier index exceeds physical resource "
                    f"capacity {resource_count}")

        # This graph is a structural projection only. Action-specific
        # feasibility remains a P3 lowering obligation over every requested
        # carrier interaction, not a promise made by one crossing edge.
        owner = {
            carrier: slot for slot, group in enumerate(self.carrier_groups)
            for carrier in group
        }
        neighbors: dict[int, set[int]] = {node: set() for node in topology.nodes}
        for edge in topology.edges:
            neighbors[edge.source].add(edge.target)
            neighbors[edge.target].add(edge.source)
        coarse_edges = []
        for left in range(self.capacity):
            for right in range(left + 1, self.capacity):
                targets = set(self.carrier_groups[right])
                frontier = list(self.carrier_groups[left])
                visited = set(frontier)
                connected = False
                while frontier and not connected:
                    node = frontier.pop(0)
                    for neighbor in sorted(neighbors[node]):
                        if neighbor in targets:
                            connected = True
                            break
                        if neighbor in visited or neighbor in owner:
                            continue
                        visited.add(neighbor)
                        frontier.append(neighbor)
                if connected:
                    coarse_edges.append((left, right))
        result = PatchTopology(
            self.carrier_groups,
            categories={
                slot: category
                for slot, category in enumerate(self.categories)
                if category is not None
            },
        )
        object.__setattr__(result, "edges", tuple(coarse_edges))
        return result


class PhysicalMachine(ImmutableValue):
    __slots__ = (
        "name",
        "resource_classes",
        "topologies",
        "metadata",
        "_device",
    )

    def __init__(
        self,
        name: str,
        *,
        resource_classes: Mapping[str, ResourceClass] | Iterable[ResourceClass],
        topologies: Mapping[str, Topology] | Iterable[Topology] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.resource_classes, _ = self._name_values(resource_classes)
        self.topologies, _ = self._name_values(topologies)
        self.metadata = freeze_mapping(metadata)
        self._device = None
        self._seal()

    def __getattr__(self, name: str):
        for collection in (
                self.resource_classes,
                self.topologies,
        ):
            for value in collection:
                if value.name == name:
                    return value
        raise AttributeError(name)

    @staticmethod
    def _name_values(values):
        if isinstance(values, Mapping):
            named = tuple((value._named(name) if isinstance(value, Topology)
                           else replace(value, name=name))
                          for name, value in values.items())
            return named, {
                id(original): replacement
                for original, replacement in zip(values.values(), named)
            }
        values = tuple(values)
        if any(value.name is None for value in values):
            raise ValueError("unnamed architecture members require a mapping")
        return values, {id(value): value for value in values}

    def materialize(self, module=None):
        from ..compiler import compile

        return compile(self, module=module)

    def _derive(self) -> "PhysicalMachine":
        """Return an unattached immutable-equivalent architecture value."""

        return PhysicalMachine(
            self.name,
            resource_classes=self.resource_classes,
            topologies=self.topologies,
            metadata=self.metadata,
        )
