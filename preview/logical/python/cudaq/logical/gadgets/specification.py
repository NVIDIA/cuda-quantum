# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from inspect import Signature, signature
from math import isfinite
from types import MappingProxyType, NoneType
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from cudaq.logical.programs.binding import (
    LogicalPortRef,
    ObjectiveOperandRef,
)
from cudaq.logical._core.immutable import ImmutableValue

EncodingT = TypeVar("EncodingT")

from .interface import (
    BlockEndpoint,
    BlockFlow,
    EndpointCollection,
    GadgetInterface,
    _freeze_gadget_metadata,
    patch,
)
from .semantics import OutcomeMap, ParameterMap

_PORT_OWNERSHIP = {"inout": "borrow", "input": "consume", "output": "produce"}
_PORT_WIRE_STATE = {
    "inout": "initialized",
    "input": "absent",
    "output": "initialized"
}
_WIRE_STATES = ("initialized", "uninitialized", "measured", "absent")


@dataclass(frozen=True, slots=True)
class Port:
    """One explicit typed encoded-boundary port of a gadget specification.

    ``wire_state`` names the state of the port's encoded wire at the gadget
    boundary after the gadget acts, mirroring the canonical
    ``fabric.gadget_spec`` port lifecycles: an inout port borrows and returns
    ``initialized`` state, an output port produces ``initialized`` state from
    uninitialized carriers, and a consumed input port leaves ``absent`` state
    (``measured`` for terminal destructive data readout).
    """

    name: str
    direction: str
    encoding: Any
    ownership: str | None = None
    logical_ports: Any = None
    data_view: tuple[int, ...] | None = None
    ancilla_view: tuple[int, ...] | None = None
    wire_state: str | None = None

    @classmethod
    def input(cls, name: str, *, encoding, measured: bool = False, **kwargs):
        """Construct one typed consumed input boundary."""

        return cls(name,
                   "input",
                   encoding,
                   wire_state="measured" if measured else "absent",
                   **kwargs)

    @classmethod
    def output(cls, name: str, *, encoding, **kwargs):
        """Construct one typed produced output boundary."""

        return cls(name, "output", encoding, **kwargs)

    @classmethod
    def inout(cls, name: str, *, encoding, **kwargs):
        """Construct one typed borrowed-and-returned boundary."""

        return cls(name, "inout", encoding, **kwargs)

    def __post_init__(self) -> None:
        from cudaq.logical.codes import (
            Code,
            Encoding,
        )

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Port name must be a nonempty string")
        if self.direction not in _PORT_OWNERSHIP:
            raise ValueError("Port direction must be input, output, or inout")
        encoding = self.encoding
        if isinstance(encoding, Code):
            encoding = encoding.default_encoding
        if not isinstance(encoding, Encoding):
            raise TypeError(
                "Port encoding must be a cudaq.logical.Code or cudaq.logical.Encoding"
            )
        object.__setattr__(self, "encoding", encoding)

        canonical = _PORT_OWNERSHIP[self.direction]
        ownership = canonical if self.ownership is None else self.ownership
        if ownership != canonical:
            raise ValueError(
                f"{self.direction} ports must use {canonical!r} ownership, "
                f"not {ownership!r}")
        object.__setattr__(self, "ownership", ownership)

        wire_state = self.wire_state
        if wire_state is None:
            wire_state = _PORT_WIRE_STATE[self.direction]
        if wire_state not in _WIRE_STATES:
            raise ValueError(
                "Port wire_state must be initialized, uninitialized, "
                "measured, or absent")
        allowed = (("absent", "measured") if self.direction == "input" else
                   ("initialized",))
        if wire_state not in allowed:
            raise ValueError(
                f"a {self.direction} port's wire_state must be one of {allowed}"
            )
        object.__setattr__(self, "wire_state", wire_state)

        code = encoding.code
        logical_ports = self.logical_ports
        mapping: dict[str, str] = {}
        if logical_ports is None:
            pass
        elif isinstance(logical_ports,
                        int) and not isinstance(logical_ports, bool):
            if logical_ports != code.k:
                raise ValueError(
                    f"Port logical arity {logical_ports} disagrees with code "
                    f"{code.name!r} (k = {code.k})")
        elif isinstance(logical_ports, Mapping):
            names = set(encoding.logical_ports)
            normalized = {}
            targets = set()
            for raw_operand, raw_target in logical_ports.items():
                operand = (raw_operand.name if isinstance(
                    raw_operand, ObjectiveOperandRef) else raw_operand)
                if not isinstance(operand, str) or not operand:
                    raise ValueError(
                        "Port logical_ports keys must be objective operand names"
                    )
                if isinstance(raw_target, LogicalPortRef):
                    if raw_target.encoding is not encoding:
                        raise ValueError(
                            "Port logical_ports target belongs to a different "
                            "encoding")
                    target = raw_target.name
                else:
                    target = raw_target
                if target not in names:
                    raise ValueError(
                        f"encoding {encoding.name!r} has no logical port {target!r}"
                    )
                if operand in normalized:
                    raise ValueError(
                        f"Port logical_ports repeats objective operand {operand!r}"
                    )
                if target in targets:
                    raise ValueError(
                        "Port logical_ports must map operands to distinct "
                        "encoding logical ports")
                normalized[operand] = target
                targets.add(target)
            mapping = normalized
        else:
            raise TypeError(
                "Port logical_ports must be an arity int or an operand mapping")
        object.__setattr__(self, "logical_ports", MappingProxyType(mapping))

        views: dict[str, tuple[int, ...] | None] = {}
        for label, view in (
            ("data_view", self.data_view),
            ("ancilla_view", self.ancilla_view),
        ):
            if view is None:
                views[label] = None
                continue
            view = tuple(view)
            if any(not isinstance(index, int) or isinstance(index, bool) or
                   index < 0 for index in view):
                raise TypeError(
                    f"Port {label} indices must be nonnegative ints")
            if len(set(view)) != len(view):
                raise ValueError(f"Port {label} indices must be unique")
            if any(index >= code.block.size for index in view):
                raise ValueError(
                    f"Port {label} index is outside the code block")
            views[label] = view
        if views["data_view"] is not None and len(views["data_view"]) != code.n:
            raise ValueError(
                f"Port data_view must select exactly n = {code.n} data carriers"
            )
        if (views["data_view"] is not None and
                views["ancilla_view"] is not None and
                set(views["data_view"]) & set(views["ancilla_view"])):
            raise ValueError("Port data_view and ancilla_view must be disjoint")
        object.__setattr__(self, "data_view", views["data_view"])
        object.__setattr__(self, "ancilla_view", views["ancilla_view"])

    @property
    def logical_arity(self) -> int:
        return self.encoding.code.k


@dataclass(frozen=True, slots=True)
class GadgetSpec:
    """Explicit gadget boundary and semantic-map specification.

    Ordinary gadgets infer this artifact from the typed realization
    signature; an explicit spec exists for irregular boundaries and imported
    ABIs.  It constrains — never overrides — physical reality: its port
    count, directions, and encodings must match what the realization
    signature infers, and materialization rejects any disagreement with
    :class:`cudaq.logical.errors.InvalidPortBinding`.
    """

    implements: Any
    ports: tuple[Port, ...]
    record_schema: tuple[str, ...] | None = None
    outcome_map: OutcomeMap | None = None
    parameter_map: ParameterMap | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.implements is None:
            raise ValueError("GadgetSpec requires an implements= objective")
        ports = tuple(self.ports)
        if any(not isinstance(port, Port) for port in ports):
            raise TypeError(
                "GadgetSpec ports must be cudaq.logical.Port values")
        names = tuple(port.name for port in ports)
        if len(set(names)) != len(names):
            raise ValueError("GadgetSpec port names must be unique")
        object.__setattr__(self, "ports", ports)
        if self.record_schema is not None:
            schema = tuple(self.record_schema)
            if any(not isinstance(name, str) or not name for name in schema):
                raise ValueError(
                    "GadgetSpec record_schema entries must be nonempty "
                    "record names")
            if len(set(schema)) != len(schema):
                raise ValueError(
                    "GadgetSpec record_schema entries must be unique")
            object.__setattr__(self, "record_schema", schema)
        for label, value, expected in (
            ("outcome_map", self.outcome_map, OutcomeMap),
            ("parameter_map", self.parameter_map, ParameterMap),
        ):
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"GadgetSpec {label} must be a "
                                f"cudaq.logical.{expected.__name__}")
        referenced = self.outcome_map.records if self.outcome_map else ()
        if referenced and self.record_schema is None:
            raise ValueError(
                "GadgetSpec maps that name records require record_schema")
        if self.record_schema is not None:
            missing = sorted(set(referenced) - set(self.record_schema))
            if missing:
                raise ValueError(
                    f"GadgetSpec map records {missing} are missing from its "
                    "record schema")
        if self.outcome_map is not None:
            input_ports = tuple(
                port for port in ports if port.direction in {"input", "inout"})
            for row in self.outcome_map.input_syndromes:
                for term in row:
                    if term.port >= len(input_ports):
                        raise ValueError(
                            "OutcomeMap input-syndrome term names an absent "
                            "input port ordinal")
                    profile = input_ports[term.port].encoding.profile
                    if term.index >= profile.effective_stabilizers.nrows:
                        raise ValueError(
                            "OutcomeMap input-syndrome term is outside its "
                            "endpoint's effective syndrome width")
        object.__setattr__(
            self,
            "metadata",
            _freeze_gadget_metadata(dict(self.metadata or {}),
                                    what="GadgetSpec.metadata"),
        )

    @property
    def accepted_encodings(self) -> tuple[Any, ...]:
        result = []
        for port in self.ports:
            if port.direction in ("input",
                                  "inout") and port.encoding not in result:
                result.append(port.encoding)
        return tuple(result)

    @property
    def produced_encodings(self) -> tuple[Any, ...]:
        result = []
        for port in self.ports:
            if port.direction in ("output",
                                  "inout") and port.encoding not in result:
                result.append(port.encoding)
        return tuple(result)


def _profile_for_patch(annotation):
    from cudaq.logical.codes import (
        Code,
        Encoding,
    )

    if get_origin(annotation) is not patch:
        return None
    (target,) = get_args(annotation)
    if isinstance(target, Code):
        return target.default_profile
    if isinstance(target, Encoding):
        return target.profile
    return None


def _encoding_for_patch(annotation):
    from cudaq.logical.codes import (
        Code,
        Encoding,
    )

    if get_origin(annotation) is not patch:
        return None
    (target,) = get_args(annotation)
    if isinstance(target, Code):
        return target.default_encoding
    if isinstance(target, Encoding):
        return target
    return None


def _build_gadget_interface(gadget: "GadgetDefinition") -> GadgetInterface:
    """Infer a typed block hypergraph from the realization signature."""

    from ..std import LogicalInstrumentRef

    parameters = []
    for name, parameter in gadget.signature.parameters.items():
        annotation = gadget.type_hints.get(name, parameter.annotation)
        encoding = _encoding_for_patch(annotation)
        profile = _profile_for_patch(annotation)
        if encoding is not None and profile is not None:
            parameters.append((name, encoding, profile))

    preparing = isinstance(gadget.implements, LogicalInstrumentRef) and (
        gadget.implements.name.startswith("prepare_"))
    input_specs = () if preparing else tuple(parameters)

    annotation = gadget.type_hints.get("return",
                                       gadget.signature.return_annotation)
    return_types = get_args(annotation) if get_origin(
        annotation) is tuple else (annotation,)
    output_specs = []
    for item in return_types:
        encoding = _encoding_for_patch(item)
        profile = _profile_for_patch(item)
        if encoding is None or profile is None:
            continue
        # Patch lanes are ordinal independently of interleaved classical
        # results. A paired output retains only the corresponding input's
        # diagnostic alias; semantic identity is carried by the flow pair.
        lane = len(output_specs)
        name = parameters[lane][0] if lane < len(
            parameters) else f"output{lane}"
        output_specs.append((name, encoding, profile))

    paired_count = min(len(input_specs), len(output_specs))
    inputs = tuple(
        BlockEndpoint(
            gadget,
            "input",
            index,
            name,
            encoding,
            profile,
            ownership="borrow" if index < paired_count else "consume",
        ) for index, (name, encoding, profile) in enumerate(input_specs))
    outputs = tuple(
        BlockEndpoint(
            gadget,
            "output",
            index,
            name,
            encoding,
            profile,
            ownership="borrow" if index < paired_count else "produce",
        ) for index, (name, encoding, profile) in enumerate(output_specs))
    pairs = tuple(zip(inputs[:paired_count], outputs[:paired_count]))
    if not inputs:
        kind = "prepare"
    elif not outputs:
        kind = "measure"
    elif len(inputs) < len(outputs):
        kind = "split"
    elif len(inputs) > len(outputs):
        kind = "merge"
    else:
        kind = "transform"
    flows = (BlockFlow(inputs, outputs, kind, pairs,
                       gadget.transform),) if (inputs or outputs) else ()
    return GadgetInterface(
        gadget,
        EndpointCollection(inputs),
        EndpointCollection(outputs),
        flows,
    )


def _inferred_port_table(interface: GadgetInterface):
    """The ordered (name, direction, encoding) boundary the signature infers.

    This mirrors the normalized ``fabric.gadget_spec`` port construction: one
    port per boundary name, inputs in signature order followed by output-only
    names, with an inout port carrying its output endpoint's encoding.
    """

    pairs = tuple(pair for flow in interface.flows for pair in flow.pairs)
    output_by_input = {
        input_endpoint: output_endpoint
        for input_endpoint, output_endpoint in pairs
    }
    paired_outputs = {output_endpoint for _input, output_endpoint in pairs}
    table = []
    for endpoint in interface.inputs:
        output = output_by_input.get(endpoint)
        table.append((endpoint.name, "inout" if output else "input",
                      output.encoding if output else endpoint.encoding))
    for endpoint in interface.outputs:
        if endpoint not in paired_outputs:
            table.append((endpoint.name, "output", endpoint.encoding))
    return tuple(table)


def _verify_explicit_spec(definition: "GadgetDefinition") -> None:
    """An explicit spec constrains the realization boundary, never overrides it."""

    from ..errors import InvalidPortBinding, ObjectiveMismatch
    from cudaq.logical.types.values import logical_qubit
    from ..std import LogicalInstrumentRef
    from cudaq.logical.programs.definition import ProgramDefinition
    from cudaq.logical.types.semantic import (
        logical_event,
        logical_frame,
        logical_record,
        logical_resource,
        record,
        resource,
    )

    spec = definition.spec
    inferred = _inferred_port_table(definition.interface)
    if len(spec.ports) != len(inferred):
        raise InvalidPortBinding(
            f"gadget {definition.name!r} explicit spec declares "
            f"{len(spec.ports)} port(s), but its realization boundary has "
            f"{len(inferred)}")
    for port, (name, direction, encoding) in zip(spec.ports, inferred):
        if port.direction != direction:
            raise InvalidPortBinding(
                f"gadget {definition.name!r} explicit port {port.name!r} "
                f"declares direction {port.direction!r}, but the realization "
                f"boundary {name!r} is {direction!r}")
        if port.encoding is not encoding:
            raise InvalidPortBinding(
                f"gadget {definition.name!r} explicit port {port.name!r} "
                f"declares encoding {port.encoding.name!r}, but the "
                f"realization boundary {name!r} carries {encoding.name!r}")

    def flattened(annotation):
        if annotation in (None, NoneType):
            return ()
        if get_origin(annotation) in (tuple, list):
            return tuple(item for member in get_args(annotation)
                         for item in flattened(member))
        return (annotation,)

    objective = definition.implements
    if isinstance(objective, LogicalInstrumentRef):
        objective_outcomes = (0 if objective.name.startswith("prepare_") else
                              objective.result_arity)
    elif isinstance(objective,
                    ProgramDefinition) and objective.kind == "objective":
        result = objective.type_hints.get("return",
                                          objective.signature.return_annotation)
        objective_outcomes = sum(item is bool for item in flattened(result))
    else:
        objective_outcomes = 0

    mapped_outcomes = (spec.outcome_map.outcome_count
                       if spec.outcome_map is not None else 0)
    if mapped_outcomes != objective_outcomes:
        raise ObjectiveMismatch(
            f"gadget {definition.name!r} explicit OutcomeMap has "
            f"{mapped_outcomes} row(s), but objective "
            f"{getattr(objective, 'name', objective)!r} exposes "
            f"{objective_outcomes} classical Boolean outcome(s)")

    realization_port_origins = {patch, record, resource}
    realization_parameters = tuple(
        name for name, parameter in definition.signature.parameters.items()
        if get_origin(definition.type_hints.get(
            name, parameter.annotation)) not in realization_port_origins)

    objective_parameters = ()
    if isinstance(objective,
                  ProgramDefinition) and objective.kind == "objective":
        objective_port_origins = {
            logical_event,
            logical_frame,
            logical_record,
            logical_resource,
        }
        objective_parameters = tuple(
            name for name, parameter in objective.signature.parameters.items()
            if (annotation := objective.type_hints.get(
                name, parameter.annotation)) is not logical_qubit and
            get_origin(annotation) not in objective_port_origins)

    parameter_pairs = (spec.parameter_map.pairs
                       if spec.parameter_map is not None else ())
    mapped_realization = tuple(pair[0] for pair in parameter_pairs)
    mapped_objective = tuple(pair[1] for pair in parameter_pairs)
    missing_realization = sorted(
        set(realization_parameters) - set(mapped_realization))
    extra_realization = sorted(
        set(mapped_realization) - set(realization_parameters))
    missing_objective = sorted(
        set(objective_parameters) - set(mapped_objective))
    extra_objective = sorted(set(mapped_objective) - set(objective_parameters))
    if any((
            missing_realization,
            extra_realization,
            missing_objective,
            extra_objective,
    )):
        raise ObjectiveMismatch(
            f"gadget {definition.name!r} ParameterMap must be total and "
            "bijective over non-port parameters; "
            f"missing realization={missing_realization}, "
            f"unknown realization={extra_realization}, "
            f"missing objective={missing_objective}, "
            f"unknown objective={extra_objective}")


def _gadget_boundary_profiles(gadget):
    return (
        {
            endpoint: endpoint.code_profile
            for endpoint in gadget.interface.inputs
        },
        {
            endpoint: endpoint.code_profile
            for endpoint in gadget.interface.outputs
        },
    )


def _normalize_gadget_logical_ports(implements, interface, logical_ports):
    """Validate typed/legacy bindings and retain canonical symbolic names."""

    from ..errors import InvalidPortBinding

    if logical_ports is None:
        return MappingProxyType({})
    if not isinstance(logical_ports, Mapping):
        raise TypeError(
            "@cudaq.logical.gadget logical_ports= expects a mapping")

    try:
        objective_operands = tuple(implements.operands)
    except (AttributeError, TypeError):
        objective_operands = ()
    objective_names = {operand.name for operand in objective_operands}

    endpoints = {}
    for endpoint in (*interface.inputs, *interface.outputs):
        endpoints[(endpoint.name, id(endpoint.encoding))] = endpoint
    boundary = tuple(endpoints.values())

    def normalize_target(raw_target):
        if isinstance(raw_target, LogicalPortRef):
            matching_endpoints = tuple(
                endpoint for endpoint in boundary
                if endpoint.encoding is raw_target.encoding and
                raw_target.name in endpoint.encoding.logical_port_indices)
            if not matching_endpoints:
                raise InvalidPortBinding(
                    f"logical port {raw_target!r} is not exposed by the gadget "
                    "boundary")
            leaf = raw_target.name
            expected_index = raw_target.encoding.logical_port_indices.get(leaf)
            if expected_index != raw_target.index:
                raise InvalidPortBinding(
                    f"logical port reference {raw_target!r} is not declared by "
                    f"encoding {raw_target.encoding.name!r}")
            if len(matching_endpoints) != 1:
                raise InvalidPortBinding(
                    f"logical port {leaf!r} is exposed by several gadget "
                    "boundaries carrying the same encoding; use a qualified "
                    "legacy 'parameter.port' binding until typed "
                    "boundary-port references are available")
            endpoint = matching_endpoints[0]
            exposures = tuple(
                candidate for candidate in boundary
                if leaf in candidate.encoding.logical_port_indices)
            target = leaf if len(exposures) == 1 else f"{endpoint.name}.{leaf}"
            identity = (endpoint.name, id(raw_target.encoding), leaf)
            return target, identity

        if not isinstance(raw_target, str) or not raw_target:
            raise TypeError(
                "logical_ports values must be cudaq.logical.LogicalPortRef values or "
                "legacy nonempty strings")
        if "." in raw_target:
            alias, leaf = raw_target.rsplit(".", 1)
            matches = tuple(endpoint for endpoint in boundary
                            if endpoint.name == alias and
                            leaf in endpoint.encoding.logical_port_indices)
            if len(matches) != 1:
                raise InvalidPortBinding(
                    f"logical port binding {raw_target!r} does not name one "
                    "gadget boundary port")
            endpoint = matches[0]
            return raw_target, (endpoint.name, id(endpoint.encoding), leaf)

        matches = tuple(endpoint for endpoint in boundary
                        if raw_target in endpoint.encoding.logical_port_indices)
        if not matches:
            raise InvalidPortBinding(
                f"no gadget boundary encoding exposes logical port "
                f"{raw_target!r}")
        if len(matches) != 1:
            raise InvalidPortBinding(
                f"logical port {raw_target!r} is ambiguous across gadget "
                "boundaries; qualify it as 'parameter.port'")
        endpoint = matches[0]
        return raw_target, (endpoint.name, id(endpoint.encoding), raw_target)

    normalized = {}
    claimed_targets = set()
    for raw_operand, raw_target in logical_ports.items():
        if isinstance(raw_operand, ObjectiveOperandRef):
            if raw_operand.objective is not implements:
                raise InvalidPortBinding(
                    f"objective operand {raw_operand!r} belongs to a different "
                    "objective")
            if raw_operand not in objective_operands:
                raise InvalidPortBinding(
                    f"objective operand reference {raw_operand!r} is not "
                    "declared by the objective")
            operand = raw_operand.name
        elif isinstance(raw_operand, str) and raw_operand:
            operand = raw_operand
        else:
            raise TypeError(
                "logical_ports keys must be cudaq.logical.ObjectiveOperandRef values or "
                "legacy nonempty strings")
        if operand not in objective_names:
            raise InvalidPortBinding(
                f"logical_ports contains unknown objective operand {operand!r}")
        if operand in normalized:
            raise InvalidPortBinding(
                f"logical_ports repeats objective operand {operand!r}")
        target, identity = normalize_target(raw_target)
        if identity in claimed_targets:
            raise InvalidPortBinding(
                f"logical port {target!r} is bound to more than one objective "
                "operand")
        normalized[operand] = target
        claimed_targets.add(identity)
    return MappingProxyType(normalized)
