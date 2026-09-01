# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from typing import get_args, get_origin, get_type_hints
from typing import Any

from .. import ir as mlir_ir

from ..qec.lowering import (
    ActionSiteHandle,
    GeneratedQECArtifact,
    QECCompilerContext,
    QECLowering,
)
from ..codes import (
    Code,
    Encoding,
    QECActionSelection,
    QECBlockBinding,
    QECBlockOwner,
    QECBlockRequest,
    QECSelectionWitness,
)
from ..programs.definition import (
    DefinitionHandle,
    ProgramDefinition,
)
from ..devices.definition import Device
from ..gadgets import (
    GadgetDefinition,
    patch,
)
from ..protocols.definition import ProtocolDefinition
from ..architecture.logical import CapabilityKey
from ..std import LogicalActionRef, LogicalInstrumentRef, ResourceFlowRef
from .build import (
    Build,
    EvidenceRecord,
    _qec_selection_sha256,
)
from .context import CompilationContext
from .lowering import discover_linked_definitions
from .protocol_identity import (
    protocol_definition_payload as _protocol_definition_payload,
    protocol_definition_sha256 as _protocol_definition_sha256,
    protocol_operation_payload as _protocol_operation_payload,
)


def _text(attribute) -> str:
    return str(getattr(attribute, "value", attribute)).strip('"')


def _symbol(attribute) -> str:
    raw = getattr(attribute, "value", attribute)
    if isinstance(raw, (tuple, list)):
        return str(raw[-1])
    value = str(raw).strip('"')
    if value.startswith("@"):
        value = value[1:]
    return value.split("::@")[-1]


def _root_symbol(attribute) -> str:
    raw = getattr(attribute, "value", attribute)
    if isinstance(raw, (tuple, list)):
        return str(raw[0])
    value = str(raw).strip('"')
    if value.startswith("@"):
        value = value[1:]
    return value.split("::@", 1)[0]


def _objective(attribute) -> str:
    text = str(attribute)
    for prefix in ("#qlx.action<", "#qlx.instrument<"):
        if text.startswith(prefix) and text.endswith(">"):
            return text[len(prefix):-1].strip('"')
    return _symbol(attribute).removeprefix("qlx_standard_")


def _basis(attribute) -> str:
    text = str(attribute)
    prefix = "#qlx.pauli<"
    if text.startswith(prefix) and text.endswith(">"):
        return text[len(prefix):-1].strip('"').lower()
    return _text(attribute).lower()


def _capability_key(attribute) -> str:
    text = str(attribute)
    prefix = "#lvm.capability<"
    if text.startswith(prefix) and text.endswith(">"):
        return text[len(prefix):-1].strip('"')
    return _text(attribute)


def _i64(context, value: int):
    return mlir_ir.IntegerAttr.get(
        mlir_ir.IntegerType.get_signless(64, context=context), value)


def _dictionary(context, values):
    with mlir_ir.Location.unknown(context=context):
        return mlir_ir.DictAttr.get(
            {
                str(key): (
                    mlir_ir.IntegerAttr.get(
                        mlir_ir.IntegerType.get_signless(64, context=context),
                        value,
                    ) if isinstance(value, int) and not isinstance(value, bool)
                    else mlir_ir.BoolAttr.get(value, context=context)
                    if isinstance(value, bool) else
                    mlir_ir.FloatAttr.get(mlir_ir.F64Type.get(
                        context=context), value) if isinstance(value, float)
                    else mlir_ir.StringAttr.get(str(value), context=context)
                ) for key, value in values.items()
            },
            context=context,
        )


def _parameters(attribute) -> dict[str, Any]:
    if attribute is None:
        return {}
    result = {}
    for key in ("x_mask", "z_mask", "sign", "angle_pi_numer", "angle_pi_denom"):
        try:
            value = attribute[key]
        except (KeyError, IndexError, TypeError):
            continue
        result[key] = int(getattr(value, "value", value))
    try:
        precision = attribute["precision"]
    except (KeyError, IndexError, TypeError):
        pass
    else:
        result["precision"] = float(getattr(precision, "value", precision))
    return result


def _objective_family(kind: str, objective: str | None) -> str:
    name = objective or ""
    name = name.removeprefix("qlx_standard_")
    if name == "mpp" or name.startswith("mpp_"):
        return "pauli_product_measurement"
    if name == "pauli_rotation" or name.startswith("pauli_rotation_"):
        return "pauli_product_rotation"
    if name.startswith("measure_"):
        return "logical_measurement"
    if name.startswith("prepare_") or kind == "prepare":
        return "logical_preparation"
    if kind == "memory":
        return "logical_memory"
    return kind


def _candidate_code(candidate) -> tuple[Code, Encoding]:
    if isinstance(candidate, QECLowering):
        values = candidate.codes
        name = candidate.name
    else:
        hints = candidate.type_hints
        values = []
        for parameter_name, parameter in candidate.signature.parameters.items():
            annotation = hints.get(parameter_name, parameter.annotation)
            if get_origin(annotation) is patch:
                values.append(get_args(annotation)[0])
        name = candidate.name
    if not values:
        raise ValueError(
            f"P2 candidate {name!r} must declare at least one accepted "
            "code or encoding before it can establish a P2 boundary")
    value = values[0]
    if isinstance(value, Encoding):
        return value.code, value
    return value, value.default_encoding


def _matches_objective(lowering: QECLowering, site: ActionSiteHandle) -> bool:
    if lowering.objective_family not in {
            site.objective_family,
            site.kind,
            "any",
    }:
        return False
    objective = lowering.objective
    if objective is None:
        return True
    if isinstance(objective, (LogicalActionRef, LogicalInstrumentRef)):
        expected = objective.name
    elif isinstance(objective, ProgramDefinition):
        expected = objective.name.removeprefix("qlx_standard_")
    else:
        expected = getattr(objective, "name", str(objective))
    return site.objective == expected


def _flatten_annotation(annotation):
    if get_origin(annotation) in (tuple, list):
        result = []
        for item in get_args(annotation):
            result.extend(_flatten_annotation(item))
        return tuple(result)
    return (annotation,)


def _matches_fixed(candidate, site: ActionSiteHandle, operation,
                   binding_by_name) -> bool:
    objective = candidate.implements
    if objective is None:
        return False
    if isinstance(objective, (LogicalActionRef, LogicalInstrumentRef)):
        expected = objective.name
    elif isinstance(objective, ProgramDefinition):
        expected = objective.name.removeprefix("qlx_standard_")
    else:
        return False
    if site.objective != expected:
        return False
    hints = candidate.type_hints
    result_annotation = hints.get("return",
                                  candidate.signature.return_annotation)
    flattened_results = _flatten_annotation(result_annotation)
    if (isinstance(objective, LogicalInstrumentRef) and
            objective.name.startswith("prepare_")):
        parameters = [
            hints.get(name, parameter.annotation)
            for name, parameter in candidate.signature.parameters.items()
        ]
        return (site.input_arity == 0 and site.result_arity == 1 and
                len(parameters) == 1 and get_origin(parameters[0]) is patch and
                len(flattened_results) == 1 and
                get_origin(flattened_results[0]) is patch)
    if (len(candidate.signature.parameters) == site.input_arity and
            len(flattened_results) == site.result_arity):
        return True

    # One high-rate block may realize an objective over several placed logical
    # values. Its Python boundary owns one patch, while P1 still has one value
    # per logical port.
    parameter_annotations = [
        hints.get(name, parameter.annotation)
        for name, parameter in candidate.signature.parameters.items()
    ]
    patch_inputs = sum(
        get_origin(value) is patch for value in parameter_annotations)
    if patch_inputs != len(parameter_annotations) or patch_inputs != 1:
        return False
    bindings = [binding_by_name.get(name) for name in site.placements]
    if any(binding is None or binding.block is None for binding in bindings):
        return False
    if len({binding.block for binding in bindings}) != 1:
        return False
    code, _encoding = _candidate_code(candidate)
    if code.k < len(bindings):
        return False
    quantum_results = sum(
        str(value.type).startswith("!lvm.logical_qubit<")
        for value in operation.results)
    patch_results = sum(
        get_origin(value) is patch for value in flattened_results)
    classical_results = len(flattened_results) - patch_results
    preserves_packed_owner = (quantum_results == 0 and patch_results == 1 and
                              len({binding.block for binding in bindings}) == 1)
    return ((patch_results == (1 if quantum_results else 0) or
             preserves_packed_owner) and
            classical_results == len(operation.results) - quantum_results)


def _requirements_met(candidate,
                      placements,
                      device,
                      binding_by_name,
                      *,
                      site=None) -> bool:
    available = set()
    spaces = {space.name: space for space in device.logical.spaces}
    for placement in placements:
        binding = binding_by_name.get(placement)
        if binding is None:
            continue
        binding_data = dict(binding.binding_data)
        support_spaces = (binding.space,)
        for space_name in support_spaces:
            space = spaces.get(space_name)
            if space is not None:
                available.update(item.key for item in space.capabilities)
    requirements = tuple(getattr(candidate, "requires", ()))
    return all(
        getattr(item, "key", str(item)) in available for item in requirements)


def _device_encodings(device):
    return {
        binding.logical_region.name: binding.qec_region.encoding
        for binding in device.logical_to_qec
    }


def _same_encoding(transaction, left, right) -> bool:
    if left is right:
        return True
    return (transaction.materialize(left).symbol == transaction.materialize(
        right).symbol)


def _device_encoding_for_site(transaction, device, placements, binding_by_name):
    encodings = _device_encodings(device)
    selected = []
    for placement in placements:
        binding = binding_by_name.get(placement)
        if binding is None:
            continue
        if binding.encoding is not None and not any(
                _same_encoding(transaction, item, binding.encoding)
                for item in selected):
            selected.append(binding.encoding)
        encoding = encodings.get(binding.space)
        if encoding is not None and not any(
                _same_encoding(transaction, item, encoding)
                for item in selected):
            selected.append(encoding)
    if len(selected) > 1:
        raise ValueError(
            "P2 QEC block policy conflicts with device or cross-space encoding "
            "requirements")
    return selected[0] if selected else None


def _matches_device_encoding(transaction, candidate, site, device,
                             binding_by_name) -> bool:
    required = _device_encoding_for_site(transaction, device, site.placements,
                                         binding_by_name)
    if required is None:
        return True
    _code, encoding = _candidate_code(candidate)
    return _same_encoding(transaction, encoding, required)


@dataclass(slots=True)
class _SelectedSite:
    operation: Any
    handle: ActionSiteHandle
    feasible: tuple[Any, ...]
    selected: Any


@dataclass(frozen=True, slots=True)
class _P2Binding:
    """Internal realization view layered over a code-agnostic P1 binding."""

    placement: str
    space: str
    slot: int
    source_allocation: int | None
    source_group: str | None
    source_path: tuple[int, ...]
    binding_kind: str
    binding_data: tuple[tuple[str, Any], ...]
    block: str | None = None
    logical_index: int | None = None
    encoding: Encoding | None = None

    @classmethod
    def from_p1(cls, binding):
        return cls(
            placement=binding.placement,
            space=binding.space,
            slot=binding.slot,
            source_allocation=binding.source_allocation,
            source_group=binding.source_group,
            source_path=binding.source_path,
            binding_kind=binding.binding_kind,
            binding_data=binding.binding_data,
        )


@dataclass(frozen=True, slots=True)
class _ConcreteResourceSupply:
    stream: Any
    consume: Any
    qec_route: Any


class _P1ToP2:

    def __init__(
        self,
        source: Build,
        device: Device,
        policy,
    ) -> None:
        self.source = source
        self.device = device
        self.policy = dict(policy or {})
        self.transaction = CompilationContext.replay(source)
        self.context = self.transaction.context
        self.module = self.transaction.module
        self.location = self.transaction.location
        self.kernel = self.transaction.find_symbol(source.root.symbol,
                                                   "lvm.kernel")
        if self.kernel is None:
            raise ValueError(
                f"P1 root lvm.kernel @{source.root.symbol} is missing")
        if "estimate_only" in self.kernel.attributes:
            raise ValueError(
                "estimate-only P1 demand surrogates cannot lower to P2; "
                "provide a unitary-faithful logical program first")
        linked = discover_linked_definitions(source.source_modules, device)
        definitions = linked.definitions
        self.ordinary_definition_ids = linked.ordinary_ids
        self.transaction.bind_resource_streams(device)
        self._bind_retained_stream_protocols(definitions)
        self.transaction.bind_resource_streams(device)
        self.lowerings = tuple(
            sorted(
                (value for value in definitions
                 if isinstance(value, QECLowering)),
                key=lambda value: (value.plugin, value.version, value.name),
            ))
        self.fixed = tuple(
            sorted(
                (value for value in definitions
                 if isinstance(value, (GadgetDefinition, ProtocolDefinition
                                      )) and value.implements is not None),
                key=lambda value: (value.name, type(value).__name__),
            ))
        self.binding_by_name = {
            binding.placement: _P2Binding.from_p1(binding)
            for binding in source.placement.bindings
        }
        self.architecture_definitions_by_space = {
            binding.logical_region.name:
                frozenset(
                    id(definition)
                    for definition in binding.architecture.link_roots)
            for binding in device.logical_to_qec
            if binding.architecture is not None
        }
        self.qec_region_by_space = {
            binding.logical_region.name: binding.qec_region.name
            for binding in device.logical_to_qec
        }
        self.qec_selection = self._bind_qec_blocks()
        self.value_bindings = {}
        self._local_bindings = {}
        for binding in self.binding_by_name.values():
            if binding.binding_kind == "local":
                self._local_bindings.setdefault(binding.space,
                                                []).append(binding)
        self._local_binding_offsets = {
            space: 0 for space in self._local_bindings
        }
        self.selected_sites: dict[str, _SelectedSite] = {}
        self.generated: dict[tuple[Any, ...], tuple[Any, DefinitionHandle]] = {}
        self.placement_generated: dict[tuple[Any, ...],
                                       tuple[Any, DefinitionHandle,
                                             DefinitionHandle]] = {}
        self.value_map = {}
        self._concrete_resource_events = {}
        self.block_state = {}
        self.open_packed_preparation_blocks = set()
        self.packed_readout_preserves: dict[str, bool] = {}
        self.standard_instruments = {}
        self._selection_prepared = False

    @staticmethod
    def _single_use(value, expected: str, *, what: str):
        uses = tuple(value.uses)
        if len(uses) != 1:
            raise ValueError(f"{what} must have exactly one linear consumer")
        owner = getattr(uses[0].owner, "operation", uses[0].owner)
        if owner.name != expected:
            raise ValueError(
                f"{what} must flow directly to {expected}, not {owner.name}")
        return owner

    def _concrete_resource_supply(self, request) -> _ConcreteResourceSupply:
        """Authenticate the selected stream, route, and terminal CCZ use."""

        kind = _symbol(request.attributes["kind"])
        stream_name = _symbol(request.attributes["stream"])
        streams = tuple(
            stream for stream in self.device.logical.streams
            if stream.name == stream_name and stream.produces.name == kind)
        if len(streams) != 1:
            raise ValueError(
                "concrete resource request must select one exact device stream")
        stream = streams[0]
        if stream.region is None or stream.external:
            raise ValueError(
                "CCZ_STATE supply requires one concrete backed producer region")
        if stream.produced_by is None or stream.transfer is None:
            raise ValueError(
                "CCZ_STATE supply requires typed producer and transfer protocols"
            )

        awaited = self._single_use(request.result,
                                   "lvm.event_await",
                                   what="CCZ_STATE request event")
        consume = self._single_use(
            awaited.result,
            "lvm.consume_resource",
            what="awaited CCZ_STATE payload",
        )
        if _symbol(consume.attributes["resource_kind"]) != kind:
            raise ValueError(
                "CCZ_STATE consumer kind does not match its request")
        if _symbol(consume.attributes["resource_stream"]) != stream.name:
            raise ValueError("CCZ_STATE consumer references a foreign stream")
        selected = self.selected_sites.get(
            f"site{int(consume.attributes['site'])}")
        if selected is None:
            raise ValueError(
                "CCZ_STATE consumer has no selected P2 action site")
        placements = selected.handle.placements
        destination_spaces = {
            self.binding_by_name[placement].space for placement in placements
        }
        if len(destination_spaces) != 1:
            raise NotImplementedError(
                "cross-region CCZ_STATE delivery requires an explicit bridge; "
                "the concrete route currently supports one destination space")
        destination = next(iter(destination_spaces))

        logical_routes = tuple(
            channel for channel in self.device.logical._channels
            if getattr(channel.source, "name", None) == stream.name and
            getattr(channel.destination, "name", None) == destination)
        if len(logical_routes) != 1:
            raise ValueError(
                "CCZ_STATE supply requires one available logical delivery route"
            )
        logical_route = logical_routes[0]
        retained_channel = self.transaction.find_symbol(logical_route.name,
                                                        "lvm.channel")
        if retained_channel is None:
            raise ValueError("P1 dropped the selected CCZ_STATE delivery route")
        if (_symbol(retained_channel.attributes["from"]) != stream.name or
                _symbol(retained_channel.attributes["to"]) != destination):
            raise ValueError("P1 CCZ_STATE delivery route endpoints changed")

        raise NotImplementedError(
            "cross-region QEC resource delivery is not supported by CUDA-Q Logical"
        )

    def _bind_retained_stream_protocols(self, linked) -> None:
        """Reconnect P1's exact stream protocol closure after build replay."""

        linked_protocols = tuple(
            value for value in linked if isinstance(value, ProtocolDefinition))
        for stream in self.device.logical.streams:
            operation = self.transaction.find_symbol(stream.name, "lvm.stream")
            if operation is None:
                raise ValueError(
                    f"P1 is missing retained logical stream @{stream.name}")
            actual_kind = _symbol(operation.attributes["produces"])
            if actual_kind != stream.produces.name:
                raise ValueError(
                    f"P1 logical stream @{stream.name} resource identity "
                    f"changed from @{stream.produces.name} to @{actual_kind}")
            actual_external = "external" in operation.attributes
            if actual_external != stream.external:
                raise ValueError(
                    f"P1 logical stream @{stream.name} external provenance changed"
                )
            expected_backing = (None if stream.region is None else
                                stream.region.name)
            actual_backing = (_symbol(operation.attributes["backing_region"])
                              if "backing_region" in operation.attributes else
                              None)
            if actual_backing != expected_backing:
                raise ValueError(
                    f"P1 logical stream @{stream.name} backing identity changed "
                    f"from @{expected_backing} to @{actual_backing}")
            for field, definition in (
                ("produced_by", stream.produced_by),
                ("transfer", stream.transfer),
            ):
                if definition is None:
                    if field in operation.attributes:
                        raise ValueError(
                            f"P1 logical stream @{stream.name} gained unexpected "
                            f"{field} provenance")
                    continue
                if not isinstance(definition, ProtocolDefinition):
                    raise TypeError(
                        f"logical stream {field} provenance must be a typed protocol"
                    )
                expected_role = "produce" if field == "produced_by" else "transport"
                objective = definition.implements
                is_resource_flow = (isinstance(objective, ResourceFlowRef) and
                                    objective.kind == expected_role and
                                    objective.resource == stream.produces)
                is_consumer = (field == "transfer" and
                               isinstance(objective, LogicalActionRef) and
                               objective == stream.produces.consume_action)
                if not is_resource_flow and not is_consumer:
                    raise ValueError(
                        f"logical stream {field} provenance must implement "
                        f"the exact {expected_role} or resource-consumer "
                        f"objective for {stream.produces.name}")
                inputs = definition._resource_input_kinds()
                outputs = definition._resource_output_kinds()
                expected_inputs = (() if field == "produced_by" else
                                   (stream.produces,))
                expected_outputs = (() if is_consumer else (stream.produces,))
                if inputs != expected_inputs or outputs != expected_outputs:
                    raise ValueError(
                        f"logical stream {field} provenance has the wrong exact "
                        f"resource boundary for {stream.produces.name}")
                if field not in operation.attributes:
                    raise ValueError(
                        f"P1 logical stream @{stream.name} dropped {field} provenance"
                    )
                digest_field = f"{field}_sha256"
                expected_digest = _protocol_definition_sha256(definition)
                actual_digest = (_text(operation.attributes[digest_field]) if
                                 digest_field in operation.attributes else None)
                if actual_digest != expected_digest:
                    raise ValueError(
                        f"P1 logical stream @{stream.name} {field} bound "
                        "definition changed from the device-selected protocol")
                symbol = _symbol(operation.attributes[field])
                if symbol != definition.name:
                    raise ValueError(
                        f"P1 logical stream @{stream.name} {field} identity changed "
                        f"from @{definition.name} to @{symbol}")
                retained = self.transaction.find_symbol(symbol,
                                                        "fabric.protocol")
                retained_was_present = retained is not None
                if retained is None:
                    handle = self.transaction.materialize(definition)
                    if handle.symbol != symbol:
                        raise ValueError(
                            f"P1 logical stream @{stream.name} {field} identity "
                            f"@{symbol} collides with another definition")
                    retained = self.transaction.find_symbol(
                        symbol, "fabric.protocol")
                    if retained is None:
                        raise ValueError(
                            f"P2 failed to materialize retained protocol @{symbol}"
                        )
                else:
                    self.transaction.bind_existing_protocol(definition, symbol)
                expected_payload = _protocol_definition_payload(definition)
                retained_payload = (_protocol_operation_payload(
                    retained, symbol) if retained_was_present else
                                    expected_payload)
                if retained_was_present and retained_payload != expected_payload:
                    raise ValueError(
                        f"P1 logical stream @{stream.name} {field} bound "
                        "definition changed from the device-selected protocol")
                owners = {id(definition): definition}
                for value in linked_protocols:
                    if (value.implements != definition.implements or
                            value._resource_input_kinds()
                            != definition._resource_input_kinds() or
                            value._resource_output_kinds()
                            != definition._resource_output_kinds()):
                        continue
                    if _protocol_definition_payload(value) == retained_payload:
                        owners[id(value)] = value
                for owner in owners.values():
                    self.transaction.bind_existing_protocol(owner, symbol)

    def _bind_qec_blocks(self) -> QECSelectionWitness:
        requests = self.policy.get("qec_blocks", ())
        if isinstance(requests, QECBlockRequest):
            requests = (requests,)
        else:
            requests = tuple(requests or ())
        if any(not isinstance(item, QECBlockRequest) for item in requests):
            raise TypeError(
                "P2 policy 'qec_blocks' requires cudaq.logical.qec_block(...) values"
            )

        by_source = {
            (binding.source_allocation, binding.source_path): binding
            for binding in self.binding_by_name.values()
            if binding.source_allocation is not None
        }
        claimed = set()
        blocks = []
        for block_index, request in enumerate(requests):
            selected = []
            selected_placements = set()
            for reference in request.values:
                if reference.program not in {
                        self.source.root.symbol,
                        self.source.placement.input_p0,
                }:
                    raise ValueError(
                        "qlx.qec_block references a value outside the P0-to-P1 "
                        "lineage; construct it from p0.values, p1.values, or a "
                        "policy callback")
                binding = by_source.get((reference.allocation, reference.path))
                if binding is None:
                    raise ValueError(
                        "qlx.qec_block references a value absent from the P1 witness"
                    )
                if (binding.placement in claimed or
                        binding.placement in selected_placements):
                    raise ValueError(
                        f"P1 owner {binding.placement!r} appears in several QEC blocks"
                    )
                if binding.binding_kind != "local":
                    raise NotImplementedError(
                        "one-block QEC packing currently requires local P1 owners"
                    )
                selected.append(binding)
                selected_placements.add(binding.placement)
            spaces = {binding.space for binding in selected}
            if len(spaces) != 1:
                raise ValueError(
                    "one QEC block requires all P1 owners to reside in one logical space"
                )
            space = next(iter(spaces))
            device_encoding = _device_encodings(self.device).get(space)
            if device_encoding is not None:
                requested_handle = self.transaction.materialize(
                    request.encoding)
                device_handle = self.transaction.materialize(device_encoding)
                if requested_handle.symbol != device_handle.symbol:
                    raise ValueError(
                        f"P2 QEC block encoding {request.encoding.name!r} "
                        "conflicts with device encoding "
                        f"{device_encoding.name!r} for space {space!r}; "
                        "their semantic declarations differ")
            symbol = f"qec_block{block_index}"
            owners = []
            for logical_index, binding in enumerate(selected):
                claimed.add(binding.placement)
                realized = _P2Binding(
                    placement=binding.placement,
                    space=binding.space,
                    slot=binding.slot,
                    source_allocation=binding.source_allocation,
                    source_group=binding.source_group,
                    source_path=binding.source_path,
                    binding_kind=binding.binding_kind,
                    binding_data=binding.binding_data,
                    block=symbol,
                    logical_index=logical_index,
                    encoding=request.encoding,
                )
                self.binding_by_name[binding.placement] = realized
                owners.append(
                    QECBlockOwner(
                        placement=binding.placement,
                        logical_index=logical_index,
                        source_allocation=binding.source_allocation,
                        source_group=binding.source_group,
                        source_path=binding.source_path,
                    ))
            blocks.append(
                QECBlockBinding(
                    block=symbol,
                    space=space,
                    code=request.encoding.code.name,
                    encoding=request.encoding.name,
                    logical_capacity=request.encoding.code.k,
                    owners=tuple(owners),
                ))

        architecture_bindings = {
            binding.logical_region.name: binding
            for binding in self.device.logical_to_qec
            if binding.architecture is not None
        }
        for space, device_binding in sorted(architecture_bindings.items()):
            if device_binding.packing != "dense":
                raise NotImplementedError(
                    f"QEC architecture {device_binding.architecture.name!r} "
                    f"uses unsupported packing {device_binding.packing!r}")
            encoding = device_binding.qec_region.encoding
            logical_capacity = encoding.code.k
            local = sorted(
                (binding for binding in self.binding_by_name.values()
                 if binding.space == space and binding.binding_kind == "local"),
                key=lambda binding: (binding.slot, binding.placement),
            )
            groups: dict[int, list[_P2Binding]] = {}
            for binding in local:
                if binding.slot < 0:
                    raise ValueError(
                        f"P1 owner {binding.placement!r} has an invalid negative slot"
                    )
                block_index = binding.slot // logical_capacity
                if block_index >= device_binding.qec_region.block_capacity:
                    raise ValueError(
                        f"P1 slot {binding.slot} for owner {binding.placement!r} "
                        f"overflows QEC region @{device_binding.qec_region.name}"
                    )
                groups.setdefault(block_index, []).append(binding)

            for block_index, owners_for_block in sorted(groups.items()):
                group_names = {
                    binding.placement for binding in owners_for_block
                }
                claimed_in_group = group_names & claimed
                if claimed_in_group:
                    if claimed_in_group != group_names:
                        raise ValueError(
                            "explicit qec_blocks policy partially overrides "
                            f"architecture dense block {block_index} in space "
                            f"{space!r}")
                    for binding in owners_for_block:
                        realized = self.binding_by_name[binding.placement]
                        expected_port = binding.slot % logical_capacity
                        if (realized.logical_index != expected_port or
                                not _same_encoding(self.transaction,
                                                   realized.encoding,
                                                   encoding)):
                            raise ValueError(
                                "explicit qec_blocks policy conflicts with "
                                "architecture dense mapping for owner "
                                f"{binding.placement!r}")
                    continue

                symbol = f"qec_{space}_block{block_index}"
                owners = []
                for binding in owners_for_block:
                    logical_index = binding.slot % logical_capacity
                    claimed.add(binding.placement)
                    self.binding_by_name[binding.placement] = _P2Binding(
                        placement=binding.placement,
                        space=binding.space,
                        slot=binding.slot,
                        source_allocation=binding.source_allocation,
                        source_group=binding.source_group,
                        source_path=binding.source_path,
                        binding_kind=binding.binding_kind,
                        binding_data=binding.binding_data,
                        block=symbol,
                        logical_index=logical_index,
                        encoding=encoding,
                    )
                    owners.append(
                        QECBlockOwner(
                            placement=binding.placement,
                            logical_index=logical_index,
                            source_allocation=binding.source_allocation,
                            source_group=binding.source_group,
                            source_path=binding.source_path,
                        ))
                blocks.append(
                    QECBlockBinding(
                        block=symbol,
                        space=space,
                        code=encoding.code.name,
                        encoding=encoding.name,
                        logical_capacity=logical_capacity,
                        owners=tuple(owners),
                    ))
        return QECSelectionWitness(self.source.root.symbol, tuple(blocks))

    def _architecture_allows(self, candidate, placements) -> bool:
        candidate_id = id(candidate)
        resolved = False
        for placement in placements:
            binding = self.binding_by_name.get(placement)
            if binding is None:
                continue
            resolved = True
            definitions = self.architecture_definitions_by_space.get(
                binding.space)
            if definitions is None:
                if candidate_id not in self.ordinary_definition_ids:
                    return False
            elif candidate_id not in definitions:
                return False
        if resolved:
            return True
        return candidate_id in self.ordinary_definition_ids

    def _prepare_selection(self):
        if self._selection_prepared:
            return
        unsupported_bindings = tuple(
            binding for binding in self.source.placement.bindings
            if binding.binding_kind != "local")
        if unsupported_bindings:
            kinds = sorted(
                {binding.binding_kind for binding in unsupported_bindings})
            raise NotImplementedError("P1-to-P2 excludes nonlocal placement "
                                      f"descriptors: {kinds!r}")
        self._index_value_bindings()
        self._validate_packed_root_boundaries()
        self._validate_packed_preparations()
        self._analyze_packed_lifetimes()
        self._select_sites()
        has_quantum = any(
            str(value.type).startswith("!lvm.logical_qubit<")
            for operation in self._walk_operations(
                self.kernel.regions[0].blocks[0])
            for value in (*operation.operands, *operation.results)) or any(
                str(argument.type).startswith("!lvm.logical_qubit<")
                for argument in self.kernel.regions[0].blocks[0].arguments)
        resource_only = not self.selected_sites and not has_quantum
        if resource_only:
            self.code = self.encoding = None
            self.code_handle = self.encoding_handle = None
            self.epoch_symbol = None
        elif not self.selected_sites:
            encodings = {
                binding.encoding
                for binding in self.binding_by_name.values()
                if binding.encoding is not None
            }
            if len(encodings) != 1:
                raise ValueError(
                    "P1-to-P2 found neither a selected QEC realization nor one "
                    "explicit P2 QEC-block encoding")
            self.encoding = next(iter(encodings))
            self.code = self.encoding.code
            self.code_handle = self.transaction.materialize(self.code)
            self.encoding_handle = self.transaction.materialize(self.encoding)
            self.epoch_symbol = self.transaction.materialize(
                self.encoding.initial_epoch).symbol
        else:
            candidates = [
                item.selected for item in self.selected_sites.values()
            ]
            encodings = {
                id(_candidate_code(item)[1]): _candidate_code(item)
                for item in candidates
            }
            if len(encodings) != 1:
                raise NotImplementedError(
                    "this CUDA-Q Logical slice requires one selected encoding across the root "
                    "protocol; heterogeneous encoding transitions must be explicit"
                )
            self.code, self.encoding = next(iter(encodings.values()))
            self.code_handle = self.transaction.materialize(self.code)
            self.encoding_handle = self.transaction.materialize(self.encoding)
            self.epoch_symbol = self.transaction.materialize(
                self.encoding.initial_epoch).symbol
        if not resource_only and self.encoding is not None:
            self._complete_standard_instrument_sites()
        self.patch_type = (None if resource_only else mlir_ir.Type.parse(
            f"!fabric.patch<@{self.code_handle.symbol}, "
            f"@{self.encoding_handle.symbol}, @{self.epoch_symbol}>",
            context=self.context,
        ))
        # Placement already retained the exact immutable device stack.  Reuse
        # that definition instead of importing a second renamed copy whose
        # nested QEC bindings would describe the same architecture
        # under conflicting symbol identities.
        retained_device = self.transaction.find_symbol(self.device.name,
                                                       "qlx.device")
        self.device_handle = (DefinitionHandle(
            self.device.name, "device", "device") if retained_device is not None
                              else self.transaction.materialize(self.device))
        self._selection_prepared = True

    def _finalize_qec_selection(self, *, action_filter=None):
        self._prepare_selection()
        blocks = list(self.qec_selection.blocks)
        claimed = {
            owner.placement for block in blocks for owner in block.owners
        }
        if self.code is not None and self.encoding is not None:
            for binding in self.binding_by_name.values():
                if binding.placement in claimed:
                    continue
                blocks.append(
                    QECBlockBinding(
                        block=f"qec_owner_{binding.placement}",
                        space=binding.space,
                        code=self.code.name,
                        encoding=self.encoding.name,
                        logical_capacity=self.code.k,
                        owners=(QECBlockOwner(
                            placement=binding.placement,
                            logical_index=0,
                            source_allocation=binding.source_allocation,
                            source_group=binding.source_group,
                            source_path=binding.source_path,
                        ),),
                    ))
        selected_actions = tuple(item for item in sorted(
            self.selected_sites.values(),
            key=lambda item: item.handle.symbol,
        ) if action_filter is None or action_filter(item))
        return QECSelectionWitness(
            input_p1=self.qec_selection.input_p1,
            blocks=tuple(blocks),
            actions=tuple(
                QECActionSelection(
                    site=item.handle.symbol,
                    kind=item.handle.kind,
                    objective=str(item.handle.objective),
                    placements=tuple(item.handle.placements),
                    feasible_candidates=tuple(candidate.name
                                              for candidate in item.feasible),
                    selected=item.selected.name,
                    provider=getattr(item.selected, "plugin", "fixed"),
                    version=getattr(item.selected, "version", "linked"),
                    manifest_sha256=getattr(
                        item.selected,
                        "manifest_sha256",
                        None,
                    ),
                )
                for item in selected_actions),
            code=None if self.code is None else self.code.name,
            encoding=None if self.encoding is None else self.encoding.name,
            objective=self.qec_selection.objective,
            tie_break=self.qec_selection.tie_break,
        )

    def run(self):
        self._prepare_selection()
        self._emit_protocol()
        self.transaction.add_profile("p2n")
        self.qec_selection = self._finalize_qec_selection()
        metadata = self.protocol.attributes["metadata"]
        metadata_values = {
            "input_p1": _text(metadata["input_p1"]),
            "device": _text(metadata["device"]),
            "qec_selection_sha256": _qec_selection_sha256(self.qec_selection),
        }
        self.protocol.attributes["metadata"] = _dictionary(
            self.context,
            metadata_values,
        )
        return self.module, self.root_symbol

    def _convert_sequence(self, operations):
        for operation in operations:
            self._convert(operation)

    def _standard_instrument(self, handle, encoding):
        """Return the encoding-derived realization for a standard instrument."""

        from .. import gadgets

        objective = handle.objective
        if objective in {"prepare_zero", "prepare_plus"}:
            key = (objective, id(encoding))
            candidate = self.standard_instruments.get(key)
            if candidate is None:
                candidate = getattr(gadgets, objective)(encoding)
                self.standard_instruments[key] = candidate
            return candidate
        if objective in {"x", "z"}:
            if len(handle.placements) != 1:
                raise ValueError(
                    f"standard logical {objective.upper()} requires one owner")
            binding = self.binding_by_name.get(handle.placements[0])
            logical = (binding.logical_index if binding is not None and
                       binding.logical_index is not None else 0)
            key = (objective, id(encoding), logical)
            candidate = self.standard_instruments.get(key)
            if candidate is None:
                candidate = gadgets.logical_pauli(
                    encoding,
                    basis=objective,
                    logical=logical,
                )
                self.standard_instruments[key] = candidate
            return candidate
        if objective not in {"measure_x", "measure_z"}:
            return None
        if len(handle.placements) != 1:
            raise ValueError(
                f"standard {objective} requires exactly one placed logical value"
            )
        binding = self.binding_by_name.get(handle.placements[0])
        logical = (binding.logical_index if binding is not None and
                   binding.logical_index is not None else 0)
        shared_owners = (sum(
            candidate.block == binding.block
            for candidate in self.binding_by_name.values()
            if candidate.block is not None) if binding is not None and
                         binding.block is not None else 0)
        preserve_block = shared_owners > 1
        key = (objective, id(encoding), logical, preserve_block)
        candidate = self.standard_instruments.get(key)
        if candidate is None:
            candidate = getattr(gadgets, objective)(
                encoding,
                logical=logical,
                preserve_block=preserve_block,
            )
            self.standard_instruments[key] = candidate
        return candidate

    def _complete_standard_instrument_sites(self):
        """Close standard preparation/readout sites for the chosen encoding.

        Author- or device-provided candidates are selected first. If another
        action establishes the root encoding, this fills any remaining common
        preparation and destructive X/Z readout sites with inspectable gadgets
        derived from that encoding. Packed readout preserves the shared carrier
        patch while consuming only the addressed P1 logical value.
        """

        block = self.kernel.regions[0].blocks[0]
        for operation in self._walk_operations(block):
            if operation.name not in {
                    "lvm.prepare", "lvm.apply", "lvm.measure"
            }:
                continue
            handle = self._site_handle(operation)
            if handle.symbol in self.selected_sites:
                continue
            if operation.name == "lvm.apply" and handle.objective not in {
                    "x", "z"
            }:
                continue
            candidate = self._standard_instrument(handle, self.encoding)
            if candidate is None:
                raise ValueError(
                    f"no code-specific P2 implementation for {handle.objective}"
                )
            self.selected_sites[handle.symbol] = _SelectedSite(
                operation, handle, (candidate,), candidate)

    def _site_handle(self, operation):
        name = operation.name
        if name == "lvm.prepare":
            kind = "instrument"
            objective = f"prepare_{_text(operation.attributes['state'])}"
            placements_attr = (operation.attributes["at"],)
        elif name == "lvm.apply":
            kind = "action"
            objective = _objective(operation.attributes["action"])
            placements_attr = operation.attributes["placements"]
        elif name == "lvm.instrument":
            kind = "instrument"
            objective = _objective(operation.attributes["instrument"])
            placements_attr = operation.attributes["placements"]
        elif name == "lvm.measure":
            kind = "instrument"
            objective = f"measure_{_basis(operation.attributes['basis'])}"
            placements_attr = (operation.attributes["at"],)
        elif name == "lvm.consume_resource":
            kind = "resource_action"
            objective = _objective(operation.attributes["action"])
            placements_attr = operation.attributes["placements"]
        else:
            raise ValueError(f"{name} is not a selectable P1 operation")
        fallback = tuple(_symbol(value) for value in placements_attr)
        quantum_inputs = tuple(
            value for value in operation.operands
            if str(value.type).startswith("!lvm.logical_qubit<"))
        if name == "lvm.prepare":
            binding = self.value_bindings.get(operation.result)
            placements = fallback if binding is None else (binding.placement,)
        else:
            placements = tuple((self.value_bindings[value].placement if value in
                                self.value_bindings else fallback[index])
                               for index, value in enumerate(quantum_inputs))
        params_attr = (operation.attributes["parameters"]
                       if "parameters" in operation.attributes else None)
        parameters = _parameters(params_attr)
        if name == "lvm.measure":
            parameters["basis"] = _basis(operation.attributes["basis"])
        objective_family = _objective_family(kind, objective)
        if objective_family == "pauli_product_rotation":
            angle_operand = operation.operands[-1]
            owner = angle_operand.owner
            if owner.name == "arith.constant" and "value" in owner.attributes:
                parameters["angle"] = float(
                    getattr(
                        owner.attributes["value"],
                        "value",
                        owner.attributes["value"],
                    ))
            else:
                parameters["dynamic_angle"] = True
        resource_kind = None
        resource_stream = None
        resource_stream_owner = None
        if name == "lvm.consume_resource":
            if ("resource_kind" not in operation.attributes or
                    "resource_stream" not in operation.attributes):
                raise ValueError(
                    "resource action site has no typed resource provenance")
            resource_kind = _symbol(operation.attributes["resource_kind"])
            stream_reference = operation.attributes["resource_stream"]
            resource_stream_owner = _root_symbol(stream_reference)
            resource_stream = _symbol(stream_reference)
        if name == "lvm.instrument" and "channel" in operation.attributes:
            raise ValueError(
                "communication-qualified P1 sites are not supported by CUDA-Q Logical"
            )
        return ActionSiteHandle(
            symbol=f"site{int(operation.attributes['site'])}",
            kind=kind,
            objective_family=objective_family,
            objective=objective,
            placements=placements,
            parameters=parameters,
            input_arity=len(operation.operands),
            result_arity=len(operation.results),
            resource_kind=resource_kind,
            resource_stream=resource_stream,
            resource_stream_owner=resource_stream_owner,
        )

    def _canonical_stream(self, reference, kind):
        """Resolve a device stream only after matching retained P1 truth."""

        name = _symbol(reference)
        operation = self.transaction.find_symbol(name, "lvm.stream")
        if operation is None:
            raise ValueError(
                f"resource request references unknown canonical P1 stream @{name}"
            )
        candidates = tuple(stream for stream in self.device.logical.streams
                           if stream.name == name)
        if len(candidates) != 1:
            detail = "no" if not candidates else "several"
            raise ValueError(
                f"P2 device has {detail} stream declaration(s) named @{name}")
        stream = candidates[0]
        produced_kind = getattr(stream.produces, "name", str(stream.produces))
        if (produced_kind != kind or
                _symbol(operation.attributes["produces"]) != kind):
            raise ValueError(
                f"P1 stream @{name} does not canonically produce {kind!r}")
        expected_region = None if stream.region is None else stream.region.name
        actual_region = (_symbol(operation.attributes["backing_region"])
                         if "backing_region" in operation.attributes else None)
        if actual_region != expected_region:
            raise ValueError(f"P1 stream @{name} region is not canonical")
        actual_external = "external" in operation.attributes
        if actual_external != stream.external:
            raise ValueError(
                f"P1 stream @{name} external boundary is not canonical")
        return operation, stream

    @staticmethod
    def _is_logical_value(value):
        return str(value.type).startswith("!lvm.logical_qubit<")

    def _binding_for_reference(self, reference):
        name = _symbol(reference)
        binding = self.binding_by_name.get(name)
        if binding is not None:
            return binding
        choices = self._local_bindings.get(name, ())
        offset = self._local_binding_offsets.get(name, 0)
        if offset >= len(choices):
            return None
        self._local_binding_offsets[name] = offset + 1
        return choices[offset]

    def _seed_block_arguments(self, block, values):
        for argument, source in zip(block.arguments, values):
            if self._is_logical_value(
                    argument) and source in self.value_bindings:
                self.value_bindings[argument] = self.value_bindings[source]

    def _bind_results_from_values(self, results, values):
        for result, source in zip(results, values):
            if self._is_logical_value(result) and source in self.value_bindings:
                self.value_bindings[result] = self.value_bindings[source]

    def _index_binding_block(self, block, initial_values=()):
        if initial_values:
            self._seed_block_arguments(block, initial_values)
        for child in block.operations:
            operation = child.operation
            name = operation.name
            if name == "lvm.prepare":
                binding = self._binding_for_reference(
                    operation.attributes["at"])
                if binding is not None:
                    self.value_bindings[operation.result] = binding
                continue
            if name == "lvm.if":
                branch_values = []
                for region in operation.regions:
                    nested = region.blocks[0]
                    self._index_binding_block(nested)
                    branch_values.append(tuple(nested.operations[-1].operands))
                for index, result in enumerate(operation.results):
                    if not self._is_logical_value(result):
                        continue
                    candidates = [
                        self.value_bindings.get(values[index])
                        for values in branch_values
                    ]
                    if candidates and all(
                            item == candidates[0] for item in candidates):
                        if candidates[0] is not None:
                            self.value_bindings[result] = candidates[0]
                continue
            if name == "lvm.repeat":
                body = operation.regions[0].blocks[0]
                self._index_binding_block(body, operation.operands)
                self._bind_results_from_values(operation.results,
                                               body.operations[-1].operands)
                continue
            if name == "lvm.while":
                before = operation.regions[0].blocks[0]
                self._index_binding_block(before, operation.operands)
                forwarded = tuple(before.operations[-1].operands)[1:]
                after = operation.regions[1].blocks[0]
                self._index_binding_block(after, forwarded)
                self._bind_results_from_values(operation.results,
                                               after.operations[-1].operands)
                continue
            if name == "lvm.event_try_take":
                branch_values = []
                for region in operation.regions:
                    nested = region.blocks[0]
                    self._index_binding_block(nested, operation.operands)
                    branch_values.append(tuple(nested.operations[-1].operands))
                for index, result in enumerate(operation.results):
                    if not self._is_logical_value(result):
                        continue
                    candidates = [
                        self.value_bindings.get(values[index])
                        for values in branch_values
                    ]
                    if candidates and all(
                            item == candidates[0] for item in candidates):
                        if candidates[0] is not None:
                            self.value_bindings[result] = candidates[0]
                continue
            quantum_inputs = tuple(value for value in operation.operands
                                   if self._is_logical_value(value))
            quantum_results = tuple(value for value in operation.results
                                    if self._is_logical_value(value))
            self._bind_results_from_values(quantum_results, quantum_inputs)
            for region in operation.regions:
                for nested in region.blocks:
                    self._index_binding_block(nested)

    def _index_value_bindings(self):
        block = self.kernel.regions[0].blocks[0]
        for argument in block.arguments:
            if not self._is_logical_value(argument):
                continue
            reference = self._placement_from_type(argument.type)
            binding = self._binding_for_reference(reference)
            if binding is not None:
                self.value_bindings[argument] = binding
        self._index_binding_block(block)

    def _validate_packed_preparations(self):
        """Fail closed until packed blocks have a composite preparation gadget."""

        owners_by_block = {}
        for binding in self.binding_by_name.values():
            if binding.block is not None:
                owners_by_block.setdefault(binding.block,
                                           set()).add(binding.placement)
        states_by_block = {}
        prepared_by_block = {}
        for operation in self._walk_operations(
                self.kernel.regions[0].blocks[0]):
            if operation.name != "lvm.prepare":
                continue
            binding = self.value_bindings.get(operation.result)
            if (binding is None or binding.block is None or
                    len(owners_by_block.get(binding.block, ())) <= 1):
                continue
            states_by_block.setdefault(binding.block, set()).add(
                _text(operation.attributes["state"]))
            prepared_by_block.setdefault(binding.block,
                                         set()).add(binding.placement)
        for block, states in sorted(states_by_block.items()):
            owners = owners_by_block[block]
            prepared = prepared_by_block[block]
            if prepared != owners:
                already_live = ", ".join(sorted(owners - prepared))
                raise NotImplementedError(
                    f"packed QEC block @{block} cannot mix whole-block "
                    "preparation with already-live logical owners "
                    f"({already_live}); use an unpacked realization")
            if len(states) > 1:
                requested = ", ".join(sorted(states))
                raise NotImplementedError(
                    f"packed QEC block @{block} cannot yet prepare heterogeneous "
                    f"logical states ({requested}); use homogeneous preparation "
                    "or an unpacked realization")

    def _analyze_packed_lifetimes(self):
        """Derive packed readout liveness and reject premature disposal."""

        owners_by_block = {}
        for binding in self.binding_by_name.values():
            if binding.block is not None:
                owners_by_block.setdefault(binding.block,
                                           set()).add(binding.placement)
        packed_blocks = dict(owners_by_block)
        shared_blocks = {
            block: owners
            for block, owners in packed_blocks.items()
            if len(owners) > 1
        }
        if not packed_blocks:
            return

        def packed_owners(values):
            owners = set()
            for value in values:
                binding = self.value_bindings.get(value)
                if binding is not None and binding.block in packed_blocks:
                    owners.add(binding.placement)
            return owners

        def validate_block(block, live):
            for child in block.operations:
                operation = child.operation
                if operation.name == "lvm.measure":
                    binding = self.value_bindings.get(operation.operands[0])
                    if binding is not None and binding.block in packed_blocks:
                        siblings = ((live & packed_blocks[binding.block]) -
                                    {binding.placement})
                        self.packed_readout_preserves[
                            f"site{int(operation.attributes['site'])}"] = bool(
                                siblings)
                if operation.name == "lvm.discard":
                    discarded_by_block = {}
                    for value in operation.operands:
                        binding = self.value_bindings.get(value)
                        if (binding is None or
                                binding.block not in shared_blocks):
                            continue
                        discarded_by_block.setdefault(binding.block, set()).add(
                            binding.placement)
                    for block_name, discarded in sorted(
                            discarded_by_block.items()):
                        remaining = ((live & shared_blocks[block_name]) -
                                     discarded)
                        if remaining:
                            names = ", ".join(sorted(remaining))
                            raise NotImplementedError(
                                f"packed QEC block @{block_name} cannot discard "
                                "a strict subset while logical owners "
                                f"({names}) remain live; discard the complete "
                                "block at one site or use an unpacked realization"
                            )
                for region in operation.regions:
                    for nested in region.blocks:
                        validate_block(nested, set(live))
                live.difference_update(packed_owners(operation.operands))
                live.update(packed_owners(operation.results))

        root = self.kernel.regions[0].blocks[0]
        validate_block(root, packed_owners(root.arguments))

    def _architecture_bound(self, placements) -> bool:
        return any(binding is not None and
                   binding.space in self.architecture_definitions_by_space
                   for binding in (self.binding_by_name.get(placement)
                                   for placement in placements))

    @staticmethod
    def _mpp_readout_handle(handle, *, objective):
        basis = handle.parameters.get("basis")
        if basis not in {"x", "z"}:
            raise ValueError(
                "architecture-derived logical readout requires X or Z basis")
        return ActionSiteHandle(
            symbol=handle.symbol,
            kind=handle.kind,
            objective_family="pauli_product_measurement",
            objective=objective,
            placements=handle.placements,
            parameters={
                "x_mask": 1 if basis == "x" else 0,
                "z_mask": 1 if basis == "z" else 0,
                "sign": 1,
            },
            input_arity=handle.input_arity,
            result_arity=handle.result_arity,
        )

    def _validate_packed_root_boundaries(self):
        """Reject root ABIs that cannot represent one shared encoded block."""

        source_block = self.kernel.regions[0].blocks[0]
        return_values = tuple(value for child in source_block.operations
                              if child.operation.name == "lvm.return"
                              for value in child.operation.operands)
        for boundary, values in (("input", tuple(source_block.arguments)),
                                 ("result", return_values)):
            owners_by_block = {}
            for value in values:
                binding = self.value_bindings.get(value)
                if binding is None or binding.block is None:
                    continue
                owners_by_block.setdefault(binding.block,
                                           set()).add(binding.placement)
            for block, owners in sorted(owners_by_block.items()):
                if len(owners) <= 1:
                    continue
                owner_names = ", ".join(sorted(owners))
                raise NotImplementedError(
                    f"packed QEC block @{block} cannot yet represent several "
                    f"root {boundary} owners ({owner_names}); use an unpacked "
                    "realization until the root protocol ABI supports one "
                    "patch per witnessed block")

    def _select_sites(self):
        block = self.kernel.regions[0].blocks[0]
        for operation in self._walk_operations(block):
            if operation.name not in {
                    "lvm.prepare",
                    "lvm.apply",
                    "lvm.instrument",
                    "lvm.measure",
                    "lvm.consume_resource",
            }:
                continue
            if "site" not in operation.attributes:
                raise ValueError(
                    f"{operation.name} is missing its inline site ordinal")
            handle = self._site_handle(operation)
            preserve_packed_owner = (
                handle.objective in {"measure_x", "measure_z"} and
                self.packed_readout_preserves.get(handle.symbol, False))
            generated = tuple(
                lowering for lowering in self.lowerings
                if self._architecture_allows(lowering, handle.placements) and
                _matches_objective(lowering, handle) and
                _matches_device_encoding(
                    self.transaction,
                    lowering,
                    handle,
                    self.device,
                    self.binding_by_name,
                ) and _requirements_met(
                    lowering,
                    handle.placements,
                    self.device,
                    self.binding_by_name,
                    site=handle,
                ))
            if preserve_packed_owner:
                generated = ()
            fixed = () if preserve_packed_owner else tuple(
                candidate for candidate in self.fixed
                if self._architecture_allows(candidate, handle.placements) and
                _matches_fixed(candidate, handle, operation, self.
                               binding_by_name) and _matches_device_encoding(
                                   self.transaction,
                                   candidate,
                                   handle,
                                   self.device,
                                   self.binding_by_name,
                               ) and _requirements_met(
                                   candidate,
                                   handle.placements,
                                   self.device,
                                   self.binding_by_name,
                                   site=handle,
                               ))
            if (not fixed and not generated and
                    handle.objective in {"measure_x", "measure_z"} and
                    self._architecture_bound(handle.placements)):
                selected_handle = self._mpp_readout_handle(
                    handle, objective=handle.objective)
                match_handle = self._mpp_readout_handle(handle, objective="mpp")
                generated = tuple(lowering for lowering in self.lowerings
                                  if self._architecture_allows(
                                      lowering, selected_handle.placements) and
                                  _matches_objective(lowering, match_handle) and
                                  _matches_device_encoding(
                                      self.transaction,
                                      lowering,
                                      selected_handle,
                                      self.device,
                                      self.binding_by_name,
                                  ) and _requirements_met(
                                      lowering,
                                      selected_handle.placements,
                                      self.device,
                                      self.binding_by_name,
                                      site=selected_handle,
                                  ))
                if not generated:
                    raise ValueError(
                        "architecture-bound logical readout requires a "
                        "compatible architecture-owned MPP QECLowering")
                handle = selected_handle
            if not fixed and not generated and handle.objective in {
                    "prepare_zero",
                    "prepare_plus",
                    "measure_x",
                    "measure_z",
            }:
                encoding = _device_encoding_for_site(
                    self.transaction,
                    self.device,
                    handle.placements,
                    self.binding_by_name,
                )
                if encoding is not None:
                    candidate = self._standard_instrument(handle, encoding)
                    fixed = (candidate,)
            # Fixed exact artifacts win by default; a policy can later compare
            # cost/evidence uniformly after dynamic specialization.
            feasible = (*fixed, *generated)
            if not feasible:
                continue
            selected = feasible[0]
            self.selected_sites[handle.symbol] = _SelectedSite(
                operation, handle, feasible, selected)

    def _walk_operations(self, block):
        """Yield action-bearing operations throughout a structured kernel body.

        P1 action sites are declarations on the surrounding ``lvm.domain``, but
        their uses may occur inside ``lvm.if`` and ``lvm.repeat`` regions.  QEC
        selection therefore follows structured regions instead of considering
        only the kernel's entry block.
        """

        for child in block.operations:
            operation = child.operation
            yield operation
            for region in operation.regions:
                for nested_block in region.blocks:
                    yield from self._walk_operations(nested_block)

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

    def _emit_protocol(self):
        source_block = self.kernel.regions[0].blocks[0]
        input_types = tuple(
            self._p2_type(argument.type) for argument in source_block.arguments)
        kernel_type = mlir_ir.TypeAttr(
            self.kernel.attributes["function_type"]).value
        result_types = tuple(
            self._p2_type(result) for result in kernel_type.results)
        function_type = mlir_ir.FunctionType.get(input_types,
                                                 result_types,
                                                 context=self.context)
        self.root_symbol = self.transaction.unique_symbol(
            f"{self.source.root.symbol}_qec")
        with self.context:
            function_type_attr = mlir_ir.TypeAttr.get(function_type)
        attrs = {
            "sym_name":
                mlir_ir.StringAttr.get(self.root_symbol, context=self.context),
            "function_type":
                function_type_attr,
            "metadata":
                _dictionary(
                    self.context,
                    {
                        "input_p1": self.source.root.symbol,
                        "device": self.device_handle.symbol,
                    },
                ),
        }
        if "specialization" in self.kernel.attributes:
            attrs["specialization"] = self.kernel.attributes["specialization"]
        with self.location:
            protocol = mlir_ir.Operation.create("fabric.protocol",
                                                attributes=attrs,
                                                regions=1,
                                                loc=self.location)
            self.module.body.append(protocol)
            block = protocol.regions[0].blocks.append(*input_types)
        self.protocol = protocol
        for old, new in zip(source_block.arguments, block.arguments):
            self.value_map[old] = new
        self.ip = mlir_ir.InsertionPoint(block)
        self._convert_sequence(
            child.operation for child in source_block.operations)

    def _mapped(self, value):
        binding = self._binding_for_value(value)
        if binding is not None and binding.block is not None:
            state = self.block_state.get(binding.block)
            if state is not None:
                self.open_packed_preparation_blocks.discard(binding.block)
                return state
        try:
            return self.value_map[value]
        except KeyError as error:
            raise NotImplementedError(
                f"P1 value {value} has no P2 mapping") from error

    @staticmethod
    def _placement_from_type(type_):
        text = str(type_)
        if not text.startswith("!lvm.logical_qubit<"):
            return None
        reference = text[len("!lvm.logical_qubit<"):-1]
        return reference.split("::@")[-1].lstrip("@")

    def _block_key(self, placement):
        binding = self.binding_by_name.get(placement)
        return binding.block if binding is not None and binding.block else placement

    def _binding_for_value(self, value):
        binding = self.value_bindings.get(value)
        if binding is not None:
            return binding
        placement = self._placement_from_type(value.type)
        return self.binding_by_name.get(
            placement) if placement is not None else None

    def _owner_key(self, value):
        binding = self._binding_for_value(value)
        if binding is not None:
            return binding.block or binding.placement
        return self._placement_from_type(value.type)

    def _coalesced_boundary(self, values):
        """Return one SSA edge per packed owner and each source's edge index."""

        representatives = []
        indices = []
        owner_indices = {}
        for value in values:
            owner = self._owner_key(value)
            if owner is None:
                index = len(representatives)
                representatives.append(value)
            else:
                index = owner_indices.get(owner)
                if index is None:
                    index = len(representatives)
                    owner_indices[owner] = index
                    representatives.append(value)
            indices.append(index)
        return tuple(representatives), tuple(indices)

    def _p2_type(self, type_):
        text = str(type_)
        if text.startswith("!lvm.logical_qubit<"):
            return self.patch_type
        resource_marker = '!lvm.logical_resource<"'
        if text.startswith(resource_marker):
            kind = text[len(resource_marker):].split('"', 1)[0]
            return mlir_ir.Type.parse(f"!fabric.resource<@{kind}>",
                                      context=self.context)
        event_marker = '!lvm.logical_event<!lvm.logical_resource<"'
        if text.startswith(event_marker):
            kind = text[len(event_marker):].split('"', 1)[0]
            return mlir_ir.Type.parse(
                f'!fabric.event<!fabric.resource<@{kind}>, "linear">',
                context=self.context,
            )
        if text.startswith("!lvm.logical_frame<"):
            domain = text.split('"', 2)[1]
            return mlir_ir.Type.parse(f"!fabric.frame<@{domain}>",
                                      context=self.context)
        return type_

    def _convert_region(self, source_region, target_region, *, inits=()):
        source_block = source_region.blocks[0]
        arguments, argument_indices = self._coalesced_boundary(
            source_block.arguments)
        argument_types = [
            self._p2_type(argument.type) for argument in arguments
        ]
        with self.location:
            target_block = target_region.blocks.append(*argument_types)
        for source, index in zip(source_block.arguments, argument_indices):
            target = target_block.arguments[index]
            self.value_map[source] = target
            binding = self._binding_for_value(source)
            if binding is not None and binding.block is not None:
                self.block_state[binding.block] = target
        saved_ip = self.ip
        self.ip = mlir_ir.InsertionPoint(target_block)
        try:
            for child in source_block.operations:
                operation = child.operation
                if operation.name == "lvm.yield":
                    yielded, _ = self._coalesced_boundary(operation.operands)
                    self._insert(
                        self.ip,
                        "fabric.yield",
                        operands=[self._mapped(value) for value in yielded],
                    )
                elif operation.name == "lvm.while_condition":
                    yielded, _ = self._coalesced_boundary(operation.operands)
                    self._insert(
                        self.ip,
                        "fabric.while_condition",
                        operands=[self._mapped(value) for value in yielded],
                    )
                else:
                    self._convert(operation)
        finally:
            self.ip = saved_ip

    def _placement(self, operation):
        if "at" in operation.attributes:
            return _symbol(operation.attributes["at"])
        values = operation.attributes["placements"]
        return _symbol(values[0])

    def _prepare(self, operation):
        fallback = self._placement(operation)
        binding = self.value_bindings.get(operation.result)
        placement = binding.placement if binding is not None else fallback
        block = binding.block if binding is not None and binding.block else placement
        packed = binding is not None and binding.block is not None
        if packed and block in self.block_state:
            if block not in self.open_packed_preparation_blocks:
                raise NotImplementedError(
                    f"packed QEC block @{block} cannot prepare another logical "
                    "owner after the shared block has become live; keep "
                    "homogeneous preparations at one initial frontier or use "
                    "an unpacked realization")
            self.value_map[operation.result] = self.block_state[block]
            return
        allocated = self._insert(
            self.ip,
            "fabric.alloc",
            results=[self.patch_type],
            attributes={
                "code":
                    mlir_ir.FlatSymbolRefAttr.get(self.code_handle.symbol,
                                                  context=self.context),
                "region":
                    mlir_ir.FlatSymbolRefAttr.get(
                        (self.qec_region_by_space.get(binding.space,
                                                      binding.space)
                         if binding is not None else fallback),
                        context=self.context,
                    ),
                "strict_region":
                    mlir_ir.UnitAttr.get(context=self.context),
            },
        )
        selected = self.selected_sites.get(
            f"site{int(operation.attributes['site'])}")
        if selected is None:
            state = _text(operation.attributes["state"])
            raise ValueError(
                f"no code-specific P2 implementation for prepare_{state}")
        generated, handle = self._materialize_selected(selected)
        inputs, results, _ = self.transaction.signature_of(generated)
        if len(inputs) != 1 or len(results) != 1:
            raise ValueError(
                "encoded preparation must consume one allocated patch and "
                "produce one prepared patch")
        prepared = self._insert(
            self.ip,
            "fabric.call",
            operands=[allocated.result],
            results=results,
            attributes={
                "callee":
                    mlir_ir.FlatSymbolRefAttr.get(handle.symbol,
                                                  context=self.context)
            },
        )
        self.value_map[operation.result] = prepared.result
        if packed:
            self.block_state[block] = prepared.result
            self.open_packed_preparation_blocks.add(block)

    def _constant_int(self, value):
        owner = value.owner
        if owner.name != "arith.constant":
            raise NotImplementedError(
                "dynamic logical idle rounds require folded P2 control")
        return int(
            getattr(owner.attributes["value"], "value",
                    owner.attributes["value"]))

    def _memory_realization(self, placements):
        """Select one linked memory realization for the root code, if any.

        The model binds an explicit ``qlx.idle`` workload to a selected
        memory protocol/round group at P2. When exactly one linked
        realization implements the standard ``idle`` objective for the
        placed code it is selected; with none the workload stays an
        abstract ``fabric.idle``; several matches are an explicit
        ambiguity rather than a silent winner.
        """
        key = tuple(sorted(placements))
        if not hasattr(self, "_memory_choices"):
            self._memory_choices = {}
        if key not in self._memory_choices:
            candidates = []
            for candidate in self.fixed:
                if not self._architecture_allows(candidate, placements):
                    continue
                implements = candidate.implements
                if getattr(implements, "name", None) != "idle":
                    continue
                try:
                    code, _encoding = _candidate_code(candidate)
                except ValueError:
                    continue
                if self.code is not None and code.name != self.code.name:
                    continue
                candidates.append(candidate)
            if len(candidates) > 1:
                names = ", ".join(sorted(item.name for item in candidates))
                raise ValueError(
                    "several linked memory realizations implement idle for "
                    f"code {self.code.name if self.code else '?'}; select one "
                    f"explicitly: {names}")
            self._memory_choices[key] = candidates[0] if candidates else None
        return self._memory_choices[key]

    def _emit_memory_realization(self, value, realization, rounds):
        """Emit the selected memory realization for one idle workload.

        One iteration is one call; several stay folded as ``fabric.repeat``
        so estimate cost never scales with the round count.
        """
        handle = self.transaction.materialize(realization)

        def emit_call(ip, operand):
            call = self._insert(
                ip,
                "fabric.call",
                operands=[operand],
                results=[self.patch_type],
                attributes={
                    "callee":
                        mlir_ir.FlatSymbolRefAttr.get(handle.symbol,
                                                      context=self.context)
                },
            )
            return call.result

        if rounds == 1:
            return emit_call(self.ip, value)
        repeat = self._insert(
            self.ip,
            "fabric.repeat",
            operands=[value],
            results=[self.patch_type],
            attributes={"count": _i64(self.context, rounds)},
            regions=1,
        )
        with self.location:
            body = repeat.regions[0].blocks.append(self.patch_type)
        body_ip = mlir_ir.InsertionPoint(body)
        iterated = emit_call(body_ip, body.arguments[0])
        self._insert(body_ip, "fabric.yield", operands=[iterated])
        return repeat.result

    def _materialize_action_site(self, selected):
        """Materialize the code-agnostic P1 site referenced by P2 evidence."""

        existing = self.transaction.find_symbol(selected.handle.symbol,
                                                "lvm.action_site",
                                                scan=False)
        operation = selected.operation
        root_reference = (
            operation.attributes["channel"] if "channel" in operation.attributes
            else operation.attributes["at"] if operation.name == "lvm.measure"
            else operation.attributes["placements"][0])
        domain_symbol = _root_symbol(root_reference)
        if existing is None:
            domain = self.transaction.find_symbol(domain_symbol, "lvm.domain")
            if domain is None:
                raise ValueError("selected QEC realization has no LVM domain")
            if operation.name == "lvm.measure":
                placement_refs = [operation.attributes["at"]]
            else:
                placement_refs = list(operation.attributes["placements"])
            attrs = {
                "sym_name":
                    mlir_ir.StringAttr.get(selected.handle.symbol,
                                           context=self.context),
                "kind":
                    mlir_ir.StringAttr.get(selected.handle.kind,
                                           context=self.context),
                "placements":
                    mlir_ir.ArrayAttr.get(
                        placement_refs,
                        context=self.context,
                    ),
            }
            if selected.handle.parameters:
                attrs["parameters"] = _dictionary(
                    self.context, dict(selected.handle.parameters))
            if operation.name in {"lvm.apply", "lvm.consume_resource"}:
                attrs["objective"] = operation.attributes["action"]
            elif operation.name == "lvm.instrument":
                attrs["objective"] = operation.attributes["instrument"]
            with mlir_ir.InsertionPoint(domain.regions[0].blocks[0]):
                existing = mlir_ir.Operation.create("lvm.action_site",
                                                    attributes=attrs,
                                                    loc=self.location)
            self.transaction._index_symbol(existing, selected.handle.symbol)
        with self.context:
            return mlir_ir.SymbolRefAttr.get(
                [domain_symbol, selected.handle.symbol],
                context=self.context,
            )

    def _materialize_selected(self, selected: _SelectedSite):
        lowering = selected.selected
        key = (
            id(lowering),
            selected.handle.symbol,
            selected.handle.objective,
            tuple(sorted(selected.handle.parameters.items())),
            selected.handle.placements,
            selected.handle.resource_kind,
            selected.handle.resource_stream_owner,
            selected.handle.resource_stream,
        )
        cached = self.generated.get(key)
        if cached is not None:
            return cached
        if isinstance(lowering, QECLowering):
            manifest = self.transaction.materialize(lowering)
            code, encoding = _candidate_code(lowering)
            context = QECCompilerContext(
                device=self.device,
                lowering=lowering,
                code=code,
                encoding=encoding,
                policy=self.policy,
                dependencies=lowering.dependencies,
                placements=tuple(self.binding_by_name[name]
                                 for name in selected.handle.placements),
                qec_selection=self.qec_selection,
            )
            generated = lowering.compile_site(selected.handle, context)
            generation_specialization = {}
            if isinstance(generated, GeneratedQECArtifact):
                generation_specialization = dict(generated.specialization)
                generated = generated.definition
            if not isinstance(generated,
                              (GadgetDefinition, ProtocolDefinition)):
                raise TypeError(
                    f"QEC lowering {lowering.name!r} must return a gadget or protocol"
                )
            declared_objective = lowering.objective
            if isinstance(declared_objective,
                          (LogicalActionRef, LogicalInstrumentRef)):
                declared_objective = declared_objective.name
            elif isinstance(declared_objective, ProgramDefinition):
                declared_objective = declared_objective.name.removeprefix(
                    "qlx_standard_")
            generated_objective = generated.implements
            if isinstance(generated_objective,
                          (LogicalActionRef, LogicalInstrumentRef)):
                generated_objective = generated_objective.name
            elif isinstance(generated_objective, ProgramDefinition):
                generated_objective = generated_objective.name.removeprefix(
                    "qlx_standard_")
            if (declared_objective is not None and
                    generated_objective != declared_objective):
                raise ValueError(
                    f"QEC lowering {lowering.name!r} must return a realization "
                    f"implementing the exact {declared_objective!r} objective")
        else:
            manifest = None
            generated = lowering
            generation_specialization = {}
        payload_block_ids = generation_specialization.pop(
            "_payload_block_ids", None)
        if payload_block_ids is not None:
            if not isinstance(generated, ProtocolDefinition):
                raise TypeError(
                    "selected payload block witness requires a generated protocol"
                )
            self.transaction.bind_protocol_payload_blocks(
                generated, payload_block_ids)
        handle = self.transaction.materialize(generated)
        if manifest is not None:
            manifest_operation = self.transaction.find_symbol(
                manifest.symbol, "qlx.qec_lowering")
            dependencies = list(manifest_operation.attributes["dependencies"])
            dependency_names = {
                _symbol(dependency) for dependency in dependencies
            }
            if handle.symbol not in dependency_names:
                dependencies.append(
                    mlir_ir.FlatSymbolRefAttr.get(handle.symbol,
                                                  context=self.context))
                manifest_operation.attributes["dependencies"] = (
                    mlir_ir.ArrayAttr.get(
                        dependencies,
                        context=self.context,
                    ))
        inputs, results, _ = self.transaction.signature_of(generated)
        blocks = {
            self._block_key(placement)
            for placement in selected.handle.placements
        }
        patch_inputs = sum(
            str(value).startswith("!fabric.patch<") for value in inputs)
        nonpatch_inputs = len(inputs) - patch_inputs
        site_nonquantum_inputs = (selected.handle.input_arity -
                                  len(selected.handle.placements))
        specialized_nonquantum_inputs = (
            1 if isinstance(lowering, QECLowering) and
            lowering.objective_family == "pauli_product_rotation" and
            "angle" in selected.handle.parameters else 0)
        input_boundary_matches = (
            (selected.handle.objective_family == "logical_preparation" and
             patch_inputs == 1 and nonpatch_inputs == 0 and len(blocks) == 1) or
            len(inputs) == selected.handle.input_arity or
            (patch_inputs == len(blocks) and nonpatch_inputs
             == site_nonquantum_inputs - specialized_nonquantum_inputs))
        if not input_boundary_matches:
            raise ValueError(
                "generated QEC artifact input arity does not match action site: "
                f"family={selected.handle.objective_family!r}, "
                f"inputs={len(inputs)}, patch_inputs={patch_inputs}, "
                f"blocks={len(blocks)}, site_inputs={selected.handle.input_arity}"
            )
        quantum_results = sum(
            str(value.type).startswith("!lvm.logical_qubit<")
            for value in selected.operation.results)
        patch_results = sum(
            str(value).startswith("!fabric.patch<") for value in results)
        packed_owner_result = (
            quantum_results == 0 and patch_results == 1 and
            any(self.binding_by_name[placement].block is not None
                for placement in selected.handle.placements))
        result_boundary_matches = (
            len(results) == selected.handle.result_arity or
            ((patch_results == (len(blocks) if quantum_results else 0) or
              packed_owner_result) and len(results) - patch_results
             == selected.handle.result_arity - quantum_results))
        if not result_boundary_matches:
            raise ValueError(
                "generated QEC artifact result arity does not match action site"
            )
        artifact_name = ("fabric.protocol" if isinstance(
            generated, ProtocolDefinition) else "fabric.gadget")
        artifact = self.transaction.find_symbol(handle.symbol, artifact_name)
        if manifest is not None:
            artifact.attributes["generated_by"] = mlir_ir.FlatSymbolRefAttr.get(
                manifest.symbol, context=self.context)
            artifact.attributes["action_site"] = self._materialize_action_site(
                selected)
            artifact.attributes["specialization"] = _dictionary(
                self.context,
                {
                    **selected.handle.parameters,
                    **generation_specialization,
                },
            )
            objective_attr = (
                selected.operation.attributes["action"]
                if selected.operation.name in {
                    "lvm.apply", "lvm.consume_resource"
                } else selected.operation.attributes["instrument"]
                if selected.operation.name == "lvm.instrument" else None)
            if isinstance(generated, ProtocolDefinition):
                if ("objective" not in artifact.attributes and
                        objective_attr is not None):
                    artifact.attributes["objective"] = objective_attr
        self.generated[key] = generated, handle
        return generated, handle

    def _call(self, operation, selected):
        generated, handle = self._materialize_selected(selected)
        inputs, results, _ = self.transaction.signature_of(generated)
        quantum_operands = tuple(
            value for value in operation.operands
            if str(value.type).startswith("!lvm.logical_qubit<"))
        blocks = tuple(
            dict.fromkeys(
                self._block_key(placement)
                for placement in selected.handle.placements))
        if len(quantum_operands) != len(selected.handle.placements):
            raise ValueError(
                "selected QEC realization has inconsistent placement and "
                "logical-operand arity")
        patch_by_block = {}
        for operand, placement in zip(quantum_operands,
                                      selected.handle.placements):
            block = self._block_key(placement)
            mapped = self._mapped(operand)
            previous = patch_by_block.setdefault(block, mapped)
            if previous != mapped:
                raise ValueError(
                    "one selected QEC block maps to multiple live patch owners")
        nonpatch_available = [
            self._mapped(value)
            for value in operation.operands
            if value not in quantum_operands
        ]
        remaining_blocks = list(blocks)
        ordered = []
        for expected in inputs:
            if str(expected).startswith("!fabric.patch<"):
                index = next(
                    (index for index, block in enumerate(remaining_blocks)
                     if patch_by_block[block].type == expected),
                    None,
                )
                if index is None:
                    raise ValueError(
                        "selected QEC realization patch boundary cannot be "
                        "matched one-for-one to distinct placed blocks")
                ordered.append(patch_by_block[remaining_blocks.pop(index)])
                continue
            index = next(
                (index for index, value in enumerate(nonpatch_available)
                 if value.type == expected),
                None,
            )
            if index is None:
                raise ValueError(
                    "selected QEC realization boundary cannot be matched to "
                    "the placed action-site operands")
            ordered.append(nonpatch_available.pop(index))
        if remaining_blocks:
            raise ValueError(
                "selected QEC realization did not consume every distinct "
                "placed QEC block")
        if nonpatch_available:
            specialized_angle = (isinstance(selected.selected, QECLowering) and
                                 selected.selected.objective_family
                                 == "pauli_product_rotation" and
                                 "angle" in selected.handle.parameters and
                                 len(nonpatch_available) == 1 and
                                 str(nonpatch_available[0].type) == "f64")
            if not specialized_angle:
                raise ValueError(
                    "selected QEC realization did not consume or specialize "
                    "every action-site operand")
        call_attributes = {
            "callee":
                mlir_ir.FlatSymbolRefAttr.get(handle.symbol,
                                              context=self.context)
        }
        resource_action_site = None
        if operation.name == "lvm.consume_resource":
            resource_action_site = self._materialize_action_site(selected)
            call_attributes["resource_action_site"] = resource_action_site
            call_attributes["resource_objective"] = operation.attributes[
                "action"]
        call = self._insert(
            self.ip,
            "fabric.call",
            operands=ordered,
            results=results,
            attributes=call_attributes,
        )
        if resource_action_site is not None:
            self._bind_resource_consumer(
                operation,
                resource_action_site,
                call_attributes["callee"],
                call_attributes["resource_objective"],
            )
        old_quantum = [
            value for value in operation.results
            if str(value.type).startswith("!lvm.logical_qubit<")
        ]
        new_patches = [
            value for value in call.results
            if str(value.type).startswith("!fabric.patch<")
        ]
        old_classical = [
            value for value in operation.results if value not in old_quantum
        ]
        new_classical = [
            value for value in call.results if value not in new_patches
        ]
        # Destructive actions may intentionally return no patch owner. Any
        # boundary that does preserve logical data must return exactly one
        # linear owner per distinct block; repeated logical results are port
        # aliases of that owner.
        if old_quantum or new_patches:
            if len(new_patches) != len(blocks):
                raise ValueError(
                    "selected QEC realization must return one patch owner per "
                    "distinct placed QEC block")
            patch_results_by_block = dict(zip(blocks, new_patches))
            for old in old_quantum:
                binding = self._binding_for_value(old)
                if binding is None:
                    raise ValueError(
                        "selected QEC realization result has no placement binding"
                    )
                self.value_map[old] = patch_results_by_block[binding.block or
                                                             binding.placement]
            self.block_state.update(patch_results_by_block)
        if len(old_classical) != len(new_classical):
            raise ValueError(
                "selected QEC realization classical results do not match action site"
            )
        for old, new in zip(old_classical, new_classical):
            self.value_map[old] = new
        return tuple(new_patches)

    def _bind_resource_consumer(self, operation, action_site, callee,
                                objective):
        """Authenticate one direct request/await/consume ownership chain."""
        resource = operation.operands[0]
        awaited = getattr(resource, "owner", None)
        if getattr(awaited, "name", None) != "lvm.event_await":
            return
        request = getattr(awaited.operands[0], "owner", None)
        if getattr(request, "name", None) != "lvm.resource_request":
            return
        concrete = self._concrete_resource_events.get(request.result)
        if concrete is not None:
            transport = getattr(concrete, "owner", None)
            if getattr(transport, "name", None) != "fabric.transport":
                raise ValueError(
                    "concrete resource consume lost its authenticated transport"
                )
            return
        projected_event = self._mapped(request.result)
        projected_request = getattr(projected_event, "owner", None)
        if (projected_request is None or
                projected_request.name != "fabric.resource_request"):
            raise ValueError(
                "resource consume lost its canonical projected request")
        existing_site = projected_request.attributes.get("consumer_action_site")
        existing_callee = projected_request.attributes.get("consumer_callee")
        existing_objective = projected_request.attributes.get(
            "consumer_objective")
        if ((existing_site is not None and existing_site != action_site) or
            (existing_callee is not None and existing_callee != callee) or
            (existing_objective is not None and
             existing_objective != objective)):
            raise ValueError(
                "one linear resource request cannot select several consumers")
        projected_request.attributes["consumer_action_site"] = action_site
        projected_request.attributes["consumer_callee"] = callee
        projected_request.attributes["consumer_objective"] = objective

    def _convert(self, operation):
        name = operation.name
        if name == "lvm.prepare":
            self._prepare(operation)
            return
        if name in {
                "lvm.apply",
                "lvm.instrument",
                "lvm.measure",
                "lvm.consume_resource",
        }:
            selected = self.selected_sites.get(
                f"site{int(operation.attributes['site'])}")
            if selected is None:
                objective = (
                    f"measure_{_basis(operation.attributes['basis'])}" if name
                    == "lvm.measure" else _objective(operation.attributes[
                        "action" if name in
                        {"lvm.apply", "lvm.consume_resource"} else "instrument"]
                                                    ))
                raise ValueError(
                    f"no feasible P2 implementation for {objective}")
            returned_patches = self._call(operation, selected)
            if (name == "lvm.measure" and not self.packed_readout_preserves.get(
                    selected.handle.symbol, False)):
                block = self._block_key(selected.handle.placements[0])
                patch = self.block_state.pop(block, None)
                if returned_patches and (len(returned_patches) != 1 or
                                         patch != returned_patches[0]):
                    raise ValueError(
                        "logical readout returned an ambiguous patch owner")
                if returned_patches:
                    self._insert(
                        self.ip,
                        "fabric.dealloc",
                        operands=[returned_patches[0]],
                    )
            return
        if name == "lvm.resource_request":
            kind = _text(operation.attributes["kind"])
            _stream_op, stream = self._canonical_stream(
                operation.attributes["stream"], kind)
            if kind == "ccz_state":
                supply = self._concrete_resource_supply(operation)
                producer = self.transaction.materialize(
                    supply.stream.produced_by)
                producer_inputs, producer_results, _ = (
                    self.transaction.signature_of(supply.stream.produced_by))
                if producer_inputs or len(producer_results) != 1:
                    raise ValueError(
                        "selected CCZ_STATE producer must return exactly one "
                        "resource without protocol inputs")
                produced = self._insert(
                    self.ip,
                    "fabric.call",
                    results=producer_results,
                    attributes={
                        "callee":
                            mlir_ir.FlatSymbolRefAttr.get(producer.symbol,
                                                          context=self.context)
                    },
                )
                transfer = self.transaction.materialize(supply.stream.transfer)
                route = supply.qec_route
                device_handle = self.transaction.materialize(self.device)
                device_operation = self.transaction.find_symbol(
                    device_handle.symbol, "qlx.device")
                if device_operation is None or "qec" not in device_operation.attributes:
                    raise ValueError(
                        "selected CCZ_STATE route has no retained QEC machine")
                qec_symbol = _symbol(device_operation.attributes["qec"])
                with self.context:
                    route_reference = mlir_ir.SymbolRefAttr.get(
                        [qec_symbol, route.name],
                        context=self.context,
                    )
                transported = self._insert(
                    self.ip,
                    "fabric.transport",
                    operands=[produced.result],
                    results=[produced.result.type],
                    attributes={
                        "src_region":
                            mlir_ir.FlatSymbolRefAttr.get(
                                route.source.region.name, context=self.context),
                        "dst_region":
                            mlir_ir.FlatSymbolRefAttr.get(
                                route.destination.region.name,
                                context=self.context),
                        "protocol":
                            mlir_ir.FlatSymbolRefAttr.get(transfer.symbol,
                                                          context=self.context),
                        "route":
                            route_reference,
                    },
                )
                self._concrete_resource_events[operation.result] = (
                    transported.result)
                return
            event = self._insert(
                self.ip,
                "fabric.resource_request",
                results=[self._p2_type(operation.result.type)],
                attributes={
                    "kind": operation.attributes["kind"],
                    "stream": operation.attributes["stream"],
                },
            )
            self.value_map[operation.result] = event.result
            return
        if name == "lvm.event_test":
            tested = self._insert(
                self.ip,
                "fabric.event_test",
                operands=[self._mapped(operation.operands[0])],
                results=[operation.result.type],
            )
            self.value_map[operation.result] = tested.result
            return
        if name == "lvm.event_poll":
            polled = self._insert(
                self.ip,
                "fabric.event_poll",
                operands=[self._mapped(operation.operands[0])],
                results=[operation.result.type],
            )
            self.value_map[operation.result] = polled.result
            return
        if name == "lvm.event_is":
            tested = self._insert(
                self.ip,
                "fabric.event_is",
                operands=[self._mapped(operation.operands[0])],
                results=[operation.result.type],
                attributes={"state": operation.attributes["state"]},
            )
            self.value_map[operation.result] = tested.result
            return
        if name == "lvm.event_select_ready":
            selected = self._insert(
                self.ip,
                "fabric.event_select_ready",
                operands=[self._mapped(value) for value in operation.operands],
                results=[operation.result.type],
                attributes={"policy": operation.attributes["policy"]},
            )
            self.value_map[operation.result] = selected.result
            return
        if name == "lvm.event_try_take":
            operands, _ = self._coalesced_boundary(operation.operands)
            results, result_indices = self._coalesced_boundary(
                operation.results)
            result_types = [self._p2_type(result.type) for result in results]
            nested = self._insert(
                self.ip,
                "fabric.event_try_take",
                operands=[self._mapped(value) for value in operands],
                results=result_types,
                regions=3,
            )
            incoming_blocks = dict(self.block_state)
            branch_states = []
            for source_region, target_region in zip(operation.regions,
                                                    nested.regions):
                self.block_state = dict(incoming_blocks)
                self._convert_region(source_region, target_region)
                branch_states.append(dict(self.block_state))
            self.block_state = dict(incoming_blocks)
            for old, index in zip(operation.results, result_indices):
                result = nested.results[index]
                self.value_map[old] = result
                binding = self._binding_for_value(old)
                if binding is not None and binding.block is not None:
                    block = binding.block
                    if any(block not in state for state in branch_states):
                        raise ValueError(
                            "event_try_take branches must yield every encoded owner"
                        )
                    self.block_state[block] = result
            return
        if name == "lvm.event_cancel":
            attrs = {}
            if "reason" in operation.attributes:
                attrs["reason"] = operation.attributes["reason"]
            cancelled = self._insert(
                self.ip,
                "fabric.event_cancel",
                operands=[self._mapped(operation.operands[0])],
                results=[operation.result.type],
                attributes=attrs,
            )
            self.value_map[operation.result] = cancelled.result
            return
        if name == "lvm.event_await":
            concrete = self._concrete_resource_events.get(operation.operands[0])
            if concrete is not None:
                if concrete.type != self._p2_type(operation.result.type):
                    raise ValueError(
                        "concrete CCZ_STATE producer returned the wrong kind")
                self.value_map[operation.result] = concrete
                return
            awaited = self._insert(
                self.ip,
                "fabric.event_await",
                operands=[self._mapped(operation.operands[0])],
                results=[self._p2_type(operation.result.type)],
            )
            self.value_map[operation.result] = awaited.result
            return
        if name == "lvm.fence":
            self._insert(
                self.ip,
                "fabric.fence",
                attributes={"effects": operation.attributes["effects"]},
            )
            return
        if name == "lvm.selection":
            self._insert(
                self.ip,
                "fabric.selection",
                operands=[self._mapped(operation.operands[0])],
                attributes={
                    "mode": operation.attributes["mode"],
                    "accept_when": operation.attributes["accept_when"],
                },
            )
            return
        if name == "lvm.frame_init":
            created = self._insert(
                self.ip,
                "fabric.frame_create",
                results=[self._p2_type(operation.result.type)],
                attributes={"domain": operation.attributes["domain"]},
            )
            self.value_map[operation.result] = created.result
            return
        if name in {"lvm.frame_update", "lvm.frame_transform"}:
            transformed = self._insert(
                self.ip,
                name.replace("lvm.", "fabric."),
                operands=[self._mapped(value) for value in operation.operands],
                results=[self._p2_type(operation.result.type)],
                attributes=dict(operation.attributes),
            )
            self.value_map[operation.result] = transformed.result
            return
        if name == "lvm.idle":
            rounds = self._constant_int(operation.operands[-1])
            groups = {}
            for old_input, old_result in zip(operation.operands[:-1],
                                             operation.results):
                groups.setdefault(self._owner_key(old_input), []).append(
                    (old_input, old_result))
            for block, values in groups.items():
                placements = tuple(
                    binding.placement
                    for old_input, _ in values
                    for binding in (self._binding_for_value(old_input),)
                    if binding is not None)
                realization = (self._memory_realization(placements)
                               if rounds > 0 else None)
                mapped_input = self._mapped(values[0][0])
                if realization is None:
                    idle = self._insert(
                        self.ip,
                        "fabric.idle",
                        operands=[mapped_input],
                        results=[self.patch_type],
                        attributes={"rounds": _i64(self.context, rounds)},
                    )
                    result = idle.result
                else:
                    result = self._emit_memory_realization(
                        mapped_input, realization, rounds)
                for _old_input, old_result in values:
                    self.value_map[old_result] = result
                self.block_state[block] = result
            return
        if name == "lvm.discard":
            owners = {}
            for value in operation.operands:
                owners.setdefault(self._owner_key(value), self._mapped(value))
            for block, value in owners.items():
                self._insert(self.ip, "fabric.dealloc", operands=[value])
                self.block_state.pop(block, None)
            return
        if name == "lvm.if":
            results, result_indices = self._coalesced_boundary(
                operation.results)
            result_types = [self._p2_type(result.type) for result in results]
            nested = self._insert(
                self.ip,
                "fabric.if",
                operands=[self._mapped(operation.operands[0])],
                results=result_types,
                regions=2,
            )
            incoming_blocks = dict(self.block_state)
            self.block_state = dict(incoming_blocks)
            self._convert_region(operation.regions[0], nested.regions[0])
            then_blocks = dict(self.block_state)
            self.block_state = dict(incoming_blocks)
            self._convert_region(operation.regions[1], nested.regions[1])
            else_blocks = dict(self.block_state)
            self.block_state = dict(incoming_blocks)
            for old, index in zip(operation.results, result_indices):
                result = nested.results[index]
                self.value_map[old] = result
                binding = self._binding_for_value(old)
                if binding is not None and binding.block is not None:
                    block = binding.block
                    self.block_state[block] = result
                    # Both branches must refine the same encoded owner.  The
                    # merged fabric.if result is the only post-dominator state;
                    # branch-local Python conversion state must not leak.
                    if block not in then_blocks or block not in else_blocks:
                        raise ValueError(
                            "structured control failed to yield one encoded "
                            f"owner for shared block {block!r}")
            return
        if name == "lvm.repeat":
            operands, _ = self._coalesced_boundary(operation.operands)
            results, result_indices = self._coalesced_boundary(
                operation.results)
            result_types = [self._p2_type(result.type) for result in results]
            nested = self._insert(
                self.ip,
                "fabric.repeat",
                operands=[self._mapped(value) for value in operands],
                results=result_types,
                attributes={"count": operation.attributes["count"]},
                regions=1,
            )
            incoming_blocks = dict(self.block_state)
            self.block_state = dict(incoming_blocks)
            self._convert_region(operation.regions[0], nested.regions[0])
            body_blocks = dict(self.block_state)
            self.block_state = dict(incoming_blocks)
            for old, index in zip(operation.results, result_indices):
                result = nested.results[index]
                self.value_map[old] = result
                binding = self._binding_for_value(old)
                if binding is not None and binding.block is not None:
                    block = binding.block
                    if block not in body_blocks:
                        raise ValueError(
                            "folded repeat failed to yield one encoded owner "
                            f"for shared block {block!r}")
                    self.block_state[block] = result
            return
        if name == "lvm.while":
            operands, _ = self._coalesced_boundary(operation.operands)
            results, result_indices = self._coalesced_boundary(
                operation.results)
            result_types = [self._p2_type(result.type) for result in results]
            attrs = {}
            if "max_iterations" in operation.attributes:
                attrs["max_iterations"] = operation.attributes["max_iterations"]
            nested = self._insert(
                self.ip,
                "fabric.while",
                operands=[self._mapped(value) for value in operands],
                results=result_types,
                attributes=attrs,
                regions=2,
            )
            incoming_blocks = dict(self.block_state)
            self.block_state = dict(incoming_blocks)
            self._convert_region(operation.regions[0], nested.regions[0])
            condition_blocks = dict(self.block_state)
            self.block_state = dict(incoming_blocks)
            self._convert_region(operation.regions[1], nested.regions[1])
            body_blocks = dict(self.block_state)
            self.block_state = dict(incoming_blocks)
            for old, index in zip(operation.results, result_indices):
                result = nested.results[index]
                self.value_map[old] = result
                binding = self._binding_for_value(old)
                if binding is not None and binding.block is not None:
                    block = binding.block
                    if block not in condition_blocks or block not in body_blocks:
                        raise ValueError(
                            "dynamic while failed to preserve one encoded owner "
                            f"for shared block {block!r} on every edge")
                    self.block_state[block] = result
            return
        if name == "arith.constant":
            clone = self._insert(
                self.ip,
                name,
                results=[result.type for result in operation.results],
                attributes=dict(operation.attributes),
            )
            for old, new in zip(operation.results, clone.results):
                self.value_map[old] = new
            return
        if name == "lvm.xor":
            clone = self._insert(
                self.ip,
                "fabric.xor",
                operands=[self._mapped(value) for value in operation.operands],
                results=[operation.result.type],
            )
            self.value_map[operation.result] = clone.result
            return
        if name == "lvm.return":
            self._insert(
                self.ip,
                "fabric.protocol_return",
                operands=[self._mapped(value) for value in operation.operands],
            )
            return
        raise NotImplementedError(f"lvm-to-fabric does not yet convert {name}")


def lower_qec(source,
              *,
              device,
              pipeline,
              policy=None,
              experiment=None) -> Build:
    if not isinstance(source, Build) or source.profile != "p1":
        raise ValueError("QEC lowering requires a P1 Build")
    if not isinstance(device, Device):
        raise TypeError("P1-to-P2 lowering requires a concrete Device")
    compiler = _P1ToP2(source, device, policy)
    module, root = compiler.run()
    unresolved = tuple(operation.name
                       for operation in compiler.transaction.walk()
                       if operation.name == "fabric.inject")
    if unresolved:
        raise ValueError(
            "canonical P2 cannot contain fabric.inject; resource-backed "
            "logical intent must select a concrete code-specific protocol")
    return Build(
        context=module.context,
        module=module,
        root=DefinitionHandle(root, "protocol", "p2n"),
        profile="p2n",
        pipeline=pipeline,
        evidence=source.evidence + (EvidenceRecord(
            kind="qec_selection_and_generation",
            producer="qlx-python@0.3",
            result="pass",
            obligations=(
                "objective-match",
                "capability-feasibility",
                "typed-boundary",
                "provider-version",
                "generated-artifact-verification",
            ),
        ),),
        value_groups={
            name: len(group) for name, group in source.values._groups.items()
        },
        placement=source.placement,
        qec_selection=compiler.qec_selection,
        experiment=experiment or source.experiment,
        device=device,
        source_modules=source.source_modules,
    )
