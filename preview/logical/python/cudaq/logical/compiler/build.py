# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import logging
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

from cudaq.mlir import ir as mlir_ir

from ..programs.definition import DefinitionHandle
from ..experiments.definition import Experiment
from ..architecture.logical import ProgramValueSchema
from ..stages import (
    facets_for_kind,
    normalize_facets,
    stage_and_facets,
)
from .build_bundle import (
    _BUILD_V2_IR_VERSION,
    _BUILD_V2_MODEL_VERSION,
    _BUILD_V2_SCHEMA,
    _build_bundle_content_sha256,
    _root_operation_name,
    _root_profiles,
    _validate_v2_bundle,
    _validate_v2_nested_metadata,
)

_logger = logging.getLogger("cudaq.logical")

_UNSET = object()

_FACET_WITNESS_OPERATIONS = {
    "qec_spec":
        frozenset({
            "fabric.code",
            "fabric.code_profile",
            "fabric.encoding",
            "fabric.encoding_epoch",
            "fabric.encoding_epoch_schema",
        }),
    "qec_realization":
        frozenset({"fabric.gadget"}),
    "protocol_network":
        frozenset({"fabric.protocol"}),
    "patch_graph":
        frozenset({"fabric.patch_graph"}),
}


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    kind: str
    producer: str
    result: str
    obligations: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()


def _placement_witness_sha256(placement) -> str:
    """Canonical content commitment for a replayable placement witness."""

    payload = json.dumps(asdict(placement),
                         sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def _qec_selection_sha256(qec_selection) -> str:
    """Canonical content commitment for a replayable P2 selection witness."""

    payload = json.dumps(asdict(qec_selection),
                         sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def _attr_text(attribute) -> str:
    """Plain text of a string/symbol attribute (``@`` and quotes stripped)."""

    value = getattr(attribute, "value", None)
    text = str(attribute if value is None else value)
    return text.strip('"').lstrip("@").split("::@")[-1]


def _walk_operation(operation):
    stack = [operation]
    while stack:
        current = stack.pop()
        yield current
        children = [
            view.operation
            for region in current.regions
            for block in region.blocks
            for view in block.operations
        ]
        stack.extend(reversed(children))


def _string_array_attribute(operation, name: str) -> tuple[str, ...]:
    try:
        attribute = operation.attributes[name]
    except KeyError as error:
        raise ValueError(
            f"qlx.build/v2 module is missing required {name}") from error
    return tuple(_attr_text(item) for item in attribute)


def _retained_device_profile(operation) -> str:
    """Derive a device root's stage from its contiguous retained layers."""

    if "qec" in operation.attributes:
        return "p2"
    return "p1"


def _validate_v2_module_classification(module, root, bundle) -> None:
    """Prove replay classification from retained, already parsed MLIR."""

    operation = module.operation
    for attribute, expected, label in (
        ("qlx.model_version", _BUILD_V2_MODEL_VERSION, "model_version"),
        ("qlx.ir_version", _BUILD_V2_IR_VERSION, "ir_version"),
    ):
        try:
            actual = _attr_text(operation.attributes[attribute])
        except KeyError as error:
            raise ValueError(
                f"qlx.build/v2 module is missing {attribute}") from error
        if actual != expected:
            raise ValueError(
                f"qlx.build/v2 module {label} differs from the supported version"
            )

    stage = bundle["stage"]
    if stage is not None:
        retained_stages = _string_array_attribute(operation, "qlx.stages")
        if stage not in retained_stages:
            raise ValueError(
                "qlx.build/v2 stage is not retained by the embedded MLIR")
    if bundle["profile"] != "common":
        retained_profiles = _string_array_attribute(operation, "qlx.profiles")
        if bundle["profile"] not in retained_profiles:
            raise ValueError(
                "qlx.build/v2 profile is not retained by the embedded MLIR")

    if root.name == "lvm.kernel" and bundle["qec_selection"] is not None:
        raise ValueError(
            "qlx.build/v2 P1 root cannot carry a P2 QEC selection witness")

    if root.name == "qlx.device":
        actual_profile = _retained_device_profile(root)
        if actual_profile != bundle["profile"]:
            raise ValueError(
                "qlx.build/v2 profile differs from the retained device layers")
    retained_facets = set(_string_array_attribute(operation, "qlx.facets"))
    missing_facets = tuple(
        facet for facet in bundle["facets"] if facet not in retained_facets)
    if missing_facets:
        raise ValueError(
            "qlx.build/v2 facets are not retained by the embedded MLIR: " +
            ", ".join(missing_facets))

    module_operation_names = {
        candidate.name for candidate in _walk_operation(operation)
    }
    pipeline_facets = []
    if bundle["pipeline"] is not None:
        for item in bundle["pipeline"]["passes"]:
            removed = {
                *item["invalidates_facets"],
                *item["recomputes_facets"],
            }
            pipeline_facets = [
                facet for facet in pipeline_facets if facet not in removed
            ]
            for facet in (
                    *item["provides_facets"],
                    *item["recomputes_facets"],
            ):
                if facet not in pipeline_facets:
                    pipeline_facets.append(facet)
    missing_pipeline_facets = tuple(
        facet for facet in pipeline_facets if facet not in bundle["facets"])
    if missing_pipeline_facets:
        raise ValueError("qlx.build/v2 facets omit retained recipe facets: " +
                         ", ".join(missing_pipeline_facets))

    for facet in bundle["facets"]:
        witnesses = _FACET_WITNESS_OPERATIONS.get(facet)
        if (witnesses is not None and facet not in pipeline_facets and
                witnesses.isdisjoint(module_operation_names)):
            raise ValueError(
                f"qlx.build/v2 facet {facet!r} is not established by "
                "retained verified IR")

    _, profile_facets = stage_and_facets(bundle["profile"])
    intrinsic = {
        facet.value for facet in (
            *profile_facets,
            *facets_for_kind(bundle["root"]["kind"].rsplit(".", 1)[-1]),
        )
    }
    for facet in bundle["facets"]:
        if (facet not in _FACET_WITNESS_OPERATIONS and
                facet not in intrinsic and facet not in pipeline_facets):
            raise ValueError(
                f"qlx.build/v2 facet {facet!r} is not established by "
                "the retained root or replay recipe")


def _symbol_path(attribute) -> tuple[str, ...]:
    value = getattr(attribute, "value", None)
    if isinstance(value, (tuple, list)):
        return tuple(str(item).lstrip("@") for item in value)
    text = str(attribute if value is None else value).strip('"')
    return tuple(item.lstrip("@") for item in text.split("::") if item)


def _operation_symbol_path(operation) -> tuple[str, ...]:
    path = []
    current = operation
    while current is not None:
        if "sym_name" in current.attributes:
            path.append(_symbol_path(current.attributes["sym_name"])[-1])
        current = current.parent
    return tuple(reversed(path))


def _objective_text(attribute) -> str:
    text = str(attribute)
    for prefix in ("#qlx.action<", "#qlx.instrument<"):
        if text.startswith(prefix) and text.endswith(">"):
            return text[len(prefix):-1].strip('"')
    return _symbol_path(attribute)[-1].removeprefix("qlx_standard_")


def _capability_text(attribute) -> str:
    text = str(attribute)
    prefix = "#lvm.capability<"
    if text.startswith(prefix) and text.endswith(">"):
        return text[len(prefix):-1].strip('"')
    return _attr_text(attribute)


def _top_level_symbol(module, symbol: str, kinds, prefix: str):
    matches = tuple(view.operation
                    for view in module.body.operations
                    if view.operation.name in kinds and
                    "sym_name" in view.operation.attributes and
                    _attr_text(view.operation.attributes["sym_name"]) == symbol)
    if len(matches) != 1:
        expected = " or ".join(sorted(kinds))
        raise ValueError(
            f"{prefix}: selected @{symbol} {expected} is missing or ambiguous")
    return matches[0]


def _verify_placement_witness(kernel, placement, prefix: str) -> str:
    """Authenticate detached placement data against its retained P1 kernel."""

    if "input_p0" not in kernel.attributes:
        raise ValueError(
            f"{prefix}: retained P1 input P0 provenance is missing")
    input_p0 = _attr_text(kernel.attributes["input_p0"])
    if placement is None:
        raise ValueError(f"{prefix}: placement witness is missing")
    if (placement.input_p0 != input_p0 or "domain" not in kernel.attributes or
            _attr_text(kernel.attributes["domain"]) != placement.machine):
        raise ValueError(f"{prefix}: placement provenance differs")
    if "placement_witness_sha256" not in kernel.attributes:
        raise ValueError(f"{prefix}: placement witness commitment is missing")
    if (_attr_text(kernel.attributes["placement_witness_sha256"])
            != _placement_witness_sha256(placement)):
        raise ValueError(f"{prefix}: placement witness commitment differs")
    retained = []
    for operation in _walk_operation(kernel):
        if operation.name != "lvm.prepare":
            continue
        required = (
            "placement_owner",
            "placement_slot",
            "source_allocation",
            "source_group",
            "source_path",
            "at",
        )
        if any(name not in operation.attributes for name in required):
            raise ValueError(f"{prefix}: placement ownership facts are missing")
        allocation = int(operation.attributes["source_allocation"])
        group = _attr_text(operation.attributes["source_group"])
        retained.append((
            _attr_text(operation.attributes["placement_owner"]),
            _symbol_path(operation.attributes["at"])[-1],
            int(operation.attributes["placement_slot"]),
            None if allocation < 0 else allocation,
            None if not group else group,
            tuple(int(value) for value in operation.attributes["source_path"]),
        ))
    witnessed = tuple((
        binding.placement,
        binding.space,
        binding.slot,
        binding.source_allocation,
        binding.source_group,
        binding.source_path,
    ) for binding in placement.bindings)
    if tuple(retained) != witnessed:
        raise ValueError(f"{prefix}: placement ownership facts differ")
    return input_p0


def _verify_qec_selection_commitment(module, root, placement,
                                     qec_selection) -> None:
    """Bind a local P2 selection witness to retained P1 and Fabric truth."""

    prefix = "QEC selection does not match retained verified IR"
    candidates = tuple(
        view.operation
        for view in module.body.operations
        if "sym_name" in view.operation.attributes and
        _attr_text(view.operation.attributes["sym_name"]) == root.symbol)
    if len(candidates) != 1:
        raise ValueError(f"{prefix}: selected root @{root.symbol} is missing")
    selected = candidates[0]
    if selected.name == "lvm.kernel":
        if qec_selection is not None:
            raise ValueError(f"{prefix}: P1 root carries a P2 witness")
        if "input_p0" in selected.attributes or placement is not None:
            _verify_placement_witness(selected, placement, prefix)
        return
    if selected.name not in {"fabric.gadget", "fabric.protocol"}:
        if qec_selection is None and placement is None:
            return
        raise ValueError(f"{prefix}: selected root is not a P1/P2 artifact")

    try:
        metadata = selected.attributes["metadata"]
        input_p1 = _attr_text(metadata["input_p1"])
        committed = _attr_text(metadata["qec_selection_sha256"])
    except KeyError as error:
        if qec_selection is None and placement is None:
            return
        raise ValueError(
            f"{prefix}: P2 provenance metadata is incomplete") from error
    if qec_selection is None:
        raise ValueError(f"{prefix}: QEC selection witness is missing")
    if qec_selection.input_p1 != input_p1:
        raise ValueError(f"{prefix}: input P1 provenance differs")
    if committed != _qec_selection_sha256(qec_selection):
        raise ValueError(f"{prefix}: QEC selection commitment differs")
    kernel = _top_level_symbol(module, input_p1, {"lvm.kernel"}, prefix)
    _verify_placement_witness(kernel, placement, prefix)
    callees = {
        _symbol_path(operation.attributes["callee"])[-1]
        for operation in _walk_operation(selected)
        if operation.name == "fabric.call" and "callee" in operation.attributes
    }
    for action in qec_selection.actions:
        if action.selected not in action.feasible_candidates:
            raise ValueError(f"{prefix}: selected realization is not feasible")
        if action.selected not in callees:
            raise ValueError(
                f"{prefix}: selected realization is not retained by Fabric")


def _verify_ccz_resource_realization(module, root, placement,
                                     qec_selection) -> None:
    """Cross-authenticate CCZ payload owners and concrete supply lineage."""

    if qec_selection is None:
        return
    actions = tuple(
        action for action in qec_selection.actions
        if action.objective == "ccz" and action.kind == "resource_action")
    if not actions:
        return
    if placement is None:
        raise ValueError("CCZ realization requires retained placement evidence")
    owners = {
        owner.placement: (block.block, owner.logical_index)
        for block in qec_selection.blocks for owner in block.owners
    }
    records = {}
    for action in actions:
        if action.site in records:
            raise ValueError("CCZ selection duplicates an action site")
        try:
            expected_ids = tuple(owners[name][0] for name in action.placements)
            expected_ports = tuple(
                owners[name][1] for name in action.placements)
        except KeyError as error:
            raise ValueError(
                "CCZ payload selection omits an action owner") from error
        order = tuple(dict.fromkeys(expected_ids))
        expected_blocks = tuple(order.index(name) for name in expected_ids)
        spaces = {
            binding.space
            for binding in placement.bindings
            if binding.placement in action.placements
        }
        if len(spaces) != 1:
            raise ValueError("CCZ supply requires one exact destination space")
        records[action.site] = {
            "payload": (expected_ids, expected_blocks, expected_ports),
            "segment_count": len(order) + 1,
            "destination": next(iter(spaces)),
        }
    expected_payloads = Counter(
        record["payload"] for record in records.values())
    expected_destinations = Counter(
        record["destination"] for record in records.values())
    operations = tuple(_walk_operation(module.operation))
    fabric_definitions: dict[str, list[Any]] = {}
    fabric_circuits: dict[str, list[Any]] = {}
    fabric_interconnects: dict[tuple[str, ...], list[Any]] = {}
    for operation in operations:
        if (operation.name in {"fabric.gadget", "fabric.protocol"} and
                "sym_name" in operation.attributes):
            symbol = _symbol_path(operation.attributes["sym_name"])[-1]
            fabric_definitions.setdefault(symbol, []).append(operation)
        if (operation.name == "fabric.circuit" and
                "sym_name" in operation.attributes):
            symbol = _symbol_path(operation.attributes["sym_name"])[-1]
            fabric_circuits.setdefault(symbol, []).append(operation)
        if operation.name == "fabric.interconnect":
            path = _operation_symbol_path(operation)
            fabric_interconnects.setdefault(path, []).append(operation)

    root_symbol = getattr(root, "symbol", None)
    if not root_symbol:
        raise ValueError("CCZ realization requires one exact retained root")
    source_symbol = root_symbol

    sources = tuple(fabric_definitions.get(source_symbol, ()))
    if len(sources) != 1:
        raise ValueError(
            "CCZ realization source protocol is missing or ambiguous")
    source = sources[0]
    invocation_ordinal = 0
    active_callables = set()
    reachable_definitions = {}
    reachable_body_owners = {}
    projected_fabric_calls = []

    def body_owner(definition):
        if (definition.name == "fabric.gadget" and
                "realization" in definition.attributes):
            realization = _symbol_path(definition.attributes["realization"])[-1]
            matches = tuple(fabric_circuits.get(realization, ()))
            if len(matches) != 1:
                raise ValueError(
                    "CCZ retained Fabric gadget realization is missing or "
                    "ambiguous")
            return matches[0]
        return definition

    frames = []

    def push_callable(definition, parent_instance):
        identity = id(definition)
        if identity in active_callables:
            raise ValueError("CCZ realization source call graph is recursive")
        active_callables.add(identity)
        symbol = _attr_text(definition.attributes["sym_name"])
        reachable_definitions[symbol] = definition
        owner = body_owner(definition)
        reachable_body_owners[id(owner)] = owner
        frames.append({
            "definition": definition,
            "instance": parent_instance,
            "operations": tuple(_walk_operation(owner)),
            "next": 0,
        })

    push_callable(source, source_symbol)
    invocation_suffixes = {
        "fabric.call": "call",
        "fabric.relocate": "relocate",
        "fabric.establish_support": "support",
        "fabric.establish_topological_record": "topological",
    }
    while frames:
        frame = frames[-1]
        if frame["next"] == len(frame["operations"]):
            active_callables.remove(id(frame["definition"]))
            frames.pop()
            continue
        operation = frame["operations"][frame["next"]]
        frame["next"] += 1
        suffix = invocation_suffixes.get(operation.name)
        if suffix is None:
            continue
        try:
            callee_path = _symbol_path(operation.attributes["callee"])
        except KeyError as error:
            raise ValueError(
                "CCZ retained Fabric invocation is missing its callee"
            ) from error
        callee = callee_path[-1]
        targets = tuple(fabric_definitions.get(callee, ()))
        if len(targets) != 1:
            raise ValueError(
                "CCZ retained Fabric invocation is missing or ambiguous")
        instance = (
            f"{frame['instance']}.{callee}.{suffix}{invocation_ordinal}")
        invocation_ordinal += 1
        if operation.name == "fabric.call":
            projected_fabric_calls.append((operation, instance))
        push_callable(targets[0], instance)

    source_owners = {
        **{
            id(value): value for value in reachable_definitions.values()
        },
        **reachable_body_owners,
    }
    source_operations = tuple(operation for owner in source_owners.values()
                              for operation in _walk_operation(owner))

    def ints(operation, name):
        try:
            return tuple(int(value) for value in operation.attributes[name])
        except KeyError as error:
            raise ValueError(f"CCZ realization is missing {name}") from error

    def strings(operation, name):
        try:
            return tuple(
                _attr_text(value) for value in operation.attributes[name])
        except KeyError as error:
            raise ValueError(f"CCZ realization is missing {name}") from error

    unpacks = tuple(
        operation for operation in source_operations
        if operation.name == "fabric.unpack_resource" and
        "payload_action" in operation.attributes and
        _objective_text(operation.attributes["payload_action"]) == "ccz")
    actual_payloads = {(
        strings(unpack, "payload_logical_block_ids"),
        ints(unpack, "payload_logical_blocks"),
        ints(unpack, "payload_logical_ports"),
    ) for unpack in unpacks}
    if actual_payloads != set(expected_payloads):
        raise ValueError(
            "CCZ payload maps differ from selected QEC block ownership: "
            f"expected {set(expected_payloads)!r}, got {actual_payloads!r}")

    calls = tuple(
        (operation, instance)
        for operation, instance in projected_fabric_calls
        if "resource_action_site" in operation.attributes and
        "resource_objective" in operation.attributes and
        _objective_text(operation.attributes["resource_objective"]) == "ccz")
    actual_sites = Counter(
        _symbol_path(call.attributes["resource_action_site"])[-1]
        for call, _ in calls)
    expected_sites = Counter(records.keys())
    if actual_sites != expected_sites:
        raise ValueError(
            "CCZ resource calls differ from selected action sites: "
            f"expected {dict(expected_sites)!r}, got {dict(actual_sites)!r}")
    selected_fabric_occurrences = Counter()
    selected_fabric_occurrence_order = []
    for call, instance in calls:
        site_path = _symbol_path(call.attributes["resource_action_site"])
        site = site_path[-1]
        callee = _symbol_path(call.attributes["callee"])[-1]
        objective = _objective_text(call.attributes["resource_objective"])
        occurrence = (
            site_path,
            objective,
            _symbol_path(call.attributes["callee"]),
            instance,
        )
        selected_fabric_occurrences[occurrence] += 1
        selected_fabric_occurrence_order.append(occurrence)
        targets = tuple(fabric_definitions.get(callee, ()))
        if len(targets) != 1:
            raise ValueError(
                "CCZ resource call must resolve to one exact realization")
        payloads = {
            (
                strings(unpack, "payload_logical_block_ids"),
                ints(unpack, "payload_logical_blocks"),
                ints(unpack, "payload_logical_ports"),
            )
            for unpack in _walk_operation(body_owner(targets[0]))
            if unpack.name == "fabric.unpack_resource" and
            "payload_action" in unpack.attributes and
            _objective_text(unpack.attributes["payload_action"]) == "ccz"
        }
        if payloads != {records[site]["payload"]}:
            raise ValueError(
                "CCZ resource call payload differs from its selected action site"
            )

    streams = tuple(
        op for op in operations
        if op.name == "lvm.stream" and "produces" in op.attributes and
        _attr_text(op.attributes["produces"]) == "ccz_state")
    if len(streams) != 1:
        raise ValueError("CCZ supply requires one exact retained stream")
    stream = streams[0]
    for field in ("produced_by", "transfer", "backing_region"):
        if field not in stream.attributes:
            raise ValueError(f"CCZ supply stream is missing {field}")
    producer = _attr_text(stream.attributes["produced_by"])
    transfer = _attr_text(stream.attributes["transfer"])
    backing = _attr_text(stream.attributes["backing_region"])

    fabric = tuple(
        op for op in source_operations if op.name == "fabric.transport" and
        "route" in op.attributes and "ccz_state" in str(op.result.type))
    actual_destinations = Counter()
    for transport in fabric:
        destination = _attr_text(transport.attributes["dst_region"])
        actual_destinations[destination] += 1
        if (_attr_text(transport.attributes["src_region"]) != backing or
                _attr_text(transport.attributes["protocol"]) != transfer):
            raise ValueError(
                "CCZ fabric transport differs from stream provenance")
        call = transport.operands[0].owner
        if call.name != "fabric.call" or _attr_text(
                call.attributes["callee"]) != producer:
            raise ValueError("CCZ fabric transport has a foreign producer")
        route_path = _symbol_path(transport.attributes["route"])
        routes = tuple(fabric_interconnects.get(route_path, ()))
        if len(routes) != 1 or any(
                _attr_text(routes[0].attributes[name]) != expected
                for name, expected in (
                    ("region_a", backing),
                    ("region_b", destination),
                    ("protocol", transfer),
                )):
            raise ValueError(
                "CCZ selected route differs from stream provenance")
    if actual_destinations != expected_destinations:
        raise ValueError(
            "CCZ routed fabric transports differ from selected action sites")
    if any(op.name == "fabric.resource_request" and "kind" in op.attributes and
           _attr_text(op.attributes["kind"]) == "ccz_state"
           for op in source_operations):
        raise ValueError("CCZ supply retained a generic resource request")


class BuildDefinition:
    """Typed read-only view of one top-level symbol in a Build's module.

    Handles are produced by :attr:`Build.definitions` and stay valid for the
    lifetime of the owning build; ``.op`` exposes the underlying MLIR
    operation of an inspection replay for advanced inspection.  Mutating that
    operation never changes the owning build's authenticated snapshot.
    """

    __slots__ = ("_build", "_operation", "_calls", "symbol", "kind")

    def __init__(self, build, symbol: str, kind: str, operation) -> None:
        self._build = build
        self._operation = operation
        self._calls = None
        self.symbol = symbol
        self.kind = kind

    @property
    def op(self):
        """The MLIR operation backing this definition (read-only)."""

        return self._operation

    def calls(self) -> tuple[str, ...]:
        """Callee symbols of every ``fabric.call`` in this definition's body,
        in program order (duplicates preserved)."""

        if self._calls is None:
            self._calls = tuple(
                _attr_text(operation.attributes["callee"])
                for operation in _walk_operation(self._operation)
                if operation.name == "fabric.call")
        return self._calls

    def __repr__(self) -> str:
        return f"<BuildDefinition @{self.symbol} ({self.kind})>"


class CallSite:
    """One ``fabric.call`` site inside a Build's linked module."""

    __slots__ = ("_operation", "callee", "parent_symbol")

    def __init__(self, operation, callee: str,
                 parent_symbol: str | None) -> None:
        self._operation = operation
        self.callee = callee
        self.parent_symbol = parent_symbol

    @property
    def op(self):
        """The MLIR ``fabric.call`` operation (read-only)."""

        return self._operation

    def __repr__(self) -> str:
        return f"<CallSite @{self.callee} in @{self.parent_symbol}>"


@dataclass(frozen=True, slots=True)
class BuildStatus:
    """Frozen inspection summary of a Build's verification state."""

    root: str
    stage: Any
    facets: tuple
    evidence_counts: tuple[tuple[str, int], ...]

    def count(self, result: str) -> int:
        """Number of evidence records with the given result (``pass``,
        ``fail``, ``unresolved``, ...)."""

        return dict(self.evidence_counts).get(result, 0)

    @property
    def total_evidence(self) -> int:
        return sum(count for _, count in self.evidence_counts)


@dataclass(frozen=True, slots=True)
class SynthesisSummary:
    """Inspectable logical gate-set legalization result."""

    gate_set: str
    precision: float
    h_count: int
    s_count: int
    t_count: int
    cx_count: int

    @property
    def clifford_count(self) -> int:
        return self.h_count + self.s_count + self.cx_count

    @property
    def gate_count(self) -> int:
        return self.clifford_count + self.t_count


class Build:
    """A frozen CUDA-Q Logical compilation artifact backed by serialized MLIR."""

    __slots__ = (
        "_context",
        "_module",
        "_snapshot",
        "_cache",
        "root",
        "profile",
        "stage",
        "facets",
        "evidence",
        "pipeline",
        "values",
        "placement",
        "qec_selection",
        "experiment",
        "source_modules",
        "_sealed",
    )

    def __setattr__(self, name, value) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("Build artifacts are immutable")
        object.__setattr__(self, name, value)

    def __init__(
            self,
            *,
            context,
            module,
            root: DefinitionHandle[Any],
            profile: str,
            facets=(),
            pipeline,
            evidence: Iterable[EvidenceRecord] = (),
            value_groups=None,
            placement=None,
            qec_selection=None,
            experiment=None,
            device=None,
            objective=None,
            source_modules=(),
            _facets_are_final=False,
    ) -> None:
        stage, legacy_facets = stage_and_facets(profile)
        initial_facets = normalize_facets(
            (*legacy_facets, *facets_for_kind(root.kind), *tuple(facets or ())))
        # A replay bundle commits the already-verified output facets. Replaying
        # a retained recipe here could reorder them after invalidation/recompute
        # passes and would amount to rerunning compilation during deserialization.
        normalized_facets = (initial_facets
                             if pipeline is None or _facets_are_final else
                             pipeline.apply_facets(initial_facets))
        self._declare_stage_facets(module, stage, normalized_facets)
        experiment = self._bind_experiment(
            module,
            root=root,
            profile=profile,
            stage=stage,
            facets=normalized_facets,
            pipeline=pipeline,
            experiment=experiment,
            device=device,
            placement=placement,
            objective=objective,
        )

        _logger.debug("Build from module: \n%s", module)

        if not module.operation.verify():
            raise ValueError("CUDA-Q Logical MLIR module failed verification")
        _verify_qec_selection_commitment(module, root, placement, qec_selection)
        _verify_ccz_resource_realization(module, root, placement, qec_selection)
        self._context = context
        self._module = module
        self._snapshot = str(module)
        self._cache = {}
        self.root = root
        self.profile = profile
        self.stage = stage
        self.facets = normalized_facets
        self.pipeline = pipeline
        self.evidence = tuple(evidence)
        self.values = ProgramValueSchema(root.symbol, value_groups or {})
        self.placement = placement
        self.qec_selection = qec_selection
        self.experiment = experiment
        self.source_modules = tuple(dict.fromkeys(source_modules))
        self._sealed = True

    @staticmethod
    def _declare_stage_facets(module, stage, facets) -> None:
        context = module.context

        def merge(name, values):
            existing = []
            if name in module.operation.attributes:
                existing = [
                    str(getattr(item, "value", item)).strip('"')
                    for item in module.operation.attributes[name]
                ]
            merged = tuple(dict.fromkeys((*existing, *values)))
            module.operation.attributes[name] = mlir_ir.ArrayAttr.get(
                [
                    mlir_ir.StringAttr.get(value, context=context)
                    for value in merged
                ],
                context=context,
            )

        if stage is not None:
            merge("qlx.stages", (stage.value,))
        merge("qlx.facets", tuple(facet.value for facet in facets))

    @staticmethod
    def _symbol(operation) -> str | None:
        try:
            value = operation.attributes["sym_name"]
        except KeyError:
            return None
        return str(getattr(value, "value", value)).strip('"')

    @classmethod
    def _definition_symbols(cls, module):
        return tuple(
            symbol for operation in module.body.operations
            if operation.operation.name != "qlx.experiment"
            if (symbol := cls._symbol(operation.operation)) is not None)

    @classmethod
    def _bind_experiment(
        cls,
        module,
        *,
        root,
        profile,
        stage,
        facets,
        pipeline,
        experiment,
        device,
        placement,
        objective,
    ):
        executable_kinds = {
            "program",
            "kernel",
            "gadget",
            "protocol",
        }
        # Experiment metadata belongs to the frozen Build/ExperimentBundle
        # envelope, not the executable semantic module. Strip manifests from
        # legacy/imported modules while preserving their Python value below.
        for operation in list(module.body.operations):
            if operation.operation.name == "qlx.experiment":
                operation.detach_from_parent()
        if experiment is None and root.kind not in executable_kinds:
            return None
        if experiment is None:
            experiment = Experiment(root=root)
        elif not isinstance(experiment, Experiment):
            raise TypeError(
                "Build experiment= must be a cudaq.logical.Experiment")

        closure = cls._definition_symbols(module)
        if root.symbol not in closure:
            raise ValueError(
                f"experiment root @{root.symbol} is missing from the linked module"
            )
        bound = experiment.bind(
            root=root,
            profile=profile,
            stage=stage,
            facets=facets,
            pipeline=pipeline,
            closure=closure,
            device=device,
            placement=placement,
            objective=objective,
        )
        return bound

    @property
    def module(self):
        """The linked MLIR module, parsed once from the frozen snapshot.

        The same module object is returned on every access, so repeated
        typed inspection never re-parses the textual snapshot. The build is
        immutable: treat this module as a read-only view. Compiler passes
        that need a scratch copy to mutate must use :meth:`_fresh_module`.

        A cheap top-level fingerprint defensively detects mutation through
        external or legacy compatibility code, drops every derived handle,
        and re-parses the pristine snapshot so the build stays observably
        immutable. Core target lowering always uses :meth:`_fresh_module`.
        """

        module = self._cache.get("module")
        if module is not None:
            fingerprint = tuple(
                view.operation.name for view in module.body.operations)
            if fingerprint != self._cache["fingerprint"]:
                self._cache.clear()
                module = None
        if module is None:
            context = mlir_ir.Context()
            module = mlir_ir.Module.parse(self._snapshot, context)
            self._cache["context"] = context
            self._cache["module"] = module
            self._cache["fingerprint"] = tuple(
                view.operation.name for view in module.body.operations)
        return module

    def _fresh_module(self):
        """Parse a private mutable copy of the snapshot in its own context.

        Internal compiler entry points lower against this replayed clone so
        the cached read-only :attr:`module` view is never mutated.
        """

        return mlir_ir.Module.parse(self._snapshot, mlir_ir.Context())

    def to_mlir(self) -> str:
        return self._snapshot

    @property
    def content_sha256(self) -> str:
        """Stable authenticated commitment of this complete Build bundle."""

        return self._bundle()["content_sha256"]

    @property
    def definitions(self):
        """Mapping of symbol name to :class:`BuildDefinition` handles.

        Covers every top-level symbol-bearing definition in the linked
        module (``fabric.code``, ``fabric.encoding``, ``fabric.gadget``,
        ``fabric.gadget_spec``, ``fabric.protocol``, ``qlx.qec_lowering``,
        ``qlx.program``, ``lvm.kernel``, and ``qlx.device``). Handles share
        one private inspection replay and are
        cached; mutating them cannot change the authenticated snapshot.
        """

        table = self._cache.get("definitions")
        if table is None:
            module = self._fresh_module()
            entries = {}
            for view in module.body.operations:
                operation = view.operation
                symbol = self._symbol(operation)
                if symbol is None:
                    continue
                entries[symbol] = BuildDefinition(self, symbol, operation.name,
                                                  operation)
            table = MappingProxyType(entries)
            # Keep the replay alive for the lifetime of its cached operation
            # handles.  It remains an inspection value, never an authority for
            # compiler consumers.
            self._cache["definitions_module"] = module
            self._cache["definitions"] = table
        return table

    def _call_sites(self) -> tuple[CallSite, ...]:
        sites = self._cache.get("call_sites")
        if sites is None:
            collected = []
            for symbol, handle in self.definitions.items():
                for operation in _walk_operation(handle.op):
                    if operation.name != "fabric.call":
                        continue
                    collected.append(
                        CallSite(
                            operation,
                            _attr_text(operation.attributes["callee"]),
                            symbol,
                        ))
            sites = tuple(collected)
            self._cache["call_sites"] = sites
        return sites

    def calls(self, symbol) -> tuple[CallSite, ...]:
        """All ``fabric.call`` sites targeting ``@symbol`` in the module.

        Accepts a plain symbol name or anything with a ``.symbol`` (a
        :class:`BuildDefinition` or a model ``DefinitionHandle``).  Each
        returned :class:`CallSite` carries the containing top-level
        definition as ``parent_symbol``.
        """

        name = getattr(symbol, "symbol", symbol)
        if not isinstance(name, str):
            raise TypeError(
                "Build.calls expects a symbol name or a definition handle")
        name = name.lstrip("@")
        return tuple(site for site in self._call_sites() if site.callee == name)

    def _objective_index(self):
        """Cached ``(handle, implemented-objective-names)`` pairs for every
        gadget/protocol in the module."""

        index = self._cache.get("objective_index")
        if index is not None:
            return index
        definitions = self.definitions

        declared: dict[str, frozenset[str]] = {}
        for symbol, handle in definitions.items():
            if handle.kind not in ("qlx.action", "qlx.instrument_decl"):
                continue
            names = {symbol}
            if symbol.startswith("qlx_standard_"):
                names.add(symbol[len("qlx_standard_"):])
            if "kind" in handle.op.attributes:
                kind = _attr_text(handle.op.attributes["kind"])
                # "composite" is a family marker, not an objective name; the
                # composite objective is addressed by its own symbol.
                if kind != "composite":
                    names.add(kind)
            declared[symbol] = frozenset(names)

        def reference_names(attribute) -> frozenset[str]:
            text = str(attribute)
            if text.startswith("#"):
                # Inline standard-objective attribute, e.g. #qlx.action<idle>.
                if "<" in text and text.endswith(">"):
                    return frozenset((text[text.index("<") + 1:-1],))
                return frozenset()
            symbol = _attr_text(attribute)
            return declared.get(symbol, frozenset((symbol,)))

        entries = []
        for handle in definitions.values():
            if handle.kind not in ("fabric.gadget", "fabric.protocol"):
                continue
            attributes = handle.op.attributes
            names: set[str] = set()
            if handle.kind == "fabric.gadget" and "spec" in attributes:
                # fabric.gadget -> gadget_spec -> fabric.objective -> logical
                spec = definitions.get(_attr_text(attributes["spec"]))
                if spec is not None and "objective" in spec.op.attributes:
                    target = definitions.get(
                        _attr_text(spec.op.attributes["objective"]))
                    if target is not None and "logical" in target.op.attributes:
                        names |= reference_names(
                            target.op.attributes["logical"])
            if handle.kind == "fabric.protocol" and "objective" in attributes:
                names |= reference_names(attributes["objective"])
            if "generated_by" in attributes:
                # ``qlx.qec_lowering`` provenance: a generated realization
                # implements its manifest's objective.
                manifest = definitions.get(
                    _attr_text(attributes["generated_by"]))
                if (manifest is not None and
                        manifest.kind == "qlx.qec_lowering" and
                        "objective" in manifest.op.attributes):
                    names |= reference_names(
                        manifest.op.attributes["objective"])
            if names:
                entries.append((handle, frozenset(names)))
        index = tuple(entries)
        self._cache["objective_index"] = index
        return index

    def protocol_for(self, objective):
        """The gadget/protocol definition that implements ``objective``.

        Accepts an objective name (``"idle"``), a logical reference with a
        ``.name`` (``cudaq.logical.std.idle``, an ``@cudaq.logical.objective`` definition),
        or a materialized handle with a ``.symbol``.  Resolution walks
        ``fabric.gadget_spec`` objective links, ``fabric.protocol``
        objective attributes, and ``qlx.qec_lowering`` ``generated_by``
        provenance.  Returns ``None`` when nothing implements the
        objective; when several candidates match, the build root wins if it
        is among them, otherwise a :class:`LookupError` reports the
        candidate set.
        """

        name = objective
        for attribute in ("name", "symbol"):
            value = getattr(objective, attribute, None)
            if isinstance(value, str):
                name = value
                break
        if not isinstance(name, str):
            raise TypeError(
                "protocol_for expects an objective name, a logical objective "
                "reference, or a definition handle")
        name = name.lstrip("@")
        matches = tuple(handle for handle, names in self._objective_index()
                        if name in names)
        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        for handle in matches:
            if handle.symbol == self.root.symbol:
                return handle
        candidates = ", ".join(
            f"@{handle.symbol} ({handle.kind})" for handle in matches)
        raise LookupError(
            f"objective {name!r} is ambiguous in this build: {candidates}")

    @property
    def status(self) -> BuildStatus:
        """Frozen summary: stage, facets, root, evidence counts by result."""

        status = self._cache.get("status")
        if status is None:
            counts: dict[str, int] = {}
            for record in self.evidence:
                counts[record.result] = counts.get(record.result, 0) + 1
            status = BuildStatus(
                root=self.root.symbol,
                stage=self.stage,
                facets=self.facets,
                evidence_counts=tuple(sorted(counts.items())),
            )
            self._cache["status"] = status
        return status

    @property
    def synthesis(self) -> SynthesisSummary | None:
        """Logical synthesis summary, or ``None`` for an unsynthesized Build."""

        summary = self._cache.get("synthesis", _UNSET)
        if summary is not _UNSET:
            return summary
        pass_spec = None
        if self.pipeline is not None:
            pass_spec = next(
                (item for item in self.pipeline.passes
                 if item.name == "qlx-synthesize-rotations"),
                None,
            )
        if pass_spec is None:
            self._cache["synthesis"] = None
            return None

        counts = {"h": 0, "s": 0, "t": 0, "cx": 0}
        module = self._fresh_module()
        for operation in _walk_operation(module.operation):
            if operation.name != "qlx.apply":
                continue
            action = str(operation.attributes["action"])
            for name in counts:
                if action == f"#qlx.action<{name}>":
                    counts[name] += 1
                    break
        options = dict(pass_spec.options)
        summary = SynthesisSummary(
            gate_set=str(options["gate_set"]),
            precision=float(options["precision"]),
            h_count=counts["h"],
            s_count=counts["s"],
            t_count=counts["t"],
            cx_count=counts["cx"],
        )
        self._cache["synthesis"] = summary
        return summary

    @property
    def patch_graph(self):
        """Typed P2 patch interaction/mapping view, or ``None``."""

        view = self._cache.get("patch_graph", _UNSET)
        if view is _UNSET:
            from .topology_view import PatchGraphView

            view = PatchGraphView.from_build(self)
            self._cache["patch_graph"] = view
        return view

    @staticmethod
    def _json_value(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, (tuple, list)):
            return [Build._json_value(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): Build._json_value(item) for key, item in value.items()
            }
        raise TypeError("build metadata is not replayable: "
                        f"{type(value).__module__}.{type(value).__qualname__}")

    def _bundle(self) -> dict[str, Any]:
        from .pipeline import PassSpec

        pipeline = None
        if self.pipeline is not None:
            pipeline = {
                "output_profile":
                    self.pipeline.output_profile,
                "passes": [{
                    "name": item.name,
                    "options": self._json_value(item.options),
                    "requires_facets": list(item.requires_facets),
                    "provides_facets": list(item.provides_facets),
                    "preserves_facets": list(item.preserves_facets),
                    "invalidates_facets": list(item.invalidates_facets),
                    "recomputes_facets": list(item.recomputes_facets),
                }
                           for item in self.pipeline.passes
                           if isinstance(item, PassSpec)],
            }
        placement = None
        if self.placement is not None:
            placement = self._json_value(asdict(self.placement))
        qec_selection = None
        if self.qec_selection is not None:
            qec_selection = self._json_value(asdict(self.qec_selection))
        value_groups = [{
            "allocation": group.allocation,
            "name": name,
            "count": group.count,
        } for name, group in self.values._groups.items()]
        kind = self.root.kind
        if not isinstance(kind, str):
            kind = f"{kind.__module__}.{kind.__qualname__}"
        bundle = {
            "schema":
                _BUILD_V2_SCHEMA,
            "model_version":
                _BUILD_V2_MODEL_VERSION,
            "ir_version":
                _BUILD_V2_IR_VERSION,
            "module":
                self._snapshot,
            "root": {
                "symbol": self.root.symbol,
                "kind": kind,
                "profile": self.root.profile,
            },
            "profile":
                self.profile,
            "stage":
                None if self.stage is None else self.stage.value,
            "facets": [facet.value for facet in self.facets],
            "pipeline":
                pipeline,
            "evidence": [asdict(item) for item in self.evidence],
            "value_groups":
                value_groups,
            "placement":
                placement,
            "qec_selection":
                qec_selection,
            "source_modules":
                list(self.source_modules),
            "experiment": (None if self.experiment is None else
                           self.experiment.to_bundle()),
        }
        bundle["content_sha256"] = _build_bundle_content_sha256(bundle)
        return bundle

    def serialize(self, path=None) -> bytes:
        """Serialize a self-describing, clean-process replay bundle.

        The module remains canonical textual MLIR in v2.  The envelope closes
        over the root, verified stage/facets, pass recipe, evidence, value schema,
        placement witness, and source-module provenance.  Its content
        commitment prevents optional provenance and detached witnesses from
        being erased to reclassify compiler output as direct-authored input.
        """

        payload = json.dumps(self._bundle(),
                             sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        if path is not None:
            Path(path).write_bytes(payload)
        return payload

    @classmethod
    def replay(
        cls,
        payload,
        *,
        root: DefinitionHandle[Any] | None = _UNSET,
        profile: str | None = _UNSET,
        pipeline=_UNSET,
        evidence: Iterable[EvidenceRecord] | None = _UNSET,
        value_groups=_UNSET,
        placement=_UNSET,
        qec_selection=_UNSET,
        source_modules=_UNSET,
        experiment=_UNSET,
    ) -> "Build":
        from ..architecture.constraints import (
            PlacementBinding,
            PlacementWitness,
        )
        from ..codes import (
            QECActionSelection,
            QECBlockBinding,
            QECBlockOwner,
            QECSelectionWitness,
        )
        from .pipeline import PassSpec, Pipeline

        if isinstance(payload, (str, Path)):
            payload = Path(payload).read_bytes()
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            raise TypeError(
                "Build.replay expects bundle bytes or a filesystem path")
        raw = bytes(payload)
        bundle = None
        try:
            candidate = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            candidate = None
        if isinstance(candidate,
                      dict) and candidate.get("schema") == "qlx.build/v1":
            raise ValueError(
                "qlx.build/v1 lacks the required replay content commitment; "
                "reserialize the trusted Build as qlx.build/v2")
        if isinstance(candidate,
                      dict) and candidate.get("schema") == _BUILD_V2_SCHEMA:
            bundle = candidate
            override_values = {
                "root": root,
                "profile": profile,
                "pipeline": pipeline,
                "evidence": evidence,
                "value_groups": value_groups,
                "placement": placement,
                "qec_selection": qec_selection,
                "source_modules": source_modules,
                "experiment": experiment,
            }
            overrides = sorted(name for name, value in override_values.items()
                               if value is not _UNSET)
            if overrides:
                raise ValueError(
                    "qlx.build/v2 does not accept replay overrides: " +
                    ", ".join(overrides))
            _validate_v2_bundle(bundle)
            root_data = bundle["root"]
            root = DefinitionHandle(
                symbol=root_data["symbol"],
                kind=root_data["kind"],
                profile=root_data["profile"],
            )
            profile = bundle["profile"]
            replay_facets = tuple(bundle["facets"])
            pipeline = None
            placement = None
            qec_selection = None
            experiment = None
            if bundle["pipeline"] is not None:
                pipeline_data = bundle["pipeline"]
                pipeline = Pipeline(
                    passes=tuple(
                        PassSpec(
                            item["name"],
                            tuple((str(key), cls._restore_json(value))
                                  for key, value in item.get("options", ())),
                            tuple(item.get("requires_facets", ())),
                            tuple(item.get("provides_facets", ())),
                            tuple(item.get("preserves_facets", ())),
                            tuple(item.get("invalidates_facets", ())),
                            tuple(item.get("recomputes_facets", ())),
                        )
                        for item in pipeline_data["passes"]),
                    output_profile=pipeline_data["output_profile"],
                )
            evidence = tuple(
                EvidenceRecord(
                    kind=item["kind"],
                    producer=item["producer"],
                    result=item["result"],
                    obligations=tuple(item.get("obligations", ())),
                    assumptions=tuple(item.get("assumptions", ())),
                ) for item in bundle["evidence"])
            value_groups = bundle["value_groups"]
            if bundle["placement"] is not None:
                item = bundle["placement"]
                placement = PlacementWitness(
                    machine=item["machine"],
                    input_p0=item["input_p0"],
                    bindings=tuple(
                        PlacementBinding(
                            placement=binding["placement"],
                            space=binding["space"],
                            slot=binding["slot"],
                            source_allocation=binding.get("source_allocation"),
                            source_group=binding.get("source_group"),
                            source_path=tuple(binding.get("source_path", ())),
                            binding_kind=binding.get("binding_kind", "local"),
                            binding_data=tuple(
                                (str(key), cls._restore_json(value))
                                for key, value in binding.get(
                                    "binding_data", ())),
                        )
                        for binding in item.get("bindings", ())),
                    relaxed_preferences=tuple(
                        item.get("relaxed_preferences", ())),
                    objective=item.get("objective", "first_fit"),
                    tie_break=item.get("tie_break", "declaration_order"),
                )
            if bundle["qec_selection"] is not None:
                item = bundle["qec_selection"]
                qec_selection = QECSelectionWitness(
                    input_p1=item["input_p1"],
                    blocks=tuple(
                        QECBlockBinding(
                            block=block["block"],
                            space=block["space"],
                            code=block["code"],
                            encoding=block["encoding"],
                            logical_capacity=block["logical_capacity"],
                            owners=tuple(
                                QECBlockOwner(
                                    placement=owner["placement"],
                                    logical_index=owner["logical_index"],
                                    source_allocation=owner.get(
                                        "source_allocation"),
                                    source_group=owner.get("source_group"),
                                    source_path=tuple(
                                        owner.get("source_path", ())),
                                ) for owner in block.get("owners", ())),
                        ) for block in item.get("blocks", ())),
                    actions=tuple(
                        QECActionSelection(
                            site=action["site"],
                            kind=action["kind"],
                            objective=action["objective"],
                            placements=tuple(action.get("placements", ())),
                            feasible_candidates=tuple(
                                action.get("feasible_candidates", ())),
                            selected=action["selected"],
                            provider=action.get("provider", "fixed"),
                            version=action.get("version", "linked"),
                            manifest_sha256=action.get("manifest_sha256"),
                            tie_break=action.get(
                                "tie_break",
                                "fixed-before-generated-then-symbol-order",
                            ),
                        ) for action in item.get("actions", ())),
                    code=item.get("code"),
                    encoding=item.get("encoding"),
                    objective=item.get("objective",
                                       "policy_then_device_then_candidate"),
                    tie_break=item.get("tie_break", "declaration_order"),
                )
            source_modules = tuple(bundle["source_modules"])
            if bundle["experiment"] is not None:
                experiment = Experiment.from_bundle(bundle["experiment"])
            if experiment is not None and experiment.root != root:
                raise ValueError(
                    "replay bundle experiment root does not match its Build root"
                )
            module_text = bundle["module"]
        elif isinstance(candidate, dict):
            raise ValueError("unsupported Build replay bundle schema")
        else:
            # Compatibility with the initial text-only prototype.  Text-only
            # payloads cannot infer root/profile metadata and therefore require
            # the old explicit arguments.
            if root is _UNSET or root is None or profile is _UNSET or profile is None:
                raise ValueError(
                    "legacy textual MLIR replay requires root= and profile=")
            pipeline = None if pipeline is _UNSET else pipeline
            evidence = None if evidence is _UNSET else evidence
            value_groups = None if value_groups is _UNSET else value_groups
            placement = None if placement is _UNSET else placement
            qec_selection = (None if qec_selection is _UNSET else qec_selection)
            source_modules = (None
                              if source_modules is _UNSET else source_modules)
            experiment = None if experiment is _UNSET else experiment
            module_text = raw.decode("utf-8")

        if root is _UNSET or root is None or profile is _UNSET or profile is None:
            raise ValueError("replay bundle is missing root/profile metadata")
        context = mlir_ir.Context()
        module = mlir_ir.Module.parse(module_text, context)
        if bundle is not None:
            expected_operation = _root_operation_name(root.kind)
            matches = tuple(operation.operation
                            for operation in module.body.operations
                            if cls._symbol(operation.operation) == root.symbol)
            if len(matches) != 1:
                raise ValueError(
                    "qlx.build/v2 selected MLIR root is missing or ambiguous")
            if matches[0].name != expected_operation:
                raise ValueError(
                    "qlx.build/v2 root kind differs from the selected MLIR root"
                )
            _validate_v2_module_classification(module, matches[0], bundle)
        result = cls(
            context=context,
            module=module,
            root=root,
            profile=profile,
            facets=(replay_facets if bundle is not None else ()),
            pipeline=pipeline,
            evidence=() if evidence is None else evidence,
            value_groups=value_groups,
            placement=placement,
            qec_selection=qec_selection,
            source_modules=() if source_modules is None else source_modules,
            experiment=experiment,
            _facets_are_final=bundle is not None,
        )
        if bundle is not None:
            actual_stage = None if result.stage is None else result.stage.value
            if actual_stage != bundle["stage"]:
                raise ValueError(
                    "qlx.build/v2 reconstructed stage differs from the bundle")
            if [facet.value for facet in result.facets] != bundle["facets"]:
                raise ValueError(
                    "qlx.build/v2 reconstructed facets differ from the bundle")
            reconstructed_experiment = (None if result.experiment is None else
                                        result.experiment.to_bundle())
            if reconstructed_experiment != bundle["experiment"]:
                raise ValueError(
                    "qlx.build/v2 reconstructed experiment differs from "
                    "the authenticated experiment metadata")
        return result

    @staticmethod
    def _restore_json(value):
        if isinstance(value, list):
            return tuple(Build._restore_json(item) for item in value)
        if isinstance(value, dict):
            return {
                key: Build._restore_json(item) for key, item in value.items()
            }
        return value

    def verify(self) -> bool:
        return bool(self._fresh_module().operation.verify())

    def __str__(self) -> str:
        return self._snapshot
