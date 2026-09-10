# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable
import weakref

import cudaq.mlir.ir as mlir_ir

from cudaq.logical.programs.definition import DefinitionHandle
from cudaq.logical.experiments.definition import Experiment
from cudaq.logical.architecture.logical import ProgramValueSchema
from cudaq.logical.stages import (
    facets_for_kind,
    normalize_facets,
    stage_and_facets,
)
from .build_bundle import (
    _BUILD_V2_IR_VERSION,
    _BUILD_V2_MODEL_VERSION,
    _BUILD_V2_SCHEMA,
    _FACET_MINIMUM_STAGE,
    _STAGE_ORDINAL,
    _build_bundle_content_sha256,
    _root_operation_name,
    _root_profiles,
    _validate_v2_bundle,
    _validate_v2_nested_metadata,
)

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
    "patch_mapping":
        frozenset({"fabric.patch_mapping"}),
    "carrier_mapping":
        frozenset({"phys.mapping", "phys.allocation_mapping"}),
    "physical_routing":
        frozenset({"phys.routing"}),
    "zoned_movement":
        frozenset({"phys.move"}),
    "physical_schedule":
        frozenset({"phys.schedule"}),
}
_ROOT_SCOPED_FACETS = frozenset({
    "carrier_mapping",
    "physical_routing",
    "zoned_movement",
    "physical_schedule",
})


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

    def dataclass_fields(value):
        if not is_dataclass(value):
            raise TypeError(
                "QEC selection commitment contains a non-JSON value "
                f"{type(value).__module__}.{type(value).__qualname__}")
        return {
            field.name: getattr(value, field.name) for field in fields(value)
        }

    # ``dataclasses.asdict`` recursively duplicates the complete witness before
    # JSON encoding.  For a paper-scale P2 selection that means copying hundreds
    # of thousands of immutable action rows.  JSON's default hook visits the
    # same dataclass fields lazily and yields exactly the prior canonical byte
    # stream without the second full Python object graph.
    payload = json.dumps(
        qec_selection,
        default=dataclass_fields,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def _qec_network_source_sha256(
    module,
    input_p1: str,
    placement,
    qec_selection,
) -> str:
    """Commit the verifier-closed lineage from P0 through QEC selection.

    The digest is deliberately reconstructible from a retained P2/P3 module.
    It does not depend on a transient ``Build.serialize()`` envelope, so replay
    can independently authenticate a provider request against the exact P0
    program, P1 kernel, and placement witness.  The QEC selection is committed
    separately by ``qec_selection_sha256`` and cross-checked against the
    request's typed action/block ownership.
    """

    prefix = "QEC network source commitment is not reconstructible"
    if not isinstance(input_p1, str) or not input_p1:
        raise ValueError(f"{prefix}: input P1 identity is missing")
    if qec_selection is None or qec_selection.input_p1 != input_p1:
        raise ValueError(f"{prefix}: QEC selection input P1 differs")
    kernel = _top_level_symbol(module, input_p1, {"lvm.kernel"}, prefix)
    input_p0 = _verify_placement_witness(kernel, placement, prefix)
    program = _top_level_symbol(module, input_p0, {"qlx.program"}, prefix)
    fields = (
        ("schema", b"qlx.qec_network.source/v2"),
        ("input_p1", input_p1.encode("utf-8")),
        ("input_p0", input_p0.encode("utf-8")),
        (
            "p1",
            kernel.get_asm(assume_verified=True).encode("utf-8"),
        ),
        (
            "p0",
            program.get_asm(assume_verified=True).encode("utf-8"),
        ),
        ("placement", _placement_witness_sha256(placement).encode("ascii")),
    )
    digest = sha256()
    for label, payload in fields:
        label_bytes = label.encode("ascii")
        digest.update(len(label_bytes).to_bytes(4, "big"))
        digest.update(label_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def _qec_network_region_runs(
    module,
    input_p1: str,
    selected_sites,
) -> tuple[tuple[str, ...], ...]:
    """Re-derive maximal top-level selected-action runs from retained P1."""

    prefix = "QEC network regions are not reconstructible"
    kernel = _top_level_symbol(module, input_p1, {"lvm.kernel"}, prefix)
    selected = frozenset(selected_sites)
    runs = []
    current = []
    block = kernel.regions[0].blocks[0]
    for operation_view in block.operations:
        operation = operation_view.operation
        symbol = (f"site{int(operation.attributes['site'])}"
                  if "site" in operation.attributes else None)
        if symbol not in selected:
            if current:
                runs.append(tuple(current))
                current = []
            continue
        current.append(symbol)
    if current:
        runs.append(tuple(current))
    covered = tuple(site for run in runs for site in run)
    if sorted(covered) != sorted(selected) or len(covered) != len(selected):
        raise ValueError(
            f"{prefix}: selected action is nested, missing, or duplicated")
    return tuple(runs)


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

    if "physical" in operation.attributes:
        return "p3"
    if "qec" in operation.attributes:
        return "p2"
    return "p1"


def _selected_physical_graph_symbol(root) -> str | None:
    if root.name == "phys.graph":
        return _attr_text(root.attributes["sym_name"])
    for attribute in ("graph", "physical_graph"):
        if attribute in root.attributes:
            return _attr_text(root.attributes[attribute])
    return None


def _selected_schedule_operations(module, root):
    graph_symbol = _selected_physical_graph_symbol(root)
    if graph_symbol is None:
        return ()
    explicit = None
    for attribute in ("schedule", "physical_schedule"):
        if attribute in root.attributes:
            explicit = _attr_text(root.attributes[attribute])
            break
    # A non-graph P3 artifact owns schedule evidence only through its explicit
    # schedule reference. Merely linking another schedule for the same graph
    # does not classify the selected root.
    if root.name != "phys.graph" and explicit is None:
        return ()
    return tuple(
        operation for operation in _walk_operation(module.operation)
        if operation.name == "phys.schedule" and "graph" in operation.attributes
        and _attr_text(operation.attributes["graph"]) == graph_symbol and
        (explicit is None or
         _attr_text(operation.attributes["sym_name"]) == explicit))


def _root_scoped_operation_names(module, root) -> tuple[str, ...]:
    graph_symbol = _selected_physical_graph_symbol(root)
    schedule_symbols = {
        _attr_text(operation.attributes["sym_name"])
        for operation in _selected_schedule_operations(module, root)
    }
    scoped_symbols = {_attr_text(root.attributes["sym_name"])}
    if graph_symbol is not None:
        scoped_symbols.add(graph_symbol)
    scoped_symbols.update(schedule_symbols)
    names = [operation.name for operation in _walk_operation(root)]
    if graph_symbol is not None:
        graph = next(
            (operation for operation in _walk_operation(module.operation)
             if operation.name == "phys.graph" and
             _attr_text(operation.attributes["sym_name"]) == graph_symbol),
            None,
        )
        if graph is not None and graph is not root:
            names.extend(operation.name for operation in _walk_operation(graph))
    for operation in _walk_operation(module.operation):
        if operation is root:
            continue
        if (operation.name == "phys.schedule" and _attr_text(
                operation.attributes["sym_name"]) not in schedule_symbols):
            continue
        for attribute in (
                "graph",
                "source_graph",
                "source_protocol",
                "protocol",
        ):
            if (attribute in operation.attributes and _attr_text(
                    operation.attributes[attribute]) in scoped_symbols):
                names.append(operation.name)
                break
    return tuple(names)


def _root_scoped_facet_names(module, root) -> tuple[str, ...]:
    operation_names = set(_root_scoped_operation_names(module, root))
    return tuple(
        facet for facet, witnesses in _FACET_WITNESS_OPERATIONS.items()
        if facet in _ROOT_SCOPED_FACETS and
        not witnesses.isdisjoint(operation_names))


def _symbol_index(candidates):
    """Index symbol definitions once for replay-link authentication."""

    result = {}
    for operation in candidates:
        if "sym_name" not in operation.attributes:
            continue
        symbol = _attr_text(operation.attributes["sym_name"])
        result.setdefault(symbol, []).append(operation)
    return result


def _require_unique_link(candidates, symbol: str, kinds: frozenset[str],
                         label: str):
    definitions = (candidates.get(symbol,
                                  ()) if isinstance(candidates, dict) else
                   (operation for operation in candidates
                    if "sym_name" in operation.attributes and
                    _attr_text(operation.attributes["sym_name"]) == symbol))
    matches = tuple(
        operation for operation in definitions if operation.name in kinds)
    if len(matches) != 1:
        expected = " or ".join(sorted(kinds))
        raise ValueError(
            f"qlx.build/v2 selected P3 {label} @{symbol} must resolve "
            f"uniquely to {expected}")
    return matches[0]


def _state_resource_symbol(type_) -> str | None:
    text = str(type_)
    prefix = "!phys.state<@"
    if not text.startswith(prefix) or not text.endswith(">"):
        return None
    return text[len(prefix):-1].split("::@")[-1]


def _validate_v2_retained_p3_links(module) -> None:
    """Require closure for every retained compiler-produced P3 authority."""

    top_level = tuple(view.operation for view in module.body.operations)
    top_level_index = _symbol_index(top_level)

    # Standalone MLIR libraries may carry unresolved physical provenance
    # symbols, but a replayable compiler-produced Build is a closed artifact.
    # This closure is independent of the selected root kind and declared Build
    # stage: retained sidecars and projection maps must never become unchecked
    # merely because replay selects an earlier-stage or non-graph root.
    for projection in (operation for operation in top_level
                       if operation.name == "phys.record_projection"):
        _require_unique_link(
            top_level_index,
            _attr_text(projection.attributes["graph"]),
            frozenset({"phys.graph"}),
            "record projection graph",
        )
        _require_unique_link(
            top_level_index,
            _attr_text(projection.attributes["source_protocol"]),
            frozenset({"fabric.gadget", "fabric.protocol"}),
            "record projection source protocol",
        )

    sidecar_kinds = frozenset({"phys.selection_sidecar"})
    for sidecar in (operation for operation in top_level
                    if operation.name in sidecar_kinds):
        sidecar_graph = _attr_text(sidecar.attributes["graph"])
        _require_unique_link(
            top_level_index,
            sidecar_graph,
            frozenset({"phys.graph"}),
            "sidecar graph",
        )
        if "source_profile" not in sidecar.attributes:
            continue
        if ("record_projection" not in sidecar.attributes or
                "projection_indices" not in sidecar.attributes):
            raise ValueError(
                "qlx.build/v2 selected P3 sidecar source_profile requires "
                "record_projection and projection_indices")
        projection = _require_unique_link(
            top_level_index,
            _attr_text(sidecar.attributes["record_projection"]),
            frozenset({"phys.record_projection"}),
            "sidecar record projection",
        )
        if _attr_text(projection.attributes["graph"]) != sidecar_graph:
            raise ValueError(
                "qlx.build/v2 selected P3 sidecar record projection must "
                "reference the same physical graph")
        source_kind = _attr_text(sidecar.attributes["source_kind"])
        if source_kind not in {"profile", "outcome_map"}:
            raise ValueError(
                "qlx.build/v2 selected P3 sidecar source_kind must be "
                "profile or outcome_map")
        profile = _require_unique_link(
            top_level_index,
            _attr_text(sidecar.attributes["source_profile"]),
            frozenset({"fabric.gadget_profile"}),
            "sidecar source profile",
        )
        gadget = _require_unique_link(
            top_level_index,
            _attr_text(profile.attributes["gadget"]),
            frozenset({"fabric.gadget"}),
            "sidecar source gadget",
        )
        if "spec" in gadget.attributes:
            _require_unique_link(
                top_level_index,
                _attr_text(gadget.attributes["spec"]),
                frozenset({"fabric.gadget_spec"}),
                "sidecar source gadget spec",
            )
        elif source_kind == "outcome_map":
            raise ValueError(
                "qlx.build/v2 selected P3 outcome_map sidecar source gadget "
                "must retain a GadgetSpec")


def _validate_v2_selected_p3_links(module, root, bundle) -> None:
    """Require complete links for the selected compiler-produced P3 graph."""

    top_level = tuple(view.operation for view in module.body.operations)
    top_level_index = _symbol_index(top_level)
    graph_symbol = _selected_physical_graph_symbol(root)
    if graph_symbol is None:
        return
    graph = _require_unique_link(top_level_index, graph_symbol,
                                 frozenset({"phys.graph"}), "graph")
    architecture_symbol = _attr_text(graph.attributes["architecture"])
    architecture = _require_unique_link(
        top_level_index,
        architecture_symbol,
        frozenset({"phys.machine"}),
        "architecture",
    )
    architecture_symbols = tuple(_walk_operation(architecture))
    architecture_index = _symbol_index(architecture_symbols)
    has_concrete_action_closure = any(
        operation.name == "phys.action" for operation in top_level)
    has_concrete_instrument_closure = any(
        operation.name == "phys.instrument" for operation in top_level)
    selection = bundle["qec_selection"]
    has_interconnect_actions = (selection is not None and any(
        action["channel"] is not None for action in selection["actions"]))
    require_action_definitions = (has_concrete_action_closure or
                                  has_interconnect_actions)
    require_instrument_definitions = (has_concrete_instrument_closure or
                                      has_interconnect_actions)

    if "source_protocol" in graph.attributes:
        _require_unique_link(
            top_level_index,
            _attr_text(graph.attributes["source_protocol"]),
            frozenset({"fabric.gadget", "fabric.protocol"}),
            "source protocol",
        )

    resource_symbols = set()
    for operation in _walk_operation(graph):
        for value in (*operation.operands, *operation.results):
            symbol = _state_resource_symbol(value.type)
            if symbol is not None:
                resource_symbols.add(symbol)
        for region in operation.regions:
            for block in region.blocks:
                for argument in block.arguments:
                    symbol = _state_resource_symbol(argument.type)
                    if symbol is not None:
                        resource_symbols.add(symbol)

        if operation.name == "phys.apply":
            action_symbol = _attr_text(operation.attributes["action"])
            action_definitions = tuple(
                candidate
                for candidate in top_level_index.get(action_symbol, ())
                if candidate.name == "phys.action")
            if require_action_definitions or action_definitions:
                _require_unique_link(
                    top_level_index,
                    action_symbol,
                    frozenset({"phys.action"}),
                    "action",
                )
            if "topology" in operation.attributes:
                _require_unique_link(
                    architecture_index,
                    _attr_text(operation.attributes["topology"]),
                    frozenset({"phys.topology"}),
                    "action topology",
                )
        elif operation.name == "phys.measure":
            instrument_symbol = _attr_text(operation.attributes["measurement"])
            instrument_definitions = tuple(
                candidate
                for candidate in top_level_index.get(instrument_symbol, ())
                if candidate.name == "phys.instrument")
            if require_instrument_definitions or instrument_definitions:
                _require_unique_link(
                    top_level_index,
                    instrument_symbol,
                    frozenset({"phys.instrument"}),
                    "measurement instrument",
                )
        elif operation.name == "phys.measure_product":
            instrument_symbol = _attr_text(operation.attributes["instrument"])
            instrument_definitions = tuple(
                candidate
                for candidate in top_level_index.get(instrument_symbol, ())
                if candidate.name == "phys.instrument")
            if require_instrument_definitions or instrument_definitions:
                _require_unique_link(
                    top_level_index,
                    instrument_symbol,
                    frozenset({"phys.instrument"}),
                    "product-measurement instrument",
                )
        elif operation.name == "phys.call":
            _require_unique_link(
                top_level_index,
                _attr_text(operation.attributes["callee"]),
                frozenset({"fabric.gadget", "fabric.protocol"}),
                "call target",
            )
            if "profile" in operation.attributes:
                _require_unique_link(
                    top_level_index,
                    _attr_text(operation.attributes["profile"]),
                    frozenset({"fabric.gadget_profile"}),
                    "call profile",
                )
        elif operation.name == "phys.move":
            _require_unique_link(
                architecture_index,
                _attr_text(operation.attributes["route"]),
                frozenset({"phys.topology"}),
                "movement route",
            )

    for symbol in resource_symbols:
        resource = _require_unique_link(
            top_level_index,
            symbol,
            frozenset({"phys.resource"}),
            "resource",
        )
        _require_unique_link(
            architecture_index,
            _attr_text(resource.attributes["resource_class"]),
            frozenset({"phys.resource_class"}),
            "resource class",
        )

    mappings = tuple(
        operation for operation in top_level
        if (operation.name == "phys.mapping" and
            _attr_text(operation.attributes["graph"]) == graph_symbol))
    if len(mappings) > 1 or ("source_protocol" in graph.attributes and
                             len(mappings) != 1):
        raise ValueError(
            f"qlx.build/v2 selected P3 graph @{graph_symbol} must have "
            "unique patch-graph mapping provenance")
    for mapping in mappings:
        _require_unique_link(
            top_level_index,
            _attr_text(mapping.attributes["source_graph"]),
            frozenset({"fabric.patch_graph"}),
            "source patch graph",
        )

    for operation in top_level:
        if (operation.name == "phys.routing" and
                _attr_text(operation.attributes["graph"]) == graph_symbol):
            _require_unique_link(
                architecture_index,
                _attr_text(operation.attributes["topology"]),
                frozenset({"phys.topology"}),
                "routing topology",
            )


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
    _validate_v2_retained_p3_links(module)
    if bundle["stage"] == "p3":
        _validate_v2_selected_p3_links(module, root, bundle)

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
    root_operation_names = set(_root_scoped_operation_names(module, root))
    schedules = _selected_schedule_operations(module, root)
    if len(schedules) > 1:
        raise ValueError(
            "qlx.build/v2 selected root has ambiguous physical schedules")
    if "physical_schedule" in bundle["facets"] and len(schedules) != 1:
        raise ValueError(
            "qlx.build/v2 physical_schedule facet is not established by "
            "one selected-root schedule")
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

    missing_root_facets = [
        facet for facet in _root_scoped_facet_names(module, root)
        if facet not in bundle["facets"]
    ]
    if missing_root_facets:
        raise ValueError("qlx.build/v2 facets omit retained root facets: " +
                         ", ".join(sorted(missing_root_facets)))

    for facet in bundle["facets"]:
        witnesses = _FACET_WITNESS_OPERATIONS.get(facet)
        requires_retained_witness = (facet == "physical_schedule" or
                                     facet not in pipeline_facets)
        operation_names = (root_operation_names if facet in _ROOT_SCOPED_FACETS
                           else module_operation_names)
        if (witnesses is not None and requires_retained_witness and
                witnesses.isdisjoint(operation_names)):
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
    witnessed = {
        binding.placement: binding
        for binding in placement.bindings
        if binding.source_group != "argument"
    }
    if len(witnessed) != sum(binding.source_group != "argument"
                             for binding in placement.bindings):
        raise ValueError(f"{prefix}: placement ownership is ambiguous")
    retained = set()
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
        owner = _attr_text(operation.attributes["placement_owner"])
        binding = witnessed.get(owner)
        if binding is None or owner in retained:
            raise ValueError(f"{prefix}: placement ownership facts differ")
        # Local owners refer directly to their space, while distributed,
        # trajectory, and topological owners refer to their explicit
        # lvm.placement symbol. Both forms remain tied to the same detached
        # binding by the compiler-derived owner identity.
        if (_symbol_path(operation.attributes["at"])[-1]
                not in {binding.space, binding.placement} or
                int(operation.attributes["placement_slot"]) != binding.slot or
            (None if allocation < 0 else allocation)
                != binding.source_allocation or
            (None if not group else group) != binding.source_group or tuple(
                int(value) for value in operation.attributes["source_path"])
                != binding.source_path):
            raise ValueError(f"{prefix}: placement ownership facts differ")
        retained.add(owner)
    if retained != set(witnessed):
        raise ValueError(f"{prefix}: placement ownership facts differ")
    return input_p0


def _p0_mpp_action_rows(program, placement, prefix: str):
    """Derive exact MPP ownership without trusting retained P1/P2 markers."""

    bindings = {binding.placement: binding for binding in placement.bindings}
    by_source = {
        (binding.source_allocation, tuple(binding.source_path)):
            binding.placement
        for binding in placement.bindings
        if binding.source_allocation is not None
    }
    owner_by_value = {}
    block = program.regions[0].blocks[0]
    unallocated = iter(binding.placement
                       for binding in placement.bindings
                       if binding.source_allocation is None)
    for argument in block.arguments:
        if str(argument.type) == "!qlx.logical_qubit":
            try:
                owner_by_value[argument] = next(unallocated)
            except StopIteration as error:
                raise ValueError(
                    f"{prefix}: retained P0 input ownership is incomplete"
                ) from error

    active = set()

    def owner_of(value):
        known = owner_by_value.get(value)
        if known is not None:
            return known
        if value in active:
            raise ValueError(f"{prefix}: retained P0 owner lineage is cyclic")
        active.add(value)
        try:
            operation = value.owner
            name = operation.name
            if name == "qlx.prepare":
                try:
                    allocation = int(operation.attributes["allocation"])
                    index = int(operation.attributes["value_index"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"{prefix}: retained P0 preparation provenance is incomplete"
                    ) from error
                owner = by_source.get((allocation, (index,)))
            elif name in {"qlx.apply", "qlx.instrument", "qlx.idle"}:
                results = tuple(result for result in operation.results
                                if str(result.type) == "!qlx.logical_qubit")
                inputs = tuple(operand for operand in operation.operands
                               if str(operand.type) == "!qlx.logical_qubit")
                owner = owner_of(inputs[results.index(value)])
            elif name == "qlx.consume_resource":
                index = tuple(operation.results).index(value)
                owner = owner_of(tuple(operation.operands)[index + 1])
            elif name == "cflow.if":
                index = tuple(operation.results).index(value)
                yielded = tuple(
                    region.blocks[0].operations[-1].operation.operands[index]
                    for region in operation.regions)
                owners = tuple(owner_of(item) for item in yielded)
                owner = owners[0] if owners and len(set(owners)) == 1 else None
            elif name in {"cflow.repeat", "cflow.while"}:
                index = tuple(operation.results).index(value)
                owner = owner_of(tuple(operation.operands)[index])
            else:
                owner = None
            if owner is None or owner not in bindings:
                raise ValueError(
                    f"{prefix}: retained P0 logical owner is not in the "
                    "placement witness")
            owner_by_value[value] = owner
            return owner
        finally:
            active.remove(value)

    site_operations = {
        "qlx.prepare",
        "qlx.measure",
        "qlx.instrument",
        "qlx.apply",
        "qlx.consume_resource",
    }
    site = 0
    remote_sites = set()
    rows = {}
    for operation in _walk_operation(program):
        if operation.name not in site_operations:
            continue
        current_site = site
        site += 1
        if (operation.name != "qlx.instrument" or
                "instrument" not in operation.attributes or
                _objective_text(operation.attributes["instrument"]) != "mpp"):
            continue
        quantum_inputs = tuple(operand for operand in operation.operands
                               if str(operand.type) == "!qlx.logical_qubit")
        owners = tuple(owner_of(value) for value in quantum_inputs)
        site_name = f"site{current_site}"
        rows[site_name] = {
            "kind": "instrument",
            "objective": "mpp",
            "placements": owners,
        }
        spaces = {bindings[owner].space for owner in owners}
        if len(spaces) > 1:
            remote_sites.add(site_name)
    return rows, remote_sites


def _verify_qec_network_witness(
    program,
    placement,
    qec_selection,
    *,
    accepted_encodings,
    prefix: str,
) -> set[str]:
    """Authenticate network block/action rows against P0/P1 and exact codes."""

    if placement is None:
        raise ValueError(f"{prefix}: placement witness is missing")
    bindings = {binding.placement: binding for binding in placement.bindings}
    if len(bindings) != len(placement.bindings):
        raise ValueError(f"{prefix}: placement owners are ambiguous")
    if qec_selection.code is None or qec_selection.encoding is None:
        raise ValueError(f"{prefix}: network code and encoding are missing")

    accepted_encodings = tuple(accepted_encodings)
    accepted = {
        (code, encoding): capacity
        for code, encoding, capacity in accepted_encodings
    }
    if len(accepted) != len(accepted_encodings):
        raise ValueError(
            f"{prefix}: accepted code/encoding identities are ambiguous")
    selected_identity = (
        qec_selection.code,
        qec_selection.encoding,
    )
    if selected_identity not in accepted:
        raise ValueError(
            f"{prefix}: selected code/encoding is not accepted by the manifest")
    selected_capacity = accepted[selected_identity]

    witnessed_owners = {}
    block_names = set()
    for block in qec_selection.blocks:
        if block.block in block_names:
            raise ValueError(f"{prefix}: QEC block identity is duplicated")
        block_names.add(block.block)
        if (block.code != qec_selection.code or
                block.encoding != qec_selection.encoding or
                block.logical_capacity != selected_capacity):
            raise ValueError(
                f"{prefix}: QEC block differs from the selected encoding")
        if not block.owners or len(block.owners) > block.logical_capacity:
            raise ValueError(f"{prefix}: QEC block owner capacity is invalid")
        logical_indices = tuple(owner.logical_index for owner in block.owners)
        if len(set(logical_indices)) != len(logical_indices):
            raise ValueError(
                f"{prefix}: QEC block logical ports are duplicated")
        for owner in block.owners:
            binding = bindings.get(owner.placement)
            if binding is None or owner.placement in witnessed_owners:
                raise ValueError(
                    f"{prefix}: QEC owner is missing or duplicated")
            if block.space != binding.space:
                raise ValueError(f"{prefix}: QEC owner space differs")
            if (owner.source_allocation != binding.source_allocation or
                    owner.source_group != binding.source_group or
                    tuple(owner.source_path) != tuple(binding.source_path)):
                raise ValueError(f"{prefix}: QEC owner source lineage differs")
            if (owner.logical_index < 0 or
                    owner.logical_index >= block.logical_capacity):
                raise ValueError(f"{prefix}: QEC owner logical port is invalid")
            witnessed_owners[owner.placement] = owner
    if set(witnessed_owners) != set(bindings):
        raise ValueError(
            f"{prefix}: QEC owners do not cover the exact P1 placement")

    expected_rows, remote_sites = _p0_mpp_action_rows(
        program,
        placement,
        prefix,
    )
    witnessed_rows = {}
    for action in qec_selection.actions:
        if action.manifest_sha256 != qec_selection.network_manifest_sha256:
            continue
        if action.site in witnessed_rows:
            raise ValueError(f"{prefix}: QEC action site is duplicated")
        witnessed_rows[action.site] = {
            "kind": action.kind,
            "objective": action.objective,
            "placements": tuple(action.placements),
        }
    if witnessed_rows != expected_rows:
        raise ValueError(f"{prefix}: network MPP action rows differ")
    return remote_sites


def _network_manifest_encodings_from_ir(module, lowering, selection, prefix):
    """Resolve exact accepted Code/Encoding identities from one manifest."""

    accepted = []
    for reference in lowering.attributes["codes"]:
        path = _symbol_path(reference)
        if len(path) != 1:
            raise ValueError(
                f"{prefix}: accepted code or encoding is not top-level")
        target = _top_level_symbol(
            module,
            path[0],
            {"fabric.code", "fabric.encoding"},
            prefix,
        )
        if target.name == "fabric.encoding":
            code_path = _symbol_path(target.attributes["code"])
            if len(code_path) != 1:
                raise ValueError(
                    f"{prefix}: accepted encoding code is not top-level")
            code = _top_level_symbol(
                module,
                code_path[0],
                {"fabric.code"},
                prefix,
            )
            accepted.append((code_path[0], path[0], int(code.attributes["k"])))
            continue

        matching_encodings = tuple(
            view.operation
            for view in module.body.operations
            if view.operation.name == "fabric.encoding" and
            "code" in view.operation.attributes and
            _symbol_path(view.operation.attributes["code"]) == path and
            "sym_name" in view.operation.attributes and _attr_text(
                view.operation.attributes["sym_name"]) == selection.encoding)
        if len(matching_encodings) != 1:
            raise ValueError(
                f"{prefix}: selected encoding for accepted code is missing or ambiguous"
            )
        accepted.append(
            (path[0], selection.encoding, int(target.attributes["k"])))
    return tuple(accepted)


def _verify_qec_network_source(source, lowering, selection) -> None:
    """Authenticate a transient compiler context against its canonical P1."""

    prefix = "QEC network context does not match retained verified P1"
    kernel = _top_level_symbol(
        source.module,
        source.root.symbol,
        {"lvm.kernel"},
        prefix,
    )
    input_p0 = _verify_placement_witness(
        kernel,
        source.placement,
        prefix,
    )
    program = _top_level_symbol(
        source.module,
        input_p0,
        {"qlx.program"},
        prefix,
    )
    accepted = []
    from cudaq.logical.codes import (
        Code,
        Encoding,
    )

    for value in lowering.codes:
        if not isinstance(value, (Code, Encoding)):
            raise TypeError(
                "network QECLowering accepted definitions must be Code or Encoding"
            )
        encoding = value if isinstance(value,
                                       Encoding) else value.default_encoding
        accepted.append((encoding.code.name, encoding.name, encoding.code.k))
    verification_selection = selection
    if selection.network_manifest_sha256 is None:
        verification_selection = replace(
            selection,
            network_manifest_sha256=lowering.manifest_sha256,
        )
    _verify_qec_network_witness(
        program,
        source.placement,
        verification_selection,
        accepted_encodings=tuple(accepted),
        prefix=prefix,
    )


def _verify_communication_selection(module, root, placement, qec_selection,
                                    device) -> None:
    """Authenticate detached communication selection data against verified IR.

    The MLIR verifiers close each communication-qualified ``fabric.call`` to
    its action site and lowering manifest.  This cross-check makes that retained
    chain authoritative for the corresponding public replay witness.
    """

    operations = tuple(_walk_operation(module.operation))
    prefix = ("communication QEC selection does not match retained verified IR")

    envelope_candidates = tuple(
        view.operation
        for view in module.body.operations
        if "sym_name" in view.operation.attributes and
        _attr_text(view.operation.attributes["sym_name"]) == root.symbol)
    if not envelope_candidates and qec_selection is None:
        return
    if len(envelope_candidates) != 1:
        raise ValueError(
            f"{prefix}: selected root @{root.symbol} is missing or ambiguous")
    envelope_root = envelope_candidates[0]
    if envelope_root.name == "lvm.kernel":
        if qec_selection is not None:
            raise ValueError(
                f"{prefix}: P1 root cannot carry a P2 QEC selection witness")
        if "input_p0" in envelope_root.attributes or placement is not None:
            _verify_placement_witness(
                envelope_root,
                placement,
                "placement witness does not match retained verified P1",
            )
        return
    if qec_selection is None and envelope_root.name not in {
            "fabric.gadget",
            "fabric.protocol",
            "phys.graph",
    }:
        return
    if envelope_root.name in {"fabric.gadget", "fabric.protocol", "phys.graph"}:
        selected = envelope_root
    else:
        projected_graphs = tuple(view.operation
                                 for view in module.body.operations
                                 if view.operation.name == "phys.graph" and
                                 "source_protocol" in view.operation.attributes)
        if len(projected_graphs) == 1:
            selected = projected_graphs[0]
        else:
            selected_protocols = tuple(
                view.operation
                for view in module.body.operations
                if view.operation.name in {"fabric.gadget", "fabric.protocol"}
                and "metadata" in view.operation.attributes and
                "input_p1" in view.operation.attributes["metadata"] and
                (qec_selection is None or
                 _attr_text(view.operation.attributes["metadata"]
                            ["input_p1"]) == qec_selection.input_p1))
            if len(selected_protocols) != 1:
                raise ValueError(
                    f"{prefix}: selected root does not identify one retained "
                    "P2/P3 provenance chain")
            selected = selected_protocols[0]
    if selected.name == "phys.graph":
        if "source_protocol" not in selected.attributes:
            if qec_selection is None and placement is None:
                return
            raise ValueError(
                f"{prefix}: selected P3 source protocol is missing")
        source_path = _symbol_path(selected.attributes["source_protocol"])
        if len(source_path) != 1:
            raise ValueError(
                f"{prefix}: selected P3 source protocol is not top-level")
        selected = _top_level_symbol(
            module,
            source_path[0],
            {"fabric.gadget", "fabric.protocol"},
            prefix,
        )

    if "metadata" not in selected.attributes or (
            "input_p1" not in selected.attributes["metadata"]):
        if qec_selection is None and placement is None:
            return
        raise ValueError(
            f"{prefix}: selected P2 input P1 provenance is missing")
    input_p1 = _attr_text(selected.attributes["metadata"]["input_p1"])
    kernel = _top_level_symbol(module, input_p1, {"lvm.kernel"}, prefix)
    input_p0 = _verify_placement_witness(kernel, placement, prefix)
    metadata = selected.attributes["metadata"]
    if "qec_selection_sha256" not in metadata:
        raise ValueError(f"{prefix}: QEC selection commitment is missing")
    if qec_selection is None:
        raise ValueError(f"{prefix}: QEC selection witness is missing")
    if (_attr_text(metadata["qec_selection_sha256"])
            != _qec_selection_sha256(qec_selection)):
        raise ValueError(f"{prefix}: QEC selection commitment differs")
    program = _top_level_symbol(module, input_p0, {"qlx.program"}, prefix)
    _, p0_expected_sites = _p0_mpp_action_rows(
        program,
        placement,
        prefix,
    )

    if qec_selection.input_p1 != input_p1:
        raise ValueError(f"{prefix}: input P1 provenance differs")

    retained_realizations = {
        _attr_text(operation.attributes["sym_name"])
        for operation in operations
        if operation.name in {"fabric.gadget", "fabric.protocol"} and
        "sym_name" in operation.attributes
    }
    for action in qec_selection.actions:
        if action.selected not in action.feasible_candidates:
            raise ValueError(f"{prefix}: selected realization is not feasible")
        if (action.manifest_sha256 is None and
                action.selected not in retained_realizations):
            raise ValueError(
                f"{prefix}: selected realization is not retained by Fabric")

    network_manifest = qec_selection.network_manifest_sha256
    if network_manifest is not None:
        network_actions = tuple(action for action in qec_selection.actions
                                if action.manifest_sha256 == network_manifest)
        identities = {(
            action.selected,
            action.provider,
            action.version,
            action.manifest_sha256,
        ) for action in network_actions}
        if not network_actions or len(identities) != 1:
            raise ValueError(
                f"{prefix}: network actions select different QEC manifests")
        manifest_name, plugin, version, action_digest = next(iter(identities))
        if action_digest != network_manifest:
            raise ValueError(f"{prefix}: network manifest commitment differs")
        if "generated_by" not in selected.attributes:
            raise ValueError(
                f"{prefix}: network P2 generated_by provenance is missing")
        generated_path = _symbol_path(selected.attributes["generated_by"])
        if len(generated_path) != 1:
            raise ValueError(
                f"{prefix}: network P2 generated_by is not top-level")
        lowering = _top_level_symbol(
            module,
            generated_path[0],
            {"qlx.qec_lowering"},
            prefix,
        )
        required = {
            "manifest_name": manifest_name,
            "manifest_sha256": network_manifest,
            "compiler_plugin": plugin,
            "compiler_version": version,
        }
        if any(name not in lowering.attributes or
               _attr_text(lowering.attributes[name]) != expected
               for name, expected in required.items()):
            raise ValueError(
                f"{prefix}: generated_by differs from the selected network "
                "QEC manifest")
        accepted_encodings = _network_manifest_encodings_from_ir(
            module,
            lowering,
            qec_selection,
            prefix,
        )
        _verify_qec_network_witness(
            program,
            placement,
            qec_selection,
            accepted_encodings=accepted_encodings,
            prefix=prefix,
        )
        required_network_attributes = {
            "qlx.qec_network_request",
            "qlx.qec_network_plan",
        }
        if not required_network_attributes.issubset(selected.attributes):
            raise ValueError(
                f"{prefix}: canonical network P2 requires its request and plan")
        from ..qec import lattice_surgery

        serialized_request = _attr_text(
            selected.attributes["qlx.qec_network_request"])
        payload = lattice_surgery._network_request_payload(serialized_request)
        serialized_plan = _attr_text(
            selected.attributes["qlx.qec_network_plan"])
        plan = lattice_surgery.QECNetworkPlan.from_json(serialized_plan)
        request_digest = json.loads(serialized_request)["digest"]
        if payload["lowering_manifest_sha256"] != network_manifest:
            raise ValueError(
                f"{prefix}: network request manifest commitment differs")
        expected_source = _qec_network_source_sha256(
            module,
            input_p1,
            placement,
            qec_selection,
        )
        if payload["source_sha256"] != expected_source:
            raise ValueError(
                f"{prefix}: network request source commitment differs")
        commitments = (
            (plan.request_sha256, request_digest, "request"),
            (
                plan.lowering_manifest_sha256,
                network_manifest,
                "manifest",
            ),
            (
                plan.device_architecture_sha256,
                payload["device_architecture_sha256"],
                "device architecture",
            ),
            (
                plan.policy_sha256,
                lattice_surgery._digest(payload["policy"]),
                "policy",
            ),
        )
        for actual, expected, what in commitments:
            if actual != expected:
                raise ValueError(
                    f"{prefix}: network plan {what} commitment differs")
        metadata_required = {
            "network_request_sha256":
                request_digest,
            "network_plan_sha256":
                plan.digest,
            "network_provider":
                plan.provider_key,
            "required_projector":
                plan.required_projector_key,
            "required_projector_pipeline_sha256":
                (plan.required_projector_pipeline_sha256),
        }
        if any(name not in metadata or _attr_text(metadata[name]) != expected
               for name, expected in metadata_required.items()):
            raise ValueError(f"{prefix}: network request/plan metadata differs")
        compiler_fields = (
            "compiler_key",
            "compiler_plugin",
            "compiler_symbol",
            "compiler_version",
        )
        if any(name not in lowering.attributes for name in compiler_fields):
            raise ValueError(
                f"{prefix}: selected network compiler identity is incomplete")
        compiler_key = _attr_text(lowering.attributes["compiler_key"])
        canonical_key = (
            f"{_attr_text(lowering.attributes['compiler_plugin'])}:"
            f"{_attr_text(lowering.attributes['compiler_symbol'])}@"
            f"{_attr_text(lowering.attributes['compiler_version'])}")
        if compiler_key != canonical_key or plan.provider_key != compiler_key:
            raise ValueError(
                f"{prefix}: network plan provider differs from its selected "
                "compiler")
        if device is not None:
            request = lattice_surgery.QECNetworkRequest.from_dict(payload,
                                                                  device=device)
            lattice_surgery.validate_network_plan(request, plan)
            lattice_surgery._validate_network_request_selection(
                request,
                qec_selection,
            )
            lattice_surgery._validate_network_request_regions(
                request,
                _qec_network_region_runs(
                    module,
                    input_p1,
                    (value.site.symbol for value in request.actions),
                ),
            )
            compiler, exact_lowering = (
                lattice_surgery._network_compiler_for_manifest(
                    device,
                    network_manifest,
                ))
            if compiler.key != compiler_key or (exact_lowering.manifest_sha256
                                                != network_manifest):
                raise ValueError(
                    f"{prefix}: network plan provider differs from the "
                    "device-selected compiler")
            architecture_digest = lattice_surgery._require_digest(
                compiler.architecture_digest(device),
                what=(f"network compiler {compiler.key!r} architecture digest"),
            )
            if architecture_digest != request.device_architecture_sha256:
                raise ValueError(
                    f"{prefix}: network compiler architecture differs")

    calls = tuple(operation for operation in _walk_operation(selected)
                  if operation.name == "fabric.call" and
                  "channel" in operation.attributes)
    communication_instruments = tuple(
        operation for operation in _walk_operation(kernel)
        if operation.name == "lvm.instrument" and
        "channel" in operation.attributes)
    domain = _top_level_symbol(
        module,
        _attr_text(kernel.attributes["domain"]),
        {"lvm.domain"},
        prefix,
    )
    witnessed_communication = (any(
        any((
            action.channel,
            action.channel_capability,
            action.endpoints,
            action.direction,
        )) for action in qec_selection.actions))
    expected_sites: set[str] = set()
    for operation in communication_instruments:
        try:
            site_name = f"site{int(operation.attributes['site'])}"
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{prefix}: retained P1 communication site is incomplete"
            ) from error
        if site_name in expected_sites:
            raise ValueError(
                f"{prefix}: retained P1 site @{site_name} is duplicated")
        expected_sites.add(site_name)
    if expected_sites != p0_expected_sites:
        raise ValueError(
            f"{prefix}: retained P0 and P1 communication sites differ")
    if not expected_sites and not calls and not witnessed_communication:
        return

    def unique_by_symbol(path, kind):
        matches = tuple(
            operation for operation in operations
            if operation.name == kind and "sym_name" in operation.attributes and
            _operation_symbol_path(operation) == path)
        if len(matches) != 1:
            raise ValueError(
                f"{prefix}: {kind} @{'::@'.join(path)} is missing or ambiguous")
        return matches[0]

    domain_path = _operation_symbol_path(domain)
    for site_name in expected_sites:
        unique_by_symbol((*domain_path, site_name), "lvm.action_site")

    retained: dict[str, dict[str, Any]] = {}
    for call in calls:
        attributes = call.attributes
        required = (
            "action_site",
            "channel",
            "channel_capability",
            "endpoints",
            "generated_by",
        )
        if any(name not in attributes for name in required):
            raise ValueError(f"{prefix}: retained call is incomplete")
        site_path = _symbol_path(attributes["action_site"])
        site_name = site_path[-1]
        site = unique_by_symbol(site_path, "lvm.action_site")
        site_attributes = site.attributes
        lowering_path = _symbol_path(attributes["generated_by"])
        lowering_name = lowering_path[-1]
        lowering = unique_by_symbol(lowering_path, "qlx.qec_lowering")
        lowering_attributes = lowering.attributes
        try:
            row = {
                "site":
                    site_name,
                "kind":
                    _attr_text(site_attributes["kind"]),
                "objective":
                    _objective_text(site_attributes["objective"]),
                "feasible_candidates": (lowering_name,),
                "selected":
                    lowering_name,
                "provider":
                    _attr_text(lowering_attributes["compiler_plugin"]),
                "version":
                    _attr_text(lowering_attributes["compiler_version"]),
                "channel":
                    _symbol_path(site_attributes["channel"])[-1],
                "channel_capability":
                    _capability_text(site_attributes["channel_capability"]),
                "endpoints":
                    tuple(
                        _symbol_path(item)[-1]
                        for item in site_attributes["endpoints"]),
                "direction":
                    _attr_text(site_attributes["direction"]),
            }
        except (IndexError, KeyError) as error:
            raise ValueError(
                f"{prefix}: retained provenance is incomplete") from error
        if site_name in retained:
            raise ValueError(
                f"{prefix}: site @{site_name} has a duplicate call occurrence")
        retained[site_name] = row

    witnessed: dict[str, dict[str, Any]] = {}
    for action in qec_selection.actions:
        communication_values = (
            action.channel,
            action.channel_capability,
            action.endpoints,
            action.direction,
        )
        if not any(communication_values):
            continue
        if (action.channel is None or action.channel_capability is None or
                not action.endpoints or action.direction is None):
            raise ValueError(
                f"{prefix}: site @{action.site} has an incomplete row")
        if action.site in witnessed:
            raise ValueError(f"{prefix}: site @{action.site} is duplicated")
        witnessed[action.site] = {
            "site": action.site,
            "kind": action.kind,
            "objective": action.objective,
            "feasible_candidates": tuple(action.feasible_candidates),
            "selected": action.selected,
            "provider": action.provider,
            "version": action.version,
            "channel": action.channel,
            "channel_capability": action.channel_capability,
            "endpoints": tuple(action.endpoints),
            "direction": action.direction,
        }

    if witnessed != retained:
        raise ValueError(f"{prefix}: communication rows differ")
    if set(retained) != expected_sites:
        raise ValueError(f"{prefix}: communication action sites differ")


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
    all_physical_graphs = tuple(view.operation
                                for view in module.body.operations
                                if view.operation.name == "phys.graph")
    physical_graph = None
    if all_physical_graphs:
        matches = tuple(
            operation for operation in all_physical_graphs
            if "sym_name" in operation.attributes and
            _attr_text(operation.attributes["sym_name"]) == root_symbol)
        if len(matches) != 1:
            raise ValueError(
                "CCZ realization requires one exact retained physical graph")
        physical_graph = matches[0]
        if "source_protocol" not in physical_graph.attributes:
            raise ValueError(
                "CCZ physical graph is missing retained source_protocol")
        source_symbol = _symbol_path(
            physical_graph.attributes["source_protocol"])[-1]
    else:
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
    physical_operations = (tuple(_walk_operation(physical_graph))
                           if physical_graph is not None else ())

    if physical_graph is not None:
        expected_call_tree = tuple(
            (_symbol_path(call.attributes["callee"]), instance)
            for call, instance in projected_fabric_calls)
        actual_call_tree = tuple((
            _symbol_path(operation.attributes["callee"]),
            _attr_text(operation.attributes["instance"]),
        ) for operation in physical_operations if operation.name == "phys.call")
        if actual_call_tree != expected_call_tree:
            raise ValueError(
                "CCZ physical call callee and instance hierarchy differs "
                "from the selected Fabric call tree in the retained source "
                "protocol")

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
        operation for operation in (*source_operations, *physical_operations)
        if operation.name in {"fabric.unpack_resource", "phys.unpack_resource"
                             } and "payload_action" in operation.attributes and
        _objective_text(operation.attributes["payload_action"]) == "ccz")
    required = {"fabric.unpack_resource"}
    if physical_graph is not None:
        required.add("phys.unpack_resource")
    for kind in required:
        matches = tuple(op for op in unpacks if op.name == kind)
        actual_payloads = Counter((
            strings(unpack, "payload_logical_block_ids"),
            ints(unpack, "payload_logical_blocks"),
            ints(unpack, "payload_logical_ports"),
        ) for unpack in matches)
        expected = (expected_payloads if kind == "phys.unpack_resource" else
                    set(expected_payloads))
        actual = (actual_payloads
                  if kind == "phys.unpack_resource" else set(actual_payloads))
        if actual != expected:
            raise ValueError(
                "CCZ payload maps differ from selected QEC block ownership: "
                f"expected {expected!r}, got {actual!r}")
        if kind == "phys.unpack_resource":
            segment_counts = {
                record["payload"]: record["segment_count"]
                for record in records.values()
            }
            for unpack in matches:
                payload = (
                    strings(unpack, "payload_logical_block_ids"),
                    ints(unpack, "payload_logical_blocks"),
                    ints(unpack, "payload_logical_ports"),
                )
                if len(ints(unpack, "payload_carrier_segments")) != (
                        segment_counts[payload]):
                    raise ValueError(
                        "CCZ physical segments differ from selected QEC blocks")

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

    physical_resource_calls = tuple(
        operation for operation in physical_operations
        if operation.name == "phys.call" and
        ("resource_action_site" in operation.attributes or
         "resource_objective" in operation.attributes))
    actual_physical_occurrences = Counter()
    actual_physical_occurrence_order = []
    for call in physical_resource_calls:
        if ("resource_action_site" not in call.attributes or
                "resource_objective" not in call.attributes):
            raise ValueError(
                "physical resource call has incomplete action provenance")
        objective = _objective_text(call.attributes["resource_objective"])
        if objective == "ccz":
            occurrence = (
                _symbol_path(call.attributes["resource_action_site"]),
                objective,
                _symbol_path(call.attributes["callee"]),
                _attr_text(call.attributes["instance"]),
            )
            actual_physical_occurrences[occurrence] += 1
            actual_physical_occurrence_order.append(occurrence)
    if physical_graph is not None and (actual_physical_occurrences
                                       != selected_fabric_occurrences or
                                       actual_physical_occurrence_order
                                       != selected_fabric_occurrence_order):
        raise ValueError(
            "CCZ physical resource calls differ from selected Fabric calls: "
            f"expected {dict(selected_fabric_occurrences)!r}, "
            f"got {dict(actual_physical_occurrences)!r}")

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
    if any(op.name in {"fabric.resource_request", "phys.resource_request"} and
           "kind" in op.attributes and
           _attr_text(op.attributes["kind"]) == "ccz_state"
           for op in (*source_operations, *physical_operations)):
        raise ValueError("CCZ supply retained a generic resource request")
    physical = tuple(
        op for op in physical_operations
        if op.name == "phys.transport_resource" and "route" in op.attributes and
        "ccz_state" in str(op.result.type))
    if "phys.unpack_resource" in required:

        def terminal_physical_consumer(value):
            visited = set()
            qualified_calls = []
            while value not in visited:
                visited.add(value)
                uses = tuple(value.uses)
                if len(uses) != 1:
                    raise ValueError(
                        "CCZ physical payload must have one linear consumer")
                use = uses[0]
                owner = getattr(use.owner, "operation", use.owner)
                if owner.name == "phys.unpack_resource":
                    if use.operand_number != 0 or len(qualified_calls) != 1:
                        raise ValueError(
                            "CCZ physical transport lacks one exact selected "
                            "unpack consumer call")
                    return qualified_calls[0], owner
                if owner.name == "phys.call":
                    has_site = "resource_action_site" in owner.attributes
                    has_objective = "resource_objective" in owner.attributes
                    if has_site != has_objective:
                        raise ValueError(
                            "physical resource call has incomplete action "
                            "provenance")
                    if has_site:
                        qualified_calls.append(owner)
                        if len(qualified_calls) > 1:
                            raise ValueError(
                                "CCZ physical transport crosses multiple "
                                "selected consumer calls")
                    block = owner.regions[0].blocks[0]
                    if use.operand_number >= len(block.arguments):
                        raise ValueError(
                            "CCZ physical consumer lineage is incomplete")
                    value = block.arguments[use.operand_number]
                    continue
                if owner.name == "phys.yield":
                    parent = owner.parent
                    if (parent is None or parent.name != "phys.call" or
                            use.operand_number >= len(parent.results)):
                        raise ValueError(
                            "CCZ physical consumer lineage is incomplete")
                    value = parent.results[use.operand_number]
                    continue
                raise ValueError(
                    "CCZ physical transport has a foreign consumer")
            raise ValueError("CCZ physical consumer lineage is cyclic")

        physical_destinations = Counter()
        for transport in physical:
            destination = _attr_text(transport.attributes["destination"])
            physical_destinations[destination] += 1
            if (_attr_text(transport.attributes["source"]) != backing or
                    _attr_text(transport.attributes["protocol"]) != transfer):
                raise ValueError(
                    "CCZ physical transport differs from stream provenance")
            value = transport.operands[0]
            visited = set()
            while value not in visited:
                visited.add(value)
                if isinstance(value, mlir_ir.BlockArgument):
                    block = value.owner
                    parent = getattr(block.owner, "operation", block.owner)
                    if (parent.name != "phys.call" or
                            value.arg_number >= len(parent.operands)):
                        raise ValueError(
                            "CCZ physical producer lineage is incomplete")
                    value = parent.operands[value.arg_number]
                    continue
                owner = value.owner
                if owner.name == "phys.produce_resource":
                    if (_attr_text(owner.attributes["region"]) != backing or
                            _attr_text(owner.attributes["resource_kind"])
                            != "ccz_state" or _attr_text(
                                owner.attributes["protocol"]) != producer):
                        raise ValueError(
                            "CCZ physical producer differs from stream provenance"
                        )
                    matching_provider_calls = []
                    parent = owner.parent
                    while parent is not None:
                        if (parent.name == "phys.call" and
                                "callee" in parent.attributes and _attr_text(
                                    parent.attributes["callee"]) == producer):
                            matching_provider_calls.append(parent)
                        parent = parent.parent
                    if len(matching_provider_calls) != 1:
                        raise ValueError(
                            "CCZ physical producer call differs from stream "
                            "provenance")
                    break
                if owner.name != "phys.call":
                    raise ValueError(
                        "CCZ physical transport has a foreign producer")
                index = next(
                    (index for index, result in enumerate(owner.results)
                     if result == value),
                    None,
                )
                block = owner.regions[0].blocks[0]
                terminator = block.operations[-1].operation
                if index is None or terminator.name != "phys.yield":
                    raise ValueError(
                        "CCZ physical producer lineage is incomplete")
                value = terminator.operands[index]
            else:
                raise ValueError("CCZ physical producer lineage is cyclic")
            consumer, unpack = terminal_physical_consumer(transport.result)
            site_path = _symbol_path(
                consumer.attributes["resource_action_site"])
            witness = (
                site_path,
                _objective_text(consumer.attributes["resource_objective"]),
                _symbol_path(consumer.attributes["callee"]),
                _attr_text(consumer.attributes["instance"]),
            )
            if selected_fabric_occurrences[witness] != 1:
                raise ValueError(
                    "CCZ physical consumer differs from its selected Fabric "
                    "call")
            site = site_path[-1]
            payload = (
                strings(unpack, "payload_logical_block_ids"),
                ints(unpack, "payload_logical_blocks"),
                ints(unpack, "payload_logical_ports"),
            )
            if (records[site]["destination"] != destination or
                    records[site]["payload"] != payload):
                raise ValueError(
                    "CCZ physical transport and unpack differ from their "
                    "selected action site")
        if physical_destinations != expected_destinations:
            raise ValueError(
                "CCZ physical transports differ from selected action sites")


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


_VERIFIED_INCREMENT_AUTH = object()


@dataclass(frozen=True, slots=True)
class _VerifiedLinkSnapshot:
    """Compact in-process replay of one already-verified Build closure."""

    bytecode: bytes
    root: DefinitionHandle[Any]


class Build:
    """A frozen QLX compilation product backed by a verified MLIR module."""

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
        "_device",
        "_transient",
        "_verified",
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
        _transient=False,
        _verified_increment=None,
        _take_verified_increment_ownership=False,
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
        verified_increment = _verified_increment is _VERIFIED_INCREMENT_AUTH
        if _verified_increment is not None and not verified_increment:
            raise ValueError("invalid verified Build increment authority")
        if _take_verified_increment_ownership and not verified_increment:
            raise ValueError(
                "module ownership transfer requires verified increment authority"
            )
        if not verified_increment:
            if not module.operation.verify():
                raise ValueError("QLX MLIR module failed verification")
            _verify_communication_selection(module, root, placement,
                                            qec_selection, device)
            _verify_ccz_resource_realization(module, root, placement,
                                             qec_selection)
        self._context = context
        self._transient = bool(_transient)
        if self._transient or _take_verified_increment_ownership:
            self._module = module
        else:
            # A caller may intentionally keep linking more definitions into a
            # user-owned module after this Build is published.  Retain an
            # in-memory clone at the publication boundary so lazy assembly
            # generation cannot observe those later mutations.  Progressive
            # compiler intermediates bypass this clone via _transient.
            from cudaq.mlir._mlir_libs import _qlxRuntime

            self._module = _qlxRuntime.clone_module(module)
        # A verified live module is the in-process authority. Canonical
        # assembly is generated only when a portable representation is
        # explicitly requested. This keeps ordinary stage-to-stage lowering
        # out of the MLIR printer/parser path.
        self._snapshot = None
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
        self._device = device
        self._verified = True
        self._sealed = True

    @classmethod
    def _from_verified_increment(cls, **values):
        """Seal an increment independently verified against a verified Build.

        This is an internal compiler boundary for native passes that preserve
        every existing operation and construct one schema-verified operation
        from those already-authenticated definitions.  It is deliberately not
        a public escape hatch for arbitrary ModuleOp construction.
        """

        return cls(
            **values,
            _verified_increment=_VERIFIED_INCREMENT_AUTH,
            _take_verified_increment_ownership=True,
        )

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
            "gadget_profile",
            "protocol",
            "physical_graph",
            "control_plan",
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
        """A cached read-only inspection clone of the verified module.

        The same clone is returned on every access, so repeated typed
        inspection performs neither printing nor parsing. The private module
        that authenticates this Build is never exposed. Compiler passes that
        need a scratch copy to mutate must use :meth:`_fresh_module`.

        A cheap top-level fingerprint defensively detects mutation through
        external or legacy compatibility code, drops every derived handle,
        and reclones the private module so the build stays observably
        immutable. Core target lowering always uses :meth:`_fresh_module`.
        """

        # Transient builds are compiler-owned transfer objects. They never
        # escape the progressive compile transaction, so the next internal
        # stage may consume the live ModuleOp directly.
        if self._transient:
            return self._module

        module = self._cache.get("module")
        if module is not None:
            fingerprint = tuple(
                view.operation.name for view in module.body.operations)
            if fingerprint != self._cache["fingerprint"]:
                self._cache.clear()
                module = None
        if module is None:
            module = self._clone_module()
            self._cache["module"] = module
            self._cache["fingerprint"] = tuple(
                view.operation.name for view in module.body.operations)
        return module

    def _clone_module(self):
        """Deep-clone the private module without assembly round-tripping."""

        from cudaq.mlir._mlir_libs import _qlxRuntime

        return _qlxRuntime.clone_module(self._module)

    def _fresh_module(self):
        """Return a private mutable module for one compiler consumer.

        An unpublished intermediate Build transfers its live ModuleOp through
        the progressive compiler transaction. A published Build instead
        supplies an in-memory deep clone, preserving immutability without
        printing or parsing MLIR.
        """

        if self._transient:
            return self._module
        return self._clone_module()

    def _ensure_snapshot(self) -> str:
        """Materialize canonical assembly only for a portable request."""

        snapshot = self._snapshot
        if snapshot is None:
            snapshot = self._module.operation.get_asm(assume_verified=True)
            object.__setattr__(self, "_snapshot", snapshot)
        return snapshot

    def _verified_link_snapshot(self) -> _VerifiedLinkSnapshot:
        """Return a compact cross-context linker snapshot of this Build.

        MLIR bytecode preserves the exact verified operation graph without the
        expensive canonical-text printer/parser round trip.  This is private
        compiler interchange only; portable Build serialization remains the
        versioned textual bundle required by the public replay contract.
        """

        snapshot = self._cache.get("verified_link_snapshot")
        if snapshot is None:
            from io import BytesIO

            output = BytesIO()
            self._module.operation.write_bytecode(output)
            snapshot = _VerifiedLinkSnapshot(output.getvalue(), self.root)
            self._cache["verified_link_snapshot"] = snapshot
        return snapshot

    def to_mlir(self) -> str:
        return self._ensure_snapshot()

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
                # qlx.qec_lowering provenance: a generated realization
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
        ``.name`` (``cudaq.logical.logical.idle``, an
        ``@cudaq.logical.objective`` definition),
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
    def schedule(self):
        """The :class:`PhysicalSchedule` this build carries, or ``None``.

        ``cudaq.logical.schedule`` records its verified result as a
        ``phys.schedule``
        symbol inside the linked module, so a scheduled build (and its
        ``serialize()``/``replay()`` round trip) reconstructs the typed
        schedule directly from IR.  Builds that never went through the
        physical scheduler carry no ``phys.schedule`` op and report
        ``None``; schedules are not stored as detached Python state on the
        Build envelope.
        """

        schedule = self._cache.get("schedule", _UNSET)
        if isinstance(schedule, weakref.ReferenceType):
            schedule = schedule()
            if schedule is None:
                schedule = _UNSET
        if schedule is _UNSET:
            schedule = self._parse_schedule()
            self._cache["schedule"] = (None if schedule is None else
                                       weakref.ref(schedule))
        return schedule

    @property
    def patch_graph(self):
        """Typed P2 patch interaction/mapping view, or ``None``."""

        view = self._cache.get("patch_graph", _UNSET)
        if view is _UNSET:
            from .topology_view import PatchGraphView

            view = PatchGraphView.from_build(self)
            self._cache["patch_graph"] = view
        return view

    @property
    def carrier_graph(self):
        """Typed P3 carrier topology/event view, or ``None``."""

        view = self._cache.get("carrier_graph", _UNSET)
        if view is _UNSET:
            from .topology_view import CarrierGraphView

            view = CarrierGraphView.from_build(self)
            self._cache["carrier_graph"] = view
        return view

    def _parse_schedule(self, module=None, *, parse_entries=True):
        from .schedule import PhysicalSchedule, scheduling

        # Public/later access reconstructs schedule authority from a private
        # in-memory clone. The scheduler may instead hand this method the exact
        # verified private module while sealing the Build; parsing immutable
        # scalar/tuple values from that authority does not expose MLIR handles
        # and avoids cloning the largest stage immediately after construction.
        if module is None:
            module = self._fresh_module()
        elif module is not self._module:
            raise ValueError(
                "schedule parsing requires the Build's verified private module")
        selected_root = next(
            (operation.operation
             for operation in module.body.operations
             if self._symbol(operation.operation) == self.root.symbol),
            None,
        )
        if selected_root is None:
            return None
        selected_schedules = _selected_schedule_operations(
            module, selected_root)
        candidates = [
            operation for operation in selected_schedules
            if operation.name == "phys.schedule"
        ]
        if not candidates:
            return None
        if len(candidates) != 1:
            raise ValueError(
                "selected physical graph must have at most one schedule")
        chosen = candidates[0]
        attributes = chosen.attributes
        graph_symbol = _attr_text(attributes["graph"])
        graph = next(
            (operation.operation
             for operation in module.body.operations
             if operation.operation.name == "phys.graph" and
             self._symbol(operation.operation) == graph_symbol),
            None,
        )
        if graph is None:
            raise ValueError(
                "selected physical schedule graph is missing from its Build")
        machine_symbol = _attr_text(graph.attributes["architecture"])
        operating_point_symbol = (_attr_text(
            graph.attributes["operating_point"]) if "operating_point"
                                  in graph.attributes else None)
        entries = (tuple(
            self._parse_schedule_entry(
                str(getattr(item, "value", item)).strip('"'))
            for item in attributes["entries"]) if parse_entries else ())
        makespan = attributes["makespan_ns"]
        makespan = float(getattr(makespan, "value", makespan))
        strategy_name = _attr_text(attributes["strategy"])
        if strategy_name != scheduling.greedy_asap.name:
            raise ValueError(
                f"physical schedule names unknown strategy {strategy_name!r}")
        constraints = tuple(
            _attr_text(value) for value in attributes["constraints"])
        timing_profile = tuple(
            (str(named.name), float(getattr(named.attr, "value", named.attr)))
            for named in attributes["timing_profile"])
        objective_value = None
        if "objective_value" in attributes:
            value = attributes["objective_value"]
            objective_value = float(getattr(value, "value", value))
        return PhysicalSchedule._from_verified_ir(
            build=self,
            _requested_root_symbol=self.root.symbol,
            _graph_symbol=graph_symbol,
            _schedule_symbol=self._symbol(chosen),
            _machine_symbol=machine_symbol,
            _operating_point_symbol=operating_point_symbol,
            entries=entries,
            strategy=scheduling.greedy_asap,
            strategy_domain=_attr_text(attributes["strategy_domain"]),
            provider=_attr_text(attributes["provider"]),
            provider_version=_attr_text(attributes["provider_version"]),
            constraint_profile=_attr_text(attributes["constraint_profile"]),
            constraints=constraints,
            timing_profile=timing_profile,
            tie_break=_attr_text(attributes["tie_break"]),
            optimization_status=_attr_text(attributes["optimization_status"]),
            objective_value=objective_value,
            makespan_ns=makespan,
            _defer_entries=not parse_entries,
        )

    @staticmethod
    def _parse_schedule_entry(text: str):
        from .schedule import ScheduleEntry

        event_id, kind, start, duration, resources, *rest = text.split("|")
        details = {}
        for item in rest:
            key, _, value = item.partition("=")
            details[key] = value

        def optional(key):
            return details.get(key) or None

        def optional_int(key):
            value = details.get(key, "")
            return int(value) if value else None

        def optional_float(key):
            value = details.get(key, "")
            return float(value) if value else None

        return ScheduleEntry(
            event_id=event_id,
            kind=kind,
            start_ns=float(start),
            duration_ns=float(duration),
            resources=tuple(part for part in resources.split(",") if part),
            dependencies=tuple(
                part for part in details.get("deps", "").split(",") if part),
            parent=optional("parent"),
            branch=optional("branch"),
            condition=optional("condition"),
            max_attempts=optional_int("max_attempts"),
            exhaustion=optional("exhaustion"),
            commit_point=optional("commit_point"),
            repeat_count=optional_int("repeat_count"),
            repeat_period_ns=optional_float("repeat_period_ns"),
            repeat_epilogue_ns=optional_float("repeat_epilogue_ns"),
            max_iterations=optional_int("max_iterations"),
            callee=optional("callee"),
            instance=optional("instance"),
            profile=optional("profile"),
            template_event=optional("template_event"),
            attempt=optional("attempt"),
            attempt_event=optional("attempt_event"),
            decision_event=optional("decision_event"),
            success_probability=optional_float("success_probability"),
            success_probability_source=optional("success_probability_source"),
            success_probability_evidence=optional(
                "success_probability_evidence"),
            data_dependencies=tuple(
                part for part in details.get("data_deps", "").split(",")
                if part),
            resource_dependencies=tuple(
                part for part in details.get("resource_deps", "").split(",")
                if part),
            domain_dependencies=tuple(
                part for part in details.get("domain_deps", "").split(",")
                if part),
        )

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
                self._ensure_snapshot(),
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
        from cudaq.logical.architecture.constraints import (
            PlacementBinding,
            PlacementWitness,
        )
        from cudaq.logical.codes import (
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
                            channel=action.get("channel"),
                            channel_capability=action.get("channel_capability"),
                            endpoints=tuple(action.get("endpoints", ())),
                            direction=action.get("direction"),
                        ) for action in item.get("actions", ())),
                    code=item.get("code"),
                    encoding=item.get("encoding"),
                    objective=item.get("objective",
                                       "policy_then_device_then_candidate"),
                    network_manifest_sha256=item.get("network_manifest_sha256"),
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
        return self._ensure_snapshot()
