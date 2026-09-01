# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Authenticated qlx.build/v2 schema and pure validation.

This module owns the canonical replay-envelope shape, content commitment,
and fail-closed structural checks. MLIR replay and the public Build
facade remain in cudaq.logical.compiler.build.
"""

from __future__ import annotations

from hashlib import sha256
import json

from ..stages import (
    facets_for_kind,
    normalize_facets,
    stage_and_facets,
)

_BUILD_V2_SCHEMA = "qlx.build/v2"
_BUILD_V2_MODEL_VERSION = "0.3.10-proposed"
_BUILD_V2_IR_VERSION = "0.4-draft"
_BUILD_V2_FIELDS = frozenset({
    "schema",
    "model_version",
    "ir_version",
    "module",
    "root",
    "profile",
    "stage",
    "facets",
    "pipeline",
    "evidence",
    "value_groups",
    "placement",
    "qec_selection",
    "source_modules",
    "experiment",
    "content_sha256",
})
_BUILD_V2_PASS_FIELDS = frozenset({
    "name",
    "options",
    "requires_facets",
    "provides_facets",
    "preserves_facets",
    "invalidates_facets",
    "recomputes_facets",
})
_BUILD_V2_EVIDENCE_FIELDS = frozenset(
    {"kind", "producer", "result", "obligations", "assumptions"})
_BUILD_V2_PLACEMENT_FIELDS = frozenset({
    "machine",
    "input_p0",
    "bindings",
    "relaxed_preferences",
    "objective",
    "tie_break",
})
_BUILD_V2_PLACEMENT_BINDING_FIELDS = frozenset({
    "placement",
    "space",
    "slot",
    "source_allocation",
    "source_group",
    "source_path",
    "binding_kind",
    "binding_data",
})
_BUILD_V2_QEC_SELECTION_FIELDS = frozenset({
    "input_p1",
    "blocks",
    "actions",
    "code",
    "encoding",
    "objective",
    "tie_break",
})
_BUILD_V2_QEC_BLOCK_FIELDS = frozenset({
    "block",
    "space",
    "code",
    "encoding",
    "logical_capacity",
    "owners",
})
_BUILD_V2_QEC_OWNER_FIELDS = frozenset({
    "placement",
    "logical_index",
    "source_allocation",
    "source_group",
    "source_path",
})
_BUILD_V2_QEC_ACTION_FIELDS = frozenset({
    "site",
    "kind",
    "objective",
    "placements",
    "feasible_candidates",
    "selected",
    "provider",
    "version",
    "manifest_sha256",
    "tie_break",
})
_BUILD_V2_EXPERIMENT_FIELDS = frozenset({
    "root",
    "profile",
    "stage",
    "facets",
    "pass_recipe",
    "closure",
    "bindings",
})
_BUILD_V2_EXPERIMENT_BINDINGS = frozenset({
    "device",
    "device_provenance",
    "placement",
    "target",
    "policy",
    "parameters",
    "objective",
})
_ROOT_OPERATION_BY_KIND = {
    "action": "qlx.action",
    "instrument": "qlx.instrument_decl",
    "program": "qlx.program",
    "kernel": "lvm.kernel",
    "gadget": "fabric.gadget",
    "protocol": "fabric.protocol",
    "device": "qlx.device",
    "qec_lowering": "qlx.qec_lowering",
    "target_manifest": "qlx.target_manifest",
    "Code": "fabric.code",
    "CodeProfile": "fabric.code_profile",
    "Encoding": "fabric.encoding",
    "EncodingEpoch": "fabric.encoding_epoch",
    "EncodingEpochSchema": "fabric.encoding_epoch_schema",
    "EncodingHierarchy": "fabric.encoding_hierarchy",
    "EncodingProjection": "fabric.encoding_projection",
    "PatchTransform": "fabric.patch_transform",
}
_ROOT_PROFILES_BY_KIND = {
    "action": frozenset({"p0"}),
    "instrument": frozenset({"p0"}),
    "program": frozenset({"p0"}),
    "kernel": frozenset({"p1"}),
    "gadget": frozenset({"p2a"}),
    "protocol": frozenset({"p2n"}),
    "device": frozenset({"p1", "p2", "p2n"}),
    "qec_lowering": frozenset({"common"}),
    "target_manifest": frozenset({"common"}),
    "Code": frozenset({"p2s"}),
    "CodeProfile": frozenset({"p2s"}),
    "Encoding": frozenset({"p2s"}),
    "EncodingEpoch": frozenset({"p2s"}),
    "EncodingEpochSchema": frozenset({"p2s"}),
    "EncodingHierarchy": frozenset({"p2s"}),
    "EncodingProjection": frozenset({"p2s"}),
    "PatchTransform": frozenset({"p2s"}),
}
_TYPED_ROOT_KIND_OWNERS = {
    "Code": "cudaq.logical.codes.definition.Code",
    "CodeProfile": "cudaq.logical.codes.profiles.CodeProfile",
    "Encoding": "cudaq.logical.codes.encodings.Encoding",
    "EncodingEpoch": "cudaq.logical.codes.profiles.EncodingEpoch",
    "EncodingEpochSchema": "cudaq.logical.codes.profiles.EncodingEpochSchema",
    "EncodingHierarchy": "cudaq.logical.codes.encodings.EncodingHierarchy",
    "EncodingProjection": "cudaq.logical.codes.encodings.EncodingProjection",
    "PatchTransform": "cudaq.logical.codes.structure.PatchTransform",
}
_FACET_MINIMUM_STAGE = {
    "qec_spec": "p2",
    "qec_realization": "p2",
    "protocol_network": "p2",
    "patch_graph": "p2",
}
_STAGE_ORDINAL = {"p0": 0, "p1": 1, "p2": 2}


def _build_bundle_content_sha256(bundle) -> str:
    """Commit every serialized Build field except the commitment itself."""

    committed = {
        key: value for key, value in bundle.items() if key != "content_sha256"
    }
    payload = json.dumps(committed, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def _root_operation_name(kind: str) -> str | None:
    """Return the canonical top-level MLIR operation for a serialized kind."""

    return _ROOT_OPERATION_BY_KIND.get(kind.rsplit(".", 1)[-1])


def _root_profiles(kind: str) -> frozenset[str]:
    return _ROOT_PROFILES_BY_KIND.get(kind.rsplit(".", 1)[-1], frozenset())


def _require_exact_fields(value, fields, label):
    if not isinstance(value, dict) or frozenset(value) != fields:
        raise ValueError(f"qlx.build/v2 {label} fields differ from the schema")


def _require_nonempty_string(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"qlx.build/v2 {label} must be a nonempty string")


def _require_optional_string(value, label):
    if value is not None:
        _require_nonempty_string(value, label)


def _require_optional_sha256(value, label):
    if value is None:
        return
    _require_nonempty_string(value, label)
    prefix = "sha256:"
    payload = value[len(prefix):] if value.startswith(prefix) else ""
    if len(payload) != 64 or any(
            item not in "0123456789abcdef" for item in payload):
        raise ValueError(f"qlx.build/v2 {label} must be canonical sha256")


def _require_nonnegative_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"qlx.build/v2 {label} must be a nonnegative integer")


def _require_string_list(value, label, *, unique=False):
    if not isinstance(value, list) or not all(
            isinstance(item, str) and item for item in value):
        raise ValueError(
            f"qlx.build/v2 {label} must be a list of nonempty strings")
    if unique and len(set(value)) != len(value):
        raise ValueError(f"qlx.build/v2 {label} must contain unique strings")


def _validate_json_value(value, label):
    if value is None or isinstance(value, (bool, int, float, str)):
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, label)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _validate_json_value(item, label)
        return
    raise ValueError(f"qlx.build/v2 {label} is not canonical JSON metadata")


def _validate_option_pairs(value, label):
    if not isinstance(value, list):
        raise ValueError(f"qlx.build/v2 {label} must be a list")
    names = set()
    for option in value:
        if (not isinstance(option, list) or len(option) != 2 or
                not isinstance(option[0], str) or not option[0]):
            raise ValueError(f"qlx.build/v2 {label} must contain nonempty "
                             "string-key pairs")
        if option[0] in names:
            raise ValueError(f"qlx.build/v2 {label} names must be unique")
        names.add(option[0])
        _validate_json_value(option[1], label)


def _validate_v2_nested_metadata(bundle):
    evidence = bundle["evidence"]
    if not isinstance(evidence, list):
        raise ValueError("qlx.build/v2 evidence must be a list")
    for item in evidence:
        _require_exact_fields(item, _BUILD_V2_EVIDENCE_FIELDS, "evidence row")
        for field in ("kind", "producer", "result"):
            _require_nonempty_string(item[field], f"evidence {field}")
        for field in ("obligations", "assumptions"):
            _require_string_list(item[field], f"evidence {field}", unique=True)

    placement = bundle["placement"]
    if placement is not None:
        _require_exact_fields(placement, _BUILD_V2_PLACEMENT_FIELDS,
                              "placement")
        for field in ("machine", "input_p0", "objective", "tie_break"):
            _require_nonempty_string(placement[field], f"placement {field}")
        _require_string_list(
            placement["relaxed_preferences"],
            "placement relaxed_preferences",
            unique=True,
        )
        bindings = placement["bindings"]
        if not isinstance(bindings, list):
            raise ValueError("qlx.build/v2 placement bindings must be a list")
        placement_identities = set()
        placement_sources = set()
        for binding in bindings:
            _require_exact_fields(
                binding,
                _BUILD_V2_PLACEMENT_BINDING_FIELDS,
                "placement binding",
            )
            for field in ("placement", "space", "binding_kind"):
                _require_nonempty_string(binding[field],
                                         f"placement binding {field}")
            _require_nonnegative_integer(binding["slot"],
                                         "placement binding slot")
            if binding["source_allocation"] is not None:
                _require_nonnegative_integer(
                    binding["source_allocation"],
                    "placement binding source_allocation",
                )
            _require_optional_string(
                binding["source_group"],
                "placement binding source_group",
            )
            source_path = binding["source_path"]
            if not isinstance(source_path, list):
                raise ValueError("qlx.build/v2 placement binding source_path "
                                 "must be a list")
            for value in source_path:
                _require_nonnegative_integer(
                    value, "placement binding source_path entry")
            _validate_option_pairs(
                binding["binding_data"],
                "placement binding binding_data",
            )
            if binding["placement"] in placement_identities:
                raise ValueError(
                    "qlx.build/v2 placement binding identities must be unique")
            placement_identities.add(binding["placement"])
            if binding["source_allocation"] is not None:
                source_identity = (
                    binding["source_allocation"],
                    binding["source_group"],
                    tuple(binding["source_path"]),
                )
                if source_identity in placement_sources:
                    raise ValueError(
                        "qlx.build/v2 placement binding source identities "
                        "must be unique")
                placement_sources.add(source_identity)

    selection = bundle["qec_selection"]
    if selection is not None:
        _require_exact_fields(selection, _BUILD_V2_QEC_SELECTION_FIELDS,
                              "QEC selection")
        for field in ("input_p1", "objective", "tie_break"):
            _require_nonempty_string(selection[field], f"QEC selection {field}")
        for field in ("code", "encoding"):
            _require_optional_string(selection[field], f"QEC selection {field}")
        blocks = selection["blocks"]
        actions = selection["actions"]
        if not isinstance(blocks, list) or not isinstance(actions, list):
            raise ValueError(
                "qlx.build/v2 QEC selection blocks/actions must be lists")
        block_identities = set()
        owner_placements = set()
        for block in blocks:
            _require_exact_fields(block, _BUILD_V2_QEC_BLOCK_FIELDS,
                                  "QEC block")
            for field in ("block", "space", "code", "encoding"):
                _require_nonempty_string(block[field], f"QEC block {field}")
            _require_nonnegative_integer(block["logical_capacity"],
                                         "QEC block logical_capacity")
            if not isinstance(block["owners"], list):
                raise ValueError("qlx.build/v2 QEC block owners must be a list")
            if block["block"] in block_identities:
                raise ValueError(
                    "qlx.build/v2 QEC block identities must be unique")
            block_identities.add(block["block"])
            logical_indices = set()
            for owner in block["owners"]:
                _require_exact_fields(owner, _BUILD_V2_QEC_OWNER_FIELDS,
                                      "QEC owner")
                _require_nonempty_string(owner["placement"],
                                         "QEC owner placement")
                _require_nonnegative_integer(owner["logical_index"],
                                             "QEC owner logical_index")
                if owner["source_allocation"] is not None:
                    _require_nonnegative_integer(
                        owner["source_allocation"],
                        "QEC owner source_allocation",
                    )
                _require_optional_string(owner["source_group"],
                                         "QEC owner source_group")
                if not isinstance(owner["source_path"], list):
                    raise ValueError(
                        "qlx.build/v2 QEC owner source_path must be a list")
                for value in owner["source_path"]:
                    _require_nonnegative_integer(value,
                                                 "QEC owner source_path entry")
                if owner["placement"] in owner_placements:
                    raise ValueError(
                        "qlx.build/v2 QEC owner placement identities must "
                        "be unique")
                owner_placements.add(owner["placement"])
                if owner["logical_index"] in logical_indices:
                    raise ValueError(
                        "qlx.build/v2 QEC owner logical indices must be "
                        "unique within a block")
                logical_indices.add(owner["logical_index"])
        action_identities = set()
        for action in actions:
            _require_exact_fields(action, _BUILD_V2_QEC_ACTION_FIELDS,
                                  "QEC action")
            for field in (
                    "site",
                    "kind",
                    "objective",
                    "selected",
                    "provider",
                    "version",
                    "tie_break",
            ):
                _require_nonempty_string(action[field], f"QEC action {field}")
            for field in ("placements", "feasible_candidates"):
                _require_string_list(action[field],
                                     f"QEC action {field}",
                                     unique=True)
            _require_optional_sha256(
                action["manifest_sha256"],
                "QEC action manifest",
            )
            if action["site"] in action_identities:
                raise ValueError(
                    "qlx.build/v2 QEC action site identities must be unique")
            action_identities.add(action["site"])

    _require_string_list(bundle["source_modules"],
                         "source_modules",
                         unique=True)

    experiment = bundle["experiment"]
    if experiment is not None:
        _require_exact_fields(experiment, _BUILD_V2_EXPERIMENT_FIELDS,
                              "experiment")
        root = experiment["root"]
        _require_exact_fields(root, {"symbol", "kind", "profile"},
                              "experiment root")
        for field in ("symbol", "kind", "profile"):
            _require_nonempty_string(root[field], f"experiment root {field}")
        if root != bundle["root"]:
            raise ValueError(
                "qlx.build/v2 experiment root differs from the build root")
        for field in ("profile", "stage"):
            _require_nonempty_string(experiment[field], f"experiment {field}")
        if (experiment["profile"] != bundle["profile"] or
                experiment["stage"] != bundle["stage"]):
            raise ValueError(
                "qlx.build/v2 experiment classification differs from the build")
        _require_string_list(experiment["facets"],
                             "experiment facets",
                             unique=True)
        if experiment["facets"] != bundle["facets"]:
            raise ValueError(
                "qlx.build/v2 experiment facets differ from the build")
        _require_string_list(experiment["closure"],
                             "experiment closure",
                             unique=True)
        recipe = experiment["pass_recipe"]
        if not isinstance(recipe, list):
            raise ValueError(
                "qlx.build/v2 experiment pass_recipe must be a list")
        for item in recipe:
            if (not isinstance(item, list) or len(item) != 2 or
                    not isinstance(item[0], str) or not item[0]):
                raise ValueError(
                    "qlx.build/v2 experiment pass_recipe rows must contain "
                    "a nonempty pass name and options")
            _validate_option_pairs(item[1], "experiment pass_recipe options")
        bindings = experiment["bindings"]
        if (not isinstance(bindings, dict) or
                not frozenset(bindings).issubset(_BUILD_V2_EXPERIMENT_BINDINGS)
           ):
            raise ValueError(
                "qlx.build/v2 experiment bindings contain unsupported fields")
        _validate_json_value(bindings, "experiment bindings")


def _validate_v2_bundle(bundle) -> None:
    """Validate the authenticated, self-contained v2 replay envelope."""

    commitment = bundle.get("content_sha256")
    if (not isinstance(commitment, str) or
            commitment != _build_bundle_content_sha256(bundle)):
        raise ValueError(
            "build bundle content commitment is missing or differs")

    fields = frozenset(bundle)
    if fields != _BUILD_V2_FIELDS:
        missing = sorted(_BUILD_V2_FIELDS - fields)
        unexpected = sorted(fields - _BUILD_V2_FIELDS)
        details = []
        if missing:
            details.append(f"missing {missing!r}")
        if unexpected:
            details.append(f"unexpected {unexpected!r}")
        raise ValueError(
            "qlx.build/v2 fields differ from the committed schema: " +
            ", ".join(details))
    if bundle["model_version"] != _BUILD_V2_MODEL_VERSION:
        raise ValueError("qlx.build/v2 model_version is missing or unsupported")
    if bundle["ir_version"] != _BUILD_V2_IR_VERSION:
        raise ValueError("qlx.build/v2 ir_version is missing or unsupported")
    if not isinstance(bundle["module"], str) or not bundle["module"].strip():
        raise ValueError("qlx.build/v2 module must be nonempty textual MLIR")

    root = bundle["root"]
    if not isinstance(root, dict) or frozenset(root) != {
            "symbol", "kind", "profile"
    }:
        raise ValueError(
            "qlx.build/v2 root must contain exactly symbol, kind, and profile")
    if any(not isinstance(root[name], str) or not root[name]
           for name in ("symbol", "kind", "profile")):
        raise ValueError(
            "qlx.build/v2 root symbol, kind, and profile must be nonempty strings"
        )
    if _root_operation_name(root["kind"]) is None:
        raise ValueError("qlx.build/v2 root kind is unsupported")
    kind_name = root["kind"].rsplit(".", 1)[-1]
    allowed_kinds = {kind_name}
    if not kind_name[:1].islower():
        # Accept the historical spelling on replay while serializing newly
        # authored typed roots with their semantic package owner.
        allowed_kinds.add(f"cudaq.logical.model.qec.{kind_name}")
        owner = _TYPED_ROOT_KIND_OWNERS.get(kind_name)
        if owner is not None:
            allowed_kinds.add(owner)
    if root["kind"] not in allowed_kinds:
        raise ValueError("qlx.build/v2 root kind must be canonical")
    profile = bundle["profile"]
    if not isinstance(profile, str) or not profile:
        raise ValueError("qlx.build/v2 profile must be a nonempty string")
    if root["profile"] != profile:
        raise ValueError(
            "qlx.build/v2 root profile differs from the build profile")
    if profile not in _root_profiles(root["kind"]):
        raise ValueError(
            "qlx.build/v2 profile is not valid for the committed root kind")
    try:
        expected_stage, _ = stage_and_facets(profile)
    except ValueError as error:
        raise ValueError("qlx.build/v2 profile is unsupported") from error
    expected_stage = None if expected_stage is None else expected_stage.value
    if bundle["stage"] != expected_stage:
        raise ValueError(
            "qlx.build/v2 stage differs from the committed profile")

    facets = bundle["facets"]
    if not isinstance(facets, list) or not all(
            isinstance(facet, str) for facet in facets):
        raise ValueError("qlx.build/v2 facets must be a string list")
    try:
        normalized_facets = normalize_facets(facets)
    except ValueError as error:
        raise ValueError(
            "qlx.build/v2 contains an unsupported facet") from error
    if [facet.value for facet in normalized_facets] != facets:
        raise ValueError(
            "qlx.build/v2 facets must be unique and canonically ordered")
    _, profile_facets = stage_and_facets(profile)
    required_facets = normalize_facets((
        *profile_facets,
        *facets_for_kind(kind_name),
    ))
    missing_required = [
        facet.value for facet in required_facets if facet.value not in facets
    ]
    if missing_required:
        raise ValueError(
            "qlx.build/v2 facets omit required root/profile facets: " +
            ", ".join(missing_required))
    if expected_stage is not None:
        stage_ordinal = _STAGE_ORDINAL[expected_stage]
        for facet in facets:
            minimum = _FACET_MINIMUM_STAGE[facet]
            if stage_ordinal < _STAGE_ORDINAL[minimum]:
                raise ValueError(
                    f"qlx.build/v2 facet {facet!r} is not valid at "
                    f"stage {expected_stage}")

    pipeline = bundle["pipeline"]
    if pipeline is not None:
        if (not isinstance(pipeline, dict) or
                frozenset(pipeline) != {"output_profile", "passes"} or
                not isinstance(pipeline["passes"], list)):
            raise ValueError(
                "qlx.build/v2 pipeline must contain output_profile and passes")
        if pipeline["output_profile"] != profile:
            raise ValueError(
                "qlx.build/v2 pipeline output profile differs from the build profile"
            )
        facet_fields = (
            "requires_facets",
            "provides_facets",
            "preserves_facets",
            "invalidates_facets",
            "recomputes_facets",
        )
        for item in pipeline["passes"]:
            if not isinstance(item,
                              dict) or frozenset(item) != _BUILD_V2_PASS_FIELDS:
                raise ValueError(
                    "qlx.build/v2 pipeline pass fields differ from the "
                    "committed schema")
            if not isinstance(item["name"], str) or not item["name"]:
                raise ValueError(
                    "qlx.build/v2 pipeline pass name must be a nonempty string")
            options = item["options"]
            if not isinstance(options, list):
                raise ValueError(
                    "qlx.build/v2 pipeline pass options must be a list")
            option_names = set()
            for option in options:
                if (not isinstance(option, list) or len(option) != 2 or
                        not isinstance(option[0], str) or not option[0]):
                    raise ValueError(
                        "qlx.build/v2 pipeline pass options must contain "
                        "nonempty string-key pairs")
                if option[0] in option_names:
                    raise ValueError(
                        "qlx.build/v2 pipeline pass option names must be unique"
                    )
                option_names.add(option[0])
            for field in facet_fields:
                values = item[field]
                if not isinstance(values, list) or not all(
                        isinstance(value, str) for value in values):
                    raise ValueError(
                        f"qlx.build/v2 pipeline pass {field} must be a "
                        "string list")
                try:
                    normalized = normalize_facets(values)
                except ValueError as error:
                    raise ValueError(
                        f"qlx.build/v2 pipeline pass {field} contains an "
                        "unsupported facet") from error
                if len(normalized) != len(values):
                    raise ValueError(
                        f"qlx.build/v2 pipeline pass {field} must be unique")

    value_groups = bundle["value_groups"]
    if not isinstance(value_groups, list):
        raise ValueError("qlx.build/v2 value_groups must be a list")
    names = set()
    allocations = set()
    for item in value_groups:
        if not isinstance(item, dict) or frozenset(item) != {
                "allocation", "name", "count"
        }:
            raise ValueError(
                "qlx.build/v2 value group must contain exactly allocation, "
                "name, and count")
        allocation = item["allocation"]
        count = item["count"]
        name = item["name"]
        if (isinstance(allocation, bool) or not isinstance(allocation, int) or
                allocation < 0):
            raise ValueError("qlx.build/v2 value group allocation must be a "
                             "nonnegative integer")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(
                "qlx.build/v2 value group count must be a nonnegative integer")
        if not isinstance(name, str) or not name:
            raise ValueError(
                "qlx.build/v2 value group name must be a nonempty string")
        if name in names:
            raise ValueError("qlx.build/v2 value group names must be unique")
        if allocation in allocations:
            raise ValueError(
                "qlx.build/v2 value group allocations must be unique")
        names.add(name)
        allocations.add(allocation)

    _validate_v2_nested_metadata(bundle)
