# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Canonical content identities for typed protocol definitions."""

from __future__ import annotations

from hashlib import sha256
import json
import re

from cudaq.logical.protocols.definition import ProtocolDefinition

from .context import CompilationContext


def _normalize_protocol_root(text: str, root: str) -> str:
    """Normalize only a protocol's own symbol in typed IR payloads."""

    text = text.replace(f'@"{root}"', "@__protocol_root__")
    return re.sub(
        rf"@{re.escape(root)}(?![A-Za-z0-9_.$-])",
        "@__protocol_root__",
        text,
    )


def protocol_operation_payload(operation, root: str):
    """Return an exact structural payload, including SSA wiring and regions."""

    value_ids = {}

    def normalized(value):
        return _normalize_protocol_root(str(value), root)

    def region_payload(region, path):
        blocks = []
        for block_index, block in enumerate(region.blocks):
            block_path = (*path, block_index)
            arguments = tuple(
                normalized(value.type) for value in block.arguments)
            for index, value in enumerate(block.arguments):
                value_ids[value] = ("argument", block_path, index)
            blocks.append((
                arguments,
                tuple(
                    operation_payload(child.operation, (*block_path, index))
                    for index, child in enumerate(block.operations)),
            ))
        return tuple(blocks)

    def operation_payload(current, path):
        operands = tuple(
            value_ids.get(value, ("external", normalized(value.type)))
            for value in current.operands)
        results = tuple(normalized(value.type) for value in current.results)
        for index, value in enumerate(current.results):
            value_ids[value] = ("result", path, index)
        attributes = tuple(
            sorted((name, normalized(attribute))
                   for name, attribute in current.attributes.items()
                   if not (path == ("root",) and name == "sym_name")))
        regions = tuple(
            region_payload(region, (*path, "region", index))
            for index, region in enumerate(current.regions))
        return current.name, operands, results, attributes, regions

    return operation_payload(operation, ("root",))


def _symbol_leaf(attribute) -> str:
    raw = getattr(attribute, "value", attribute)
    if isinstance(raw, (tuple, list)):
        return str(raw[-1])
    value = str(raw).strip('"')
    if value.startswith("@"):
        value = value[1:]
    return value.split("::@")[-1]


def _normalize_protocol_stream_qualification(payload):
    """Erase only placement-owned logical-machine scope from stream refs."""

    operation_name, operands, results, attributes, regions = payload
    normalized_attributes = tuple(
        (name, f"@{_symbol_leaf(value)}" if name == "stream" else value)
        for name, value in attributes)
    normalized_regions = tuple(
        tuple((
            arguments,
            tuple(
                _normalize_protocol_stream_qualification(child)
                for child in operations),
        )
              for arguments, operations in region)
        for region in regions)
    return (
        operation_name,
        operands,
        results,
        normalized_attributes,
        normalized_regions,
    )


def _normalize_selected_protocol_facts(payload, *, root: bool = True):
    """Erase only P2-selection facts absent from a detached definition."""

    operation_name, operands, results, attributes, regions = payload
    normalized_attributes = []
    for name, value in attributes:
        if root and name in {"action_site", "generated_by", "specialization"}:
            continue
        if name == "stream":
            value = "@__selected_stream__"
        elif name == "payload_logical_block_ids":
            block_ids = json.loads(value)
            value = f"__selected_qec_blocks__:{len(block_ids)}"
        normalized_attributes.append((name, value))
    normalized_regions = tuple(
        tuple((
            arguments,
            tuple(
                _normalize_selected_protocol_facts(child, root=False)
                for child in operations),
        )
              for arguments, operations in region)
        for region in regions)
    return (
        operation_name,
        operands,
        results,
        tuple(normalized_attributes),
        normalized_regions,
    )


def retained_protocol_matches(
    transaction,
    operation,
    root: str,
    expected_payload,
    *,
    normalize_selection: bool = False,
) -> bool:
    """Compare a retained protocol and its link closure with a definition."""

    if (not isinstance(expected_payload, tuple) or len(expected_payload) != 3 or
            expected_payload[0] != "qlx.protocol-definition-closure.v1"):
        raise ValueError("unsupported retained protocol identity payload")
    actual_root = _normalize_protocol_stream_qualification(
        protocol_operation_payload(operation, root))
    expected_root = _normalize_protocol_stream_qualification(
        expected_payload[1])
    if normalize_selection:
        actual_root = _normalize_selected_protocol_facts(actual_root)
        expected_root = _normalize_selected_protocol_facts(expected_root)
    if actual_root != expected_root:
        return False
    for symbol, operation_name, payload in expected_payload[2]:
        retained_symbol = str(symbol).replace("__protocol_root__", root)
        dependency = transaction.find_symbol(retained_symbol, operation_name)
        if dependency is None:
            return False
        actual_dependency = _normalize_protocol_stream_qualification(
            protocol_operation_payload(dependency, root))
        expected_dependency = _normalize_protocol_stream_qualification(payload)
        if normalize_selection:
            actual_dependency = _normalize_selected_protocol_facts(
                actual_dependency)
            expected_dependency = _normalize_selected_protocol_facts(
                expected_dependency)
        if actual_dependency != expected_dependency:
            return False
    return True


def protocol_definition_payload(
    definition: ProtocolDefinition,
    *,
    payload_blocks: tuple[str, ...] | None = None,
):
    """Materialize and fingerprint the definition's isolated link closure.

    A root-only body is not a replay commitment: a same-named nested protocol
    or gadget could otherwise change while the root's ``fabric.call`` remains
    byte-for-byte identical.  The fresh transaction contains only declarations
    reached while tracing this definition, so committing every named operation
    in that transaction authenticates the complete typed closure without
    materializing P2 bodies at P1.
    """

    transaction = CompilationContext()
    if payload_blocks is not None:
        transaction.bind_protocol_payload_blocks(definition, payload_blocks)
    handle = transaction.materialize(definition)
    operation = transaction.find_symbol(handle.symbol, "fabric.protocol")
    if operation is None:
        raise ValueError(
            f"protocol definition {definition.name!r} did not materialize a "
            "fabric.protocol")
    named = {}
    for candidate in transaction.walk():
        if "sym_name" not in candidate.attributes:
            continue
        attribute = candidate.attributes["sym_name"]
        symbol = str(getattr(attribute, "value", attribute)).strip('"')
        if symbol in named:
            raise ValueError(
                "protocol definition closure contains duplicate symbol "
                f"{symbol!r}")
        named[symbol] = candidate

    for candidate in transaction.walk():
        if candidate.name != "fabric.call" or "callee" not in candidate.attributes:
            continue
        attribute = candidate.attributes["callee"]
        callee = str(getattr(attribute, "value", attribute)).strip('@"')
        if callee not in named:
            raise ValueError(
                "protocol definition closure contains unresolved fabric.call "
                f"callee {callee!r}")

    closure = []
    for symbol, candidate in named.items():
        operation_name = candidate.name
        if operation_name == "fabric.protocol" and symbol == handle.symbol:
            continue
        closure.append((
            _normalize_protocol_root(symbol, handle.symbol),
            operation_name,
            protocol_operation_payload(candidate, handle.symbol),
        ))
    closure.sort(key=lambda value: json.dumps(
        value, ensure_ascii=True, separators=(",", ":")))
    return (
        "qlx.protocol-definition-closure.v1",
        protocol_operation_payload(operation, handle.symbol),
        tuple(closure),
    )


def protocol_definition_sha256(
    definition: ProtocolDefinition,
    *,
    payload_blocks: tuple[str, ...] | None = None,
) -> str:
    """Commit the canonical typed protocol payload retained across P1 replay."""

    payload = json.dumps(
        protocol_definition_payload(definition, payload_blocks=payload_blocks),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


def factory_protocol_semantics_sha256(definition: ProtocolDefinition) -> str:
    """Commit factory protocol semantics independent of its device region.

    ``logical.add_factory`` derives the allocation region from the selected
    device.  That region name is intentionally different for a one-lane
    characterization device and a larger deployment, but it is not part of
    the producer algorithm.  Re-materialize the same typed definition against
    one canonical factory region so compiled characterizations can be reused
    across capacities without permitting a different protocol body.
    """

    from cudaq.logical.architecture.logical import Space

    canonical = ProtocolDefinition(
        definition.provider,
        implements=definition.implements,
        name=definition.name,
        metadata=definition.metadata,
        type_hints=definition.type_hints,
        _factory_region=Space(name="__qlx_factory_semantics__"),
    )
    return protocol_definition_sha256(canonical)


__all__ = [
    "factory_protocol_semantics_sha256",
    "protocol_definition_payload",
    "protocol_definition_sha256",
    "protocol_operation_payload",
    "retained_protocol_matches",
]
