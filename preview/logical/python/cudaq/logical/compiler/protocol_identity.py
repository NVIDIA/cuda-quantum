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

from ..protocols.definition import ProtocolDefinition

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


def protocol_definition_payload(definition: ProtocolDefinition):
    """Materialize and fingerprint the definition's isolated link closure.

    A root-only body is not a replay commitment: a same-named nested protocol
    or gadget could otherwise change while the root's ``fabric.call`` remains
    byte-for-byte identical.  The fresh transaction contains only declarations
    reached while tracing this definition, so committing every named operation
    in that transaction authenticates the complete typed closure without
    materializing P2 bodies at P1.
    """

    transaction = CompilationContext()
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


def protocol_definition_sha256(definition: ProtocolDefinition) -> str:
    """Commit the canonical typed protocol payload retained across P1 replay."""

    payload = json.dumps(
        protocol_definition_payload(definition),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{sha256(payload).hexdigest()}"


__all__ = [
    "protocol_definition_payload",
    "protocol_definition_sha256",
    "protocol_operation_payload",
]
