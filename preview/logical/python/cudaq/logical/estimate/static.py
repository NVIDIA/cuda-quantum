# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from collections import Counter

from ._fabric_walk import (
    _attr_text,
    _children,
    _executable_children,
    _patch_peaks,
    _symbol,
    _walk_executable,
)
from .types import FabricCounts

_DECLARATIONS = {
    "fabric.code",
    "fabric.code_profile",
    "fabric.encoding",
    "fabric.encoding_hierarchy",
    "fabric.encoding_projection",
    "fabric.objective",
    "fabric.gadget_spec",
}
_TERMINATORS = {
    "fabric.return",
    "fabric.protocol_return",
    "fabric.yield",
}


def _symbol_path(attribute) -> str:
    """Return the canonical ``root::nested`` spelling of a symbol reference."""

    return "::".join(part.strip().strip('"').removeprefix("@")
                     for part in str(attribute).split("::"))


def count(build) -> FabricCounts:
    if build.profile not in {"p2a", "p2n"}:
        raise ValueError("Tier.STATIC requires a selected P2A/P2N Build")
    # Estimates are evidence derived from the immutable Build snapshot.  The
    # public ``module`` view is cached for inspection and can be mutated through
    # the underlying MLIR bindings, so never use it as an estimator input.
    module = build._fresh_module()
    try:
        verified = module.operation.verify()
    except Exception as exc:
        raise ValueError(
            "Tier.STATIC requires a module that passes native verification"
        ) from exc
    if not verified:
        raise ValueError(
            "Tier.STATIC requires a module that passes native verification")
    symbols = {}
    operations = list(module.body.operations)
    for operation in operations:
        name = _symbol(operation.operation)
        if name is not None:
            symbols[name] = operation.operation

    root = symbols.get(build.root.symbol)
    if root is None:
        raise ValueError(
            f"build root @{build.root.symbol} is absent from its module")

    operation_counts = Counter()
    gadget_calls = Counter()
    protocol_calls = Counter()
    resource_requests = Counter()
    resource_stream_requests = Counter()
    success_count = syndrome_rounds = 0
    patch_peak, logical_peak = _patch_peaks(root, symbols)

    def consume(visit):
        nonlocal success_count, syndrome_rounds
        operation = visit.operation
        multiplier = visit.multiplier
        suppress_inline_analysis = visit.suppress_inline_analysis
        name = operation.name
        if visit.callee_name is not None:
            calls = (gadget_calls if visit.callee_kind == "fabric.gadget" else
                     protocol_calls)
            calls[visit.callee_name] += visit.callee_multiplier
        if name == "fabric.selection":
            success_count += multiplier
        elif name == "fabric.resource_request":
            resource_requests[_attr_text(
                operation.attributes["kind"])] += multiplier
            resource_stream_requests[_symbol_path(
                operation.attributes["stream"])] += multiplier
        elif name in {
                "fabric.read_syndrome_ancillas",
                "fabric.assemble_syndrome",
        }:
            syndrome_rounds += multiplier
        if (name.startswith("fabric.") and name not in _DECLARATIONS and
                name not in _TERMINATORS and
                name not in {"fabric.gadget", "fabric.protocol"}):
            operation_counts[name.removeprefix("fabric.")] += multiplier

    def consume_walk(operations, **kwargs):
        for visit in _walk_executable(operations,
                                      symbols,
                                      error_type=ValueError,
                                      **kwargs):
            consume(visit)

    consume_walk(_executable_children(root, symbols),
                 call_stack=(build.root.symbol,))

    hierarchy_depths = {
        _symbol(operation.operation):
            int(operation.operation.attributes["depth"])
        for operation in operations
        if operation.operation.name == "fabric.encoding_hierarchy"
    }
    return FabricCounts(
        operation_counts=operation_counts,
        gadget_calls=gadget_calls,
        protocol_calls=protocol_calls,
        resource_requests=resource_requests,
        resource_stream_requests=resource_stream_requests,
        success_count=success_count,
        syndrome_rounds=syndrome_rounds,
        patches_peak=patch_peak,
        logical_qubits_peak=logical_peak,
        hierarchy_depths=hierarchy_depths,
        source_stage=build.stage.value,
        source_facets=tuple(facet.value for facet in build.facets),
        build_root=build.root.symbol,
        build_sha256=build.content_sha256,
    )
