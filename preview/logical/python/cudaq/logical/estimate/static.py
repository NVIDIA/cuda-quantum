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
    _symbol,
    _walk_executable,
)
from .types import FabricCounts

_DECLARATIONS = {
    "fabric.detector",
    "fabric.observable",
    "fabric.output_syndrome",
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
    "fabric.profile_end",
    "fabric.model_return",
    "event.yield",
}

_CFLOW_OP_COUNT_NAMES = {
    "cflow.if": "if",
    "cflow.repeat": "repeat",
    "cflow.while": "while",
}

# `event.*` ops embedded in a Fabric body used to be `fabric.event_*`/
# `fabric.fence`/`fabric.selection`; preserve their pre-migration
# operation_counts key spelling (some of these are asserted on directly by
# name, e.g. "event_await", "selection") rather than switching to the bare
# `name.removeprefix("event.")` form a naive generalization would produce.
_EVENT_OP_COUNT_NAMES = {
    "event.test": "event_test",
    "event.poll": "event_poll",
    "event.is": "event_is",
    "event.select_ready": "event_select_ready",
    "event.try_take": "event_try_take",
    "event.cancel": "event_cancel",
    "event.await": "event_await",
    "event.fence": "fence",
    "event.selection": "selection",
}


def _patch_capacity(type_, symbols) -> int | None:
    text = str(type_)
    prefix = "!fabric.patch<@"
    if not text.startswith(prefix):
        return None
    code = text[len(prefix):].split(",", 1)[0].rstrip(">")
    code_op = symbols.get(code)
    return int(code_op.attributes["k"]) if code_op is not None else 1


def _root_patch_peaks(root, symbols) -> tuple[int, int]:
    """Count peak live patch values at the selected realization boundary."""

    if not root.regions or not root.regions[0].blocks:
        return 0, 0
    block = root.regions[0].blocks[0]
    live = {}
    for argument in block.arguments:
        capacity = _patch_capacity(argument.type, symbols)
        if capacity is not None:
            live[argument] = capacity
    patches_peak = len(live)
    logical_peak = sum(live.values())
    for operation_view in block.operations:
        operation = operation_view.operation
        for operand in operation.operands:
            live.pop(operand, None)
        for result in operation.results:
            capacity = _patch_capacity(result.type, symbols)
            if capacity is not None:
                live[result] = capacity
        patches_peak = max(patches_peak, len(live))
        logical_peak = max(logical_peak, sum(live.values()))
    return patches_peak, logical_peak


def count(build) -> FabricCounts:
    if build.profile not in {"p2a", "p2n"}:
        raise ValueError("Tier.STATIC requires a selected P2A/P2N Build")
    # Estimates are evidence derived from the immutable Build snapshot.  The
    # public ``module`` view is cached for inspection and can be mutated through
    # the underlying MLIR bindings, so never use it as an estimator input.
    module = build._fresh_module()
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

    # Count each body once and scale its totals by call-site multiplicity.
    summaries = {}

    def consume(visit, counts):
        operation = visit.operation
        multiplier = visit.multiplier
        name = operation.name
        if visit.callee_name is not None:
            kind = ("gadget"
                    if visit.callee_kind == "fabric.gadget" else "protocol")
            counts[kind, visit.callee_name] += visit.callee_multiplier
        if name == "fabric.success":
            counts["success", ""] += multiplier
        elif name in {
                "fabric.read_syndrome_ancillas",
                "fabric.assemble_syndrome",
        }:
            counts["syndrome", ""] += multiplier
        if (name.startswith("fabric.") and name not in _DECLARATIONS and
                name not in _TERMINATORS and name not in {
                    "fabric.gadget", "fabric.protocol", "fabric.gadget_profile"
                }):
            counts["operation", name.removeprefix("fabric.")] += multiplier
        elif name in _CFLOW_OP_COUNT_NAMES:
            counts["operation", _CFLOW_OP_COUNT_NAMES[name]] += multiplier
        elif name in _EVENT_OP_COUNT_NAMES:
            counts["operation", _EVENT_OP_COUNT_NAMES[name]] += multiplier

    def add(counts, nested, multiplier):
        for key, value in nested.items():
            counts[key] += value * multiplier

    def summarize(name, stack):
        if name in stack:
            raise ValueError(
                f"recursive executable Fabric graph through @{name}")
        if name not in summaries:
            summaries[name] = consume_walk(_children(symbols[name]),
                                           call_stack=(*stack, name))
        return summaries[name]

    def consume_walk(operations, *, call_stack=()):
        counts = Counter()
        for visit in _walk_executable(operations,
                                      symbols,
                                      error_type=ValueError,
                                      call_stack=call_stack,
                                      expand_calls=False):
            consume(visit, counts)
            if visit.callee_name is None:
                continue
            add(counts, summarize(visit.callee_name, call_stack),
                visit.callee_multiplier)
            operation = visit.operation
            if operation.name == "fabric.call" and "profile" in operation.attributes:
                add(
                    counts,
                    summarize(_attr_text(operation.attributes["profile"]),
                              call_stack), visit.multiplier)
        return counts

    # A profile is analysis attached to a realization, so count both its
    # declarative body and the referenced executable gadget exactly once.
    if root.name == "fabric.gadget_profile":
        counts = consume_walk(_children(root))
        gadget = _attr_text(root.attributes["gadget"])
        realization = symbols.get(gadget)
        if realization is None:
            raise ValueError(f"profile references missing gadget @{gadget}")
        add(counts, summarize(gadget, ()), 1)
    else:
        counts = summarize(build.root.symbol, ())

    def category(kind):
        return Counter({
            name: value for (tag, name), value in counts.items() if tag == kind
        })

    patches_peak, logical_peak = _root_patch_peaks(root, symbols)
    hierarchy_depths = {
        _symbol(operation.operation):
            int(operation.operation.attributes["depth"])
        for operation in operations
        if operation.operation.name == "fabric.encoding_hierarchy"
    }
    return FabricCounts(
        operation_counts=category("operation"),
        gadget_calls=category("gadget"),
        protocol_calls=category("protocol"),
        success_count=counts["success", ""],
        syndrome_rounds=counts["syndrome", ""],
        patches_peak=patches_peak,
        logical_qubits_peak=logical_peak,
        hierarchy_depths=hierarchy_depths,
        source_stage=build.stage.value,
        source_facets=tuple(facet.value for facet in build.facets),
        build_root=build.root.symbol,
        build_sha256=build.content_sha256,
    )
