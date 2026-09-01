# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""One fail-closed traversal of executable Fabric call structure."""

from __future__ import annotations

from dataclasses import dataclass

_EXECUTABLE_CALLEE_OPS = frozenset({
    "fabric.call",
    "fabric.map_children",
    "fabric.relocate",
    "fabric.establish_support",
    "fabric.establish_topological_record",
})


def _attr_text(attribute) -> str:
    value = getattr(attribute, "value", None)
    if value is not None:
        return str(value)
    return str(attribute).strip('"').lstrip("@")


def _symbol(operation) -> str | None:
    try:
        return _attr_text(operation.attributes["sym_name"])
    except KeyError:
        return None


def _children(operation):
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield child.operation


def _executable_operation(operation, symbols, error_type=ValueError):
    """Resolve the operation that owns the executable body of a callable."""

    if operation.name == "fabric.gadget" and "realization" in operation.attributes:
        realization = _attr_text(operation.attributes["realization"])
        circuit = symbols.get(realization)
        if circuit is None or circuit.name != "fabric.circuit":
            raise error_type(
                f"gadget realization @{realization} does not resolve to a Fabric circuit"
            )
        return circuit
    return operation


def _executable_children(operation, symbols, error_type=ValueError):
    """Return the verified executable body, following gadget realizations."""

    return tuple(
        _children(_executable_operation(operation, symbols, error_type)))


def _patch_code_name(type_) -> str | None:
    text = str(type_)
    prefix = "!fabric.patch<"
    if not text.startswith(prefix):
        return None
    code = text[len(prefix):].split(",", 1)[0].removesuffix(">")
    return code.strip().strip('"').removeprefix("@")


def _patch_weight(type_, symbols, error_type, *, required=True) -> int:
    code_name = _patch_code_name(type_)
    if code_name is None:
        return 0
    code = symbols.get(code_name)
    if code is None or code.name != "fabric.code":
        if not required:
            return 0
        raise error_type(
            f"logical-qubit counting requires resolved code @{code_name}")
    # fabric.code defines an omitted k as the legacy k=1 form.
    dimension = int(code.attributes["k"]) if "k" in code.attributes else 1
    if dimension < 0:
        raise error_type(f"code @{code_name} has negative k dimension")
    return dimension


def _patch_peaks(
    operation,
    symbols,
    error_type=ValueError,
    *,
    require_dimensions=True,
) -> tuple[int, int]:
    """Return peak live patch owners and their logical-qubit dimensions."""

    executable = _executable_operation(operation, symbols, error_type)
    live_patches = 0
    live_logicals = 0
    if executable.regions and executable.regions[0].blocks:
        arguments = executable.regions[0].blocks[0].arguments
        patch_types = tuple(argument.type
                            for argument in arguments
                            if _patch_code_name(argument.type) is not None)
        live_patches = len(patch_types)
        live_logicals = sum(
            _patch_weight(
                type_, symbols, error_type, required=require_dimensions)
            for type_ in patch_types)

    def region_items(region):
        return tuple(child.operation
                     for block in region.blocks
                     for child in block.operations)

    def patch_totals(values):
        types = tuple(value.type
                      for value in values
                      if _patch_code_name(value.type) is not None)
        return (
            len(types),
            sum(
                _patch_weight(
                    type_, symbols, error_type, required=require_dimensions)
                for type_ in types),
        )

    def apply_delta(current, added=(0, 0), removed=(0, 0)):
        return (
            max(0, current[0] + added[0] - removed[0]),
            max(0, current[1] + added[1] - removed[1]),
        )

    def walk(items, current, stack):
        peak = current
        for child in items:
            name = child.name
            if name == "fabric.alloc":
                current = apply_delta(current,
                                      added=patch_totals(child.results))
                peak = (max(peak[0], current[0]), max(peak[1], current[1]))
                continue
            if name == "fabric.dealloc":
                # Some retained resource protocols materialize patch owners
                # through typed unpack/produce boundaries rather than an
                # executable allocation operation. Match native counting by
                # treating those as an untracked baseline instead of going
                # negative.
                current = apply_delta(current,
                                      removed=patch_totals(child.operands))
                continue
            if name == "fabric.unpack_resource":
                current = apply_delta(
                    current,
                    added=patch_totals(child.results),
                    removed=patch_totals(child.operands),
                )
                peak = (max(peak[0], current[0]), max(peak[1], current[1]))
                continue
            if name == "fabric.pack_resource":
                current = apply_delta(
                    current,
                    added=patch_totals(child.results),
                    removed=patch_totals(child.operands),
                )
                continue
            if name == "fabric.repeat":
                count = int(child.attributes["count"])
                if count < 0:
                    raise error_type("fabric.repeat count must be non-negative")
                if count:
                    current, nested_peak = walk(_children(child), current,
                                                stack)
                    peak = (max(peak[0],
                                nested_peak[0]), max(peak[1], nested_peak[1]))
                continue
            if name == "fabric.if":
                branches = [
                    walk(region_items(region), current, stack)
                    for region in child.regions
                ]
                if branches:
                    exits = {branch_live for branch_live, _ in branches}
                    if len(exits) != 1:
                        raise error_type(
                            "Fabric branches disagree on live patch ownership")
                    current = branches[0][0]
                    peak = (
                        max(peak[0], *(value[0] for _, value in branches)),
                        max(peak[1], *(value[1] for _, value in branches)),
                    )
                continue
            if name in _EXECUTABLE_CALLEE_OPS:
                callee = _attr_text(child.attributes["callee"])
                target = symbols.get(callee)
                if target is None or target.name not in {
                        "fabric.gadget", "fabric.protocol"
                }:
                    raise error_type(
                        f"unresolved executable Fabric callee @{callee}")
                if callee in stack:
                    raise error_type(
                        f"recursive executable Fabric graph through @{callee}")
                current, nested_peak = walk(
                    _executable_children(target, symbols, error_type),
                    current,
                    (*stack, callee),
                )
                peak = (max(peak[0],
                            nested_peak[0]), max(peak[1], nested_peak[1]))
                continue
            nested = tuple(_children(child))
            if nested:
                current, nested_peak = walk(nested, current, stack)
                peak = (max(peak[0],
                            nested_peak[0]), max(peak[1], nested_peak[1]))
        return current, peak

    _, peak = walk(
        _executable_children(operation, symbols, error_type),
        (live_patches, live_logicals),
        (_symbol(operation) or "<root>",),
    )
    return peak


def _logical_patch_peak(operation, symbols, error_type=ValueError) -> int:
    """Count structural peak live patch owners."""

    return _patch_peaks(operation,
                        symbols,
                        error_type,
                        require_dimensions=False)[0]


def _logical_qubit_peak(operation, symbols, error_type=ValueError) -> int:
    """Count logical qubits by summing each live patch's resolved code k."""

    return _patch_peaks(operation, symbols, error_type)[1]


@dataclass(frozen=True, slots=True)
class _ExecutionVisit:
    operation: object
    multiplier: int
    suppress_inline_analysis: bool
    call_stack: tuple[str, ...] = ()
    callee_name: str | None = None
    callee_kind: str | None = None
    callee_multiplier: int = 0


def _bundle_hierarchy(operation, error_type):
    """Resolve a map's hierarchy through its typed SSA producer chain."""

    value = operation.operands[0]
    seen = set()
    while True:
        owner = getattr(value, "owner", None)
        if owner is None or id(owner) in seen:
            raise error_type(
                "mapped child bundle has no typed hierarchy source")
        seen.add(id(owner))
        if owner.name == "fabric.encoding_unpack":
            return _attr_text(owner.attributes["hierarchy"])
        if owner.name == "fabric.map_children" and owner.operands:
            value = owner.operands[0]
            continue
        raise error_type(
            "mapped child bundle must originate in fabric.encoding_unpack")


def _walk_executable(
        operations,
        symbols,
        *,
        call_stack=(),
        error_type=ValueError,
        suppress_inline_analysis=False,
):
    """Yield wrapper and invoked-body operations with exact multiplicities."""

    def walk(items, multiplier, stack, suppress):
        for operation in items:
            name = operation.name
            if name == "fabric.repeat":
                count = int(operation.attributes["count"])
                if count < 0:
                    raise error_type("fabric.repeat count must be non-negative")
                yield _ExecutionVisit(operation,
                                      multiplier,
                                      suppress,
                                      call_stack=stack)
                if count:
                    yield from walk(_children(operation), multiplier * count,
                                    stack, suppress)
                continue

            if name in _EXECUTABLE_CALLEE_OPS:
                callee = _attr_text(operation.attributes["callee"])
                target = symbols.get(callee)
                if target is None:
                    raise error_type(
                        f"unresolved executable Fabric callee @{callee}")
                if target.name not in {"fabric.gadget", "fabric.protocol"}:
                    raise error_type(
                        f"Fabric callee @{callee} is not a gadget or protocol")
                if callee in stack:
                    raise error_type(
                        f"recursive executable Fabric graph through @{callee}")

                callee_multiplier = multiplier
                if name == "fabric.map_children":
                    hierarchy_name = _bundle_hierarchy(operation, error_type)
                    hierarchy = symbols.get(hierarchy_name)
                    if hierarchy is None or hierarchy.name != (
                            "fabric.encoding_hierarchy"):
                        raise error_type(
                            f"mapped child hierarchy @{hierarchy_name} is unresolved"
                        )
                    multiplicity = int(hierarchy.attributes["multiplicity"])
                    if multiplicity <= 0:
                        raise error_type(
                            "mapped child hierarchy multiplicity must be positive"
                        )
                    callee_multiplier *= multiplicity

                yield _ExecutionVisit(
                    operation,
                    multiplier,
                    suppress,
                    call_stack=stack,
                    callee_name=callee,
                    callee_kind=target.name,
                    callee_multiplier=callee_multiplier,
                )
                yield from walk(
                    _executable_children(target, symbols, error_type),
                    callee_multiplier,
                    (*stack, callee),
                    suppress,
                )
                continue

            yield _ExecutionVisit(operation,
                                  multiplier,
                                  suppress,
                                  call_stack=stack)
            yield from walk(_children(operation), multiplier, stack, suppress)

    yield from walk(operations, 1, tuple(call_stack),
                    bool(suppress_inline_analysis))


__all__ = [
    "_EXECUTABLE_CALLEE_OPS",
    "_ExecutionVisit",
    "_attr_text",
    "_children",
    "_logical_patch_peak",
    "_logical_qubit_peak",
    "_patch_peaks",
    "_symbol",
    "_walk_executable",
]
