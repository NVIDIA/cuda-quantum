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


@dataclass(frozen=True, slots=True)
class _ExecutionVisit:
    operation: object
    multiplier: int
    suppress_inline_analysis: bool
    call_stack: tuple[str, ...] = ()
    call_path: tuple[object, ...] = ()
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

    def walk(items, multiplier, stack, path, suppress):
        for operation in items:
            name = operation.name
            if name == "cflow.repeat":
                count = int(operation.attributes["count"])
                if count < 0:
                    raise error_type("cflow.repeat count must be non-negative")
                yield _ExecutionVisit(
                    operation,
                    multiplier,
                    suppress,
                    call_stack=stack,
                    call_path=path,
                )
                if count:
                    yield from walk(
                        _children(operation),
                        multiplier * count,
                        stack,
                        path,
                        suppress,
                    )
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

                profile = None
                if name == "fabric.call" and "profile" in operation.attributes:
                    profile_name = _attr_text(operation.attributes["profile"])
                    profile = symbols.get(profile_name)
                    if profile is None or profile.name != "fabric.gadget_profile":
                        raise error_type(
                            f"Fabric call references missing profile @{profile_name}"
                        )

                yield _ExecutionVisit(
                    operation,
                    multiplier,
                    suppress,
                    call_stack=stack,
                    call_path=path,
                    callee_name=callee,
                    callee_kind=target.name,
                    callee_multiplier=callee_multiplier,
                )
                yield from walk(
                    _children(target),
                    callee_multiplier,
                    (*stack, callee),
                    (*path, operation),
                    profile is not None,
                )
                if profile is not None:
                    yield from walk(_children(profile), multiplier, stack, path,
                                    False)
                continue

            yield _ExecutionVisit(
                operation,
                multiplier,
                suppress,
                call_stack=stack,
                call_path=path,
            )
            yield from walk(_children(operation), multiplier, stack, path,
                            suppress)

    yield from walk(
        operations,
        1,
        tuple(call_stack),
        (),
        bool(suppress_inline_analysis),
    )


__all__ = [
    "_EXECUTABLE_CALLEE_OPS",
    "_ExecutionVisit",
    "_attr_text",
    "_children",
    "_symbol",
    "_walk_executable",
]
