# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Strict link-complete verification for canonical builds.

Per-op verifiers deliberately tolerate unresolved symbols so partially
linked modules can be constructed and inspected. A *final* build has no
such excuse: every symbol reference must resolve inside the module, and
the module's schema identifiers must be the ones this implementation
emits. This pass turns "partially linked" from a silent state into an
explicit, typed failure at the boundary where it matters.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import ir as mlir_ir

SUPPORTED_IR_VERSIONS = ("0.4-draft",)
SUPPORTED_MODEL_VERSIONS = ("0.3.10-proposed",)

_STAGE_VOCABULARY = {"p0", "p1", "p2"}
_FACET_VOCABULARY = {
    "qec_spec",
    "qec_realization",
    "protocol_network",
}
# Compatibility profile spellings retained by serialized builds.
_PROFILE_VOCABULARY = {
    "p0",
    "p1",
    "p2",
    "p2s",
    "p2a",
    "p2n",
    "common",
}


class LinkageError(Exception):
    """A canonical build referenced a symbol it does not contain."""


@dataclass(frozen=True, slots=True)
class LinkReport:
    defined: int
    references: int
    unresolved: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.unresolved


def _collect_symbol_refs(attr, sink) -> None:
    text = str(attr)
    if "@" not in text:
        return
    # FlatSymbolRefAttr and SymbolRefAttr print as @name or @outer::@inner;
    # walk the printed form once rather than depending on binding classes
    # that differ across attribute kinds (arrays, dictionaries, and custom
    # typed attributes).
    index = 0
    while True:
        index = text.find("@", index)
        if index < 0:
            return
        index += 1
        if index < len(text) and text[index] == '"':
            end = text.find('"', index + 1)
            if end < 0:
                return
            sink(text[index + 1:end])
            index = end + 1
            continue
        start = index
        while index < len(text) and (text[index].isalnum() or
                                     text[index] in "_$."):
            index += 1
        if index > start:
            sink(text[start:index])


def check_linkage(module) -> LinkReport:
    """Collect defined symbols and every symbol reference in one module."""

    defined: set[str] = set()
    referenced: list[str] = []

    def visit(operation) -> None:
        if "sym_name" in operation.attributes:
            defined.add(
                mlir_ir.StringAttr(operation.attributes["sym_name"]).value)
        for entry in operation.attributes:
            name = entry if isinstance(entry, str) else entry.name
            if name == "sym_name":
                continue
            attr = (operation.attributes[name]
                    if isinstance(entry, str) else entry.attr)
            _collect_symbol_refs(attr, referenced.append)
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)

    for child in module.body.operations:
        visit(child.operation)

    unresolved = tuple(
        sorted({name for name in referenced if name not in defined}))
    return LinkReport(
        defined=len(defined),
        references=len(referenced),
        unresolved=unresolved,
    )


def check_module_metadata(module) -> tuple[str, ...]:
    """Validate schema identifiers and stage/facet vocabulary on one module."""

    problems: list[str] = []
    attrs = module.operation.attributes

    def text_values(name):
        if name not in attrs:
            return None
        attr = attrs[name]
        try:
            return tuple(
                mlir_ir.StringAttr(item).value
                for item in mlir_ir.ArrayAttr(attr))
        except (ValueError, TypeError):
            return (mlir_ir.StringAttr(attr).value,)

    ir_version = text_values("qlx.ir_version")
    if ir_version is not None and ir_version[0] not in SUPPORTED_IR_VERSIONS:
        problems.append(f"unsupported qlx.ir_version {ir_version[0]!r}; this "
                        f"implementation reads {SUPPORTED_IR_VERSIONS}")
    model_version = text_values("qlx.model_version")
    if model_version is not None and model_version[
            0] not in SUPPORTED_MODEL_VERSIONS:
        problems.append(
            f"unsupported qlx.model_version {model_version[0]!r}; this "
            f"implementation reads {SUPPORTED_MODEL_VERSIONS}")
    stages = text_values("qlx.stages")
    if stages is not None:
        unknown = sorted(set(stages) - _STAGE_VOCABULARY)
        if unknown:
            problems.append(f"unknown qlx.stages entries: {unknown}")
    facets = text_values("qlx.facets")
    if facets is not None:
        unknown = sorted(set(facets) - _FACET_VOCABULARY)
        if unknown:
            problems.append(f"unknown qlx.facets entries: {unknown}")
    profiles = text_values("qlx.profiles")
    if profiles is not None:
        unknown = sorted(set(profiles) - _PROFILE_VOCABULARY)
        if unknown:
            problems.append(f"unknown qlx.profiles entries: {unknown}")
    return tuple(problems)


def verify_linked(build, *, allow_unresolved=()) -> LinkReport:
    """Fail closed unless the build is link-complete with valid metadata.

    ``allow_unresolved`` names symbols an external contract resolves later
    (e.g. a target-runtime ABI symbol); every other dangling reference is a
    typed :class:`LinkageError`.
    """

    # A linkage report is authoritative build evidence.  Never derive it from
    # the mutable cached inspection view exposed as ``Build.module``.
    module = build._fresh_module() if hasattr(build, "_fresh_module") else build
    report = check_linkage(module)
    allowed = set(allow_unresolved)
    dangling = tuple(name for name in report.unresolved if name not in allowed)
    problems = check_module_metadata(module)
    if dangling or problems:
        details = []
        if dangling:
            details.append("unresolved symbol references: " +
                           ", ".join(dangling))
        details.extend(problems)
        raise LinkageError("build is not link-complete: " + "; ".join(details))
    return report


__all__ = [
    "LinkReport",
    "LinkageError",
    "check_linkage",
    "check_module_metadata",
    "verify_linked",
]
