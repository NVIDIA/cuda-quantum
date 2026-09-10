# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping


def _walk(operation):
    yield operation
    for region in operation.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from _walk(child.operation)


class Stage:

    def apply(self, module, ctx) -> None:
        raise NotImplementedError

    def describe(self) -> str:
        return type(self).__name__


class EnsureFabric(Stage):

    def apply(self, module, ctx) -> None:
        if any(
                op.name.startswith("fabric.") or op.name == "phys.graph"
                for op in _walk(module.operation)):
            return
        raise ValueError(
            "target recipe requires Fabric or resolved physical IR; "
            "select/lower the P0/P1 build first")

    def describe(self) -> str:
        return "ensure-fabric-or-physical"


class LowerProtocolsForBackend(Stage):
    """Reify one closed P2N protocol graph as a backend entry gadget."""

    def apply(self, module, ctx) -> None:
        if not ctx.entry:
            raise ValueError("backend legalization requires a root symbol")
        root = None
        for view in module.body.operations:
            operation = view.operation
            attrs = operation.attributes
            if "sym_name" not in attrs:
                continue
            symbol = str(attrs["sym_name"]).strip('"')
            if symbol == ctx.entry:
                root = operation
                break
        if root is None or root.name not in {
                "fabric.protocol", "fabric.gadget"
        }:
            if root is not None and root.name == "phys.graph":
                # P3 already contains the selected physical realization.
                # Preserve its structured measurement feedback for adaptive
                # target finalizers instead of forcing it back through the
                # gadget-only backend legalization pass.
                return
            raise ValueError(
                f"backend root @{ctx.entry} is not a Fabric protocol or gadget")
        function_type = root.attributes["function_type"].value
        if function_type.inputs:
            raise ValueError(
                "native execution requires a closed root protocol; "
                "compile a P0/P1 program through "
                "cudaq.logical.compiler.pipelines.qec()")

        from cudaq.logical._native import native

        native.run_pass(module, "fabric-inline-realisations")
        native.run_pass(
            module,
            f"fabric-lower-protocols{{root-symbol={ctx.entry}}}",
        )

    def describe(self) -> str:
        return "native:fabric-inline-realisations+fabric-lower-protocols"


class SelectFabricSource(Stage):
    """Select the retained Fabric source when a target is handed P3 IR.

    P3 builds preserve the P2 protocol that generated their physical
    event graph. Fabric-native targets compile that semantic source rather
    than reconstructing a circuit from backend text or from P3 events.
    """

    def apply(self, module, ctx) -> None:
        if not ctx.entry:
            raise ValueError("Fabric source selection requires a root symbol")
        for view in module.body.operations:
            operation = view.operation
            attrs = operation.attributes
            if "sym_name" not in attrs:
                continue
            symbol = str(attrs["sym_name"]).strip('"')
            if symbol != ctx.entry or operation.name != "phys.graph":
                continue
            if "source_protocol" not in attrs:
                raise ValueError(
                    f"physical graph @{ctx.entry} has no retained Fabric source"
                )
            ctx.entry = str(attrs["source_protocol"]).strip('@"')
            return

    def describe(self) -> str:
        return "select-retained-fabric-source"


@dataclass(frozen=True, slots=True)
class EnsureStage(Stage):
    stages: tuple[str, ...]

    def __init__(self, *stages: str) -> None:
        object.__setattr__(
            self,
            "stages",
            tuple(str(getattr(value, "value", value)) for value in stages),
        )

    def apply(self, module, ctx) -> None:
        if ctx.source_stage not in self.stages:
            raise ValueError(
                f"stage requires one of {self.stages}, got {ctx.source_stage}")

    def describe(self) -> str:
        return "ensure-stage:" + ",".join(self.stages)


@dataclass(frozen=True, slots=True)
class EnsureFacets(Stage):
    facets: tuple[str, ...]

    def __init__(self, *facets: str) -> None:
        object.__setattr__(
            self,
            "facets",
            tuple(str(getattr(value, "value", value)) for value in facets),
        )

    def apply(self, module, ctx) -> None:
        missing = tuple(
            facet for facet in self.facets if facet not in ctx.source_facets)
        if missing:
            raise ValueError(
                f"stage requires facets {missing!r}, got {ctx.source_facets!r}")

    def describe(self) -> str:
        return "ensure-facets:" + ",".join(self.facets)


@dataclass(frozen=True, slots=True)
class Guard(Stage):
    ok: Callable[[object], bool]
    message: str
    error: type[Exception] = NotImplementedError

    def apply(self, module, ctx) -> None:
        if not self.ok(module):
            raise self.error(self.message)

    def describe(self) -> str:
        return f"guard:{self.message}"


@dataclass(frozen=True, slots=True)
class PyPass(Stage):
    fn: Callable
    kwargs: Mapping = field(default_factory=dict)
    label: str = ""

    def apply(self, module, ctx) -> None:
        self.fn(module, **dict(self.kwargs))

    def describe(self) -> str:
        return f"py:{self.label or getattr(self.fn, '__name__', 'pass')}"
