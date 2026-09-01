# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class LowerCtx:
    source_stage: str
    source_facets: tuple[str, ...] = ()
    entry: str | None = None
    bit_count: int = 0
    evidence: list[str] = field(default_factory=list)
    _verification_receipt: object | None = field(default=None, repr=False)


class _VerifiedModuleReceipt:
    """Invocation-owned proof that core verified one exact private module."""

    __slots__ = ("module",)

    def __init__(self, module):
        self.module = module


def _verified_module_receipt(module):
    return _VerifiedModuleReceipt(module)


def lower(spec, build):
    """Run one recipe through a private replay of an immutable Build."""
    from ..compiler import Build

    if not isinstance(build, Build):
        raise TypeError("target lowering requires a cudaq.logical.Build")
    if build.stage is None:
        raise ValueError(
            "cross-stage manifest builds are not executable targets")
    stage = str(build.stage.value)
    facets = tuple(facet.value for facet in build.facets)
    if spec.accepted_stages and stage not in spec.accepted_stages:
        accepted = ", ".join(spec.accepted_stages)
        raise ValueError(
            f"target recipe accepts stages [{accepted}], got {stage}")
    missing = tuple(
        facet for facet in spec.required_facets if facet not in facets)
    if missing:
        raise ValueError(
            f"target recipe requires facets {missing!r}, got {facets!r}")
    work = build._fresh_module()
    ctx = LowerCtx(
        source_stage=stage,
        source_facets=facets,
        entry=build.root.symbol,
    )
    for stage in spec.stages:
        stage.apply(work, ctx)
        ctx.evidence.append(stage.describe())
    if not work.operation.verify():
        raise ValueError("target lowering produced invalid MLIR")
    ctx._verification_receipt = _verified_module_receipt(work)
    return work, ctx
