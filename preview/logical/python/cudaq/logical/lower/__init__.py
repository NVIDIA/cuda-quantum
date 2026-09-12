# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys
import types as _py_types

from .driver import LowerCtx, lower
from .finalize import (
    emit_mlir,
    py_walker,
    translate_text,
)
from .lanes import lanes_bytecode_text
from .passes import Translation
from .physical import module_has_resource_kind, physical_manifest_text
from .stim import StimEmission, emit_stim, emit_stim_artifact
from .stage import (
    EnsureFabric,
    EnsureFacets,
    EnsureStage,
    Guard,
    LowerProtocolsForBackend,
    SelectFabricSource,
    PyPass,
    Stage,
)
from .target import LoweringSpec


class _CallableLower(_py_types.ModuleType):
    """Continue an immutable build through an explicit compiler pipeline.

    The namespace remains home to target-recipe primitives such as
    ``LoweringSpec`` while also supporting the concise model spelling
    ``cudaq.logical.lower(build, device=..., pipeline=...)``.
    """

    def __call__(
            self,
            build,
            *,
            device=None,
            pipeline=None,
            placement=(),
            constraints=None,
            objective=None,
    ):
        from ..compiler import compile, pipelines

        return compile(
            build,
            pipeline=pipeline or pipelines.physical(),
            device=device,
            placement=placement,
            constraints=constraints,
            objective=objective,
        )


sys.modules[__name__].__class__ = _CallableLower

__all__ = [
    "LowerCtx",
    "lower",
    "LoweringSpec",
    "Translation",
    "Stage",
    "EnsureFabric",
    "EnsureStage",
    "EnsureFacets",
    "Guard",
    "LowerProtocolsForBackend",
    "SelectFabricSource",
    "PyPass",
    "emit_mlir",
    "translate_text",
    "py_walker",
    "module_has_resource_kind",
    "physical_manifest_text",
    "lanes_bytecode_text",
    "StimEmission",
    "emit_stim",
    "emit_stim_artifact",
]
