# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""QEC implementation-lowering contracts used by the CUDA-Q Logical compiler."""

from .._core.lazy import public_dir as _public_dir, resolve as _resolve

_EXPORTS = {
    **{
        name: f".lowering:{name}" for name in (
            "ActionSiteHandle",
            "GeneratedQECArtifact",
            "QECCompiler",
            "QECCompilerContext",
            "QECLowering",
            "qec_lowering",
        )
    },
    "SubsystemFragmentObjective": ".objectives:SubsystemFragmentObjective",
    "subsystem_fragment": ".objectives:subsystem_fragment",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
