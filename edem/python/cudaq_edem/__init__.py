# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""EDEM v0.x bindings and packed values.

**CUDA-Q EDEM is provided for early experimental scenarios. Its APIs are evolving
quickly and may change incompatibly during v0.x development.**

The interfaces and binary compatibility are subject to change during development.
"""

from pathlib import Path

from . import _core
from ._core import (
    ABI_VERSION,
    FORMAT_VERSION,
    Block,
    CodePresentation,
    EDEM,
    Endpoint,
    FaultDistribution,
    FaultGroup,
    GroupKind,
    PortKind,
    SparseMatrix,
)

__version__ = "@CUDAQ_EDEM_VERSION@"


def get_cmake_dir() -> str:
    """Return the installed cudaq-edem CMake package directory."""
    return str((Path(__file__).resolve().parent /
                "@CUDAQ_EDEM_PYTHON_CMAKE_DIR@").resolve())


__all__ = [
    "ABI_VERSION",
    "FORMAT_VERSION",
    "Block",
    "CodePresentation",
    "EDEM",
    "Endpoint",
    "FaultDistribution",
    "FaultGroup",
    "GroupKind",
    "PortKind",
    "SparseMatrix",
    "get_cmake_dir",
]
