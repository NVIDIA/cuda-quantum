# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared EDEM data structures and operations from cudaq-edem.

Early development: the API, ABI, and packed format are unstable and subject
to rapid, potentially incompatible changes during v0.x development.

The packed representation and C API allow compilers and runtime components
to exchange and evaluate EDEMs.
"""

from pathlib import Path

from . import _core
from ._core import (
    ABI_VERSION,
    FORMAT_VERSION,
    CodePresentation,
    DEM,
    EDEM,
    Endpoint,
    EndpointDirection,
    FaultDistribution,
    FaultGroup,
    GroupKind,
    MatrixBlock,
    PortKind,
    SparseMatrix,
    compose_chain,
    coalesce_dem_columns,
    factorize_dem,
    project_DEM,
)

__version__ = "@CUDAQ_EDEM_VERSION@"


def get_cmake_dir() -> str:
    """Return the installed cudaq-edem CMake package directory."""
    return str((Path(__file__).resolve().parent /
                "@CUDAQ_EDEM_PYTHON_CMAKE_DIR@").resolve())


__all__ = [
    "ABI_VERSION",
    "FORMAT_VERSION",
    "CodePresentation",
    "DEM",
    "EDEM",
    "Endpoint",
    "EndpointDirection",
    "FaultDistribution",
    "FaultGroup",
    "GroupKind",
    "MatrixBlock",
    "PortKind",
    "SparseMatrix",
    "compose_chain",
    "coalesce_dem_columns",
    "factorize_dem",
    "get_cmake_dir",
    "project_DEM",
]
