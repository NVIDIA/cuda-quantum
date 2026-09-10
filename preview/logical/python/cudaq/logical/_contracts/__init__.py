# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Dependency-neutral contracts shared by compilation and lowering."""

from .physical_index import PhysicalStateTypeIndex
from .projection import (
    CompiledInterfaceManifest,
    ProjectedMeasurement,
    ProjectedPort,
)

__all__ = [
    "PhysicalStateTypeIndex",
    "CompiledInterfaceManifest",
    "ProjectedMeasurement",
    "ProjectedPort",
]
