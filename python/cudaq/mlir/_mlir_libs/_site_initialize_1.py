# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Dialect registration for downstream CUDA-Q extensions.

Packages that define out-of-tree MLIR dialects advertise a callable in the
``cudaq.mlir_dialects`` entry point group. Each callable is passed the
DialectRegistry that every CUDA-Q MLIR Context is seeded from.
"""

from importlib.metadata import entry_points


def register_dialects(registry):
    for ep in entry_points(group="cudaq.mlir_dialects"):
        ep.load()(registry)
