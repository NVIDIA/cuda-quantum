# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Minimal out-of-tree CUDA-Q MLIR extension example package."""


def register_dialects(registry):
    """Register the example ``trivial`` dialect with a CUDA-Q MLIR registry."""
    from cudaq_mlir_extension.mlir._mlir_libs import _mlirExtension

    _mlirExtension.register_dialects(registry)
