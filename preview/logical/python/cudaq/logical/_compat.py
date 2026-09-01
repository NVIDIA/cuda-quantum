# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Minimal pickle identity support for relocated CUDA-Q Logical definitions."""


def preserve_legacy_module(namespace: dict, legacy_name: str) -> None:
    canonical_name = namespace["__name__"]
    for value in tuple(namespace.values()):
        if getattr(value, "__module__", None) == canonical_name:
            try:
                value.__module__ = legacy_name
            except (AttributeError, TypeError):
                pass


__all__ = ["preserve_legacy_module"]
