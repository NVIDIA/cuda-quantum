# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Reusable indexes for repeated inspection of immutable P3 IR types."""

from __future__ import annotations


class PhysicalStateTypeIndex:
    """Resolve ``!phys.state`` resource symbols once per distinct MLIR type.

    Equal MLIR type handles hash identically within one context. Scheduling,
    simulation and physical analyses can therefore share this index
    instead of printing the same physical state type for every operand and
    result in a large graph.
    """

    __slots__ = ("_resources",)

    def __init__(self) -> None:
        self._resources = {}

    def resource(self, type_) -> str | None:
        try:
            return self._resources[type_]
        except KeyError:
            text = str(type_)
            prefix = "!phys.state<@"
            resource = text[len(prefix):-1] if text.startswith(prefix) else None
            self._resources[type_] = resource
            return resource


__all__ = ["PhysicalStateTypeIndex"]

PhysicalStateTypeIndex.__module__ = "cudaq.logical.compiler.physical_index"
