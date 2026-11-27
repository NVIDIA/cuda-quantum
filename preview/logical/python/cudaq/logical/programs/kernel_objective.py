# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compatibility imports for kernel-backed logical actions."""

from .kernel_action import (
    kernel_declaration_from_action,
    materialize_kernel_action,
)

kernel_declaration_from_objective = kernel_declaration_from_action
materialize_kernel_objective = materialize_kernel_action

__all__ = [
    "kernel_declaration_from_objective",
    "materialize_kernel_objective",
]
