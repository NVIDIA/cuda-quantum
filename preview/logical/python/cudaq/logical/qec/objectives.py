# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compatibility imports for QEC action declarations."""

from .actions import SubsystemFragmentAction, subsystem_fragment

SubsystemFragmentObjective = SubsystemFragmentAction

__all__ = ["SubsystemFragmentObjective", "subsystem_fragment"]
