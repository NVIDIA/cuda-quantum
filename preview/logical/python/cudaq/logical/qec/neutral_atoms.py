# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Neutral-atom architecture helpers.

Binding this module links the zoned/mobile atom vocabulary used by
neutral-atom devices: zones with roles and capacities, reusable layouts, the
generated-architecture constructor, and the movement/pulse helpers whose P3
projection is ordinary ``move``/``apply``/``measure`` events.
"""

from __future__ import annotations

from ..architecture.atoms import (  # noqa: F401  (re-exports)
    AtomLayout, GeminiFullLayout, GeminiLogicalLayout, Pulse, Role, Zone,
    architecture,
)

__all__ = [
    "AtomLayout",
    "GeminiFullLayout",
    "GeminiLogicalLayout",
    "Pulse",
    "Role",
    "Zone",
    "architecture",
]
