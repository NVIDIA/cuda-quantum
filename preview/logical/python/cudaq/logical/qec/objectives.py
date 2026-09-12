# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SubsystemFragmentObjective:
    """Request exact QEC resource-flow semantics derived from a gadget body.

    This objective is for realization-side fragments that act on the protected
    and gauge subsystems of one encoded block.  It is deliberately not logical
    identity: the compiler records every supported protected/gauge Pauli action
    and Pauli-product measurement, including which outcomes cross the gadget
    boundary.  Unsupported body operations fail rather than being hidden by a
    quotient over the gauge subsystem.
    """

    family: str = "subsystem_fragment"
    derivation: str = "body_exact"


# Exact fragment derivation is a stateless objective value, so one canonical
# instance is shared by normal authoring and the compatibility facade.
subsystem_fragment = SubsystemFragmentObjective()

__all__ = ["SubsystemFragmentObjective", "subsystem_fragment"]
