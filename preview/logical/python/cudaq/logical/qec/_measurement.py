# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Shared typed result for generated joint-measurement realizations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cudaq.logical.codes import Code
from cudaq.logical.gadgets import (
    GadgetDefinition,
    GadgetProfile,
)
from cudaq.logical.algebra.pauli import PauliProduct


@dataclass(frozen=True, slots=True)
class JointMeasurement:
    """One typed Pauli-product realization and its exact construction evidence.

    ``product`` is the public logical specification. ``realization`` is an
    ordinary Mark III gadget definition suitable for linking and composition.
    Surface constructions additionally expose the concrete temporary
    ``auxiliary_code`` that a containing protocol must allocate explicitly.
    The private diagnostics object retains row-level research data without
    making packed symplectic rows part of the implementation-library API.
    """

    product: PauliProduct
    realization: GadgetDefinition
    analysis: GadgetProfile
    evidence: Any
    rounds: int
    data_code: Code
    auxiliary_code: Code | None = None
    _diagnostics: Any = field(default=None, repr=False, compare=False)

    @property
    def kappa_qubits(self) -> int:
        return self._diagnostics.kappa_count

    @property
    def merged_check_count(self) -> int:
        return len(self._diagnostics.merged_checks)

    @property
    def chi_check_count(self) -> int:
        return len(self._diagnostics.chi_checks)


def _measurement_profile(diagnostics, *, name: str) -> GadgetProfile:
    """Return the measurement gadget's detached P2 analysis profile."""

    return GadgetProfile(diagnostics.gadget, name=name)


__all__ = ["JointMeasurement"]
