# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Provenance values for evidence-bearing claims.

These small frozen values name where a claim came from. They serialize by
``str()`` into evidence metadata; they never assert correctness by
themselves — a citation records origin, not proof status.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Provenance:
    kind: str
    reference: str

    def __str__(self) -> str:
        return f"{self.kind}:{self.reference}"


def citation(reference: str) -> Provenance:
    """A published source (paper, book, standard) backing a claim."""
    return Provenance("citation", str(reference))


def user_assertion(note: str = "user assertion") -> Provenance:
    """An explicit, unproved user statement."""
    return Provenance("user_assertion", str(note))


def report(reference: str) -> Provenance:
    """An internal analysis or experiment report backing a claim."""
    return Provenance("report", str(reference))


def computation(reference: str) -> Provenance:
    """A recorded computation (script, notebook, tool run) backing a claim."""
    return Provenance("computation", str(reference))


__all__ = ["Provenance", "citation", "user_assertion", "report", "computation"]
