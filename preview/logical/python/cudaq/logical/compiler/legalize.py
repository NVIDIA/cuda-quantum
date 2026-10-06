# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Native-action decomposition library for ``phys-legalize-native-actions``.

After P2 projection emits semantic physical actions, a device may not support
one of them natively -- e.g. a superconducting machine whose two-qubit gate is
``CZ`` cannot run ``CX`` directly. Legalization rewrites the semantic action
into actions the bound resource classes *do* support, using this library, and
records the chosen rule and its cost as evidence. The premise is typed action
identity (membership in a resource class's ``native_actions``), never a target's
output spelling: ``CZ`` and a Rydberg-blockade gate both project to Stim ``CZ``
but are distinct actions, so a rule names the concrete native action it uses.

Each rule realizes a two-qubit ``action`` as single-qubit natives on the target
before and after a native two-qubit ``entangler``. ``CX = H(t); CZ(c, t); H(t)``
-- ``CX`` is ``CZ`` conjugated by a Hadamard on the target.
"""

from __future__ import annotations

from dataclasses import dataclass

from cudaq.logical.architecture.physical_definition import PhysicalAction
from ..architecture.physical_actions import CX, CZ, H


@dataclass(frozen=True)
class NativeDecomposition:
    """A rule realizing a semantic two-qubit action in native actions.

    ``pre_target`` / ``post_target`` are single-qubit native actions applied to
    the target carrier around a native two-qubit ``entangler`` on
    ``(control, target)``. Rules carry the exact typed semantic and native
    actions; lookup may begin from an IR symbol name, but applicability requires
    equality with the advertised ``PhysicalAction`` values.
    """

    name: str
    action: PhysicalAction
    pre_target: tuple[PhysicalAction, ...]
    entangler: PhysicalAction
    post_target: tuple[PhysicalAction, ...]

    @property
    def cost(self) -> int:
        """Emitted native-action count -- the decomposition's cost evidence."""
        return len(self.pre_target) + 1 + len(self.post_target)


# CX conjugated by a Hadamard on the target is CZ.
CX_VIA_CZ = NativeDecomposition(
    name="cx_via_cz",
    action=CX,
    pre_target=(H,),
    entangler=CZ,
    post_target=(H,),
)

NATIVE_DECOMPOSITIONS = (CX_VIA_CZ,)
SEMANTIC_ACTIONS = (CX, CZ)

__all__ = [
    "NativeDecomposition",
    "NATIVE_DECOMPOSITIONS",
    "SEMANTIC_ACTIONS",
    "CX_VIA_CZ",
]
