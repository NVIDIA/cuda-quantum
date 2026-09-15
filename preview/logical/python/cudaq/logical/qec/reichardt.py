# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Reichardt tesseract implementation library.

Binding this module links the parent ``[[16,6,4]]`` tesseract code represented
in QLX as a ``[[16,4,2,4]]`` subsystem encoding: four protected logical qubits,
two gauge logical qubits, and distance four. It also links the free-CNOT
code-automorphism gadget whose signed logical action the compiler derives and
verifies from the column permutation alone (arXiv:2412.14256).
"""

from __future__ import annotations

from ..codes import Tesseract
from cudaq.logical.programs.decorators import objective as _objective
from cudaq.logical.ops._impl import cx as _cx
from cudaq.logical.ops._impl import permute as _permute
from cudaq.logical.types.values import logical_qubit as _logical_qubit
from cudaq.logical.gadgets import gadget as _gadget
from cudaq.logical.gadgets import patch as _patch

# In the QLX [[16,4,2,4]] representation, four logical qubits become protected
# graph-state ports and two remain gauge workspace used by construction
# protocols.
path4 = Tesseract.encoding(
    name="tesseract_path4",
    logical_ports={
        "path0": 0,
        "path1": 1,
        "path2": 2,
        "path3": 3
    },
)

# Swapping the first and third column of each row preserves the stabilizer
# and gauge algebra; its induced protected action is two disconnected CNOTs.
column_swap = (
    2,
    1,
    0,
    3,
    6,
    5,
    4,
    7,
    10,
    9,
    8,
    11,
    14,
    13,
    12,
    15,
)


@_objective(name="tesseract_two_free_cnots")
def two_free_cnots(
    path0: _logical_qubit,
    path1: _logical_qubit,
    path2: _logical_qubit,
    path3: _logical_qubit,
) -> tuple[_logical_qubit, _logical_qubit, _logical_qubit, _logical_qubit]:
    path0, path1 = _cx(path0, path1)
    path2, path3 = _cx(path2, path3)
    return path0, path1, path2, path3


def _free_cnots(block):
    return _permute(block, column_swap)


_free_cnots.__name__ = "tesseract_free_cnots"
_free_cnots.__qualname__ = _free_cnots.__name__
_free_cnots.__annotations__ = {
    "block": _patch[path4],
    "return": _patch[path4],
}
free_cnots = _gadget(
    _free_cnots,
    implements=two_free_cnots,
    logical_ports={
        two_free_cnots.operands.path0: path4.ports.path0,
        two_free_cnots.operands.path1: path4.ports.path1,
        two_free_cnots.operands.path2: path4.ports.path2,
        two_free_cnots.operands.path3: path4.ports.path3,
    },
    name=_free_cnots.__name__,
)

__all__ = ["Tesseract", "path4", "column_swap", "two_free_cnots", "free_cnots"]
