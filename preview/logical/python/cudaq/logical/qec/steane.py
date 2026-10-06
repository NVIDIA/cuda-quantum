# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Steane-code implementation library.

Binding this module links the ordinary Steane stack: encoded preparation,
memory rounds, destructive logical readout, the transversal Clifford
realizations (H and CX are carrierwise on the self-dual ``[[7,1,3]]`` code),
and the teleportation-based T-state injection protocol family.
"""

from __future__ import annotations

from ..codes import Steane
from cudaq.logical.ops._impl import cx as _cx
from cudaq.logical.ops._impl import h as _h
from ..gadgets import (
    css_memory_round,
    logical_measure,
    prepare_plus,
    prepare_zero,
)
from cudaq.logical.gadgets import gadget as _gadget
from cudaq.logical.gadgets import patch as _patch
from cudaq.logical.protocols.definition import protocol as _protocol
from ..protocols import (  # noqa: F401  (re-exports)
    distill_15to1, steane_logical_s, steane_t_injection,
    steane_teleportation_measurement,
)
from ..qec.synthesis import pygridsynth_rpp_compiler
from ..std import cx as _cx_objective
from ..std import h as _h_objective
from ..std import t as _t_objective

prepare_zero_gadget = prepare_zero(Steane)
prepare_plus_gadget = prepare_plus(Steane)
memory_round = css_memory_round(Steane)
logical_z = logical_measure(Steane, basis="z")
logical_x = logical_measure(Steane, basis="x")


def _transversal_h(block):
    return _h(block.data)


_transversal_h.__name__ = "steane_transversal_h"
_transversal_h.__qualname__ = _transversal_h.__name__
_transversal_h.__annotations__ = {
    "block": _patch[Steane],
    "return": _patch[Steane],
}
transversal_h = _gadget(_transversal_h,
                        implements=_h_objective,
                        name=_transversal_h.__name__)


def _transversal_cx(control, target):
    return _cx(control.data, target.data)


_transversal_cx.__name__ = "steane_transversal_cx"
_transversal_cx.__qualname__ = _transversal_cx.__name__
_transversal_cx.__annotations__ = {
    "control": _patch[Steane],
    "target": _patch[Steane],
    "return": tuple[_patch[Steane], _patch[Steane]],
}
transversal_cx = _gadget(_transversal_cx,
                         implements=_cx_objective,
                         name=_transversal_cx.__name__)


@_protocol(implements=_t_objective)
def steane_t_from_factory(block: _patch[Steane]) -> _patch[Steane]:
    """Produce and inject one exact logical T resource."""

    return steane_t_injection(block, distill_15to1())


# Importing this definition is the explicit link act for exact and approximate
# Steane Pauli-product rotations.  PyGridSynth itself remains optional until an
# off-lattice angle actually selects the synthesis path.
pygridsynth_rpp = pygridsynth_rpp_compiler(
    code=Steane,
    h=transversal_h,
    s=steane_logical_s,
    cx=transversal_cx,
    t=steane_t_from_factory,
    name="steane_pygridsynth_rpp",
)

__all__ = [
    "Steane",
    "prepare_zero_gadget",
    "prepare_plus_gadget",
    "memory_round",
    "logical_z",
    "logical_x",
    "transversal_h",
    "transversal_cx",
    "steane_t_from_factory",
    "pygridsynth_rpp",
    "steane_t_injection",
    "steane_logical_s",
    "steane_teleportation_measurement",
]
