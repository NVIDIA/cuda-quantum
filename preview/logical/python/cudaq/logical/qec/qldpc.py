# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""qLDPC implementation library: toric and bivariate-bicycle codes.

Binding this module links the high-rate stack: the toric family (two logicals
per block), the bivariate-bicycle constructor, encoded preparation, memory
rounds, and per-port destructive readout for the common toric[3] instance.
"""

from __future__ import annotations

from ..codes import Toric, bivariate_bicycle  # noqa: F401
from ..gadgets import (
    css_memory_round,
    logical_measure,
    prepare_plus,
    prepare_zero,
)

toric_3 = Toric[3]

prepare_zero_toric_3 = prepare_zero(toric_3)
prepare_plus_toric_3 = prepare_plus(toric_3)
memory_round_toric_3 = css_memory_round(toric_3)
logical_z0_toric_3 = logical_measure(toric_3, basis="z", logical=0)
logical_z1_toric_3 = logical_measure(toric_3, basis="z", logical=1)
logical_x0_toric_3 = logical_measure(toric_3, basis="x", logical=0)
logical_x1_toric_3 = logical_measure(toric_3, basis="x", logical=1)

__all__ = [
    "Toric",
    "bivariate_bicycle",
    "toric_3",
    "prepare_zero_toric_3",
    "prepare_plus_toric_3",
    "memory_round_toric_3",
    "logical_z0_toric_3",
    "logical_z1_toric_3",
    "logical_x0_toric_3",
    "logical_x1_toric_3",
    "from_metachecks",
    "css_memory_round",
    "logical_measure",
]
