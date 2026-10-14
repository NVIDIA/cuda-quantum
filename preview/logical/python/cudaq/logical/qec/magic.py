# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Magic-state production, cultivation, and injection library.

Binding this module links the non-Clifford resource stack: the concrete
five-qubit 15-to-1 distillation protocol, the typed 5-to-1 and Gidney–Fowler
CCZ resource protocols, magic-state cultivation, and the Steane injection
path. Analytical factory models ride along for estimation.
"""

from __future__ import annotations

from ..protocols import (  # noqa: F401  (re-exports)
    CCZ_GIDNEY_FOWLER, CCZProductionModel, DISTILL_15TO1_T, DISTILL_5TO1_T,
    ProductionModel, ccz_gidney_fowler, ccz_gidney_fowler_factory,
    ccz_per_state_error, cultivate_t_d3_to_matchable_d6,
    cultivated_matchable_d6, distill_15to1, distill_5to1, per_unit_cell_error,
    prepare_cultivated_t, steane_t_injection,
)
from ..std import CCZ_STATE, RAW_T_STATE, T_STATE  # noqa: F401
from .ccz import ccz_state_delivery, compiler as ccz_state_compiler

__all__ = [
    "T_STATE",
    "RAW_T_STATE",
    "CCZ_STATE",
    "distill_15to1",
    "distill_5to1",
    "ccz_gidney_fowler",
    "ccz_gidney_fowler_factory",
    "steane_t_injection",
    "prepare_cultivated_t",
    "cultivate_t_d3_to_matchable_d6",
    "cultivated_matchable_d6",
    "ProductionModel",
    "CCZProductionModel",
    "DISTILL_15TO1_T",
    "DISTILL_5TO1_T",
    "CCZ_GIDNEY_FOWLER",
    "per_unit_cell_error",
    "ccz_per_state_error",
    "ccz_state_delivery",
    "ccz_state_compiler",
]
