# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Protocol definitions, builders, and reusable resource protocols."""

from cudaq.logical._core.lazy import public_dir as _public_dir, resolve as _resolve

_LIBRARY_NAMES = (
    "ProductionModel",
    "CCZProductionModel",
    "DISTILL_15TO1_T",
    "CCZ_8TO1_CHECK_SUPPORTS",
    "CCZ_8TO1_INJECTION_TARGETS",
    "CCZ_8TO1_OUTPUT_CORRECTION_MASKS",
    "CCZ_8TO1_SYNDROME_OUTPUTS",
    "FIFTEEN_TO_ONE_ROTATION_SUPPORTS",
    "FIFTEEN_TO_ONE_ROTATION_STEPS",
    "DISTILL_5TO1_T",
    "CCZ_GIDNEY_FOWLER",
    "per_unit_cell_error",
    "ccz_per_state_error",
    "ccz_gidney_fowler_factory",
    "distill_15to1",
    "bare_measure_x",
    "bare_s",
    "distill_5to1",
    "ccz_gidney_fowler",
    "steane_teleportation_measurement_intent",
    "steane_teleportation_measurement",
    "steane_logical_s",
    "steane_t_injection",
    "COLOR_3",
    "COLOR_5",
    "COLOR_3_CARRIERS",
    "COLOR_5_CARRIERS",
    "color_3",
    "color_5",
    "color_3_to_5",
    "cultivate_color_3_to_5",
    "grow_color_3_to_5",
    "grow_color_3_to_5_profile",
    "CULTIVATED_MATCHABLE_D6",
    "cultivate_t_d3_to_matchable_d6",
    "cultivated_matchable_d6",
    "prepare_cultivated_t",
)
_EXPORTS = {
    "ProtocolDefinition":
        "cudaq.logical.protocols.definition:ProtocolDefinition",
    "protocol":
        "cudaq.logical.protocols.definition:protocol",
    "ProtocolBuilder":
        "cudaq.logical.protocols.builder:ProtocolBuilder",
    **{
        name: f"cudaq.logical.protocols.library:{name}" for name in _LIBRARY_NAMES
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
