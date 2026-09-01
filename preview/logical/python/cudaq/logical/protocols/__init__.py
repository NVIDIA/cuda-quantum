# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Protocol definitions, builders, and reusable resource protocols."""

from .._core.lazy import public_dir as _public_dir, resolve as _resolve

_LIBRARY_NAMES = (
    "DISTILL_15TO1_T",
    "FIFTEEN_TO_ONE_ROTATION_SUPPORTS",
    "FIFTEEN_TO_ONE_ROTATION_STEPS",
    "distill_15to1",
    "bare_measure_x",
    "bare_s",
)
_EXPORTS = {
    "ProtocolDefinition": ".definition:ProtocolDefinition",
    "protocol": ".definition:protocol",
    "ProtocolBuilder": ".builder:ProtocolBuilder",
    **{
        name: f".library:{name}" for name in _LIBRARY_NAMES
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
