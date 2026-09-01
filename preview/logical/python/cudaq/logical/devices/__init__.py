# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Layered devices, resources, regions, and reusable local recipes."""

from .._core.lazy import public_dir as _public_dir, resolve as _resolve

_DEFINITION_NAMES = (
    "LogicalToQECBinding",
    "QECArchitecture",
    "QECRegion",
    "QECMachine",
    "Device",
)
_BUILDER_NAMES = (
    "LogicalRegionBuilder",
    "QECRegionBuilder",
    "DeviceBuilder",
)
_EXPORTS = {
    **{
        name: f".definition:{name}" for name in _DEFINITION_NAMES
    },
    **{
        name: f".builder:{name}" for name in _BUILDER_NAMES
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
