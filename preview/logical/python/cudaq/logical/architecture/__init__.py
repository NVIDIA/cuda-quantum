# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Logical placement and QEC architecture."""

from .._core.lazy import public_dir as _public_dir, resolve as _resolve

_LOGICAL_NAMES = (
    "CapabilityKey",
    "LogicalValueRef",
    "LogicalMachine",
    "ProgramValueSchema",
    "Space",
    "SpaceSlot",
    "Stream",
    "capability",
    "machine",
    "region",
    "stream",
)
_CONSTRAINT_NAMES = (
    "AllowSpaces",
    "Colocate",
    "LocalPlacement",
    "PlacementBinding",
    "PlacementWitness",
    "Prefer",
    "RequireCapability",
    "allow_spaces",
    "colocate",
    "local",
    "lifecycle",
    "metric",
    "prefer",
    "require_capability",
)
_EXPORTS = {
    **{
        name: f".logical:{name}" for name in _LOGICAL_NAMES
    },
    **{
        name: f".constraints:{name}" for name in _CONSTRAINT_NAMES
    },
    "geometry": ".geometry",
    "placement": ".placement",
    "topology": ".topology",
}
_COMPAT_EXPORTS = {
    "DeviceBuilder": "..devices.builder:DeviceBuilder",
    "LogicalRegionBuilder": "..devices.builder:LogicalRegionBuilder",
    "QECRegionBuilder": "..devices.builder:QECRegionBuilder",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        return _resolve(globals(), _EXPORTS, name)
    except AttributeError:
        return _resolve(globals(), _COMPAT_EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
