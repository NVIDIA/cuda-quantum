# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Layered devices, resources, regions, links, and reusable recipes."""

from cudaq.logical._core.lazy import public_dir as _public_dir, resolve as _resolve

_DEFINITION_NAMES = (
    "FactoryCharacterization",
    "FactoryModel",
    "LogicalToQECBinding",
    "QECChannelPort",
    "QECChannelRealization",
    "QECChannelToPhysicalBinding",
    "QECToPhysicalBinding",
    "QECArchitecture",
    "QECRegion",
    "QECMachine",
    "PhysicalOperatingPoint",
    "Device",
)
_COMPONENT_MODEL_NAMES = (
    "InitiationIntervalSemantics",
    "PhysicalResourceClaim",
    "SpacetimePhase",
    "SpacetimePlanCharacterization",
    "SpacetimePlanModel",
    "TransportCharacterization",
    "TransportModel",
)
_BUILDER_NAMES = (
    "PhysicalResourceBuilder",
    "LogicalRegionBuilder",
    "QECRegionBuilder",
    "QECChannelBuilder",
    "DeviceBuilder",
)
_RECIPE_NAMES = (
    "CarrierSelection",
    "DeviceRecipe",
    "FactoryBank",
    "Link",
    "PhysicalResource",
    "Region",
    "compose",
    "compute_factory",
    "compute_memory",
    "compute_memory_factory",
    "compute_only",
    "compute_region",
    "carriers",
    "factory",
    "factory_bank",
    "link",
    "memory_region",
    "qubits",
    "region",
    "resources",
)
_TIMING_NAMES = ("Duration", "TimingModel", "ms", "ns", "us")
_EXPORTS = {
    **{
        name: f"cudaq.logical.devices.definition:{name}" for name in _DEFINITION_NAMES
    },
    **{
        name: f"cudaq.logical.devices.component_models:{name}" for name in _COMPONENT_MODEL_NAMES
    },
    **{
        name: f"cudaq.logical.devices.builder:{name}" for name in _BUILDER_NAMES
    },
    **{
        name: f"cudaq.logical.devices.recipes:{name}" for name in _RECIPE_NAMES
    },
    **{
        name: f"cudaq.logical.devices.timing:{name}" for name in _TIMING_NAMES
    },
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    return _resolve(globals(), _EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
