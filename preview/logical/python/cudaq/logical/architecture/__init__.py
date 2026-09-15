# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Logical placement and physical machine architecture."""

from cudaq.logical._core.lazy import public_dir as _public_dir, resolve as _resolve

_LOGICAL_NAMES = (
    "CapabilityKey",
    "Channel",
    "Direction",
    "LogicalValueRef",
    "LogicalMachine",
    "ProgramValueSchema",
    "Space",
    "SpaceSlot",
    "Stream",
    "capability",
    "channel",
    "machine",
    "region",
    "stream",
)
_CONSTRAINT_NAMES = (
    "AllowSpaces",
    "Colocate",
    "DistributedPlacement",
    "LocalPlacement",
    "PlacementBinding",
    "PlacementWitness",
    "Prefer",
    "TopologicalPlacement",
    "TrajectoryPlacement",
    "RequireCapability",
    "allow_spaces",
    "colocate",
    "distributed",
    "local",
    "lifecycle",
    "metric",
    "prefer",
    "topological_record",
    "trajectory",
    "require_capability",
)
_CAPABILITY_NAMES = (
    "HeraldedErasure",
    "PhysicalCapability",
    "PhysicalCapabilityBinding",
    "NATIVE_PAULI_PRODUCT_ROTATION",
)
_PHYSICAL_NAMES = (
    "ResourceClass",
    "ResourceGranularity",
    "QuantumProcess",
    "PhysicalFootprint",
    "Basis",
    "PhysicalAction",
    "NativeActionDecomposition",
    "NativeActionStep",
    "PhysicalInstrument",
    "PhysicalDefinition",
    "Topology",
    "PatchKind",
    "PatchTopology",
    "PhysicalMachine",
    "physical",
    "physical_qubit",
)
_RECIPE_NAMES = (
    "superconducting_grid",
    "qccd",
    "neutral_atom_array",
    "heterogeneous_qec",
)
_EXPORTS = {
    **{
        name: f"cudaq.logical.architecture.logical:{name}" for name in _LOGICAL_NAMES
    },
    **{
        name: f"cudaq.logical.architecture.constraints:{name}" for name in _CONSTRAINT_NAMES
    },
    **{
        name: f"cudaq.logical.architecture.capabilities:{name}" for name in _CAPABILITY_NAMES
    },
    **{
        name: f"cudaq.logical.architecture.physical_definition:{name}" for name in _PHYSICAL_NAMES
    },
    **{
        name: f"cudaq.logical.architecture.recipes:{name}" for name in _RECIPE_NAMES
    },
    "PhysicalBuilder": "cudaq.logical.architecture.builder:PhysicalBuilder",
    "atoms": "cudaq.logical.architecture.atoms",
    "geometry": "cudaq.logical.architecture.geometry",
    "placement": "cudaq.logical.architecture.placement",
    "physical_actions": "cudaq.logical.architecture.physical_actions",
    "physical_instruments": "cudaq.logical.architecture.physical_instruments",
    "topology": "cudaq.logical.architecture.topology",
}
_COMPAT_EXPORTS = {
    "DeviceBuilder":
        "cudaq.logical.devices.builder:DeviceBuilder",
    "LogicalRegionBuilder":
        "cudaq.logical.devices.builder:LogicalRegionBuilder",
    "QECRegionBuilder":
        "cudaq.logical.devices.builder:QECRegionBuilder",
    "QECChannelBuilder":
        "cudaq.logical.devices.builder:QECChannelBuilder",
    "PhysicalResourceBuilder":
        "cudaq.logical.devices.builder:PhysicalResourceBuilder",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        return _resolve(globals(), _EXPORTS, name)
    except AttributeError:
        return _resolve(globals(), _COMPAT_EXPORTS, name)


def __dir__():
    return _public_dir(globals(), _EXPORTS)
