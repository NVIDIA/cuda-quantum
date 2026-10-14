# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Reusable physical-architecture constructors for common device shapes.

Each constructor returns an ordinary :class:`cudaq.logical.PhysicalMachine` —
resource classes, topology, and native operations — with no machine-space
bindings. A :class:`cudaq.logical.DeviceBuilder` binds P2 QEC regions to these physical
resources explicitly. Heterogeneous and multi-topology devices construct
:class:`cudaq.logical.PhysicalMachine` directly.
"""

from __future__ import annotations

from . import physical_actions as _actions
from cudaq.logical.architecture.physical_definition import (
    PhysicalMachine,
    ResourceClass,
    Topology,
)
from .physical_instruments import MPP as _MPP
from cudaq.logical.devices.builder import (
    CarrierSelection,
    DeviceBuilder,
    LogicalRegionBuilder,
    PhysicalResourceBuilder,
    QECChannelBuilder,
    QECRegionBuilder,
)
from cudaq.logical.devices.definition import (
    Device,
    LogicalToQECBinding,
    PhysicalOperatingPoint,
    QECChannelPort,
    QECChannelRealization,
    QECChannelToPhysicalBinding,
    QECMachine,
    QECRegion,
    QECToPhysicalBinding,
)
from cudaq.logical.architecture.logical import (
    CapabilityKey,
    Channel,
    Direction,
    LogicalMachine,
    Space,
    SpaceSlot,
    Stream,
    capability,
    channel,
    machine,
    region,
    stream,
)
from cudaq.logical.architecture.physical_definition import (
    Basis,
    NativeActionDecomposition,
    NativeActionStep,
    PatchKind,
    PatchTopology,
    PhysicalAction,
    PhysicalDefinition,
    PhysicalInstrument,
    physical,
    physical_qubit,
)
from cudaq.logical.architecture.capabilities import (
    PhysicalCapability,
    PhysicalCapabilityBinding,
)
from cudaq.logical.architecture.constraints import (
    DistributedPlacement,
    LocalPlacement,
    PlacementBinding,
    PlacementWitness,
    TopologicalPlacement,
    TrajectoryPlacement,
    allow_spaces,
    colocate,
    distributed,
    lifecycle,
    local,
    metric,
    prefer,
    require_capability,
    topological_record,
    trajectory,
)


def superconducting_grid(
        *,
        rows: int,
        columns: int,
        native_actions=None,
        native_instruments=(_MPP,),
        name: str | None = None,
) -> PhysicalMachine:
    """A homogeneous transmon-style grid of ``rows x columns`` qubits."""

    qubits = ResourceClass(
        "qubit",
        rows * columns,
        native_actions=tuple(native_actions or
                             _actions.superconducting_clifford_set()),
        native_instruments=tuple(native_instruments),
    )
    return PhysicalMachine(
        name or f"superconducting_grid_{rows}x{columns}",
        resource_classes={"qubits": qubits},
        topologies={"grid": Topology.grid(rows, columns)},
    )


def qccd(
        *,
        storage_ions: int,
        gate_ions: int,
        native_actions=None,
        native_instruments=(_MPP,),
        name: str | None = None,
) -> PhysicalMachine:
    """A zoned trapped-ion QCCD with storage and gate regions."""

    actions = tuple(native_actions or _actions.qccd_native_set())
    storage = ResourceClass(
        "ion",
        storage_ions,
        native_actions=actions,
        native_instruments=tuple(native_instruments),
    )
    gate = ResourceClass(
        "ion",
        gate_ions,
        native_actions=actions,
        native_instruments=tuple(native_instruments),
    )
    return PhysicalMachine(
        name or f"qccd_{storage_ions}s_{gate_ions}g",
        resource_classes={
            "storage_ions": storage,
            "gate_ions": gate
        },
        topologies={
            "zones": Topology("zoned", storage=storage_ions, gate=gate_ions)
        },
    )


def neutral_atom_array(
        *,
        reservoir_atoms: int,
        interaction_atoms: int,
        readout_atoms: int,
        native_actions=None,
        native_instruments=(),
        name: str | None = None,
) -> PhysicalMachine:
    """A zoned neutral-atom array: reservoir, interaction, readout."""

    actions = tuple(native_actions or _actions.neutral_atom_set())
    decompositions = (_actions.neutral_atom_decompositions()
                      if native_actions is None else ())
    classes = {
        "reservoir_atoms":
            ResourceClass(
                "atom",
                reservoir_atoms,
                native_actions=actions,
                native_action_decompositions=decompositions,
                native_instruments=tuple(native_instruments),
            ),
        "interaction_atoms":
            ResourceClass(
                "atom",
                interaction_atoms,
                native_actions=actions,
                native_action_decompositions=decompositions,
                native_instruments=tuple(native_instruments),
            ),
        "readout_atoms":
            ResourceClass(
                "atom",
                readout_atoms,
                native_actions=actions,
                native_action_decompositions=decompositions,
                native_instruments=tuple(native_instruments),
            ),
    }
    return PhysicalMachine(
        name or
        f"neutral_atom_array_{reservoir_atoms}_{interaction_atoms}_{readout_atoms}",
        resource_classes=classes,
        topologies={
            "zones":
                Topology(
                    "zoned",
                    reservoir=reservoir_atoms,
                    interaction=interaction_atoms,
                    readout=readout_atoms,
                )
        },
    )


def heterogeneous_qec(
        *,
        memory_qubits: int = 4_096,
        compute_qubits: int = 2_048,
        factory_qubits: int = 1_024,
        native_actions=None,
        native_instruments=(_MPP,),
        name: str | None = None,
) -> PhysicalMachine:
    """A heterogeneous stack: dense memory, surface-like compute, factories.

    Three qubit classes with independent counts and one grid topology per
    class. This is deliberately a *shape* — codes never live in the
    architecture; QEC bindings stay on the device's logical spaces.
    """

    actions = tuple(native_actions or _actions.superconducting_clifford_set())
    instruments = tuple(native_instruments)

    def qubit_class(count):
        return ResourceClass(
            "qubit",
            count,
            native_actions=actions,
            native_instruments=instruments,
        )

    return PhysicalMachine(
        name or "heterogeneous_qec",
        resource_classes={
            "memory_qubits": qubit_class(memory_qubits),
            "compute_qubits": qubit_class(compute_qubits),
            "factory_qubits": qubit_class(factory_qubits),
        },
        topologies={
            "memory_grid": Topology("grid", qubits=memory_qubits),
            "compute_grid": Topology("grid", qubits=compute_qubits),
            "factory_grid": Topology("grid", qubits=factory_qubits),
        },
    )


__all__ = [
    "superconducting_grid",
    "qccd",
    "neutral_atom_array",
    "heterogeneous_qec",
    "CapabilityKey",
    "PhysicalCapability",
    "PhysicalCapabilityBinding",
    "Channel",
    "Direction",
    "Device",
    "DeviceBuilder",
    "LogicalRegionBuilder",
    "QECRegion",
    "QECRegionBuilder",
    "QECChannelPort",
    "QECChannelRealization",
    "QECChannelBuilder",
    "QECChannelToPhysicalBinding",
    "QECMachine",
    "LogicalToQECBinding",
    "QECToPhysicalBinding",
    "PhysicalOperatingPoint",
    "Basis",
    "DistributedPlacement",
    "LocalPlacement",
    "LogicalMachine",
    "PhysicalMachine",
    "CarrierSelection",
    "PatchKind",
    "PatchTopology",
    "PhysicalAction",
    "NativeActionDecomposition",
    "NativeActionStep",
    "PhysicalInstrument",
    "PhysicalDefinition",
    "PhysicalResourceBuilder",
    "PlacementBinding",
    "PlacementWitness",
    "ResourceClass",
    "Space",
    "SpaceSlot",
    "Stream",
    "Topology",
    "TopologicalPlacement",
    "TrajectoryPlacement",
    "allow_spaces",
    "capability",
    "channel",
    "colocate",
    "distributed",
    "local",
    "lifecycle",
    "machine",
    "metric",
    "prefer",
    "topological_record",
    "trajectory",
    "require_capability",
    "stream",
    "region",
    "physical",
    "physical_qubit",
    "atoms",
    "geometry",
    "placement",
    "physical_actions",
    "physical_instruments",
    "topology",
]

from . import atoms, geometry, physical_actions, physical_instruments, placement, topology
