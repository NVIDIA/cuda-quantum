# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Native Clifford realization projection, independent of EDEM storage."""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Literal, Mapping

from .build import Build
from .projection import (
    CompiledInterfaceManifest,
    ProjectedMeasurement,
    ProjectedPort,
)

CliffordKind = Literal["R", "RX", "H", "S", "S_DAG", "X", "Z", "CX", "CZ",
                       "SWAP", "M", "MX", "MPP"]


@dataclass(frozen=True, slots=True)
class CliffordInstruction:
    """One typed carrier instruction with a realization-local stable site.

    Binary gates contain consecutive control/target pairs in ``carriers``.
    MPP uses ordered ``(carrier, Pauli)`` factors in ``paulis`` and yields one
    bit. M and MX yield one bit per carrier starting at ``measurement_start``.
    Site IDs are stable for a fixed realization, not across compiler rewrites.
    """

    kind: CliffordKind
    carriers: tuple[int, ...]
    paulis: tuple[tuple[int, Literal["X", "Y", "Z"]], ...]
    inverted: bool
    measurement_start: int | None
    site_id: str


@dataclass(frozen=True, slots=True)
class CliffordProjection:
    """Read-only instructions and the exact encoded boundary/record manifest.

    Record aliases point to physical measurements; they introduce no new
    measurement bits. Input and output endpoints are ordered separately by
    GadgetSpec port ordinal, so preparation and destructive readout can have
    different boundaries.
    """

    instructions: tuple[CliffordInstruction, ...]
    interface: CompiledInterfaceManifest

    @property
    def record_indices(self) -> Mapping[str, int]:
        return MappingProxyType({
            alias: measurement.index
            for measurement in self.interface.measurements
            for alias in (measurement.record, *measurement.aliases)
        })


def project_clifford(source: Build,
                     *,
                     gadget: str | None = None) -> CliffordProjection:
    """Project one selected Fabric gadget through the native circuit traversal.

    ``source`` must contain a linked GadgetSpec, with stable record names on
    every measurement. The root gadget is selected by default; other roots
    require an explicit gadget symbol. Calls and static repeats are expanded.
    Unsupported operations, dynamic control, and non-Clifford gates fail with
    diagnostics. This operation does not modify the Build, insert noise, or
    prove detector-contract correctness.
    """

    if not isinstance(source, Build):
        raise TypeError(
            "project_clifford expects a cudaq.logical.compiler.Build")
    if gadget is None:
        if source.root.kind != "gadget":
            raise ValueError(
                "project_clifford requires gadget= for a non-gadget root")
        gadget = source.root.symbol
    if not isinstance(gadget, str) or not gadget:
        raise TypeError("gadget must be a nonempty Fabric gadget symbol")

    from cudaq.mlir._mlir_libs import _qlxRuntime

    # Use the authenticated snapshot, not the externally mutable inspection
    # clone exposed by Build.module.
    module = source._fresh_module()
    native = json.loads(_qlxRuntime.project_clifford_json(module, gadget))
    if native.get("schema") != "cudaq.logical.clifford-projection/v1":
        raise RuntimeError("unsupported native Clifford projection schema")

    def port(value):
        return ProjectedPort(
            value["port_ordinal"],
            value["name"],
            value["direction"],
            value["patch_id"],
            tuple((name, tuple(carriers))
                  for name, carriers in value["partitions"]),
        )

    interface = CompiledInterfaceManifest(
        inputs=tuple(port(value) for value in native["inputs"]),
        outputs=tuple(port(value) for value in native["outputs"]),
        measurements=tuple(
            ProjectedMeasurement(
                value["index"], value["record"], value["symbol"],
                tuple(value["call_path"]), value["field"], value["lane"],
                tuple(value["carriers"]), tuple(value["aliases"]))
            for value in native["measurements"]),
        carrier_count=native["carrier_count"],
    )
    instructions = tuple(
        CliffordInstruction(
            value["kind"],
            tuple(value["carriers"]),
            tuple(tuple(factor) for factor in value["paulis"]),
            value["inverted"],
            value["measurement_start"],
            value["site_id"],
        ) for value in native["instructions"])
    return CliffordProjection(instructions, interface)


__all__ = ["CliffordInstruction", "CliffordProjection", "project_clifford"]
