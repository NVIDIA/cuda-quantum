# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Estimate Gidney--Ekerå RSA resources from one CUDA-Q kernel.

The kernel is sized by one row of arXiv:1905.09749 Table 3 (RSA-2048 by
default).

``estimate_using_kernel_profile()`` compiles only to portable logical
resources, then uses the paper's closed-form architecture equations. It is the
fast default.

``estimate_physically()`` builds a complete logical/QEC/physical target and
compiles the application through its native physical schedule. Its metrics
come from the resulting schedule annotations, so it is the slower, explicit
opt-in path. The companion module characterizes the workload-independent
AutoCCZ factory while constructing that target's device.

Run this file with ``--physical`` to select the paper-scale physical path from
the command line. Without that flag, it uses the fast kernel-profile path.
``--bits`` selects the modulus size.
"""

# %%
# Import CLI support, CUDA-Q, the result types, and the factory definitions.
from __future__ import annotations

import argparse
from dataclasses import dataclass

import cudaq
import cudaq.logical as cql
import gidney_ekera_factory as factory

# %%
# Define the compact results returned by the two estimation methods.
APPLICATION_FAILURE_BUDGET = 0.8


@dataclass(frozen=True, slots=True)
class AnalyticalResult:
    logical_qubits: int
    logical_toffolis: int
    folded_lookups: int
    physical_qubits: int
    makespan_ns: float

    @property
    def runtime_hours(self) -> float:
        return self.makespan_ns / 3.6e12


@dataclass(frozen=True, slots=True)
class PhysicalResult:
    physical_qubits: int
    scheduled_events: int
    makespan_ns: float

    @property
    def runtime_hours(self) -> float:
        return self.makespan_ns / 3.6e12


# %%
# Define the CUDA-Q arithmetic helpers used by the folded RSA workload.
@cudaq.kernel
def toffoli(
    control_a: cudaq.qubit,
    control_b: cudaq.qubit,
    target: cudaq.qubit,
):
    x.ctrl([control_a, control_b], target)


@cudaq.kernel
def maj(carry: cudaq.qubit, addend: cudaq.qubit, accumulator: cudaq.qubit):
    x.ctrl(accumulator, addend)
    x.ctrl(accumulator, carry)
    toffoli(carry, addend, accumulator)


@cudaq.kernel
def uma(carry: cudaq.qubit, addend: cudaq.qubit, accumulator: cudaq.qubit):
    toffoli(carry, addend, accumulator)
    x.ctrl(accumulator, carry)
    x.ctrl(carry, addend)


# %%
# Define one table-access step of the lookup.
@cudaq.kernel
def qrom_access_step(
    address: cudaq.qubit,
    workspace_a: cudaq.qubit,
    workspace_b: cudaq.qubit,
    target: cudaq.qubit,
):
    """One unary-iteration reaction followed by an external row access.

    The helper boundary is ordinary CUDA-Q.  After QEC selection its exact owner graph has
    one selected AutoCCZ application connected by CX to a fourth owner; the
    surface-code physical provider can therefore derive one alternating access layer
    without trusting this function's name or attaching an arithmetic role.
    """

    toffoli(address, workspace_a, workspace_b)
    x.ctrl(workspace_b, target)


# %%
# Build the folded RSA kernel for one operating point.
def build_resource_kernel(point: factory.OperatingPoint):
    pieces = point.carry_pieces
    piece_length = point.piece_length
    width = point.accumulator_width
    address_width = point.address_width
    table_rows = point.table_rows
    fixup_count = point.fixup_count
    lookup_count = point.lookup_count

    @cudaq.kernel
    def lookup_addition(
        accumulator: cudaq.qview,
        address: cudaq.qview,
        bus: cudaq.qview,
        runways: cudaq.qview,
        ancillas: cudaq.qview,
        unlookup_access: cudaq.qubit,
    ):
        for table_index in range(table_rows):
            qrom_access_step(
                address[table_index % address_width],
                ancillas[0],
                ancillas[1],
                bus[table_index % width],
            )

        # The carry pieces are independent, so sweep them in lockstep.
        for piece in range(pieces):
            maj(
                runways[piece],
                bus[piece * piece_length],
                accumulator[piece * piece_length],
            )
        for offset in range(piece_length - 1):
            for piece in range(pieces):
                bit = piece * piece_length + offset + 1
                maj(accumulator[bit - 1], bus[bit], accumulator[bit])
        for offset in range(piece_length - 1):
            for piece in range(pieces):
                bit = piece * piece_length + piece_length - 1 - offset
                uma(accumulator[bit - 1], bus[bit], accumulator[bit])

        # Measurement-based unlookup. The bus persists across iterations.
        for fixup in range(fixup_count):
            qrom_access_step(
                address[fixup % address_width],
                ancillas[0],
                ancillas[1],
                unlookup_access,
            )

    @cudaq.kernel
    def rsa_resource_kernel():
        accumulator = cudaq.qvector(width)
        # Allocate the workspaces once so setup is not charged per lookup.
        address = cudaq.qvector(address_width)
        bus = cudaq.qvector(width)
        runways = cudaq.qvector(pieces)
        ancillas = cudaq.qvector(2)
        unlookup_access = cudaq.qubit()
        h(address)
        for _ in range(lookup_count):
            lookup_addition(
                accumulator,
                address,
                bus,
                runways,
                ancillas,
                unlookup_access,
            )
        for bit in range(width):
            mx(bus[bit])
        for bit in range(address_width):
            mx(address[bit])
        for bit in range(pieces):
            mz(runways[bit])
        for bit in range(2):
            mz(ancillas[bit])
        mz(unlookup_access)
        mx(accumulator[0])

    return rsa_resource_kernel


# %%
# Project compiler-counted logical resources through the paper's closed-form
# timing and layout equations; this path does not construct or schedule P3.
def calculate_analytical_metrics(
        logical,
        point: factory.OperatingPoint = factory.RSA_2048) -> AnalyticalResult:
    toffolis = logical.synthesis_demand["qlx_standard_ccx"]
    lookups, remainder = divmod(toffolis, point.toffolis_per_lookup)
    assert remainder == 0, "logical Toffoli demand is not a whole lookup count"

    timing = factory.surface_timing(point)
    cycle_ns = timing["surface_cycle_ns"]
    bank_interval_ns = (point.factory_output_interval_cycles * cycle_ns /
                        point.factory_lanes)
    qrom_step_ns = max(
        point.level_2_code_distance * cycle_ns / 2.0,
        bank_interval_ns,
    )
    addition_step_ns = max(
        timing["reaction_time_ns"],
        point.carry_pieces * bank_interval_ns,
    )
    lookup_period_ns = (
        (point.table_rows + point.fixup_count) * qrom_step_ns +
        (point.piece_length + point.piece_length - 1) * addition_step_ns)
    final_measure_ns = timing.by_code_distance[
        point.level_2_code_distance]["measure_x_instrument_ns"]
    makespan_ns = (point.factory_startup_cycles * cycle_ns +
                   lookups * lookup_period_ns + final_measure_ns)
    patch_footprint = factory.factory_for(
        point).surface.square_patch_footprint.units
    physical_qubits = (
        (point.board_patches - point.factory_patches) * patch_footprint +
        point.factory_lanes * point.factory_lane_physical_qubits)
    return AnalyticalResult(
        logical_qubits=logical.logical_qubits_peak,
        logical_toffolis=toffolis,
        folded_lookups=lookups,
        physical_qubits=physical_qubits,
        makespan_ns=makespan_ns,
    )


# %%
# Profile the kernel's portable logical resources, then apply the paper model.
def estimate_using_kernel_profile(modulus_bits: int = 2048) -> AnalyticalResult:
    """Estimate from logical counts plus explicit paper architecture formulas.

    This is not CUDA-Q Logical's Tier.ANALYTICAL estimator: the selected target
    stops at logical resources, and ``calculate_analytical_metrics`` supplies
    the Gidney--Ekerå-specific physical projection.
    """

    point = factory.OPERATING_POINTS[modulus_bits]
    cudaq.set_target(cql.targets.estimator)
    estimate = cudaq.estimate(build_resource_kernel(point))
    logical = cql.estimate.LogicalEstimate.from_annotations(
        estimate.annotations)
    result = calculate_analytical_metrics(logical, point)

    print(f"Gidney--Ekerå RSA-{modulus_bits} analytical estimate:")
    print(f"  folded lookup additions: {result.folded_lookups:,}")
    print(f"  logical Toffolis: {result.logical_toffolis:,}")
    print(f"  peak logical qubits: {result.logical_qubits:,}")
    print(f"  physical qubits: {result.physical_qubits:,}")
    print(f"  single-shot makespan: {result.runtime_hours:.6f} h")
    return result


# %%
# Build the physical target. Its operating point owns physical error, scaling,
# and timing; the failure budget and the SCHEDULE tier are estimate policy.
def build_physical_target(point: factory.OperatingPoint = factory.RSA_2048):
    device = factory.build_paper_device(
        point,
        p_phys=factory.PHYSICAL_ERROR_RATE,
        scaling=factory.DEFAULT_SCALING,
    )
    target = cql.targets.Target.from_device(
        "gidney_ekera_physical",
        device,
        runtime_backend=cql.targets.estimator,
        estimate_options={
            "failure_budget": APPLICATION_FAILURE_BUDGET,
            "tier": "SCHEDULE",
        },
        source_modules=(factory.__name__,),
    )
    return target


# %%
# Compile through the full device stack and read all physical metrics from the
# authenticated P3 schedule returned in the estimate annotations.
def estimate_physically(modulus_bits: int = 2048) -> PhysicalResult:
    """Estimate by physically compiling and scheduling the RSA application."""

    point = factory.OPERATING_POINTS[modulus_bits]
    cudaq.set_target(build_physical_target(point))
    estimate = cudaq.estimate(build_resource_kernel(point))
    schedule = estimate.annotations["SCHEDULE"]
    result = PhysicalResult(
        physical_qubits=schedule["physical_qubits"],
        scheduled_events=schedule["event_count"],
        makespan_ns=schedule["makespan_ns"],
    )

    print(f"Gidney--Ekerå RSA-{modulus_bits} physical schedule estimate:")
    print(f"  application events: {result.scheduled_events:,}")
    print(f"  physical qubits: {result.physical_qubits:,}")
    print(f"  single-shot makespan: {result.runtime_hours:.6f} h")
    return result


# %%
# Run one of the two estimation methods when invoked from the command line.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate resources for the folded RSA kernel.")
    parser.add_argument(
        "--bits",
        type=int,
        default=2048,
        choices=sorted(factory.OPERATING_POINTS),
        help="RSA modulus size; selects the paper's Table 3 operating point",
    )
    parser.add_argument(
        "--physical",
        action="store_true",
        help="compile and schedule the paper-scale physical estimate",
    )
    args = parser.parse_args()
    if args.physical:
        result = estimate_physically(args.bits)
    else:
        result = estimate_using_kernel_profile(args.bits)
