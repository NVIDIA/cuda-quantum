# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Estimate Gidney--Ekerå RSA-2048 resources from one CUDA-Q kernel.

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


@cudaq.kernel
def maj_pair(
    lower_carry: cudaq.qubit,
    lower_addend: cudaq.qubit,
    lower_accumulator: cudaq.qubit,
    upper_carry: cudaq.qubit,
    upper_addend: cudaq.qubit,
    upper_accumulator: cudaq.qubit,
):
    """Two independent carry-piece reactions exposed in one helper graph."""

    maj(lower_carry, lower_addend, lower_accumulator)
    maj(upper_carry, upper_addend, upper_accumulator)


@cudaq.kernel
def uma_pair(
    lower_carry: cudaq.qubit,
    lower_addend: cudaq.qubit,
    lower_accumulator: cudaq.qubit,
    upper_carry: cudaq.qubit,
    upper_addend: cudaq.qubit,
    upper_accumulator: cudaq.qubit,
):
    """Two independent unmajority reactions exposed in one helper graph."""

    uma(lower_carry, lower_addend, lower_accumulator)
    uma(upper_carry, upper_addend, upper_accumulator)


# %%
# Define one table-access step and one complete lookup addition.
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


@cudaq.kernel
def lookup_addition(
    accumulator: cudaq.qview,
    address: cudaq.qview,
    bus: cudaq.qview,
    runways: cudaq.qview,
    ancillas: cudaq.qview,
    unlookup_access: cudaq.qubit,
):
    for table_index in range(1023):
        qrom_access_step(
            address[table_index % 10],
            ancillas[0],
            ancillas[1],
            bus[table_index % 2124],
        )

    # The two carry-runway pieces are independent spatial sweeps.  Express
    # them in lockstep so a stable list/ASAP scheduler can expose the intended
    # two-piece concurrency without assigning a semantic "MAJ phase" or
    # "UMA phase" to either the device or the dialect.
    maj_pair(
        runways[0],
        bus[0],
        accumulator[0],
        runways[1],
        bus[1062],
        accumulator[1062],
    )
    for offset in range(1061):
        lower = offset + 1
        upper = offset + 1063
        maj_pair(
            accumulator[lower - 1],
            bus[lower],
            accumulator[lower],
            accumulator[upper - 1],
            bus[upper],
            accumulator[upper],
        )
    for offset in range(1061):
        lower = 1061 - offset
        upper = 2123 - offset
        uma_pair(
            accumulator[lower - 1],
            bus[lower],
            accumulator[lower],
            accumulator[upper - 1],
            bus[upper],
            accumulator[upper],
        )

    # The 64 alternating accesses are the retained physical recurrence for
    # measurement-based QROM uncomputation.  The persistent bus is not torn
    # down here: its final measurement belongs to the board lifetime, not to
    # every arithmetic iteration.
    for fixup in range(64):
        qrom_access_step(
            address[fixup % 10],
            ancillas[0],
            ancillas[1],
            unlookup_access,
        )


# %%
# Assemble the paper-scale folded arithmetic into one CUDA-Q kernel.
@cudaq.kernel
def rsa2048_resource_kernel():
    accumulator = cudaq.qvector(2124)
    # The paper's arithmetic board is persistent.  These workspaces are
    # prepared once, retained through every folded lookup addition, and
    # released only after the complete arithmetic recurrence.  Allocating them
    # inside lookup_addition would incorrectly charge board startup/cleanup to
    # all 505,965 iterations.
    address = cudaq.qvector(10)
    bus = cudaq.qvector(2124)
    runways = cudaq.qvector(2)
    ancillas = cudaq.qvector(2)
    unlookup_access = cudaq.qubit()
    h(address)
    for _ in range(505965):
        lookup_addition(
            accumulator,
            address,
            bus,
            runways,
            ancillas,
            unlookup_access,
        )
    for bit in range(2124):
        mx(bus[bit])
    for bit in range(10):
        mx(address[bit])
    for bit in range(2):
        mz(runways[bit])
    for bit in range(2):
        mz(ancillas[bit])
    mz(unlookup_access)
    mx(accumulator[0])


# %%
# Project compiler-counted logical resources through the paper's closed-form
# timing and layout equations; this path does not construct or schedule P3.
def calculate_analytical_metrics(logical) -> AnalyticalResult:
    toffolis = logical.synthesis_demand["qlx_standard_ccx"]
    lookups, remainder = divmod(toffolis, 5_333)
    assert remainder == 0, "logical Toffoli demand is not a whole lookup count"

    timing = factory.surface_timing()
    cycle_ns = timing["surface_cycle_ns"]
    bank_interval_ns = 135.0 * cycle_ns / factory.FACTORY_LANES
    qrom_step_ns = max(
        factory.CODE.d.conservative_value * cycle_ns / 2.0,
        bank_interval_ns,
    )
    addition_step_ns = max(
        timing["reaction_time_ns"],
        factory.CARRY_PIECES * bank_interval_ns,
    )
    lookup_period_ns = (
        (factory.TABLE_ROWS + factory.FIXUP_COUNT) * qrom_step_ns +
        (factory.PIECE_LENGTH + factory.PIECE_LENGTH - 1) * addition_step_ns)
    final_measure_ns = timing.by_code_distance[
        factory.LEVEL_2_CODE_DISTANCE]["measure_x_instrument_ns"]
    makespan_ns = 379.0 * cycle_ns + lookups * lookup_period_ns + final_measure_ns
    physical_qubits = ((factory.BOARD_PATCHES - factory.FACTORY_PATCHES) *
                       factory.PATCH_FOOTPRINT +
                       factory.FACTORY_LANES * 142_808)
    return AnalyticalResult(
        logical_qubits=logical.logical_qubits_peak,
        logical_toffolis=toffolis,
        folded_lookups=lookups,
        physical_qubits=physical_qubits,
        makespan_ns=makespan_ns,
    )


# %%
# Profile the kernel's portable logical resources, then apply the paper model.
def estimate_using_kernel_profile() -> AnalyticalResult:
    """Estimate from logical counts plus explicit paper architecture formulas.

    This is not CUDA-Q Logical's Tier.ANALYTICAL estimator: the selected target
    stops at logical resources, and ``calculate_analytical_metrics`` supplies
    the Gidney--Ekerå-specific physical projection.
    """

    cudaq.set_target(cql.targets.estimator)
    estimate = cudaq.estimate(rsa2048_resource_kernel)
    logical = cql.estimate.LogicalEstimate.from_annotations(
        estimate.annotations)
    result = calculate_analytical_metrics(logical)

    print("Gidney--Ekerå analytical estimate:")
    print(f"  folded lookup additions: {result.folded_lookups:,}")
    print(f"  logical Toffolis: {result.logical_toffolis:,}")
    print(f"  peak logical qubits: {result.logical_qubits:,}")
    print(f"  physical qubits: {result.physical_qubits:,}")
    print(f"  single-shot makespan: {result.runtime_hours:.6f} h")
    return result


# %%
# Build the physical target. Its operating point owns physical error, scaling,
# and timing; only the application's failure budget is an estimate policy.
def build_physical_target():
    device = factory.build_paper_device(
        factory_lanes=factory.FACTORY_LANES,
        p_phys=factory.PHYSICAL_ERROR_RATE,
        scaling=factory.DEFAULT_SCALING,
    )
    target = cql.targets.Target.from_device(
        "gidney_ekera_physical",
        device,
        runtime_backend=cql.targets.estimator,
        estimate_options={"failure_budget": APPLICATION_FAILURE_BUDGET},
        source_modules=(factory.__name__,),
    )
    return target


# %%
# Compile through the full device stack and read all physical metrics from the
# authenticated P3 schedule returned in the estimate annotations.
def estimate_physically() -> PhysicalResult:
    """Estimate by physically compiling and scheduling the RSA application."""

    cudaq.set_target(build_physical_target())
    estimate = cudaq.estimate(rsa2048_resource_kernel)
    schedule = estimate.annotations["SCHEDULE"]
    result = PhysicalResult(
        physical_qubits=schedule["physical_qubits"],
        scheduled_events=schedule["event_count"],
        makespan_ns=schedule["makespan_ns"],
    )

    print("Gidney--Ekerå physical schedule estimate:")
    print(f"  application events: {result.scheduled_events:,}")
    print(f"  physical qubits: {result.physical_qubits:,}")
    print(f"  single-shot makespan: {result.runtime_hours:.6f} h")
    return result


# %%
# Run one of the two estimation methods when invoked from the command line.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate resources for the folded RSA-2048 kernel.")
    parser.add_argument(
        "--physical",
        action="store_true",
        help="compile and schedule the paper-scale physical estimate",
    )
    args = parser.parse_args()
    if args.physical:
        result = estimate_physically()
    else:
        result = estimate_using_kernel_profile()
