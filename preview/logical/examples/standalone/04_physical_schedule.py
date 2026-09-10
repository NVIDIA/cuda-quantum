# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Compile a standalone logical program into a physical schedule."""

# %%
# Import the standalone compiler and surface-code recipe APIs.
import cudaq.logical as cql
from cudaq.logical.targets import recipes


# %%
# Author a logical memory experiment without a CUDA-Q kernel.
@cql.program
def three_round_memory() -> bool:
    qubit = cql.prepare_zero()
    qubit = cql.idle(qubit, rounds=3)
    return cql.measure_z(qubit)


# %%
# Build a complete encoded and physical surface-code device.
surface_architecture = recipes.surface_architecture(3)
surface_code = surface_architecture.code

builder = cql.devices.DeviceBuilder("StandaloneSurfaceCodeDevice")
compute = builder.logical.add_compute(capacity=1)
encoded_region = builder.qec.bind(compute, architecture=surface_architecture)
carriers = builder.physical.add_qubits(
    surface_code.block.size,
    native_actions=cql.architecture.physical_actions.clifford_set(),
    native_instruments=(
        cql.architecture.physical_instruments.MZ,
        cql.architecture.physical_instruments.MPP,
    ),
)
builder.physical.bind(encoded_region, to=carriers)
builder.physical.set_operating_point(timing={"cycle_ns": 1.0})
device = builder.build()

# %%
# Compile the logical program and place its owners on the device.
logical = cql.compile(three_round_memory)
placed = cql.compiler.place(logical, device=device)

# %%
# Select the QEC implementations and lower them to physical events.
encoded = cql.compile(
    placed,
    pipeline=cql.compiler.pipelines.qec(),
    device=device,
)
physical = cql.compile(
    encoded,
    pipeline=cql.compiler.pipelines.physical(),
    device=device,
)

# %%
# Schedule the physical events and calculate schedule-level resources.
schedule = cql.compiler.schedule(physical)
resources = cql.estimate(
    schedule,
    tier=cql.estimate.Tier.SCHEDULE,
    p_phys=1.0e-3,
    failure_budget=0.1,
    cycle_time=1.0e-9,
)

assert resources.physical_qubits == surface_code.block.size
assert resources.event_count == len(schedule.entries)
assert resources.makespan_ns == schedule.makespan_ns

print("Standalone physical schedule:")
print(f"  physical qubits: {resources.physical_qubits}")
print(f"  scheduled events: {resources.event_count}")
print(f"  makespan: {resources.makespan_ns:.1f} ns")
