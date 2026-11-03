# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Explore physical surface-code estimates with a configurable target."""

# %%
# Import CUDA-Q and the CUDA-Q Logical target and result APIs.
import cudaq
import cudaq.logical as cql


# %%
# Author a logical-zero memory kernel whose qubit demand is a parameter.
@cudaq.kernel
def logical_zero_memory(logical_qubits: int):
    qubits = cudaq.qvector(logical_qubits)
    mz(qubits)


# %%
# Establish a one-qubit, distance-three reference configuration.
baseline_qubits = 1
baseline_target = cql.targets.surface_physical_target(
    logical_capacity=baseline_qubits,
    distance=3,
)
cudaq.set_target(baseline_target)
baseline_target.print_stack()

baseline_estimate = cudaq.estimate(logical_zero_memory, baseline_qubits)
baseline_analytical = baseline_estimate.annotations["ANALYTICAL"]
baseline_schedule = baseline_estimate.annotations["SCHEDULE"]

# %%
# Increase the workload capacity and code distance in a second physical study.
scaled_qubits = 3
scaled_target = cql.targets.surface_physical_target(
    logical_capacity=scaled_qubits,
    distance=5,
)
cudaq.set_target(scaled_target)

scaled_estimate = cudaq.estimate(logical_zero_memory, scaled_qubits)
scaled_static = cql.estimate.FabricCounts.from_annotations(
    scaled_estimate.annotations)
scaled_analytical = scaled_estimate.annotations["ANALYTICAL"]
scaled_schedule = scaled_estimate.annotations["SCHEDULE"]

# %%
# Hold the layout fixed while changing the physical operating assumptions.
sensitivity_target = cql.targets.surface_physical_target(
    logical_capacity=scaled_qubits,
    distance=5,
    p_phys=1.0e-4,
    failure_budget=1.0e-6,
    cycle_time=5.0e-9,
)
cudaq.set_target(sensitivity_target)

sensitivity_estimate = cudaq.estimate(logical_zero_memory, scaled_qubits)
sensitivity_analytical = sensitivity_estimate.annotations["ANALYTICAL"]
sensitivity_schedule = sensitivity_estimate.annotations["SCHEDULE"]

# %%
# Verify which metrics change with layout and operating-point parameters.
assert set(scaled_estimate.annotations) == {
    "LOGICAL",
    "STATIC",
    "ANALYTICAL",
    "SCHEDULE",
}
assert baseline_schedule["physical_qubits"] == cql.codes.Surface[3].block.size
assert scaled_static.logical_qubits_peak == scaled_qubits
assert scaled_schedule["physical_qubits"] > baseline_schedule["physical_qubits"]
assert scaled_schedule["event_count"] > baseline_schedule["event_count"]
assert sensitivity_schedule["physical_qubits"] == scaled_schedule[
    "physical_qubits"]
assert sensitivity_schedule["event_count"] == scaled_schedule["event_count"]
assert sensitivity_schedule["makespan_ns"] > scaled_schedule["makespan_ns"]
assert sensitivity_analytical["logical_error"] < scaled_analytical[
    "logical_error"]
assert scaled_analytical["budget_met"]
assert not sensitivity_analytical["budget_met"]

print("Surface-code layout scaling:")
print("  baseline (1 logical qubit, distance 3):")
print(f"    physical qubits: {baseline_schedule['physical_qubits']}")
print(f"    scheduled events: {baseline_schedule['event_count']}")
print(f"    makespan: {baseline_schedule['makespan_ns']:.1f} ns")
print("  scaled memory (3 logical qubits, distance 5):")
print(f"    physical qubits: {scaled_schedule['physical_qubits']}")
print(f"    scheduled events: {scaled_schedule['event_count']}")
print(f"    makespan: {scaled_schedule['makespan_ns']:.1f} ns")

print("Operating-point sensitivity at 3 logical qubits and distance 5:")
print("  target defaults:")
print(f"    cycle time: {scaled_analytical['cycle_time'] * 1.0e9:g} ns")
print(f"    physical error rate: {scaled_analytical['p_phys']:.1e}")
print(f"    failure budget: "
      f"{scaled_analytical['failure_budget']['total']:.1e}")
print(f"    logical error: {scaled_analytical['logical_error']:.3e}")
print(f"    budget met: {scaled_analytical['budget_met']}")
print(f"    scheduled makespan: {scaled_schedule['makespan_ns']:.1f} ns")
print("  changed physical assumptions:")
print(f"    cycle time: "
      f"{sensitivity_analytical['cycle_time'] * 1.0e9:g} ns")
print(f"    physical error rate: {sensitivity_analytical['p_phys']:.1e}")
print(f"    failure budget: "
      f"{sensitivity_analytical['failure_budget']['total']:.1e}")
print(f"    logical error: {sensitivity_analytical['logical_error']:.3e}")
print(f"    budget met: {sensitivity_analytical['budget_met']}")
print(f"    scheduled makespan: "
      f"{sensitivity_schedule['makespan_ns']:.1f} ns")
