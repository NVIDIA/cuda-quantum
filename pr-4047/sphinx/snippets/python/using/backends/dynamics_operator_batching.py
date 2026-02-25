# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin Operator Batching]

import cudaq
import cupy as cp
import numpy as np
from cudaq import spin, Schedule, ScalarOperator, RungeKuttaIntegrator
# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Dimensions of sub-system.
dimensions = {0: 2}

# Qubit resonant frequency
omega_z = 10.0 * 2 * np.pi

# Transverse term
omega_x = 2 * np.pi

# Harmonic driving frequency (sweeping in the +/- 10% range around the resonant frequency).
omega_drive = np.linspace(0.1 * omega_z, 1.1 * omega_z, 16)

# Initial state of the system (ground state).
psi0 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))

# Batch the Hamiltonian operator together
hamiltonians = [
    0.5 * omega_z * spin.z(0) + omega_x *
    ScalarOperator(lambda t, omega=omega: np.cos(omega * t)) * spin.x(0)
    for omega in omega_drive
]

# Initial states for each Hamiltonian in the batch.
# Here, we use the ground state for all Hamiltonians.
initial_states = [psi0] * len(hamiltonians)

# Schedule of time steps.
steps = np.linspace(0, 0.5, 5000)
schedule = Schedule(steps, ["t"])

# Run the batch simulation.
evolution_results = cudaq.evolve(
    hamiltonians,
    dimensions,
    schedule,
    initial_states,
    observables=[spin.x(0), spin.y(0), spin.z(0)],
    collapse_operators=[],
    store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
    integrator=RungeKuttaIntegrator())

#[End Operator Batching]

#[Begin Batch Results]

# The results of the batched evolution is an array of evolution results,
# one for each Hamiltonian operator in the batch.

# For example, we can split the results into separate arrays for each observable.
all_exp_val_x = []
all_exp_val_y = []
all_exp_val_z = []
# Iterate over the evolution results in the batch:
for evolution_result in evolution_results:
    # Extract the expectation values for each observable at the respective Hamiltonian operator in the batch.
    exp_val_x = [
        exp_vals[0].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]
    exp_val_y = [
        exp_vals[1].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]
    exp_val_z = [
        exp_vals[2].expectation()
        for exp_vals in evolution_result.expectation_values()
    ]

    # Append the results to the respective lists.
    # These will be nested lists, where each inner list corresponds to the results for a specific Hamiltonian operator in the batch.
    all_exp_val_x.append(exp_val_x)
    all_exp_val_y.append(exp_val_y)
    all_exp_val_z.append(exp_val_z)
#[End Batch Results]

print(all_exp_val_x)
print(all_exp_val_y)
print(all_exp_val_z)

#[Begin Batch Size]
# Run the batch simulation with a maximum batch size of 2.
# This means that the evolution will be performed in batches of 2 Hamiltonian operators at a time, which can be useful for memory management or
# performance tuning.
results = cudaq.evolve(
    hamiltonians,
    dimensions,
    schedule,
    initial_states,
    observables=[spin.x(0), spin.y(0), spin.z(0)],
    collapse_operators=[],
    store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
    integrator=RungeKuttaIntegrator(),
    max_batch_size=2)  # Set the maximum batch size to 2
#[End Batch Size]
