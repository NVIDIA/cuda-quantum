# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
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
