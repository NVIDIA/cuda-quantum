# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

#[Begin State Batching]

import cudaq
import cupy as cp
import numpy as np
from cudaq import spin, Schedule, RungeKuttaIntegrator
# Set the target to our dynamics simulator
cudaq.set_target("dynamics")

# Qubit Hamiltonian
hamiltonian = 2 * np.pi * 0.1 * spin.x(0)

# Dimensions of sub-system. We only have a single degree of freedom of dimension 2 (two-level system).
dimensions = {0: 2}

# Initial states in the `SIC-POVM` set: https://en.wikipedia.org/wiki/SIC-POVM
psi_1 = cudaq.State.from_data(cp.array([1.0, 0.0], dtype=cp.complex128))
psi_2 = cudaq.State.from_data(
    cp.array([1.0 / np.sqrt(3.0), np.sqrt(2.0 / 3.0)], dtype=cp.complex128))
psi_3 = cudaq.State.from_data(
    cp.array([
        1.0 / np.sqrt(3.0),
        np.sqrt(2.0 / 3.0) * np.exp(1j * 2.0 * np.pi / 3.0)
    ],
             dtype=cp.complex128))
psi_4 = cudaq.State.from_data(
    cp.array([
        1.0 / np.sqrt(3.0),
        np.sqrt(2.0 / 3.0) * np.exp(1j * 4.0 * np.pi / 3.0)
    ],
             dtype=cp.complex128))

# We run the evolution for all the SIC state to determine the process tomography.
sic_states = [psi_1, psi_2, psi_3, psi_4]
# Schedule of time steps.
steps = np.linspace(0, 10, 101)
schedule = Schedule(steps, ["time"])

# Run the batch simulation.
evolution_results = cudaq.evolve(
    hamiltonian,
    dimensions,
    schedule,
    sic_states,
    observables=[spin.x(0), spin.y(0), spin.z(0)],
    collapse_operators=[],
    store_intermediate_results=cudaq.IntermediateResultSave.EXPECTATION_VALUE,
    integrator=RungeKuttaIntegrator())

#[End State Batching]
